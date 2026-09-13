from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_lab.vjepa2_ac_replay import (
    cartesian_action_grid,
    checkpoint_provenance,
    load_trajectory,
    write_json_result,
)

_ROOT = Path(__file__).parents[1]
_SCRIPT = _ROOT / "scripts" / "official_vjepa2_ac_replay.py"


def test_checkpoint_requires_and_enforces_trusted_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ac.pt"
    checkpoint.write_bytes(b"small fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="no trusted SHA-256"):
        checkpoint_provenance(checkpoint)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checkpoint_provenance(checkpoint, expected_sha256="0" * 64)

    sidecar = checkpoint.with_suffix(".pt.json")
    sidecar.write_text(json.dumps({"sha256": digest, "url": "https://example.test/ac.pt"}))
    result = checkpoint_provenance(checkpoint)
    assert result["sha256"] == digest
    assert result["trust_source"] == "sidecar"
    assert result["source_url"] == "https://example.test/ac.pt"


def test_checkpoint_rejects_disagreeing_explicit_and_sidecar_digests(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ac.pt"
    checkpoint.write_bytes(b"small fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".pt.json").write_text(json.dumps({"sha256": digest}))

    with pytest.raises(ValueError, match="disagree"):
        checkpoint_provenance(checkpoint, expected_sha256="f" * 64)


def test_bundled_trajectory_marks_shuffle_unsupported() -> None:
    observations, states, metadata = load_trajectory(
        _ROOT / "upstream/vjepa2/notebooks/franka_example_traj.npz"
    )
    assert observations.shape == (1, 2, 256, 256, 3)
    assert states.shape == (1, 2, 7)
    assert metadata["independent_transitions"] == 1
    assert metadata["shuffled_action"]["status"] == "unsupported"


def test_cartesian_grid_uses_only_xyz_and_covers_bounds() -> None:
    actions = cartesian_action_grid(3, 0.075)
    assert actions.shape == (27, 7)
    assert actions.dtype == np.float32
    assert np.all(actions[:, 3:] == 0)
    assert np.isclose(actions[:, :3].min(), -0.075)
    assert np.isclose(actions[:, :3].max(), 0.075)


def test_atomic_json_result(tmp_path: Path) -> None:
    target = write_json_result(tmp_path / "nested/result.json", {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert not (target.parent / f".{target.name}.part").exists()


def test_default_cli_is_lightweight_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    namespace = runpy.run_path(str(_SCRIPT))
    exit_code = namespace["main"]([])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "preflight-only"
    assert payload["safety"]["default_allocates_model"] is False
    assert payload["checkpoint"]["provided"] is False
    assert payload["inputs_ready"] is False
    assert isinstance(payload["recommended_resource_available"], bool)
    assert payload["trajectory"]["shuffled_action"]["status"] == "unsupported"


def test_action_energy_is_chunked_and_uses_normalized_prediction() -> None:
    namespace = runpy.run_path(str(_SCRIPT))
    action_energies = namespace["_action_energies"]

    class TinyPredictor(torch.nn.Module):
        def forward(
            self, latents: torch.Tensor, actions: torch.Tensor, states: torch.Tensor
        ) -> torch.Tensor:
            del latents, states
            value = actions[:, 0, 0]
            return torch.stack((value, -value), dim=-1).unsqueeze(1)

    current = torch.zeros(1, 1, 2)
    goal = torch.tensor([[[1.0, -1.0]]])
    state = torch.zeros(1, 1, 7)
    actions = torch.zeros(3, 7)
    actions[:, 0] = torch.tensor([1.0, 0.0, -1.0])
    result = action_energies(
        TinyPredictor(), current, goal, state, actions, chunk_size=2
    )
    assert torch.allclose(result, torch.tensor([0.0, 1.0, 2.0]), atol=1e-4)
