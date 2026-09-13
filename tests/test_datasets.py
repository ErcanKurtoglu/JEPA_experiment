import json
from pathlib import Path

import torch

from jepa_lab.datasets import (
    MovingShapesDataset,
    download_imagenette,
    planar_states_from_actions,
    sample_planar_latent_batch,
)


def test_moving_shapes_is_deterministic_and_uses_canonical_contract() -> None:
    first = MovingShapesDataset(samples=2, frames=8, image_size=32, seed=9)[0]
    second = MovingShapesDataset(samples=2, frames=8, image_size=32, seed=9)[0]
    torch.testing.assert_close(first["frames"], second["frames"])
    assert first["frames"].shape == (8, 3, 32, 32)
    assert first["actions"].shape == (7, 7)
    assert first["states"].shape == (8, 7)


def test_planar_latent_batch_matches_clipped_delta_dynamics() -> None:
    z0, actions, states, targets = sample_planar_latent_batch(4, 3, seed=1)
    assert z0.shape == (4, 1, 2)
    assert actions.shape == (4, 3, 7)
    assert states.shape == (4, 3, 7)
    assert targets.shape == (4, 3, 1, 2)
    expected_first = (z0[:, 0] + actions[:, 0, :2]).clamp(0.0, 1.0)
    torch.testing.assert_close(targets[:, 0, 0], expected_first)
    torch.testing.assert_close(states[:, 0, :2], z0[:, 0])
    rebuilt = planar_states_from_actions(z0, actions)
    torch.testing.assert_close(states, rebuilt)
    torch.testing.assert_close(
        states[:, 1, :2], (z0[:, 0] + actions[:, 0, :2]).clamp(0.0, 1.0)
    )


def test_planar_state_integration_can_match_environment_action_clipping() -> None:
    initial = torch.tensor([[[0.5, 0.5]]])
    actions = torch.zeros(1, 2, 7)
    actions[..., :2] = torch.tensor((0.8, -0.8))
    states = planar_states_from_actions(initial, actions, max_delta=0.1)
    torch.testing.assert_close(states[:, 1, :2], torch.tensor([[0.6, 0.4]]))


def test_existing_imagenette_writes_stable_split_and_class_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "imagenette2-160"
    for split in ("train", "val"):
        for class_name in ("n2", "n1"):
            folder = dataset / split / class_name
            folder.mkdir(parents=True)
            (folder / f"{split}.jpeg").write_bytes(b"image")

    assert download_imagenette(tmp_path) == dataset
    manifest = json.loads(
        (tmp_path / "imagenette2-160.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["class_to_idx"] == {"n1": 0, "n2": 1}
    assert manifest["split_counts"] == {"train": 2, "val": 2}
    assert all(len(value) == 64 for value in manifest["split_manifest_sha256"].values())
