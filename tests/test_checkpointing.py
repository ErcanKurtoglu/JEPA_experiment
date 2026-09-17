from pathlib import Path

import pytest
import torch
from torch import nn

from jepa_lab.checkpointing import load_training_checkpoint, save_training_checkpoint


def test_training_checkpoint_round_trip_restores_model_optimizer_and_step(
    tmp_path: Path,
) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_training_checkpoint(
        tmp_path / "smoke.pt", model, optimizer, step=10, metadata={"seed": 42}
    )
    with torch.no_grad():
        model.weight.zero_()
    result = load_training_checkpoint(path, model, optimizer)

    assert result.step == 10
    assert result.metadata == {"seed": 42}
    assert not result.missing_keys and not result.unexpected_keys
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert path.with_suffix(".pt.json").is_file()


def test_training_checkpoint_loads_legacy_torch_version_metadata(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    path = tmp_path / "legacy-torch-version.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 4,
            "model": model.state_dict(),
            "optimizer": None,
            "metadata": {"torch": torch.__version__},
        },
        path,
    )

    result = load_training_checkpoint(path, model)

    assert result.step == 4
    assert result.metadata["torch"] == str(torch.__version__)


def test_training_checkpoint_rejects_digest_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    path = save_training_checkpoint(tmp_path / "model.pt", model, None, step=1)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_training_checkpoint(path, model)


def test_invalid_metadata_does_not_replace_existing_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    path = save_training_checkpoint(tmp_path / "model.pt", model, None, step=1)
    sidecar = path.with_suffix(".pt.json")
    before = path.read_bytes(), sidecar.read_bytes()

    with pytest.raises(TypeError, match="JSON serializable"):
        save_training_checkpoint(path, model, None, step=2, metadata={"bad": object()})

    assert (path.read_bytes(), sidecar.read_bytes()) == before


def test_strict_mismatch_does_not_partially_mutate_model(tmp_path: Path) -> None:
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
        target.weight.fill_(7.0)
        target.bias.fill_(8.0)
    path = tmp_path / "partial.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 1,
            "model": {"weight": source.weight.detach().clone()},
            "optimizer": None,
            "metadata": {},
        },
        path,
    )
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(RuntimeError, match="missing"):
        load_training_checkpoint(path, target, strict=True)

    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])
