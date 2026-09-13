"""Small, explicit checkpoint/resume helpers for teaching-scale lab runs."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .upstreams import sha256_file


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    step: int
    sha256: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    metadata: dict[str, Any]


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model/optimizer state and an adjacent digest sidecar."""

    if step < 0:
        raise ValueError("step must be non-negative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = dict(metadata or {})
    try:
        # Validate before touching an existing checkpoint. The same representation
        # is written to both the Torch payload and its human-readable sidecar.
        json.dumps(metadata_payload, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise TypeError("checkpoint metadata must be JSON serializable") from error

    checkpoint_temporary = path.parent / f".{path.name}.part"
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar_temporary = sidecar_path.parent / f".{sidecar_path.name}.part"
    payload = {
        "schema_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata_payload,
    }
    try:
        torch.save(payload, checkpoint_temporary)
        sidecar = {
            "path": str(path),
            "bytes": checkpoint_temporary.stat().st_size,
            "sha256": sha256_file(checkpoint_temporary),
            "step": int(step),
            "metadata": metadata_payload,
        }
        sidecar_temporary.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        checkpoint_temporary.replace(path)
        sidecar_temporary.replace(sidecar_path)
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
    return path


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = True,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadResult:
    """Verify, load and report every compatibility field needed for resume."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    recorded_digest = None
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            raise ValueError(f"invalid checkpoint sidecar: {sidecar_path}")
        recorded_digest = sidecar.get("sha256")
    trusted_digest = expected_sha256 or recorded_digest
    if trusted_digest is not None and (
        not isinstance(trusted_digest, str) or digest.lower() != trusted_digest.lower()
    ):
        raise ValueError(f"SHA-256 mismatch for checkpoint {path}: {digest}")

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema_version")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    checkpoint_state = payload.get("model")
    if not isinstance(checkpoint_state, dict):
        raise TypeError("checkpoint must contain a model state mapping")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    optimizer_state = payload.get("optimizer")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise TypeError("optimizer state must be a mapping")

    expected_state = model.state_dict()
    missing = tuple(key for key in expected_state if key not in checkpoint_state)
    unexpected = tuple(key for key in checkpoint_state if key not in expected_state)
    mismatched_shapes = []
    for key in expected_state.keys() & checkpoint_state.keys():
        value = checkpoint_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint model value {key!r} must be a tensor")
        if value.shape != expected_state[key].shape:
            mismatched_shapes.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(expected_state[key].shape)}"
            )
    if mismatched_shapes:
        raise RuntimeError("checkpoint/model shape mismatch; " + ", ".join(mismatched_shapes))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint/model mismatch; missing={list(missing)}, unexpected={list(unexpected)}"
        )

    model_backup = copy.deepcopy(expected_state)
    optimizer_backup = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        incompatible = model.load_state_dict(checkpoint_state, strict=strict)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    except Exception:
        model.load_state_dict(model_backup, strict=True)
        if optimizer is not None and optimizer_backup is not None:
            optimizer.load_state_dict(optimizer_backup)
        raise
    return CheckpointLoadResult(
        step=step,
        sha256=digest,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        metadata=metadata,
    )


__all__ = [
    "CheckpointLoadResult",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
