"""Pure validation helpers for the opt-in official V-JEPA2-AC replay.

The heavyweight Meta model is deliberately loaded only by the standalone
``scripts/official_vjepa2_ac_replay.py`` process.  Helpers in this module stay
small enough for unit tests and the default preflight path.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .upstreams import sha256_file

PINNED_VJEPA2_COMMIT = "9a061fffe395573a2ee61dadcd3cf5baa188e3f2"
REPLAY_DEPENDENCIES: Mapping[str, str] = {
    "timm": "timm",
    "einops": "einops",
    "scipy": "scipy",
}


def system_memory_gib() -> float | None:
    """Return physical memory without adding a psutil dependency."""

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or physical_pages <= 0:
        return None
    return round(page_size * physical_pages / 1024**3, 2)


def replay_dependency_report() -> dict[str, bool]:
    """Report optional imports without importing their heavyweight modules."""

    return {
        display_name: importlib.util.find_spec(import_name) is not None
        for display_name, import_name in REPLAY_DEPENDENCIES.items()
    }


def checkpoint_provenance(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a local checkpoint against an explicit or recorded SHA-256.

    An arbitrary local file is never trusted on first sight.  Either the caller
    must provide ``expected_sha256`` or the adjacent ``.json`` sidecar must have
    been created by the project's checkpoint downloader.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    recorded: dict[str, Any] = {}
    if sidecar_path.exists():
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid checkpoint sidecar: {sidecar_path}")
        recorded = loaded

    explicit_digest = _validated_digest(expected_sha256, label="expected SHA-256")
    recorded_value = recorded.get("sha256")
    recorded_digest = _validated_digest(
        recorded_value,
        label=f"recorded SHA-256 in {sidecar_path}",
    )
    if explicit_digest is None and recorded_digest is None:
        raise ValueError(
            f"checkpoint {path} has no trusted SHA-256; pass --sha256 or use its provenance sidecar"
        )
    if (
        explicit_digest is not None
        and recorded_digest is not None
        and explicit_digest != recorded_digest
    ):
        raise ValueError("explicit and sidecar SHA-256 values disagree")

    trusted_digest = explicit_digest or recorded_digest
    assert trusted_digest is not None
    actual_digest = sha256_file(path).lower()
    if actual_digest != trusted_digest:
        raise ValueError(f"SHA-256 mismatch for checkpoint {path}: {actual_digest}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": actual_digest,
        "trust_source": "explicit" if explicit_digest is not None else "sidecar",
        "sidecar": str(sidecar_path.resolve()) if sidecar_path.exists() else None,
        "source_url": recorded.get("url"),
    }


def _validated_digest(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a 64-character hexadecimal string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal string")
    return normalized


def load_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate the small, non-pickle Franka replay fixture."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"observations", "states"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trajectory is missing arrays: {sorted(missing)}")
        observations = np.asarray(archive["observations"])
        states = np.asarray(archive["states"])

    if observations.ndim != 5 or observations.shape[-1] != 3:
        raise ValueError("observations must have shape [B,T,H,W,3]")
    if observations.dtype != np.uint8:
        raise TypeError("observations must use uint8 RGB values")
    if states.ndim != 3 or states.shape[-1] != 7:
        raise ValueError("states must have shape [B,T,7]")
    if observations.shape[:2] != states.shape[:2]:
        raise ValueError("observations and states must share batch/time dimensions")
    if observations.shape[1] < 2:
        raise ValueError("trajectory must contain at least one transition")
    if not np.issubdtype(states.dtype, np.floating):
        raise TypeError("states must use a floating dtype")
    if not np.isfinite(states).all():
        raise ValueError("states must contain only finite values")

    transitions = int(states.shape[0] * (states.shape[1] - 1))
    shuffle = (
        {
            "status": "unsupported",
            "reason": (
                "only one independent transition is available; permuting action axes would not "
                "be a valid shuffled-action control"
            ),
        }
        if transitions == 1
        else {
            "status": "not_computed",
            "reason": "this wrapper evaluates the first transition only",
        }
    )
    metadata = {
        "path": str(path.resolve()),
        "arrays": {
            "observations": {
                "shape": list(observations.shape),
                "dtype": str(observations.dtype),
            },
            "states": {"shape": list(states.shape), "dtype": str(states.dtype)},
        },
        "independent_transitions": transitions,
        "evaluated_transition": {"batch": 0, "from_frame": 0, "to_frame": 1},
        "shuffled_action": shuffle,
    }
    return observations, states, metadata


def cartesian_action_grid(samples_per_axis: int, bound: float) -> np.ndarray:
    """Return the official-style xyz grid in the canonical seven-action format."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    axis = np.linspace(-bound, bound, samples_per_axis, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    actions = np.zeros((samples_per_axis**3, 7), dtype=np.float32)
    actions[:, :3] = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return actions


def write_json_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a machine-readable replay result."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.part"
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "PINNED_VJEPA2_COMMIT",
    "REPLAY_DEPENDENCIES",
    "cartesian_action_grid",
    "checkpoint_provenance",
    "load_trajectory",
    "replay_dependency_report",
    "system_memory_gib",
    "write_json_result",
]
