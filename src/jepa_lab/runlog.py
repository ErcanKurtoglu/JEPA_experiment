"""Machine-readable run summaries and cross-process feature exchange."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .foundations import environment_report
from .upstreams import verify_upstreams


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return loaded


def config_digest(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class RunSummary:
    experiment: str
    seed: int
    config_sha256: str
    metrics: dict[str, float]
    notes: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    environment: dict[str, Any] = field(default_factory=environment_report)
    upstreams: list[dict[str, object]] = field(default_factory=verify_upstreams)


def build_run_summary(
    config: Mapping[str, Any], metrics: Mapping[str, float], notes: Mapping[str, Any] | None = None
) -> RunSummary:
    return RunSummary(
        experiment=str(config.get("experiment", "unnamed")),
        seed=int(config.get("seed", 42)),
        config_sha256=config_digest(config),
        metrics={key: float(value) for key, value in metrics.items()},
        notes=dict(notes or {}),
    )


def write_run_summary(summary: RunSummary, directory: str | Path = "runs") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_time = summary.created_at_utc.replace(":", "-")
    path = directory / f"{summary.experiment}-{safe_time}.json"
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_features(
    path: str | Path,
    features: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Export backend-neutral features plus a provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim < 2 or feature_array.shape[0] == 0:
        raise ValueError("features must have shape [num_samples,...,dimension]")
    if not np.isfinite(feature_array).all():
        raise ValueError("features must contain only finite values")
    arrays: dict[str, np.ndarray] = {"features": feature_array}
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != arrays["features"].shape[0]:
            raise ValueError("labels and features must have the same first dimension")
        if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"}:
            raise TypeError("labels must be a one-dimensional integer array")
        arrays["labels"] = labels.astype(np.int64, copy=False)
    if sample_ids is not None:
        sample_ids = np.asarray(sample_ids)
        if sample_ids.ndim != 1 or sample_ids.shape[0] != arrays["features"].shape[0]:
            raise ValueError("sample_ids must have shape [num_samples]")
        if sample_ids.dtype.kind not in {"U", "S"}:
            raise TypeError("sample_ids must use a string dtype")
        arrays["sample_ids"] = sample_ids
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices)
        if (
            frame_indices.ndim != 2
            or frame_indices.shape[0] != arrays["features"].shape[0]
        ):
            raise ValueError("frame_indices must have shape [num_samples, frames]")
        if frame_indices.dtype.kind not in {"i", "u"}:
            raise TypeError("frame_indices must use an integer dtype")
        arrays["frame_indices"] = frame_indices.astype(np.int64, copy=False)
    np.savez_compressed(path, **arrays)
    sidecar = path.with_suffix(path.suffix + ".json")
    payload = {
        "feature_shape": list(arrays["features"].shape),
        "feature_dtype": str(arrays["features"].dtype),
        "has_labels": labels is not None,
        "has_sample_ids": sample_ids is not None,
        "has_frame_indices": frame_indices is not None,
        "metadata": dict(metadata or {}),
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, sidecar


def import_features(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        labels = archive["labels"] if "labels" in archive.files else None
    sidecar_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    return features, labels, metadata
