"""Small datasets used by the manual JEPA laboratories."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENETTE_160_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_imagenette_manifest(dataset: Path, root: Path) -> Path:
    classes = sorted(path.name for path in (dataset / "train").iterdir() if path.is_dir())
    split_entries: dict[str, list[str]] = {}
    split_hashes: dict[str, str] = {}
    for split in ("train", "val"):
        entries = sorted(
            str(path.relative_to(dataset))
            for path in (dataset / split).rglob("*")
            if path.is_file()
        )
        split_entries[split] = entries
        split_hashes[split] = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    payload = {
        "dataset": "imagenette2-160",
        "fixed_seed": 42,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "split_counts": {name: len(entries) for name, entries in split_entries.items()},
        "split_manifest_sha256": split_hashes,
    }
    path = root / "imagenette2-160.manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def download_imagenette(root: str | Path) -> Path:
    """Download Imagenette-160, record SHA-256, and extract it safely."""

    root = Path(root)
    dataset = root / "imagenette2-160"
    if (dataset / "train").is_dir() and (dataset / "val").is_dir():
        _write_imagenette_manifest(dataset, root)
        return dataset
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-160.tgz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if not archive.exists():
        with urllib.request.urlopen(IMAGENETTE_160_URL) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        partial.replace(archive)

    root_resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise ValueError(f"unsafe path in Imagenette archive: {member.name}")
        bundle.extractall(root)
    provenance = {
        "dataset": "imagenette2-160",
        "url": IMAGENETTE_160_URL,
        "archive_sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
    }
    (root / "imagenette2-160.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    _write_imagenette_manifest(dataset, root)
    return dataset


def imagenette_loaders(
    root: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build I-JEPA-style single-view train and deterministic validation loaders."""

    from torchvision import datasets, transforms

    root = Path(root)
    if not (root / "train").is_dir():
        raise FileNotFoundError(
            f"{root}/train is missing; call download_imagenette(root.parent) first"
        )
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    generator = torch.Generator().manual_seed(42)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train = DataLoader(
        datasets.ImageFolder(root / "train", transform=train_transform),
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation = DataLoader(
        datasets.ImageFolder(root / "val", transform=validation_transform),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train, validation


class MovingShapesDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic videos with direction labels, states, and 7D delta actions."""

    DIRECTIONS = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )

    def __init__(
        self,
        *,
        samples: int = 4096,
        frames: int = 8,
        image_size: int = 112,
        square_size: int | None = None,
        speed: float = 0.045,
        seed: int = 42,
    ) -> None:
        if samples <= 0 or frames < 2 or image_size < 16 or not 0 < speed < 0.25:
            raise ValueError("invalid moving-shapes configuration")
        self.samples = int(samples)
        self.frames = int(frames)
        self.image_size = int(image_size)
        self.square_size = int(square_size or max(3, image_size // 14))
        self.speed = float(speed)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + index)
        label = int(rng.integers(0, len(self.DIRECTIONS)))
        direction = self.DIRECTIONS[label]
        margin = 0.18 + self.speed * (self.frames - 1)
        start = torch.tensor(
            rng.uniform(margin, 1.0 - margin, size=2), dtype=torch.float32
        )
        states_xy = torch.stack(
            [start + direction * self.speed * step for step in range(self.frames)]
        ).clamp(0.0, 1.0)

        video = torch.zeros(self.frames, 3, self.image_size, self.image_size)
        colour = torch.tensor(
            rng.uniform(0.55, 1.0, size=3), dtype=torch.float32
        ).view(3, 1, 1)
        for time, point in enumerate(states_xy):
            x = min(
                self.image_size - self.square_size,
                max(0, round(float(point[0]) * (self.image_size - 1))),
            )
            y = min(
                self.image_size - self.square_size,
                max(0, round(float(point[1]) * (self.image_size - 1))),
            )
            video[time, :, y : y + self.square_size, x : x + self.square_size] = colour

        states = torch.zeros(self.frames, 7)
        states[:, :2] = states_xy
        actions = torch.zeros(self.frames - 1, 7)
        actions[:, :2] = direction * self.speed
        return {
            "frames": video,
            "label": torch.tensor(label, dtype=torch.long),
            "states": states,
            "actions": actions,
        }


def sample_planar_latent_batch(
    batch_size: int,
    horizon: int,
    *,
    action_limit: float = 0.10,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate exact planar transitions for the action-world-model lab.

    Returns z0 [B,1,2], actions [B,H,7], states [B,H,7], targets [B,H,1,2].
    """

    if batch_size <= 0 or horizon <= 0 or not 0 < action_limit <= 1:
        raise ValueError("invalid planar batch configuration")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    z0 = torch.rand(batch_size, 1, 2, generator=generator) * 0.6 + 0.2
    actions = torch.zeros(batch_size, horizon, 7)
    actions[..., :2] = (
        torch.rand(batch_size, horizon, 2, generator=generator) * 2.0 - 1.0
    ) * action_limit
    positions = []
    current = z0[:, 0]
    for step in range(horizon):
        current = (current + actions[:, step, :2]).clamp(0.0, 1.0)
        positions.append(current)
    states = planar_states_from_actions(z0, actions, max_delta=action_limit)
    targets = torch.stack(positions, dim=1).unsqueeze(2)
    return tuple(
        value.to(device) for value in (z0, actions, states, targets)
    )  # type: ignore[return-value]


def planar_states_from_actions(
    initial: torch.Tensor,
    actions: torch.Tensor,
    *,
    max_delta: float | None = None,
) -> torch.Tensor:
    """Integrate candidate actions into pre-action 7D planar state tokens.

    ``max_delta`` mirrors :class:`PlanarReachEnv` clipping when callers may pass
    actions outside the environment's configured range.
    """

    if initial.ndim == 3 and initial.shape[1:] == (1, 2):
        initial = initial[:, 0]
    if initial.ndim != 2 or initial.shape[-1] != 2:
        raise ValueError("initial must have shape [B,2] or [B,1,2]")
    if actions.ndim != 3 or actions.shape[0] != initial.shape[0] or actions.shape[-1] != 7:
        raise ValueError("actions must have shape [B,H,7] with the same batch size")
    if not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(initial).all()):
        raise ValueError("initial and actions must contain only finite values")
    if max_delta is not None and not 0.0 < max_delta <= 1.0:
        raise ValueError("max_delta must be in (0,1]")
    states = torch.zeros(
        actions.shape[0],
        actions.shape[1],
        7,
        dtype=actions.dtype,
        device=actions.device,
    )
    current = initial.to(device=actions.device, dtype=actions.dtype)
    for step in range(actions.shape[1]):
        states[:, step, :2] = current
        delta = actions[:, step, :2]
        if max_delta is not None:
            delta = delta.clamp(-max_delta, max_delta)
        current = (current + delta).clamp(0.0, 1.0)
    return states


__all__ = [
    "IMAGENETTE_160_URL",
    "MovingShapesDataset",
    "download_imagenette",
    "imagenette_loaders",
    "planar_states_from_actions",
    "sample_planar_latent_batch",
]
