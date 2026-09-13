"""Device selection and reproducibility helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch


def _device_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return torch.cuda.is_available() and (
            device.index is None or device.index < torch.cuda.device_count()
        )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    return False


def select_device(
    preferred: str | torch.device | Sequence[str | torch.device] = "auto",
) -> torch.device:
    """Resolve a requested device, preferring CUDA, then MPS, then CPU.

    Explicit unavailable accelerators raise instead of silently changing an
    experiment's execution target.  Use ``"auto"`` when fallback is desired.
    """

    if isinstance(preferred, Sequence) and not isinstance(preferred, str):
        if not preferred:
            raise ValueError("device preference sequence cannot be empty")
        candidates = [torch.device(candidate) for candidate in preferred]
        for candidate in candidates:
            if _device_available(candidate):
                return candidate
        requested = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(f"none of the requested devices are available: {requested}")
    if isinstance(preferred, torch.device):
        requested = preferred
    elif preferred == "auto":
        if _device_available(torch.device("cuda")):
            return torch.device("cuda")
        if _device_available(torch.device("mps")):
            return torch.device("mps")
        return torch.device("cpu")
    else:
        requested = torch.device(preferred)

    if requested.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device type: {requested.type}")
    if not _device_available(requested):
        raise RuntimeError(f"{requested} was requested but is not available")
    return requested


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch and configure deterministic kernels."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = False if deterministic else cudnn.benchmark


def seeded_generator(seed: int) -> torch.Generator:
    """Create a CPU generator suitable for deterministic mask sampling."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    return torch.Generator(device="cpu").manual_seed(seed)
