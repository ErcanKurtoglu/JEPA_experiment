"""Boundary helpers for working with isolated official repositories.

The upstream projects all expose a top-level ``src`` package. Importing more
than one in a Python process is therefore unsafe. These helpers keep the tensor
contract common while model construction remains in one upstream-specific
kernel/process at a time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .upstreams import UPSTREAMS, repository_root


def canonical_to_vjepa(frames: torch.Tensor) -> torch.Tensor:
    """Convert canonical [B,T,C,H,W] frames to V-JEPA [B,C,T,H,W]."""
    if frames.ndim != 5:
        raise ValueError("frames must have shape [B,T,C,H,W]")
    return frames.permute(0, 2, 1, 3, 4).contiguous()


def vjepa_to_canonical(frames: torch.Tensor) -> torch.Tensor:
    """Convert V-JEPA [B,C,T,H,W] frames to canonical [B,T,C,H,W]."""
    if frames.ndim != 5:
        raise ValueError("frames must have shape [B,C,T,H,W]")
    return frames.permute(0, 2, 1, 3, 4).contiguous()


def strip_state_dict_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Sequence[str] = ("module.", "backbone."),
) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes repeatedly, as official eval loaders do."""
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


def select_checkpoint_state(
    checkpoint: Mapping[str, Any],
    preferred_keys: Sequence[str] = ("target_encoder", "encoder", "teacher", "state_dict"),
) -> dict[str, torch.Tensor]:
    """Select and normalize a model state dict from Meta checkpoint layouts."""
    selected: Mapping[str, Any] = checkpoint
    for key in preferred_keys:
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            selected = candidate
            break
    else:
        # jepa_lab training checkpoints group official modules under ``model``
        # and retain the upstream component name as a state-dict key prefix.
        grouped = checkpoint.get("model")
        if isinstance(grouped, Mapping):
            for key in preferred_keys:
                prefix = f"{key}."
                component = {
                    str(name)[len(prefix) :]: value
                    for name, value in grouped.items()
                    if str(name).startswith(prefix)
                }
                if component:
                    selected = component
                    break
    tensors = {key: value for key, value in selected.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("checkpoint does not contain a tensor state dict")
    return strip_state_dict_prefixes(tensors)


def temporal_variants(frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """Generate the four standard temporal-sensitivity controls."""
    if frames.ndim != 5:
        raise ValueError("frames must have shape [B,T,C,H,W]")
    timeline = frames.shape[1]
    if timeline < 2:
        raise ValueError("at least two frames are required")
    generator = torch.Generator(device="cpu").manual_seed(42)
    permutation = torch.randperm(timeline, generator=generator).to(frames.device)
    return {
        "normal": frames,
        "reversed": frames.flip(1),
        "shuffled": frames.index_select(1, permutation),
        "static": frames[:, :1].expand_as(frames).clone(),
    }


def cosine_to_reference(features: Mapping[str, torch.Tensor], reference: str = "normal") -> dict[str, float]:
    """Mean-pool token features and compare all variants with a reference."""
    if reference not in features:
        raise KeyError(reference)
    pooled = {
        name: value.mean(dim=1) if value.ndim == 3 else value.flatten(1)
        for name, value in features.items()
    }
    anchor = pooled[reference]
    return {
        name: float(torch.nn.functional.cosine_similarity(anchor, value, dim=-1).mean())
        for name, value in pooled.items()
    }


def run_isolated_upstream(
    name: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python entry point with exactly one upstream repository on PYTHONPATH."""
    specs = {spec.name: spec for spec in UPSTREAMS}
    if name not in specs:
        raise KeyError(f"unknown upstream {name!r}; choose from {sorted(specs)}")
    path = repository_root() / specs[name].relative_path
    if not path.exists():
        raise FileNotFoundError(f"missing submodule: {path}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(path)
    environment.update(extra_environment or {})
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
