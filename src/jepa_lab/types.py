"""Shared, framework-light data contracts for the JEPA laboratories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch

Array: TypeAlias = torch.Tensor | np.ndarray


@runtime_checkable
class VisualEncoder(Protocol):
    """Common representation boundary used by video and control adapters."""

    def encode(
        self, frames: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


def _shape(value: Array) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def _move(value: Array, device: torch.device | str) -> Array:
    return value.to(device) if isinstance(value, torch.Tensor) else value


@dataclass(frozen=True, slots=True)
class RawVideo:
    """Unprocessed RGB videos in the external ``uint8[B,T,H,W,3]`` layout."""

    frames: Array

    def __post_init__(self) -> None:
        if self.frames.ndim != 5 or self.frames.shape[-1] != 3:
            raise ValueError("raw frames must have shape [B,T,H,W,3]")
        is_uint8 = (
            self.frames.dtype == torch.uint8
            if isinstance(self.frames, torch.Tensor)
            else self.frames.dtype == np.uint8
        )
        if not is_uint8:
            raise TypeError("raw frames must use uint8")

    def to_video_batch(self, device: torch.device | str = "cpu") -> VideoBatch:
        tensor = torch.as_tensor(self.frames, device=device)
        canonical = tensor.permute(0, 1, 4, 2, 3).contiguous().float().div_(255.0)
        return VideoBatch(frames=canonical)


@dataclass(frozen=True, slots=True)
class VideoBatch:
    """Canonical video input used at the boundaries of the lab code.

    Frames always use ``[batch, time, channels, height, width]``.  Actions and
    states are optional because representation-only video models do not consume
    them.
    """

    frames: torch.Tensor
    actions: torch.Tensor | None = None
    states: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.frames.ndim != 5:
            raise ValueError(
                "frames must have shape [B, T, C, H, W], "
                f"received {tuple(self.frames.shape)}"
            )
        if self.frames.shape[2] != 3:
            raise ValueError("frames must contain exactly three RGB channels")
        if self.frames.dtype != torch.float32:
            raise TypeError("frames must use float32")
        if not bool(torch.isfinite(self.frames).all()):
            raise ValueError("frames must contain only finite values")
        batch_size = self.frames.shape[0]
        for name, value in (("actions", self.actions), ("states", self.states)):
            if value is None:
                continue
            if value.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [B, H, D], received {tuple(value.shape)}"
                )
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch dimension ({value.shape[0]}) does not match "
                    f"frames ({batch_size})"
                )
            if value.shape[-1] != 7:
                raise ValueError(f"{name} must use the canonical seven-dimensional format")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use float32")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
        if (
            self.actions is not None
            and self.states is not None
            and self.actions.shape[1] != self.states.shape[1]
        ):
            raise ValueError("actions and states must have the same horizon")

    @property
    def batch_size(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[1])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> VideoBatch:
        """Return a new batch whose tensors are on ``device``."""

        return VideoBatch(
            frames=self.frames.to(device, non_blocking=non_blocking),
            actions=(
                None
                if self.actions is None
                else self.actions.to(device, non_blocking=non_blocking)
            ),
            states=(
                None
                if self.states is None
                else self.states.to(device, non_blocking=non_blocking)
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Result of one CEM planning call.

    NumPy arrays are accepted for the light-weight simulator and Torch tensors
    for learned world models. ``energy_history`` records the raw mean elite energy
    observed after each CEM refinement so convergence can be inspected honestly.
    """

    first_action: Array
    sequence: Array
    energy: float | Array
    elite_mean: Array
    elite_std: Array
    energy_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        first_shape = _shape(self.first_action)
        sequence_shape = _shape(self.sequence)
        mean_shape = _shape(self.elite_mean)
        std_shape = _shape(self.elite_std)

        if len(first_shape) != 1:
            raise ValueError(f"first_action must have shape [A], received {first_shape}")
        if len(sequence_shape) != 2:
            raise ValueError(f"sequence must have shape [H, A], received {sequence_shape}")
        if mean_shape != sequence_shape or std_shape != sequence_shape:
            raise ValueError(
                "elite_mean and elite_std must have the same [H, A] shape as sequence"
            )
        if first_shape[0] != sequence_shape[1]:
            raise ValueError("first_action width must match the sequence action width")
        if any(not np.isfinite(value) for value in self.energy_history):
            raise ValueError("energy_history must contain only finite values")

    def to(self, device: torch.device | str) -> PlanResult:
        """Move Torch-backed fields while leaving NumPy-backed fields untouched."""

        return PlanResult(
            first_action=_move(self.first_action, device),
            sequence=_move(self.sequence, device),
            energy=(
                _move(self.energy, device)
                if isinstance(self.energy, (torch.Tensor, np.ndarray))
                else self.energy
            ),
            elite_mean=_move(self.elite_mean, device),
            elite_std=_move(self.elite_std, device),
            energy_history=self.energy_history,
        )
