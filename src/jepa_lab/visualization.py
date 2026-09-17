"""Small scientific visualizations used by the notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .masking import ImageMasks


def image_mask_map(masks: ImageMasks, sample: int = 0) -> np.ndarray:
    """Return a grid with -1 hidden, 0 context, and 1..K target block ids."""

    if not 0 <= sample < masks.batch_size:
        raise IndexError(sample)
    grid = np.full(masks.grid_size[0] * masks.grid_size[1], -1, dtype=np.int16)
    grid[masks.context[sample].cpu().numpy()] = 0
    for block, indices in enumerate(masks.targets[sample], start=1):
        grid[indices.cpu().numpy()] = block
    return grid.reshape(masks.grid_size)


def plot_image_masks(masks: ImageMasks, sample: int = 0) -> Figure:
    figure, axis = plt.subplots(figsize=(5, 5))
    rendered = axis.imshow(image_mask_map(masks, sample), interpolation="nearest")
    axis.set_title("I-JEPA patch map: context=0, targets=1..K")
    axis.set_xlabel("patch x")
    axis.set_ylabel("patch y")
    figure.colorbar(rendered, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure


def plot_tube_mask(mask: torch.Tensor, max_frames: int = 8) -> Figure:
    """Render temporal slices of a boolean [T,H,W] tube mask."""

    if mask.dtype is not torch.bool or mask.ndim != 3:
        raise ValueError("mask must be a boolean [T,H,W] tensor")
    count = min(mask.shape[0], max_frames)
    figure, axes = plt.subplots(1, count, figsize=(2.5 * count, 2.5), squeeze=False)
    for index in range(count):
        axes[0, index].imshow(mask[index].cpu(), vmin=0, vmax=1, cmap="gray_r")
        axes[0, index].set_title(f"t={index}")
        axes[0, index].axis("off")
    # ``gray_r`` maps False/0 to white and True/1 to black.  A dark target is
    # also visually consistent with the idea that this token is hidden from
    # the context encoder.
    figure.suptitle("V-JEPA target tubes (black = target, white = context)")
    figure.tight_layout()
    return figure


def plot_planar_trace(
    positions: np.ndarray,
    goal: Sequence[float],
    *,
    world_limits: tuple[float, float] = (0.0, 1.0),
) -> Figure:
    positions = np.asarray(positions)
    goal = np.asarray(goal)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [steps,2]")
    if goal.shape != (2,):
        raise ValueError("goal must have shape [2]")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(positions[:, 0], positions[:, 1], marker="o", label="executed")
    axis.scatter(goal[0], goal[1], marker="*", s=180, label="goal")
    axis.set(xlim=world_limits, ylim=world_limits, xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    axis.legend()
    axis.set_title("Receding-horizon trajectory")
    figure.tight_layout()
    return figure


__all__ = [
    "image_mask_map",
    "plot_image_masks",
    "plot_planar_trace",
    "plot_tube_mask",
]
