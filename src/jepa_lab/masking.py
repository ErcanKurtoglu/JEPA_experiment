"""Two-dimensional multi-block masking used by the image JEPA lab."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Size2D = int | tuple[int, int]
Bounds = float | tuple[float, float] | list[float]


def _pair(value: Size2D) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _bounds(value: Bounds) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar)
    if len(value) != 2:
        raise ValueError("bounds must contain exactly two values")
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True, slots=True)
class ImageMasks:
    """Patch indices for one batch of context and target blocks."""

    context: torch.Tensor  # [B, Nc]
    targets: torch.Tensor  # [B, K, Nt]
    grid_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.context.ndim != 2:
            raise ValueError("context must have shape [B, Nc]")
        if self.targets.ndim != 3:
            raise ValueError("targets must have shape [B, K, Nt]")
        if self.context.shape[0] != self.targets.shape[0]:
            raise ValueError("context and targets must have equal batch size")
        if self.context.dtype != torch.long or self.targets.dtype != torch.long:
            raise TypeError("mask indices must use torch.long")
        total = self.grid_size[0] * self.grid_size[1]
        if self.context.numel() and (
            int(self.context.min()) < 0 or int(self.context.max()) >= total
        ):
            raise ValueError("context contains an out-of-range patch index")
        if self.targets.numel() and (
            int(self.targets.min()) < 0 or int(self.targets.max()) >= total
        ):
            raise ValueError("targets contain an out-of-range patch index")

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def num_targets(self) -> int:
        return int(self.targets.shape[1])

    def boolean(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context and per-block target masks as booleans."""

        total = self.grid_size[0] * self.grid_size[1]
        context = torch.zeros(
            self.batch_size, total, dtype=torch.bool, device=self.context.device
        )
        context.scatter_(1, self.context, True)
        targets = torch.zeros(
            self.batch_size,
            self.num_targets,
            total,
            dtype=torch.bool,
            device=self.targets.device,
        )
        targets.scatter_(2, self.targets, True)
        return context, targets

    def to(self, device: torch.device | str) -> ImageMasks:
        return ImageMasks(
            context=self.context.to(device),
            targets=self.targets.to(device),
            grid_size=self.grid_size,
        )


class MultiBlockMasker:
    """Sample I-JEPA-style rectangular targets and one large context block.

    Each target position is drawn independently, so target-target overlap is
    deliberately allowed. Every target patch is removed from the context.
    Samples are truncated to the smallest context cardinality in a batch so the
    resulting tensors can be consumed by a dense Transformer.
    """

    def __init__(
        self,
        grid_size: Size2D,
        *,
        num_targets: int = 4,
        target_scale: Bounds = (0.15, 0.20),
        target_aspect_ratio: Bounds = (0.75, 1.5),
        context_scale: Bounds = (0.85, 1.0),
        context_aspect_ratio: Bounds = (1.0, 1.0),
        min_context_patches: int = 1,
        max_attempts: int = 128,
    ) -> None:
        self.grid_size = _pair(grid_size)
        self.num_targets = num_targets
        self.target_scale = _bounds(target_scale)
        self.target_aspect_ratio = _bounds(target_aspect_ratio)
        self.context_scale = _bounds(context_scale)
        self.context_aspect_ratio = _bounds(context_aspect_ratio)
        self.min_context_patches = min_context_patches
        self.max_attempts = max_attempts
        self._validate_configuration()

    @property
    def targets_may_overlap(self) -> bool:
        return True

    def _validate_configuration(self) -> None:
        height, width = self.grid_size
        if height <= 0 or width <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")
        if not 1 <= self.min_context_patches < height * width:
            raise ValueError("min_context_patches must be in [1, number of patches)")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        for name, bounds in (
            ("target_scale", self.target_scale),
            ("context_scale", self.context_scale),
        ):
            if not 0.0 < bounds[0] <= bounds[1] <= 1.0:
                raise ValueError(f"{name} must be ordered values in (0, 1]")
        for name, bounds in (
            ("target_aspect_ratio", self.target_aspect_ratio),
            ("context_aspect_ratio", self.context_aspect_ratio),
        ):
            if not 0.0 < bounds[0] <= bounds[1]:
                raise ValueError(f"{name} must contain ordered positive values")

    @staticmethod
    def _uniform(bounds: tuple[float, float], generator: torch.Generator) -> float:
        low, high = bounds
        return low + (high - low) * float(torch.rand((), generator=generator))

    def _sample_shape(
        self,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        generator: torch.Generator,
    ) -> tuple[int, int]:
        grid_height, grid_width = self.grid_size
        area = grid_height * grid_width * self._uniform(scale, generator)
        aspect = self._uniform(aspect_ratio, generator)  # height / width
        block_height = round(math.sqrt(area * aspect))
        block_width = round(math.sqrt(area / aspect))
        return (
            min(grid_height, max(1, block_height)),
            min(grid_width, max(1, block_width)),
        )

    def _rectangle(
        self,
        shape: tuple[int, int],
        generator: torch.Generator,
    ) -> torch.Tensor:
        grid_height, grid_width = self.grid_size
        block_height, block_width = shape
        top = int(
            torch.randint(grid_height - block_height + 1, (), generator=generator)
        )
        left = int(
            torch.randint(grid_width - block_width + 1, (), generator=generator)
        )
        rows = torch.arange(top, top + block_height)
        columns = torch.arange(left, left + block_width)
        return (rows[:, None] * grid_width + columns[None, :]).reshape(-1).long()

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ImageMasks:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        generator = generator or torch.default_generator
        target_shape = self._sample_shape(
            self.target_scale, self.target_aspect_ratio, generator
        )
        context_shape = self._sample_shape(
            self.context_scale, self.context_aspect_ratio, generator
        )
        total_patches = self.grid_size[0] * self.grid_size[1]

        batch_targets: list[torch.Tensor] = []
        batch_contexts: list[torch.Tensor] = []
        for _ in range(batch_size):
            targets = torch.stack(
                [self._rectangle(target_shape, generator) for _ in range(self.num_targets)]
            )
            target_union = torch.zeros(total_patches, dtype=torch.bool)
            target_union[targets.reshape(-1)] = True

            context = None
            for _ in range(self.max_attempts):
                candidate = self._rectangle(context_shape, generator)
                candidate = candidate[~target_union[candidate]]
                if candidate.numel() >= self.min_context_patches:
                    context = candidate
                    break
            if context is None:
                raise RuntimeError(
                    "could not sample a sufficiently large context; reduce target "
                    "scale/count or min_context_patches"
                )
            batch_targets.append(targets)
            batch_contexts.append(context)

        min_context = min(context.numel() for context in batch_contexts)
        contexts = []
        for context in batch_contexts:
            if context.numel() > min_context:
                order = torch.randperm(context.numel(), generator=generator)
                context = context[order[:min_context]]
            contexts.append(context)

        masks = ImageMasks(
            context=torch.stack(contexts),
            targets=torch.stack(batch_targets),
            grid_size=self.grid_size,
        )
        return masks if device is None else masks.to(device)

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ImageMasks:
        return self.sample(batch_size, generator=generator, device=device)


class RandomPatchMasker:
    """Random-patch control with the same dense ``ImageMasks`` contract.

    Each target is an independently sampled set of patch indices, so targets may
    overlap with one another.  Their union is removed from a randomly ordered
    context.  This changes spatial structure while keeping target count and scale
    explicit, which is the single-factor M2 masking ablation.
    """

    def __init__(
        self,
        grid_size: Size2D,
        *,
        num_targets: int = 4,
        target_scale: Bounds = (0.15, 0.20),
        context_scale: Bounds = (0.85, 1.0),
        min_context_patches: int = 1,
    ) -> None:
        self.grid_size = _pair(grid_size)
        self.num_targets = int(num_targets)
        self.target_scale = _bounds(target_scale)
        self.context_scale = _bounds(context_scale)
        self.min_context_patches = int(min_context_patches)
        total = self.grid_size[0] * self.grid_size[1]
        if min(*self.grid_size, self.num_targets, self.min_context_patches) <= 0:
            raise ValueError("grid, target count, and minimum context must be positive")
        if self.min_context_patches >= total:
            raise ValueError("min_context_patches must be smaller than the patch count")
        for name, bounds in (
            ("target_scale", self.target_scale),
            ("context_scale", self.context_scale),
        ):
            if not 0.0 < bounds[0] <= bounds[1] <= 1.0:
                raise ValueError(f"{name} must be ordered values in (0, 1]")

    @property
    def targets_may_overlap(self) -> bool:
        return True

    @staticmethod
    def _count(
        total: int, bounds: tuple[float, float], generator: torch.Generator
    ) -> int:
        fraction = MultiBlockMasker._uniform(bounds, generator)
        return min(total - 1, max(1, round(total * fraction)))

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ImageMasks:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        generator = generator or torch.default_generator
        total = self.grid_size[0] * self.grid_size[1]
        target_count = self._count(total, self.target_scale, generator)
        desired_context = self._count(total, self.context_scale, generator)
        targets_per_sample: list[torch.Tensor] = []
        contexts_per_sample: list[torch.Tensor] = []
        for _ in range(batch_size):
            targets = torch.stack(
                [
                    torch.randperm(total, generator=generator)[:target_count]
                    for _ in range(self.num_targets)
                ]
            )
            hidden = torch.zeros(total, dtype=torch.bool)
            hidden[targets.reshape(-1)] = True
            available = (~hidden).nonzero(as_tuple=False).flatten()
            if available.numel() < self.min_context_patches:
                raise RuntimeError(
                    "random targets left too little context; reduce target scale/count"
                )
            order = torch.randperm(available.numel(), generator=generator)
            context_count = min(desired_context, available.numel())
            contexts_per_sample.append(available[order[:context_count]])
            targets_per_sample.append(targets)

        minimum_context = min(context.numel() for context in contexts_per_sample)
        masks = ImageMasks(
            context=torch.stack(
                [context[:minimum_context] for context in contexts_per_sample]
            ),
            targets=torch.stack(targets_per_sample),
            grid_size=self.grid_size,
        )
        return masks if device is None else masks.to(device)

    def __call__(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ImageMasks:
        return self.sample(batch_size, generator=generator, device=device)
