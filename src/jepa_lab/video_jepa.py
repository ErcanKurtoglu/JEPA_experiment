"""Small, inspectable building blocks for learning how V-JEPA works.

The classes in this module are deliberately modest.  They preserve the ideas that
matter for the study labs -- spatio-temporal tubelets, a full-video EMA teacher,
masking *after* the teacher encoder, and latent L1 prediction -- while remaining
small enough to run on a laptop or a single Kaggle GPU.

All public video inputs use the project-wide ``[B, T, C, H, W]`` convention.
Meta's V-JEPA repositories generally use ``[B, C, T, H, W]`` internally; the
conversion therefore happens in :class:`TubeletTokenizer` and nowhere else.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GridSize = tuple[int, int, int]


@dataclass(frozen=True)
class TubeMasks:
    """A collection of spatial masks extended over the complete token timeline.

    ``short`` and ``long`` retain the individual sampled blocks for visualisation.
    ``target`` is their union, capped so that at least ``min_context_ratio`` of the
    spatial grid remains visible to the context encoder.
    """

    target: Tensor
    short: tuple[Tensor, ...]
    long: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if self.target.dtype is not torch.bool or self.target.ndim != 3:
            raise ValueError("target must be a boolean [T, H, W] tensor")
        if not bool(self.target.any()):
            raise ValueError("a tube mask must contain at least one target token")
        if bool(self.target.all()):
            raise ValueError("a tube mask must leave at least one context token")
        for mask in (*self.short, *self.long):
            if mask.dtype is not torch.bool or mask.shape != self.target.shape:
                raise ValueError("all component masks must match target [T, H, W]")

    @property
    def context(self) -> Tensor:
        return ~self.target

    @property
    def target_flat(self) -> Tensor:
        return self.target.flatten()

    @property
    def context_flat(self) -> Tensor:
        return self.context.flatten()

    @property
    def coverage(self) -> float:
        return float(self.target.float().mean().item())


def _normalise_scale(scale: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(scale, (float, int)):
        bounds = (float(scale), float(scale))
    else:
        if len(scale) != 2:
            raise ValueError("scale must be a float or a (minimum, maximum) pair")
        bounds = (float(scale[0]), float(scale[1]))
    if not 0.0 < bounds[0] <= bounds[1] <= 1.0:
        raise ValueError("mask scales must satisfy 0 < min <= max <= 1")
    return bounds


def _sample_spatial_block(
    height: int,
    width: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    generator: torch.Generator,
) -> Tensor:
    """Sample one rectangular boolean block without touching global RNG state."""

    sample = torch.rand(2, generator=generator)
    area_fraction = scale[0] + float(sample[0]) * (scale[1] - scale[0])
    log_min, log_max = math.log(aspect_ratio[0]), math.log(aspect_ratio[1])
    aspect = math.exp(log_min + float(sample[1]) * (log_max - log_min))
    area = max(1.0, area_fraction * height * width)
    block_height = max(1, min(height, round(math.sqrt(area / aspect))))
    block_width = max(1, min(width, round(math.sqrt(area * aspect))))

    top = int(torch.randint(height - block_height + 1, (1,), generator=generator))
    left = int(torch.randint(width - block_width + 1, (1,), generator=generator))
    block = torch.zeros((height, width), dtype=torch.bool)
    block[top : top + block_height, left : left + block_width] = True
    return block


def generate_tube_masks(
    grid_size: GridSize,
    *,
    num_short: int = 8,
    num_long: int = 2,
    short_scale: float | Sequence[float] = (0.10, 0.20),
    long_scale: float | Sequence[float] = (0.65, 0.75),
    aspect_ratio: tuple[float, float] = (0.75, 1.50),
    min_context_ratio: float = 0.10,
    seed: int = 0,
) -> TubeMasks:
    """Generate V-JEPA-style full-time spatial tube masks.

    A spatial rectangle is sampled once and copied to every temporal token.  The
    individual rectangles may overlap, as in the paper.  Since a tiny educational
    grid can otherwise be covered completely by the union of ten masks, the union
    is deterministically capped while preserving the full-time tube property.
    """

    temporal, height, width = (int(value) for value in grid_size)
    if min(temporal, height, width) <= 0:
        raise ValueError("grid dimensions must be positive")
    if num_short < 0 or num_long < 0 or num_short + num_long == 0:
        raise ValueError("at least one short or long mask is required")
    if not 0.0 < min_context_ratio < 1.0:
        raise ValueError("min_context_ratio must be between zero and one")
    if not 0.0 < aspect_ratio[0] <= aspect_ratio[1]:
        raise ValueError("aspect_ratio must contain positive ordered values")

    short_bounds = _normalise_scale(short_scale)
    long_bounds = _normalise_scale(long_scale)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample_many(count: int, scale: tuple[float, float]) -> tuple[Tensor, ...]:
        result = []
        for _ in range(count):
            spatial = _sample_spatial_block(
                height, width, scale, aspect_ratio, generator
            )
            result.append(spatial.unsqueeze(0).expand(temporal, -1, -1).clone())
        return tuple(result)

    short_masks = sample_many(num_short, short_bounds)
    long_masks = sample_many(num_long, long_bounds)
    spatial_union = torch.zeros((height, width), dtype=torch.bool)
    for mask in (*short_masks, *long_masks):
        spatial_union |= mask[0]

    spatial_tokens = height * width
    max_targets = min(
        spatial_tokens - 1,
        max(1, math.floor(spatial_tokens * (1.0 - min_context_ratio))),
    )
    target_indices = spatial_union.flatten().nonzero(as_tuple=False).flatten()
    if target_indices.numel() > max_targets:
        permutation = torch.randperm(target_indices.numel(), generator=generator)
        kept = target_indices[permutation[:max_targets]]
        spatial_union.zero_()
        spatial_union.view(-1)[kept] = True

    target = spatial_union.unsqueeze(0).expand(temporal, -1, -1).clone()
    return TubeMasks(target=target, short=short_masks, long=long_masks)


def generate_future_mask(
    grid_size: GridSize,
    *,
    context_temporal_tokens: int | None = None,
) -> Tensor:
    """Mask all future tubelets for the causal past-to-future ablation.

    Standard V-JEPA uses spatial blocks extended over the full timeline.  This
    deliberately different mask exposes complete past tubelets and makes every
    spatial token in later time steps a target, so the causal ablation cannot be
    mistaken for the representation objective used by V-JEPA v1.
    """

    temporal, height, width = (int(value) for value in grid_size)
    if min(temporal, height, width) <= 0:
        raise ValueError("grid dimensions must be positive")
    if temporal < 2:
        raise ValueError("causal future masking requires at least two temporal tokens")
    split = temporal // 2 if context_temporal_tokens is None else context_temporal_tokens
    if not 0 < split < temporal:
        raise ValueError("context_temporal_tokens must be in [1, temporal_tokens - 1]")
    target = torch.zeros((temporal, height, width), dtype=torch.bool)
    target[split:] = True
    return target


class TubeletTokenizer(nn.Module):
    """Turn ``[B,T,C,H,W]`` videos into flattened Conv3D tubelet tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        tubelet_size: GridSize = (2, 16, 16),
    ) -> None:
        super().__init__()
        if min(in_channels, embed_dim, *tubelet_size) <= 0:
            raise ValueError("channel, embedding, and tubelet dimensions must be positive")
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.tubelet_size = tuple(int(value) for value in tubelet_size)
        self.projection = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.tubelet_size,
            stride=self.tubelet_size,
        )

    def output_grid(self, num_frames: int, height: int, width: int) -> GridSize:
        dimensions = (num_frames, height, width)
        if any(size % patch != 0 for size, patch in zip(dimensions, self.tubelet_size)):
            raise ValueError(
                f"video dimensions {dimensions} must be divisible by {self.tubelet_size}"
            )
        return tuple(size // patch for size, patch in zip(dimensions, self.tubelet_size))  # type: ignore[return-value]

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, C, H, W]")
        _, temporal, channels, height, width = frames.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels at dimension 2, got {channels}"
            )
        self.output_grid(temporal, height, width)
        features = self.projection(frames.permute(0, 2, 1, 3, 4).contiguous())
        return features.flatten(2).transpose(1, 2)


class VideoTransformerEncoder(nn.Module):
    """Tiny ViT-style video encoder shared by the student and EMA teacher."""

    def __init__(
        self,
        *,
        num_frames: int,
        image_size: int | tuple[int, int],
        tubelet_size: GridSize = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        image_height, image_width = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.tokenizer = TubeletTokenizer(in_channels, embed_dim, tubelet_size)
        self.grid_size = self.tokenizer.output_grid(
            num_frames, image_height, image_width
        )
        self.num_tokens = math.prod(self.grid_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, frames: Tensor, visible_mask: Tensor | None = None) -> Tensor:
        tokens = self.tokenizer(frames)
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"configured for {self.num_tokens} tokens but received {tokens.shape[1]}"
            )
        positions = self.position_embedding
        if visible_mask is not None:
            visible_mask = _flatten_mask(visible_mask, self.num_tokens, tokens.device)
            tokens = tokens[:, visible_mask]
            positions = positions[:, visible_mask]
        return self.norm(self.blocks(tokens + positions))


class VideoJEPAPredictor(nn.Module):
    """Predict target positions from sparse context representations."""

    def __init__(
        self,
        *,
        num_tokens: int,
        encoder_dim: int,
        predictor_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads:
            raise ValueError("predictor_dim must be divisible by num_heads")
        self.num_tokens = int(num_tokens)
        self.context_projection = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_tokens, predictor_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        context_latents: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        context_mask = _flatten_mask(
            context_mask, self.num_tokens, context_latents.device
        )
        target_mask = _flatten_mask(target_mask, self.num_tokens, context_latents.device)
        if context_latents.shape[1] != int(context_mask.sum()):
            raise ValueError("context_latents count does not match context_mask")

        batch_size = context_latents.shape[0]
        sequence = self.mask_token.expand(batch_size, self.num_tokens, -1).clone()
        # CopySlices is differentiable with respect to the projected context values.
        sequence[:, context_mask] = self.context_projection(context_latents)
        sequence = self.norm(self.blocks(sequence + self.position_embedding))
        return self.output_projection(sequence[:, target_mask])


@dataclass(frozen=True)
class VideoJEPAOutput:
    loss: Tensor
    prediction: Tensor
    target: Tensor
    context_mask: Tensor
    target_mask: Tensor


def _flatten_mask(mask: Tensor, num_tokens: int, device: torch.device) -> Tensor:
    if mask.dtype is not torch.bool:
        raise ValueError("masks must have boolean dtype")
    flattened = mask.flatten().to(device=device)
    if flattened.numel() != num_tokens:
        raise ValueError(f"expected {num_tokens} mask values, got {flattened.numel()}")
    return flattened


class TinyVideoJEPA(nn.Module):
    """Educational V-JEPA with a student, full-video EMA teacher, and L1 loss."""

    def __init__(
        self,
        *,
        num_frames: int = 8,
        image_size: int | tuple[int, int] = 112,
        tubelet_size: GridSize = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 4,
        encoder_heads: int = 3,
        predictor_dim: int = 192,
        predictor_depth: int = 4,
        predictor_heads: int = 3,
    ) -> None:
        super().__init__()
        self.context_encoder = VideoTransformerEncoder(
            num_frames=num_frames,
            image_size=image_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = VideoJEPAPredictor(
            num_tokens=self.context_encoder.num_tokens,
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

    @property
    def grid_size(self) -> GridSize:
        return self.context_encoder.grid_size

    @torch.no_grad()
    def encode(self, frames: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode a complete or explicitly selected clip with the EMA encoder."""

        return self.target_encoder(frames, visible_mask=mask)

    def forward(self, frames: Tensor, target_mask: Tensor) -> VideoJEPAOutput:
        target_mask = _flatten_mask(
            target_mask, self.context_encoder.num_tokens, frames.device
        )
        context_mask = ~target_mask
        if not bool(target_mask.any()) or not bool(context_mask.any()):
            raise ValueError("target_mask must contain target and context tokens")

        context_latents = self.context_encoder(frames, visible_mask=context_mask)
        prediction = self.predictor(context_latents, context_mask, target_mask)

        # The teacher sees the complete clip.  Targets are selected only after the
        # encoder, matching the load-bearing V-JEPA design choice.
        with torch.no_grad():
            full_target_latents = self.target_encoder(frames)
            target = full_target_latents[:, target_mask]
        loss = F.l1_loss(prediction, target)
        return VideoJEPAOutput(
            loss=loss,
            prediction=prediction,
            target=target,
            context_mask=context_mask,
            target_mask=target_mask,
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Apply one exponential-moving-average update to the target encoder."""

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        for target, context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target.mul_(momentum).add_(context, alpha=1.0 - momentum)
        for target, context in zip(
            self.target_encoder.buffers(), self.context_encoder.buffers()
        ):
            if target.is_floating_point():
                target.mul_(momentum).add_(context, alpha=1.0 - momentum)
            else:
                target.copy_(context)

    def train(self, mode: bool = True) -> TinyVideoJEPA:
        super().train(mode)
        # EMA teachers are targets, never stochastic train-time networks.
        self.target_encoder.eval()
        return self


__all__ = [
    "GridSize",
    "TinyVideoJEPA",
    "TubeMasks",
    "TubeletTokenizer",
    "VideoJEPAOutput",
    "VideoJEPAPredictor",
    "VideoTransformerEncoder",
    "generate_future_mask",
    "generate_tube_masks",
]
