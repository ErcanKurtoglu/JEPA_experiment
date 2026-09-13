"""A compact, original I-JEPA implementation for white-box laboratories.

The module intentionally favors visible tensor flow over training throughput. It
implements the load-bearing I-JEPA ideas: a context encoder, a stop-gradient EMA
target encoder, output-space target masking, and a position-aware predictor.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .ema import freeze_module, update_ema
from .masking import ImageMasks

Size2D = int | tuple[int, int]


def _pair(value: Size2D) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, M]`` token indices from a ``[B, N, D]`` sequence."""

    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("tokens and indices must have shapes [B, N, D] and [B, M]")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("tokens and indices must have equal batch size")
    return torch.gather(
        tokens,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


def gather_target_blocks(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, K, M]`` blocks from a ``[B, N, D]`` sequence."""

    if tokens.ndim != 3 or indices.ndim != 3:
        raise ValueError("tokens and indices must have shapes [B, N, D] and [B, K, M]")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("tokens and indices must have equal batch size")
    expanded = tokens[:, None].expand(-1, indices.shape[1], -1, -1)
    return torch.gather(
        expanded,
        dim=2,
        index=indices.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1]),
    )


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches with a Conv2d projection."""

    def __init__(
        self,
        image_size: Size2D = 224,
        patch_size: Size2D = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        if any(image % patch for image, patch in zip(self.image_size, self.patch_size)):
            raise ValueError("image dimensions must be divisible by patch dimensions")
        self.grid_size = tuple(
            image // patch for image, patch in zip(self.image_size, self.patch_size)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, received {tuple(images.shape[-2:])}"
            )
        return self.projection(images).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class VisionTransformer(nn.Module):
    """Minimal patch-token ViT with optional pre-encoder token selection."""

    def __init__(
        self,
        image_size: Size2D = 224,
        patch_size: Size2D = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 3,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.embed_dim = embed_dim
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.patch_embed.grid_size

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def forward(
        self, images: torch.Tensor, indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.patch_embed(images) + self.position_embedding
        if indices is not None:
            tokens = gather_tokens(tokens, indices)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class JEPAPredictor(nn.Module):
    """Predict target latents from encoded context and target positions."""

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        num_patches: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(encoder_dim, predictor_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(predictor_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_targets, target_length = target_indices.shape
        context = self.context_projection(context_latents)
        context_positions = gather_tokens(
            self.position_embedding.expand(batch_size, -1, -1), context_indices
        )
        context = context + context_positions
        context = context[:, None].expand(-1, num_targets, -1, -1)

        all_positions = self.position_embedding.expand(batch_size, -1, -1)
        target_positions = gather_target_blocks(all_positions, target_indices)
        target_tokens = self.mask_token.expand(
            batch_size, num_targets, target_length, -1
        ) + target_positions

        tokens = torch.cat((context, target_tokens), dim=2)
        tokens = tokens.reshape(batch_size * num_targets, tokens.shape[2], -1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        predictions = tokens[:, -target_length:]
        predictions = self.output_projection(predictions)
        return predictions.reshape(batch_size, num_targets, target_length, -1)


@dataclass(frozen=True, slots=True)
class ImageJEPAOutput:
    loss: torch.Tensor
    predicted: torch.Tensor
    target: torch.Tensor
    context: torch.Tensor


@dataclass(frozen=True, slots=True)
class ImageJEPAStepResult:
    loss: float
    context_grad_norm: float
    predictor_grad_norm: float
    target_has_grad: bool
    all_finite: bool
    ema_max_error: float


class ImageJEPA(nn.Module):
    """White-box image JEPA with a frozen EMA teacher."""

    def __init__(
        self,
        *,
        image_size: Size2D = 224,
        patch_size: Size2D = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 3,
        encoder_heads: int = 3,
        predictor_dim: int = 192,
        predictor_depth: int = 4,
        predictor_heads: int = 3,
        mlp_ratio: float = 4.0,
        target_masking: Literal["output", "input"] = "output",
    ) -> None:
        super().__init__()
        if target_masking not in {"output", "input"}:
            raise ValueError("target_masking must be 'output' or 'input'")
        self.target_masking = target_masking
        self.context_encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
        )
        self.target_encoder = freeze_module(copy.deepcopy(self.context_encoder))
        self.predictor = JEPAPredictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            num_patches=self.context_encoder.num_patches,
            depth=predictor_depth,
            num_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
        )
        # Mirrors the parameter-free normalization applied to target features in
        # the official training loop before output-space masking.
        self.target_normalization = nn.LayerNorm(embed_dim, elementwise_affine=False)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.context_encoder.grid_size

    @property
    def num_patches(self) -> int:
        return self.context_encoder.num_patches

    def train(self, mode: bool = True) -> ImageJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, images: torch.Tensor, masks: ImageMasks) -> ImageJEPAOutput:
        if masks.grid_size != self.grid_size:
            raise ValueError(
                f"mask grid {masks.grid_size} does not match model grid {self.grid_size}"
            )
        if masks.batch_size != images.shape[0]:
            raise ValueError("mask and image batch sizes do not match")

        context = self.context_encoder(images, masks.context)
        predicted = self.predictor(context, masks.context, masks.targets)

        with torch.no_grad():
            if self.target_masking == "output":
                # The baseline teacher sees every patch. Target selection happens
                # only after complete encoding: the key I-JEPA design detail.
                full_target = self.target_encoder(images)
                full_target = self.target_normalization(full_target)
                target = gather_target_blocks(full_target, masks.targets)
            else:
                # Deliberate ablation: each target block is selected before the
                # target Transformer, removing surrounding target-side context.
                target = torch.stack(
                    [
                        self.target_normalization(
                            self.target_encoder(images, masks.targets[:, block])
                        )
                        for block in range(masks.num_targets)
                    ],
                    dim=1,
                )

        loss = F.smooth_l1_loss(predicted, target)
        return ImageJEPAOutput(
            loss=loss,
            predicted=predicted,
            target=target,
            context=context,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        update_ema(self.target_encoder, self.context_encoder, momentum)


def _gradient_norm(module: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared_norm)


def _module_is_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())


def train_one_step(
    model: ImageJEPA,
    images: torch.Tensor,
    masks: ImageMasks,
    optimizer: torch.optim.Optimizer,
    *,
    ema_momentum: float = 0.996,
    grad_clip: float | None = None,
) -> ImageJEPAStepResult:
    """Run one transparent optimization + EMA step and return diagnostics."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(images, masks)
    output.loss.backward()

    context_grad_norm = _gradient_norm(model.context_encoder)
    predictor_grad_norm = _gradient_norm(model.predictor)
    target_has_grad = any(
        parameter.grad is not None for parameter in model.target_encoder.parameters()
    )
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if grad_clip is not None:
        if grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        nn.utils.clip_grad_norm_(list(model.trainable_parameters()), grad_clip)

    target_before = {
        name: parameter.detach().clone()
        for name, parameter in model.target_encoder.named_parameters()
    }
    optimizer.step()
    model.update_target(ema_momentum)

    source_after = dict(model.context_encoder.named_parameters())
    target_after = dict(model.target_encoder.named_parameters())
    max_error = 0.0
    for name, old_target in target_before.items():
        expected = old_target * ema_momentum + source_after[name].detach() * (
            1.0 - ema_momentum
        )
        max_error = max(
            max_error,
            float((target_after[name].detach() - expected).abs().max()),
        )

    all_finite = (
        bool(torch.isfinite(output.loss))
        and gradients_finite
        and _module_is_finite(model)
    )
    return ImageJEPAStepResult(
        loss=float(output.loss.detach()),
        context_grad_norm=context_grad_norm,
        predictor_grad_norm=predictor_grad_norm,
        target_has_grad=target_has_grad,
        all_finite=all_finite,
        ema_max_error=max_error,
    )
