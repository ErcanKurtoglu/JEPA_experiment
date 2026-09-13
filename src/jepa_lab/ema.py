"""Exponential-moving-average teacher utilities."""

from __future__ import annotations

import torch
from torch import nn


def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradients for every parameter and return ``module``."""

    module.requires_grad_(False)
    return module


@torch.no_grad()
def update_ema(
    target: nn.Module,
    source: nn.Module,
    momentum: float,
    *,
    update_buffers: bool = True,
) -> None:
    """Apply ``target = momentum*target + (1-momentum)*source`` exactly.

    Named parameters are matched strictly, preventing an accidental update of
    architecturally different teacher and student modules. Floating-point
    buffers use the same EMA; integer buffers are copied from the source.
    """

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")

    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise ValueError("target and source parameter names do not match")

    source_weight = 1.0 - momentum
    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        if target_parameter.shape != source_parameter.shape:
            raise ValueError(f"parameter shape mismatch for {name}")
        target_parameter.mul_(momentum).add_(source_parameter, alpha=source_weight)

    if not update_buffers:
        return

    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("target and source buffer names do not match")
    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.shape != source_buffer.shape:
            raise ValueError(f"buffer shape mismatch for {name}")
        if target_buffer.is_floating_point() or target_buffer.is_complex():
            target_buffer.mul_(momentum).add_(source_buffer, alpha=source_weight)
        else:
            target_buffer.copy_(source_buffer)
