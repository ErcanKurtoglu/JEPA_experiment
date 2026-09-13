#!/usr/bin/env python3
"""Run one transparent training step with Meta's pinned I-JEPA components.

This script deliberately lives outside the jepa_lab process: every official
JEPA repository owns a top-level package named src. Keeping one upstream per
process prevents accidental cross-imports.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from importlib import import_module
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ijepa"
sys.path.insert(0, str(UPSTREAM))

init_model = import_module("src.helper").init_model
MaskCollator = import_module("src.masks.multiblock").MaskCollator
apply_masks = import_module("src.masks.utils").apply_masks


def _route_upstream_logs_to_stderr() -> None:
    """Keep stdout machine-readable despite Meta's root logger configuration."""

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grad_norm(module: torch.nn.Module) -> float:
    squares = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=float, default=0.996)
    args = parser.parse_args()
    _route_upstream_logs_to_stderr()

    if not UPSTREAM.exists():
        raise SystemExit("Missing upstream/ijepa; run git submodule update --init --recursive")
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        model_name="vit_tiny",
        crop_size=args.image_size,
        pred_depth=4,
        pred_emb_dim=192,
    )
    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *predictor.parameters()], lr=5e-4
    )

    collator = MaskCollator(
        input_size=args.image_size,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.20),
        aspect_ratio=(0.75, 1.50),
        nenc=1,
        npred=4,
        min_keep=1,
        allow_overlap=False,
    )
    images, context_masks, target_masks = collator(
        [
            torch.randn(3, args.image_size, args.image_size)
            for _ in range(args.batch_size)
        ]
    )
    images = images.to(device)
    context_masks = [mask.to(device) for mask in context_masks]
    target_masks = [mask.to(device) for mask in target_masks]

    with torch.no_grad():
        full_target = target_encoder(images)
        normalized_target = F.layer_norm(full_target, (full_target.shape[-1],))
        selected_target = apply_masks(normalized_target, target_masks)

    context = encoder(images, context_masks)
    predicted_target = predictor(context, context_masks, target_masks)
    loss = F.smooth_l1_loss(predicted_target, selected_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    before = [parameter.detach().clone() for parameter in target_encoder.parameters()]
    optimizer.step()
    with torch.no_grad():
        for online, target in zip(encoder.parameters(), target_encoder.parameters()):
            target.mul_(args.momentum).add_(
                online.detach(), alpha=1.0 - args.momentum
            )
    ema_error = max(
        float(
            (
                after
                - (
                    args.momentum * old
                    + (1.0 - args.momentum) * online.detach()
                )
            )
            .abs()
            .max()
            .cpu()
        )
        for online, old, after in zip(
            encoder.parameters(), before, target_encoder.parameters()
        )
    )

    context_set = set(context_masks[0][0].detach().cpu().tolist())
    target_sets = [set(mask[0].detach().cpu().tolist()) for mask in target_masks]
    result = {
        "implementation": "facebookresearch/ijepa@52c1ae95",
        "device": str(device),
        "image_shape": list(images.shape),
        "full_target_shape": list(full_target.shape),
        "context_shape": list(context.shape),
        "prediction_shape": list(predicted_target.shape),
        "target_shape": list(selected_target.shape),
        "context_tokens": len(context_set),
        "target_tokens_each": [len(indices) for indices in target_sets],
        "context_target_disjoint": all(
            context_set.isdisjoint(indices) for indices in target_sets
        ),
        "target_target_overlap_allowed": any(
            bool(left & right)
            for index, left in enumerate(target_sets)
            for right in target_sets[index + 1 :]
        ),
        "loss": float(loss.detach().cpu()),
        "encoder_grad_norm": grad_norm(encoder),
        "predictor_grad_norm": grad_norm(predictor),
        "target_has_grad": any(
            parameter.grad is not None for parameter in target_encoder.parameters()
        ),
        "ema_max_abs_error": ema_error,
        "target_masking_location": "target encoder output",
        "target_normalization": "feature-wise layer_norm",
        "loss_function": "smooth_l1",
    }
    print(json.dumps(result, indent=2))

    checks = (
        result["context_target_disjoint"],
        result["encoder_grad_norm"] > 0,
        result["predictor_grad_norm"] > 0,
        not result["target_has_grad"],
        result["ema_max_abs_error"] < 1e-6,
    )
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
