#!/usr/bin/env python3
"""Extract image features with Meta's pinned I-JEPA encoder in an isolated process."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from jepa_lab.adapters import cosine_to_reference, select_checkpoint_state
from jepa_lab.device import select_device
from jepa_lab.runlog import export_features
from jepa_lab.stimuli import IMAGE_VIEW_ORDER, image_views
from jepa_lab.upstreams import sha256_file

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ijepa"
sys.path.insert(0, str(UPSTREAM))

vit = import_module("src.models.vision_transformer")

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pinned official I-JEPA patch and pooled feature extraction"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model",
        choices=("vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"),
        default="vit_tiny",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--state-key",
        choices=("target_encoder", "encoder"),
        default="target_encoder",
    )
    parser.add_argument("--allow-incompatible", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--comparison-image",
        type=Path,
        help=(
            "Optional unrelated image. Only its normal view is encoded and appended "
            "to the exported features for a positive-vs-negative similarity check."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not UPSTREAM.exists():
        raise SystemExit("Missing upstream/ijepa; run git submodule update --init --recursive")
    if args.image_size % args.patch_size:
        parser.error("image-size must be divisible by patch-size")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.comparison_image is not None and not args.comparison_image.is_file():
        parser.error(f"comparison image does not exist: {args.comparison_image}")

    device = select_device(args.device)
    model = getattr(vit, args.model)(
        img_size=[args.image_size], patch_size=args.patch_size
    )
    load_result: dict[str, object] = {
        "checkpoint": None,
        "sha256": None,
        "state_key": None,
        "missing": [],
        "unexpected": [],
    }
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            parser.error("checkpoint root must be a mapping")
        state = select_checkpoint_state(checkpoint, preferred_keys=(args.state_key,))
        incompatible = model.load_state_dict(state, strict=False)
        load_result = {
            "checkpoint": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "state_key": args.state_key,
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }
        if (incompatible.missing_keys or incompatible.unexpected_keys) and not args.allow_incompatible:
            parser.error(
                "checkpoint/model mismatch; inspect missing/unexpected keys or pass "
                "--allow-incompatible for an explicitly diagnostic run"
            )

    model.eval().to(device)
    variants = image_views(args.image, args.image_size)
    features: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, pixels in variants.items():
            normalized = ((pixels.unsqueeze(0) - MEAN) / STD).to(device)
            features[name] = model(normalized).float().cpu()
        if args.comparison_image is not None:
            comparison_pixels = image_views(args.comparison_image, args.image_size)[
                "normal"
            ]
            normalized = ((comparison_pixels.unsqueeze(0) - MEAN) / STD).to(device)
            features["different_image"] = model(normalized).float().cpu()

    output_order = list(IMAGE_VIEW_ORDER)
    if args.comparison_image is not None:
        output_order.append("different_image")

    payload: dict[str, object] = {
        "implementation": "facebookresearch/ijepa@52c1ae95",
        "weights": "pretrained" if args.checkpoint else "random-init shape smoke only",
        "device": str(device),
        "model": args.model,
        "patch_size": args.patch_size,
        "image_size": args.image_size,
        "input_source": str(args.image) if args.image else "deterministic synthetic image",
        "comparison_source": (
            str(args.comparison_image) if args.comparison_image is not None else None
        ),
        "variant_order": output_order,
        "stimulus_id": args.image.stem if args.image else "shared-image-scene-v1",
        "feature_shape": list(features["normal"].shape),
        "expected_tokens": (args.image_size // args.patch_size) ** 2,
        "cosine_to_normal": cosine_to_reference(features),
        "pooling": "mean over patch tokens for cosine; raw patch tokens exported",
        "preprocessing": "RGB, resize/center-crop, ImageNet mean/std",
        "load_result": load_result,
    }
    if args.output is not None:
        stacked = np.concatenate(
            [features[name].numpy() for name in output_order], axis=0
        )
        export_features(
            args.output,
            stacked,
            labels=np.arange(len(output_order), dtype=np.int64),
            sample_ids=np.asarray(
                [
                    (
                        f"shared-image-scene-v1/{name}"
                        if args.image is None
                        else f"{args.image.stem}/{name}"
                    )
                    if name != "different_image"
                    else f"{args.comparison_image.stem}/normal"
                    for name in output_order
                ]
            ),
            metadata=payload,
        )
        payload["output"] = str(args.output)
    print(json.dumps(payload, indent=2))
    return int(features["normal"].shape[1] != payload["expected_tokens"])


if __name__ == "__main__":
    raise SystemExit(main())
