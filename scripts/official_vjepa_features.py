#!/usr/bin/env python3
"""Inspect token shapes and temporal sensitivity with pinned official V-JEPA code."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from jepa_lab.adapters import (
    canonical_to_vjepa,
    cosine_to_reference,
    select_checkpoint_state,
    temporal_variants,
)
from jepa_lab.device import select_device
from jepa_lab.runlog import export_features
from jepa_lab.stimuli import VIDEO_VARIANT_ORDER, moving_square_video, temporal_frame_indices
from jepa_lab.upstreams import sha256_file

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "vjepa"
sys.path.insert(0, str(UPSTREAM))

vit = import_module("src.models.vision_transformer")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=("vit_tiny", "vit_large"), default="vit_tiny")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-incompatible", action="store_true")
    parser.add_argument("--output", type=Path, help="optional common feature NPZ")
    args = parser.parse_args()

    if not UPSTREAM.exists():
        raise SystemExit("Missing upstream/vjepa; run git submodule update --init --recursive")
    device = select_device(args.device)
    model_factory = getattr(vit, args.model)
    model = model_factory(
        img_size=args.image_size,
        patch_size=16,
        num_frames=args.frames,
        tubelet_size=2,
        uniform_power=True,
    )
    load_result: dict[str, object] = {"checkpoint": None, "missing": [], "unexpected": []}
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = select_checkpoint_state(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        load_result = {
            "checkpoint": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }
        if (incompatible.missing_keys or incompatible.unexpected_keys) and not args.allow_incompatible:
            parser.error(
                "checkpoint/model mismatch; inspect missing/unexpected keys or pass "
                "--allow-incompatible for an explicitly diagnostic run"
            )
    model.eval().to(device)

    canonical = moving_square_video(args.frames, args.image_size)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    canonical = ((canonical - mean) / std).to(device)
    variants = temporal_variants(canonical)
    features: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, frames in variants.items():
            features[name] = model(canonical_to_vjepa(frames)).cpu()

    grid = (
        args.frames // 2,
        args.image_size // 16,
        args.image_size // 16,
    )
    payload: dict[str, object] = {
        "implementation": "facebookresearch/jepa@51c59d51",
        "weights": "pretrained" if args.checkpoint else "random-init shape smoke only",
        "device": str(device),
        "canonical_input_shape": list(canonical.shape),
        "upstream_input_shape": list(canonical_to_vjepa(canonical).shape),
        "token_grid": list(grid),
        "expected_tokens": grid[0] * grid[1] * grid[2],
        "feature_shape": list(features["normal"].shape),
        "cosine_to_normal": cosine_to_reference(features),
        "load_result": load_result,
        "variant_order": ["normal", "reversed", "shuffled", "static"],
        "stimulus_id": "moving-square-left-to-right-v1",
        "pooling": "raw patch tokens exported; mean pooling used for cosine",
        "preprocessing": "deterministic RGB synthetic clip, ImageNet mean/std",
    }
    if args.output is not None:
        stacked = np.concatenate(
            [features[name].numpy() for name in VIDEO_VARIANT_ORDER], axis=0
        )
        export_features(
            args.output,
            stacked,
            labels=np.arange(4, dtype=np.int64),
            sample_ids=np.asarray(
                [f"moving-square-left-to-right-v1/{name}" for name in VIDEO_VARIANT_ORDER]
            ),
            frame_indices=temporal_frame_indices(args.frames),
            metadata=payload,
        )
        payload["output"] = str(args.output)
    print(json.dumps(payload, indent=2))
    return int(features["normal"].shape[1] != payload["expected_tokens"])


if __name__ == "__main__":
    raise SystemExit(main())
