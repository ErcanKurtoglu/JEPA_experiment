#!/usr/bin/env python3
"""Feature extraction and temporal controls with the official V-JEPA2 HF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_lab.adapters import cosine_to_reference, temporal_variants
from jepa_lab.device import select_device
from jepa_lab.runlog import export_features
from jepa_lab.stimuli import VIDEO_VARIANT_ORDER, moving_square_video, temporal_frame_indices

DEFAULT_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="immutable Hugging Face model repository revision",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--frames",
        type=int,
        default=64,
        help="temporal length; 64 matches the fpc64 checkpoint contract",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as error:
        raise SystemExit(
            "Install the optional Hugging Face stack: uv sync --extra hf"
        ) from error

    device = select_device(args.device)
    processor = AutoVideoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        attn_implementation="sdpa",
    )
    model.eval().to(device)

    canonical = (moving_square_video(args.frames, args.image_size) * 255).to(torch.uint8)
    variants = temporal_variants(canonical)
    features: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, frames in variants.items():
            # HF processor consumes one video's [T,C,H,W] tensor.
            inputs = processor(frames[0], return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(device)
            # This is the representation path used by Meta's pinned demo; it
            # bypasses the JEPA predictor and returns patch-wise encoder latents.
            features[name] = model.get_vision_features(pixel_values).float().cpu()

    similarities = cosine_to_reference(features)
    grid = (
        args.frames // int(model.config.tubelet_size),
        int(model.config.crop_size) // int(model.config.patch_size),
        int(model.config.crop_size) // int(model.config.patch_size),
    )
    payload = {
        "implementation": args.model,
        "model_revision": args.revision,
        "source_repo_pin": "facebookresearch/vjepa2@9a061fff",
        "device": str(device),
        "frames": args.frames,
        "token_grid": list(grid),
        "feature_shape": list(features["normal"].shape),
        "cosine_to_normal": similarities,
        "encoder_only": True,
        "variant_order": list(VIDEO_VARIANT_ORDER),
        "stimulus_id": "moving-square-left-to-right-v1",
    }
    if args.output:
        stacked = np.concatenate(
            [features[name].numpy() for name in VIDEO_VARIANT_ORDER], axis=0
        )
        export_features(
            args.output,
            stacked,
            labels=np.arange(4),
            sample_ids=np.asarray(
                [f"moving-square-left-to-right-v1/{name}" for name in VIDEO_VARIANT_ORDER]
            ),
            frame_indices=temporal_frame_indices(args.frames),
            metadata={
                **payload,
            },
        )
        payload["output"] = str(args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
