#!/usr/bin/env python3
"""Compare deterministic image perturbations with official DINO ViT-S/16.

This is a representation-objective comparison for M2, not a claim that DINO
is a JEPA. The default Torch Hub reference is an immutable commit from the
official repository so a later change to ``main`` cannot alter the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from jepa_lab.device import select_device
from jepa_lab.runlog import export_features
from jepa_lab.stimuli import IMAGE_VIEW_ORDER, image_views

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_SOURCE_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="optional local image; synthetic is the default")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="optional common feature NPZ")
    parser.add_argument(
        "--hub-ref",
        default=f"facebookresearch/dino:{DINO_SOURCE_COMMIT}",
        help="official Torch Hub repo/ref; default is an immutable commit",
    )
    args = parser.parse_args()

    if not args.hub_ref.startswith("facebookresearch/dino:"):
        raise SystemExit("--hub-ref must refer to the official facebookresearch/dino repository")
    if args.image is not None and not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    raw_views = image_views(args.image, args.image_size)
    views = {
        name: TF.normalize(raw_views[name], IMAGENET_MEAN, IMAGENET_STD)
        for name in IMAGE_VIEW_ORDER
    }

    device = select_device(args.device)
    # Official facebookresearch/dino Torch Hub entry: DINO ViT-Small, patch 16.
    model = torch.hub.load(
        args.hub_ref,
        "dino_vits16",
        source="github",
        trust_repo=True,
        verbose=True,
    )
    model.eval().to(device)
    batch = torch.stack([views[name] for name in IMAGE_VIEW_ORDER]).to(device)
    with torch.inference_mode():
        features = model(batch).float().cpu()
    if features.ndim != 2 or features.shape[0] != len(IMAGE_VIEW_ORDER):
        raise RuntimeError(f"unexpected DINO output shape: {tuple(features.shape)}")

    normal = features[:1]
    similarities = F.cosine_similarity(normal, features, dim=-1)
    requested_ref = args.hub_ref.rsplit(":", maxsplit=1)[-1]
    mutable = requested_ref in {"main", "master"}
    provenance = {
        "implementation": "official facebookresearch/dino torch.hub dino_vits16",
        "requested_hub_ref": args.hub_ref,
        "repository_status": "mutable branch / local Torch Hub cache" if mutable else "immutable requested ref",
        "reproducibility_note": (
            "A mutable branch was explicitly requested; record the resolved cache revision."
            if mutable
            else "Official DINO code is requested at the recorded immutable Git commit."
        ),
        "torch_hub_cache": torch.hub.get_dir(),
        "objective_role": "DINO view-invariance comparison; DINO is not a JEPA",
        "weights": "official pretrained DINO ViT-S/16 loaded by Torch Hub",
        "image_source": str(args.image) if args.image else "deterministic synthetic image",
        "variant_order": list(IMAGE_VIEW_ORDER),
        "stimulus_id": "shared-image-scene-v1",
        "input_shape": list(batch.shape),
        "feature_shape": list(features.shape),
        "cosine_to_normal": {
            name: float(value) for name, value in zip(IMAGE_VIEW_ORDER, similarities)
        },
    }
    if args.output:
        export_features(
            args.output,
            features.unsqueeze(1).numpy().astype(np.float32, copy=False),
            labels=np.arange(len(IMAGE_VIEW_ORDER), dtype=np.int64),
            sample_ids=np.asarray(
                [f"shared-image-scene-v1/{name}" for name in IMAGE_VIEW_ORDER]
            ),
            metadata=provenance,
        )
        provenance["output"] = str(args.output)
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
