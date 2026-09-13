"""Command-line entry points shared by local, Kaggle, and Colab labs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import torch

from .action_world_model import ActionWorldModel
from .datasets import download_imagenette
from .device import seed_everything, select_device
from .experiments import train_action_steps
from .foundations import environment_report, foundation_report
from .image_jepa import ImageJEPA, train_one_step
from .masking import MultiBlockMasker
from .runlog import load_yaml
from .upstreams import (
    all_pins_match,
    download_and_record,
    repository_root,
    verify_upstreams,
)
from .video_jepa import TinyVideoJEPA, generate_tube_masks


def _doctor(strict: bool) -> int:
    upstreams = verify_upstreams()
    payload = {"environment": environment_report(), "upstreams": upstreams}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return int(strict and not all_pins_match(upstreams))


def _foundations() -> int:
    print(json.dumps(foundation_report(), indent=2, ensure_ascii=False))
    return 0


def _image_smoke(device_name: str) -> int:
    seed_everything(42)
    device = select_device(device_name)
    model = ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=4,
    ).to(device)
    masks = MultiBlockMasker(
        (4, 4),
        num_targets=4,
        target_scale=(0.15, 0.20),
        context_scale=(0.85, 1.0),
    )(2, generator=torch.Generator().manual_seed(42), device=device)
    images = torch.randn(2, 3, 32, 32, device=device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    result = train_one_step(model, images, masks, optimizer)
    print(json.dumps({"device": str(device), **asdict(result)}, indent=2))
    return int(
        result.target_has_grad
        or not result.all_finite
        or result.context_grad_norm <= 0
        or result.predictor_grad_norm <= 0
    )


def _video_smoke(device_name: str) -> int:
    seed_everything(42)
    device = select_device(device_name)
    model = TinyVideoJEPA(
        num_frames=4,
        image_size=32,
        tubelet_size=(2, 8, 8),
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=4,
    ).to(device)
    frames = torch.randn(2, 4, 3, 32, 32, device=device)
    mask = generate_tube_masks(
        model.grid_size,
        num_short=2,
        num_long=1,
        short_scale=0.15,
        long_scale=0.50,
        seed=42,
    ).target.to(device)
    optimizer = torch.optim.AdamW(
        [*model.context_encoder.parameters(), *model.predictor.parameters()], lr=1e-3
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(frames, mask)
    output.loss.backward()
    context_grad = sum(
        float(parameter.grad.square().sum())
        for parameter in model.context_encoder.parameters()
        if parameter.grad is not None
    ) ** 0.5
    optimizer.step()
    model.update_teacher(0.998)
    payload = {
        "device": str(device),
        "grid": list(model.grid_size),
        "loss": float(output.loss.detach()),
        "context_grad_norm": context_grad,
        "target_has_grad": any(
            parameter.grad is not None for parameter in model.target_encoder.parameters()
        ),
        "target_coverage": float(output.target_mask.float().mean()),
    }
    print(json.dumps(payload, indent=2))
    return int(
        not torch.isfinite(output.loss)
        or payload["target_has_grad"]
        or context_grad <= 0
    )


def _action_smoke(device_name: str, steps: int) -> int:
    seed_everything(42)
    device = select_device(device_name)
    model = ActionWorldModel(
        latent_dim=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_horizon=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    metrics = train_action_steps(
        model,
        optimizer,
        steps=steps,
        batch_size=128,
        horizon=4,
        device=device,
    )
    print(json.dumps({"device": str(device), **metrics.as_dict()}, indent=2))
    return int(
        metrics.final_loss >= metrics.first_loss
        or metrics.correct_action_l1 >= metrics.zero_action_l1
    )


def _download_checkpoint(name: str, root: str, expected_sha256: str | None) -> int:
    registry = load_yaml(repository_root() / "configs/checkpoints.yaml")["checkpoints"]
    if not isinstance(registry, dict) or name not in registry:
        available = ", ".join(sorted(registry)) if isinstance(registry, dict) else "none"
        raise ValueError(f"unknown checkpoint {name!r}; available: {available}")
    entry = registry[name]
    if not isinstance(entry, dict):
        raise TypeError(f"checkpoint entry {name!r} must be a mapping")
    url = entry.get("direct_official_url", entry.get("url"))
    if not isinstance(url, str):
        raise TypeError(f"checkpoint {name!r} has no direct URL")
    filename = Path(urlparse(url).path).name
    if not filename or "." not in filename:
        raise ValueError(
            f"checkpoint {name!r} uses a model repository rather than a direct file; "
            "run its official feature script so the model client can honor the revision"
        )
    metadata = download_and_record(
        name=name,
        url=url,
        destination=Path(root) / filename,
        expected_sha256=expected_sha256,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jepa-lab", description="Inspect and run the JEPA learning workspace"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="report hardware and pinned upstream state")
    doctor.add_argument("--strict", action="store_true", help="fail if a pin is missing/mismatched")
    subparsers.add_parser("foundations", help="run M0 token, gradient, and EMA checks")
    for name, help_text in (
        ("image-smoke", "run one white-box image-JEPA optimization step"),
        ("video-smoke", "run one white-box video-JEPA optimization step"),
    ):
        smoke = subparsers.add_parser(name, help=help_text)
        smoke.add_argument("--device", default="auto")
    action = subparsers.add_parser(
        "action-smoke", help="fit a tiny action-conditioned latent predictor"
    )
    action.add_argument("--device", default="auto")
    action.add_argument("--steps", type=int, default=200)
    dataset = subparsers.add_parser(
        "download-imagenette", help="download and provenance-record Imagenette-160"
    )
    dataset.add_argument("--root", default="data")
    checkpoint = subparsers.add_parser(
        "download-checkpoint", help="download an official checkpoint with SHA-256 provenance"
    )
    checkpoint.add_argument("name", help="key from configs/checkpoints.yaml")
    checkpoint.add_argument("--root", default="checkpoints")
    checkpoint.add_argument(
        "--sha256", help="optional known digest; otherwise record on first trusted download"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.strict)
    if args.command == "foundations":
        return _foundations()
    if args.command == "image-smoke":
        return _image_smoke(args.device)
    if args.command == "video-smoke":
        return _video_smoke(args.device)
    if args.command == "action-smoke":
        return _action_smoke(args.device, args.steps)
    if args.command == "download-imagenette":
        print(download_imagenette(args.root))
        return 0
    if args.command == "download-checkpoint":
        return _download_checkpoint(args.name, args.root, args.sha256)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
