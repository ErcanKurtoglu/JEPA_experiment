#!/usr/bin/env python3
"""Single-device training runner built from Meta's pinned I-JEPA components.

The script intentionally runs in its own process because the official JEPA
repositories expose colliding top-level ``src`` packages. It supports the M2
baseline recipe plus the four small ablations without editing upstream code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from jepa_lab.checkpointing import load_training_checkpoint, save_training_checkpoint
from jepa_lab.device import seed_everything, seeded_generator, select_device
from jepa_lab.evaluation import evaluate_representation
from jepa_lab.foundations import environment_report
from jepa_lab.masking import RandomPatchMasker
from jepa_lab.metrics import measure_collapse

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ijepa"
sys.path.insert(0, str(UPSTREAM))

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks


def _route_upstream_logs_to_stderr() -> None:
    """Keep stdout machine-readable despite Meta's root logger configuration."""

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)


@dataclass(frozen=True, slots=True)
class OfficialTrainResult:
    first_loss: float
    final_loss: float
    mean_loss: float
    first_window_mean: float
    last_window_mean: float
    relative_window_loss_drop: float
    optimizer_steps: int
    start_step: int
    feature_std: float
    mean_cosine: float
    effective_rank: float
    target_has_grad: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pinned official I-JEPA components on one device"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--data-root", type=Path, help="Imagenette root containing train/ and val/"
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model", default="vit_tiny")
    parser.add_argument("--predictor-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, help="derive optimizer steps from epoch count")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--num-targets", type=int, default=4)
    parser.add_argument(
        "--mask-kind", choices=("multiblock", "random-patch"), default="multiblock"
    )
    parser.add_argument(
        "--target-masking", choices=("output", "input"), default="output"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--evaluate-probes",
        action="store_true",
        help="evaluate trained and same-architecture random encoders on train/val",
    )
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-workers", type=int, default=2)
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not UPSTREAM.is_dir():
        parser.error("missing upstream/ijepa; initialise the pinned submodule")
    if args.image_size % args.patch_size:
        parser.error("image-size must be divisible by patch-size")
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.num_targets) <= 0:
        parser.error("steps, batch-size, gradient-accumulation and num-targets must be positive")
    if not 0.0 <= args.ema_start <= args.ema_end <= 1.0:
        parser.error("EMA values must satisfy 0 <= start <= end <= 1")
    if args.data_root is not None and not all(
        (args.data_root / split).is_dir() for split in ("train", "val")
    ):
        parser.error(f"data-root must contain train/ and val/: {args.data_root}")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("max-images must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.evaluate_probes and args.data_root is None:
        parser.error("evaluate-probes requires an Imagenette data-root")
    if args.probe_batch_size <= 0 or args.probe_workers < 0:
        parser.error("probe-batch-size must be positive and probe-workers non-negative")
    if args.mask_kind == "multiblock" and args.image_size // args.patch_size < 7:
        parser.error("official multiblock masking requires at least a 7x7 teaching grid")


def _dataset(args: argparse.Namespace) -> Dataset[object]:
    if args.data_root is None:
        generator = torch.Generator().manual_seed(args.seed)
        images = torch.randn(
            max(128, args.batch_size),
            3,
            args.image_size,
            args.image_size,
            generator=generator,
        )
        dataset: Dataset[object] = TensorDataset(images)
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        dataset = datasets.ImageFolder(args.data_root / "train", transform=transform)
    if args.max_images is not None:
        if args.max_images > len(dataset):
            raise ValueError(f"max-images={args.max_images} exceeds dataset size {len(dataset)}")
        dataset = Subset(dataset, range(args.max_images))
    return dataset


def _images(batch: object) -> Tensor:
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if not isinstance(batch, Tensor) or batch.ndim != 4:
        raise ValueError("loader must yield image tensors [B,C,H,W]")
    return batch


def _forever(loader: DataLoader[object]) -> Iterator[object]:
    while True:
        yield from loader


def _momentum(start: float, end: float, step: int, total: int) -> float:
    return end if total == 1 else start + (end - start) * step / (total - 1)


def _evaluation_dataset(root: Path, split: str, image_size: int) -> Dataset[object]:
    transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return datasets.ImageFolder(root / split, transform=transform)


def _dataset_manifest(root: Path) -> dict[str, object]:
    """Describe the exact ImageFolder split without hashing image contents."""

    split_counts: dict[str, int] = {}
    split_manifest_sha256: dict[str, str] = {}
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"dataset split is missing: {split_root}")
        entries = sorted(
            str(path.relative_to(root)) for path in split_root.rglob("*") if path.is_file()
        )
        split_counts[split] = len(entries)
        split_manifest_sha256[split] = hashlib.sha256(
            "\n".join(entries).encode("utf-8")
        ).hexdigest()
    classes = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())
    validation_classes = sorted(
        path.name for path in (root / "val").iterdir() if path.is_dir()
    )
    if validation_classes != classes:
        raise ValueError("train/val ImageFolder class ordering does not match")
    return {
        "dataset": root.name,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "split_counts": split_counts,
        "split_manifest_sha256": split_manifest_sha256,
    }


@torch.inference_mode()
def _pooled_features(
    model: nn.Module,
    loader: DataLoader[object],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.eval()
    features: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("probe loader must yield images and labels")
        images, target = batch[:2]
        if not isinstance(images, Tensor) or not isinstance(target, Tensor):
            raise TypeError("probe images and labels must be tensors")
        features.append(model(images.to(device)).mean(dim=1).float().cpu())
        labels.append(target.cpu())
    return torch.cat(features), torch.cat(labels)


def _official_masks(
    images: Tensor,
    collator: MaskCollator,
) -> tuple[list[Tensor], list[Tensor]]:
    _, context, target = collator([image.cpu() for image in images])
    return context, target


def _random_masks(
    images: Tensor,
    masker: RandomPatchMasker,
    *,
    seed: int,
) -> tuple[list[Tensor], list[Tensor]]:
    masks = masker(images.shape[0], generator=seeded_generator(seed))
    return [masks.context], [masks.targets[:, index] for index in range(masks.targets.shape[1])]


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate(args, parser)
    started = time.perf_counter()
    seed_everything(args.seed)
    device = select_device(args.device)
    _route_upstream_logs_to_stderr()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    encoder, predictor = init_model(
        device=device,
        patch_size=args.patch_size,
        model_name=args.model,
        crop_size=args.image_size,
        pred_depth=args.predictor_depth,
        pred_emb_dim=args.predictor_dim,
    )
    target_encoder = copy.deepcopy(encoder).to(device).requires_grad_(False)
    target_encoder.eval()
    state = nn.ModuleDict(
        {"encoder": encoder, "predictor": predictor, "target_encoder": target_encoder}
    )
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *predictor.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        loaded = load_training_checkpoint(args.resume, state, optimizer, map_location=device)
        start_step = loaded.step

    dataset = _dataset(args)
    loader: DataLoader[object] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if len(loader) == 0:
        parser.error("training dataset must contain at least one full microbatch")
    steps_per_epoch = max(
        1,
        (len(loader) + args.gradient_accumulation - 1) // args.gradient_accumulation,
    )
    total_steps = args.steps if args.epochs is None else args.epochs * steps_per_epoch
    if start_step >= total_steps:
        parser.error("resume requires requested total steps to exceed the saved step")
    batches = _forever(loader)
    grid = (args.image_size // args.patch_size,) * 2
    official_collator = MaskCollator(
        input_size=args.image_size,
        patch_size=args.patch_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.20),
        aspect_ratio=(0.75, 1.50),
        nenc=1,
        npred=args.num_targets,
        min_keep=1,
        allow_overlap=False,
    )
    random_masker = RandomPatchMasker(
        grid,
        num_targets=args.num_targets,
        target_scale=(0.15, 0.20),
        context_scale=(0.85, 1.0),
    )

    losses: list[float] = []
    last_images: Tensor | None = None
    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for microstep in range(args.gradient_accumulation):
            images = _images(next(batches)).to(device)
            if args.mask_kind == "multiblock":
                context_masks, target_masks = _official_masks(images, official_collator)
            else:
                context_masks, target_masks = _random_masks(
                    images,
                    random_masker,
                    seed=args.seed + step * args.gradient_accumulation + microstep,
                )
            context_masks = [mask.to(device) for mask in context_masks]
            target_masks = [mask.to(device) for mask in target_masks]
            with torch.no_grad():
                if args.target_masking == "output":
                    target = target_encoder(images)
                    target = F.layer_norm(target, (target.shape[-1],))
                    target = apply_masks(target, target_masks)
                else:
                    target = target_encoder(images, target_masks)
                    target = F.layer_norm(target, (target.shape[-1],))
            context = encoder(images, context_masks)
            prediction = predictor(context, context_masks, target_masks)
            loss = F.smooth_l1_loss(prediction, target)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
            last_images = images
        optimizer.step()
        momentum = _momentum(args.ema_start, args.ema_end, step, total_steps)
        with torch.no_grad():
            for online, teacher in zip(encoder.parameters(), target_encoder.parameters()):
                teacher.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        losses.append(accumulated)

    if last_images is None:
        last_images = _images(next(batches)).to(device)
    with torch.inference_mode():
        features = target_encoder(last_images)
    collapse = measure_collapse(features)
    window = min(50, len(losses))
    first_window_mean = float(np.mean(losses[:window])) if window else float("nan")
    last_window_mean = float(np.mean(losses[-window:])) if window else float("nan")
    relative_window_loss_drop = (
        (first_window_mean - last_window_mean) / first_window_mean
        if first_window_mean > 0.0
        else float("nan")
    )
    result = OfficialTrainResult(
        first_loss=losses[0] if losses else float("nan"),
        final_loss=losses[-1] if losses else float("nan"),
        mean_loss=float(np.mean(losses)) if losses else float("nan"),
        first_window_mean=first_window_mean,
        last_window_mean=last_window_mean,
        relative_window_loss_drop=relative_window_loss_drop,
        optimizer_steps=total_steps - start_step,
        start_step=start_step,
        feature_std=collapse.feature_std,
        mean_cosine=collapse.mean_cosine_similarity,
        effective_rank=collapse.effective_rank,
        target_has_grad=any(parameter.grad is not None for parameter in target_encoder.parameters()),
    )
    metadata = {
        "implementation": "facebookresearch/ijepa@52c1ae95d05f743e000e8f10a1f3a79b10cff048",
        "device": str(device),
        "environment": environment_report(),
        "seed": args.seed,
        "total_steps": total_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "model": args.model,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "num_targets": args.num_targets,
        "mask_kind": args.mask_kind,
        "target_masking": args.target_masking,
        "ema": [args.ema_start, args.ema_end],
        "data_root": str(args.data_root) if args.data_root else "deterministic synthetic smoke",
        "dataset_manifest": (
            _dataset_manifest(args.data_root)
            if args.data_root is not None
            else {"dataset": "deterministic-synthetic-smoke", "seed": args.seed}
        ),
        "max_images": args.max_images,
        "metrics": asdict(result),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    feature_dim = int(features.shape[-1])
    std_threshold = 0.01 / float(np.sqrt(feature_dim / 192.0))
    rank_threshold = max(2.0, 0.05 * feature_dim)
    collapse_checks = {
        "mean_cosine_lt_0_99": collapse.mean_cosine_similarity < 0.99,
        "effective_rank_gte_threshold": collapse.effective_rank >= rank_threshold,
        "feature_std_gte_threshold": collapse.feature_std >= std_threshold,
    }
    metadata["collapse_acceptance"] = {
        "feature_dim": feature_dim,
        "thresholds": {
            "mean_cosine_max_exclusive": 0.99,
            "effective_rank_min": rank_threshold,
            "feature_std_min": std_threshold,
        },
        "checks": collapse_checks,
        "passed": all(collapse_checks.values()),
        "note": (
            "These teaching-scale thresholds are fixed before the baseline and must be "
            "interpreted alongside the same-architecture random encoder and downstream probes."
        ),
    }
    if args.max_images == 128 and total_steps >= 300 and start_step == 0:
        metadata["overfit_acceptance"] = {
            "criterion": "last-50 mean loss is at least 30% below first-50 mean",
            "passed": relative_window_loss_drop >= 0.30,
        }
    # Persist the trained state before optional probes. Feature extraction and
    # probe fitting are deliberately downstream evaluation: a probe failure
    # must not discard an otherwise completed multi-hour training run.
    if args.checkpoint_output is not None:
        save_training_checkpoint(
            args.checkpoint_output,
            state,
            optimizer,
            step=total_steps,
            metadata={key: value for key, value in metadata.items() if key != "metrics"},
        )
        metadata["checkpoint"] = str(args.checkpoint_output)
    if args.evaluate_probes:
        assert args.data_root is not None
        train_evaluation = DataLoader(
            _evaluation_dataset(args.data_root, "train", args.image_size),
            batch_size=args.probe_batch_size,
            shuffle=False,
            num_workers=args.probe_workers,
        )
        validation_evaluation = DataLoader(
            _evaluation_dataset(args.data_root, "val", args.image_size),
            batch_size=args.probe_batch_size,
            shuffle=False,
            num_workers=args.probe_workers,
        )
        trained_train_x, train_y = _pooled_features(
            target_encoder, train_evaluation, device
        )
        trained_val_x, val_y = _pooled_features(
            target_encoder, validation_evaluation, device
        )
        trained_scores = evaluate_representation(
            trained_train_x,
            train_y,
            trained_val_x,
            val_y,
            k=20,
            probe_steps=200,
        )

        seed_everything(args.seed)
        random_encoder, _ = init_model(
            device=device,
            patch_size=args.patch_size,
            model_name=args.model,
            crop_size=args.image_size,
            pred_depth=args.predictor_depth,
            pred_emb_dim=args.predictor_dim,
        )
        random_train_x, random_train_y = _pooled_features(
            random_encoder, train_evaluation, device
        )
        random_val_x, random_val_y = _pooled_features(
            random_encoder, validation_evaluation, device
        )
        random_scores = evaluate_representation(
            random_train_x,
            random_train_y,
            random_val_x,
            random_val_y,
            k=20,
            probe_steps=200,
        )
        deltas = {
            "knn": trained_scores.knn_accuracy - random_scores.knn_accuracy,
            "linear": trained_scores.linear_accuracy - random_scores.linear_accuracy,
        }
        metadata["representation_evaluation"] = {
            "trained": trained_scores.as_dict(),
            "random": random_scores.as_dict(),
            "absolute_deltas": deltas,
            "criterion": "either k-NN or frozen-linear delta is at least 0.05",
            "passes_plus_5_points": any(value >= 0.05 for value in deltas.values()),
        }
    metadata["elapsed_seconds"] = time.perf_counter() - started
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return int(result.target_has_grad or not np.isfinite(result.final_loss))


if __name__ == "__main__":
    raise SystemExit(main())
