#!/usr/bin/env python3
"""Export inspectable k-NN examples from a trained official I-JEPA checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from jepa_lab.checkpointing import load_training_checkpoint
from jepa_lab.device import seed_everything, select_device
from jepa_lab.evaluation import fit_linear_probe
from jepa_lab.stimuli import IMAGE_VIEW_ORDER, image_views

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ijepa"
sys.path.insert(0, str(UPSTREAM))

from src.helper import init_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export correct/incorrect k-NN examples for a trained I-JEPA encoder"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="vit_tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--predictor-dim", type=int, default=192)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _evaluation_dataset(root: Path, split: str, image_size: int) -> datasets.ImageFolder:
    transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return datasets.ImageFolder(root / split, transform=transform)


@torch.inference_mode()
def _features(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    feature_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    model.eval()
    for images, labels in loader:
        feature_batches.append(model(images.to(device)).mean(dim=1).float().cpu())
        label_batches.append(labels.cpu())
    return torch.cat(feature_batches), torch.cat(label_batches)


@torch.no_grad()
def _knn_details(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    *,
    k: int,
    chunk_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    train = F.normalize(train_features.float(), dim=-1)
    validation = F.normalize(validation_features.float(), dim=-1)
    classes = torch.unique(train_labels, sorted=True)
    neighbour_indices: list[Tensor] = []
    neighbour_similarities: list[Tensor] = []
    predictions: list[Tensor] = []
    for start in range(0, validation.shape[0], chunk_size):
        similarities = validation[start : start + chunk_size] @ train.T
        values, indices = similarities.topk(min(k, train.shape[0]), dim=1)
        labels = train_labels[indices]
        votes = (labels.unsqueeze(-1) == classes).sum(dim=1)
        neighbour_indices.append(indices)
        neighbour_similarities.append(values)
        predictions.append(classes[votes.argmax(dim=1)])
    return (
        torch.cat(neighbour_indices),
        torch.cat(neighbour_similarities),
        torch.cat(predictions),
    )


def _representative_indices(
    predictions: Tensor,
    expected: Tensor,
    neighbour_labels: Tensor,
) -> Tensor:
    vote_strength = (neighbour_labels == predictions[:, None]).sum(dim=1)
    correct = torch.nonzero(predictions == expected, as_tuple=False).flatten()
    incorrect = torch.nonzero(predictions != expected, as_tuple=False).flatten()
    if correct.numel() == 0 or incorrect.numel() == 0:
        raise RuntimeError("evaluation requires at least one correct and one incorrect k-NN result")
    correct_index = correct[vote_strength[correct].argmax()]
    incorrect_index = incorrect[vote_strength[incorrect].argmax()]
    return torch.stack((correct_index, incorrect_index))


def main() -> int:
    args = _parser().parse_args()
    if not UPSTREAM.is_dir():
        raise SystemExit("missing upstream/ijepa; initialise the pinned submodule")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not all((args.data_root / split).is_dir() for split in ("train", "val")):
        raise FileNotFoundError("data-root must contain train/ and val/")
    if min(args.batch_size, args.k, args.probe_steps) <= 0 or args.workers < 0:
        raise ValueError("batch-size, k and probe-steps must be positive; workers cannot be negative")

    seed_everything(args.seed)
    device = select_device(args.device)
    encoder, predictor = init_model(
        device=device,
        patch_size=args.patch_size,
        model_name=args.model,
        crop_size=args.image_size,
        pred_depth=args.predictor_depth,
        pred_emb_dim=args.predictor_dim,
    )
    target_encoder = copy.deepcopy(encoder).to(device).requires_grad_(False)
    state = nn.ModuleDict(
        {"encoder": encoder, "predictor": predictor, "target_encoder": target_encoder}
    )
    loaded = load_training_checkpoint(
        args.checkpoint,
        state,
        optimizer=None,
        map_location=device,
    )
    del state, encoder, predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = _evaluation_dataset(args.data_root, "train", args.image_size)
    validation_dataset = _evaluation_dataset(args.data_root, "val", args.image_size)
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("train and validation class ordering differ")

    train_features, train_labels = _features(
        target_encoder,
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    validation_features, validation_labels = _features(
        target_encoder,
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    neighbour_indices, neighbour_similarities, knn_predictions = _knn_details(
        train_features,
        train_labels,
        validation_features,
        k=args.k,
    )
    all_neighbour_labels = train_labels[neighbour_indices]
    selected = _representative_indices(
        knn_predictions,
        validation_labels,
        all_neighbour_labels,
    )

    probe = fit_linear_probe(
        train_features,
        train_labels,
        steps=args.probe_steps,
        pooling="mean",
    )
    linear_logits = probe.logits(validation_features)
    linear_probabilities = linear_logits.softmax(dim=-1)
    linear_confidence, linear_encoded = linear_probabilities.max(dim=-1)
    linear_predictions = probe.classes[linear_encoded]

    train_paths = np.asarray([path for path, _ in train_dataset.samples])
    validation_paths = np.asarray([path for path, _ in validation_dataset.samples])
    selected_neighbours = neighbour_indices[selected]

    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    view_feature_groups: list[Tensor] = []
    with torch.inference_mode():
        for query_index in selected.tolist():
            views = image_views(validation_paths[query_index], args.image_size)
            pixels = torch.stack([views[name] for name in IMAGE_VIEW_ORDER])
            normalized = ((pixels - mean) / std).to(device)
            view_feature_groups.append(
                target_encoder(normalized).mean(dim=1).float().cpu()
            )
    view_features = torch.stack(view_feature_groups)
    flattened_view_features = view_features.flatten(0, 1)
    _, _, flattened_view_knn_predictions = _knn_details(
        train_features,
        train_labels,
        flattened_view_features,
        k=args.k,
    )
    view_knn_predictions = flattened_view_knn_predictions.reshape(
        len(selected), len(IMAGE_VIEW_ORDER)
    )
    view_linear_probabilities = probe.logits(flattened_view_features).softmax(dim=-1)
    view_linear_confidence, view_linear_encoded = view_linear_probabilities.max(dim=-1)
    view_linear_predictions = probe.classes[view_linear_encoded].reshape(
        len(selected), len(IMAGE_VIEW_ORDER)
    )
    view_linear_confidence = view_linear_confidence.reshape(
        len(selected), len(IMAGE_VIEW_ORDER)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        example_kind=np.asarray(["correct", "incorrect"]),
        class_names=np.asarray(train_dataset.classes),
        query_indices=selected.numpy(),
        query_paths=validation_paths[selected.numpy()],
        query_labels=validation_labels[selected].numpy(),
        knn_predictions=knn_predictions[selected].numpy(),
        linear_predictions=linear_predictions[selected].numpy(),
        linear_confidence=linear_confidence[selected].numpy(),
        linear_probabilities=linear_probabilities[selected].numpy(),
        neighbour_indices=selected_neighbours.numpy(),
        neighbour_paths=train_paths[selected_neighbours.numpy()],
        neighbour_labels=all_neighbour_labels[selected].numpy(),
        neighbour_similarities=neighbour_similarities[selected].numpy(),
        view_names=np.asarray(IMAGE_VIEW_ORDER),
        view_knn_predictions=view_knn_predictions.numpy(),
        view_linear_predictions=view_linear_predictions.numpy(),
        view_linear_confidence=view_linear_confidence.numpy(),
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_step": loaded.step,
        "device": str(device),
        "k": args.k,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "knn_accuracy": float((knn_predictions == validation_labels).float().mean()),
        "linear_accuracy": float((linear_predictions == validation_labels).float().mean()),
        "selected_validation_indices": selected.tolist(),
        "output": str(args.output),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
