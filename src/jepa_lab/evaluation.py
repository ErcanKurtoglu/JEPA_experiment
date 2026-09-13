"""Lightweight, deterministic representation evaluation utilities.

The probes in this module deliberately operate on already exported features.
That keeps the three upstream projects in separate processes while giving the
image and video laboratories one small, backend-neutral evaluation surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .runlog import import_features

ArrayLike = torch.Tensor | np.ndarray
Pooling = Literal["mean", "max", "cls"]


def _tensor(value: ArrayLike, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    result = result.cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def pool_features(features: ArrayLike, pooling: Pooling = "mean") -> torch.Tensor:
    """Return a CPU ``[samples, dimensions]`` feature matrix.

    Any axes between the sample axis and final feature axis are token/grid axes.
    They are pooled together. Two-dimensional features are returned unchanged.
    """

    matrix = _tensor(features, dtype=torch.float32)
    if matrix.ndim < 2:
        raise ValueError("features must have shape [samples, ..., dimensions]")
    if matrix.shape[0] == 0 or matrix.shape[-1] == 0:
        raise ValueError("features cannot have an empty sample or feature axis")
    if matrix.ndim > 2:
        token_axes = tuple(range(1, matrix.ndim - 1))
        if pooling == "mean":
            matrix = matrix.mean(dim=token_axes)
        elif pooling == "max":
            matrix = matrix.flatten(1, -2).amax(dim=1)
        elif pooling == "cls":
            matrix = matrix.flatten(1, -2)[:, 0]
        else:
            raise ValueError(f"unknown pooling mode: {pooling!r}")
    elif pooling not in ("mean", "max", "cls"):
        raise ValueError(f"unknown pooling mode: {pooling!r}")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("features contain NaN or Inf")
    return matrix.contiguous()


def _label_vector(labels: ArrayLike, *, expected: int) -> torch.Tensor:
    vector = _tensor(labels).reshape(-1).to(dtype=torch.long)
    if vector.numel() != expected:
        raise ValueError(f"expected {expected} labels, received {vector.numel()}")
    return vector


@torch.no_grad()
def knn_predict(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    query_features: ArrayLike,
    *,
    k: int = 20,
    pooling: Pooling = "mean",
) -> torch.Tensor:
    """Predict labels with deterministic cosine k-NN.

    Ties are resolved in favour of the numerically smallest class label. If
    ``k`` exceeds the training set size, every available training sample is
    used; this is convenient for the tiny smoke laboratories.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    train = pool_features(train_features, pooling)
    query = pool_features(query_features, pooling)
    if train.shape[1] != query.shape[1]:
        raise ValueError("train and query feature dimensions must match")
    labels = _label_vector(train_labels, expected=train.shape[0])
    classes = torch.unique(labels, sorted=True)
    if classes.numel() == 0:
        raise ValueError("the training set must contain at least one class")

    train = F.normalize(train, dim=-1)
    query = F.normalize(query, dim=-1)
    similarities = query @ train.T
    neighbours = torch.argsort(
        similarities, dim=1, descending=True, stable=True
    )[:, : min(k, train.shape[0])]
    neighbour_labels = labels[neighbours]
    votes = (neighbour_labels.unsqueeze(-1) == classes).sum(dim=1)
    return classes[votes.argmax(dim=1)]


@torch.no_grad()
def knn_accuracy(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    eval_features: ArrayLike,
    eval_labels: ArrayLike,
    *,
    k: int = 20,
    pooling: Pooling = "mean",
) -> float:
    """Compute held-out cosine k-NN accuracy in the range ``[0, 1]``."""

    predictions = knn_predict(
        train_features, train_labels, eval_features, k=k, pooling=pooling
    )
    expected = _label_vector(eval_labels, expected=predictions.numel())
    return float((predictions == expected).float().mean())


@dataclass(frozen=True, slots=True)
class FrozenLinearProbe:
    """A fitted linear classifier whose weights and inputs remain on CPU."""

    weight: torch.Tensor
    bias: torch.Tensor
    classes: torch.Tensor
    normalize: bool
    pooling: Pooling
    loss_history: tuple[float, ...]

    def logits(self, features: ArrayLike) -> torch.Tensor:
        matrix = pool_features(features, self.pooling)
        if matrix.shape[1] != self.weight.shape[1]:
            raise ValueError("feature dimension does not match the fitted probe")
        if self.normalize:
            matrix = F.normalize(matrix, dim=-1)
        return F.linear(matrix, self.weight, self.bias)

    def predict(self, features: ArrayLike) -> torch.Tensor:
        return self.classes[self.logits(features).argmax(dim=-1)]


def fit_linear_probe(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    *,
    steps: int = 200,
    learning_rate: float = 0.1,
    weight_decay: float = 0.0,
    normalize: bool = True,
    pooling: Pooling = "mean",
) -> FrozenLinearProbe:
    """Fit a full-batch linear probe without backpropagating into features.

    Zero initialization and full-batch updates make the result repeatable on
    CPU without changing the caller's random-number-generator state.
    """

    if steps <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("steps/learning_rate must be positive and weight_decay non-negative")
    matrix = pool_features(train_features, pooling).detach()
    if normalize:
        matrix = F.normalize(matrix, dim=-1)
    labels = _label_vector(train_labels, expected=matrix.shape[0])
    classes, encoded = torch.unique(labels, sorted=True, return_inverse=True)
    if classes.numel() < 2:
        raise ValueError("a linear probe requires at least two classes")

    layer = torch.nn.Linear(matrix.shape[1], classes.numel())
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(layer(matrix), encoded)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    return FrozenLinearProbe(
        weight=layer.weight.detach().clone(),
        bias=layer.bias.detach().clone(),
        classes=classes.detach().clone(),
        normalize=normalize,
        pooling=pooling,
        loss_history=tuple(losses),
    )


@torch.no_grad()
def linear_probe_accuracy(
    probe: FrozenLinearProbe,
    eval_features: ArrayLike,
    eval_labels: ArrayLike,
) -> float:
    """Evaluate a fitted frozen linear probe on held-out features."""

    predictions = probe.predict(eval_features)
    expected = _label_vector(eval_labels, expected=predictions.numel())
    return float((predictions == expected).float().mean())


@dataclass(frozen=True, slots=True)
class ProbeScores:
    knn_accuracy: float
    linear_accuracy: float
    majority_baseline: float

    def as_dict(self) -> dict[str, float]:
        return {
            "knn_accuracy": self.knn_accuracy,
            "linear_accuracy": self.linear_accuracy,
            "majority_baseline": self.majority_baseline,
        }


def evaluate_representation(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    eval_features: ArrayLike,
    eval_labels: ArrayLike,
    *,
    k: int = 20,
    probe_steps: int = 200,
    probe_learning_rate: float = 0.1,
    pooling: Pooling = "mean",
) -> ProbeScores:
    """Run the common M2/M3 k-NN and frozen-linear evaluation pair."""

    evaluation_labels = _label_vector(
        eval_labels, expected=pool_features(eval_features, pooling).shape[0]
    )
    _, counts = torch.unique(evaluation_labels, return_counts=True)
    majority = float(counts.max() / counts.sum())
    probe = fit_linear_probe(
        train_features,
        train_labels,
        steps=probe_steps,
        learning_rate=probe_learning_rate,
        pooling=pooling,
    )
    return ProbeScores(
        knn_accuracy=knn_accuracy(
            train_features,
            train_labels,
            eval_features,
            evaluation_labels,
            k=k,
            pooling=pooling,
        ),
        linear_accuracy=linear_probe_accuracy(probe, eval_features, evaluation_labels),
        majority_baseline=majority,
    )


def motion_direction_labels(
    positions_or_displacements: ArrayLike, *, static_threshold: float = 1e-6
) -> torch.Tensor:
    """Infer MovingShapes direction labels: ``+x, -x, +y, -y`` → ``0..3``.

    Input may be positions ``[B,T,2]`` or direct displacements ``[B,2]``.
    Static trajectories are rejected because the four-class laboratory has no
    static class.
    """

    values = _tensor(positions_or_displacements, dtype=torch.float32)
    if values.ndim == 3 and values.shape[1] >= 2 and values.shape[2] >= 2:
        displacement = values[:, -1, :2] - values[:, 0, :2]
    elif values.ndim == 2 and values.shape[1] >= 2:
        displacement = values[:, :2]
    else:
        raise ValueError("expected [B,T,2] positions or [B,2] displacements")
    if bool((displacement.norm(dim=-1) <= static_threshold).any()):
        raise ValueError("static trajectories do not have a four-way direction label")
    horizontal = displacement[:, 0].abs() >= displacement[:, 1].abs()
    labels = torch.where(
        horizontal,
        torch.where(displacement[:, 0] >= 0, 0, 1),
        torch.where(displacement[:, 1] >= 0, 2, 3),
    )
    return labels.to(dtype=torch.long)


def evaluate_motion_direction(
    train_features: ArrayLike,
    train_positions: ArrayLike,
    eval_features: ArrayLike,
    eval_positions: ArrayLike,
    *,
    k: int = 20,
    probe_steps: int = 200,
    probe_learning_rate: float = 0.1,
    pooling: Pooling = "mean",
) -> ProbeScores:
    """Evaluate video features using directions inferred from trajectories."""

    train_labels = motion_direction_labels(train_positions)
    eval_labels = motion_direction_labels(eval_positions)
    return evaluate_representation(
        train_features,
        train_labels,
        eval_features,
        eval_labels,
        k=k,
        probe_steps=probe_steps,
        probe_learning_rate=probe_learning_rate,
        pooling=pooling,
    )


@dataclass(frozen=True, slots=True)
class TemporalPerturbationSummary:
    source: str
    feature_shape: tuple[int, ...]
    reference: str
    cosine_to_reference: dict[str, float]
    mean_l1_to_reference: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "feature_shape": list(self.feature_shape),
            "reference": self.reference,
            "cosine_to_reference": self.cosine_to_reference,
            "mean_l1_to_reference": self.mean_l1_to_reference,
        }


def temporal_perturbation_summary(
    path: str | Path, *, reference: str = "normal"
) -> TemporalPerturbationSummary:
    """Summarize a common NPZ exported by a V-JEPA temporal-control script."""

    features, _, sidecar = import_features(path)
    if features.ndim < 2 or features.shape[0] < 2:
        raise ValueError("temporal archive must have shape [variants, ..., dimensions]")
    metadata = sidecar.get("metadata", {})
    order = metadata.get("variant_order") if isinstance(metadata, dict) else None
    if order is None and features.shape[0] == 4:
        order = ["normal", "reversed", "shuffled", "static"]
    if not isinstance(order, list) or len(order) != features.shape[0]:
        raise ValueError("NPZ sidecar must provide one variant_order name per feature")
    names = [str(name) for name in order]
    if len(set(names)) != len(names) or reference not in names:
        raise ValueError("variant_order names must be unique and include the reference")

    values = torch.as_tensor(features, dtype=torch.float32)
    pooled = values.reshape(values.shape[0], -1, values.shape[-1]).mean(dim=1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("temporal archive contains NaN or Inf")
    anchor = pooled[names.index(reference)].unsqueeze(0)
    cosine = F.cosine_similarity(anchor, pooled, dim=-1)
    l1 = (pooled - anchor).abs().mean(dim=-1)
    return TemporalPerturbationSummary(
        source=str(Path(path)),
        feature_shape=tuple(int(size) for size in features.shape),
        reference=reference,
        cosine_to_reference={name: float(value) for name, value in zip(names, cosine)},
        mean_l1_to_reference={name: float(value) for name, value in zip(names, l1)},
    )


__all__ = [
    "FrozenLinearProbe",
    "ProbeScores",
    "TemporalPerturbationSummary",
    "evaluate_motion_direction",
    "evaluate_representation",
    "fit_linear_probe",
    "knn_accuracy",
    "knn_predict",
    "linear_probe_accuracy",
    "motion_direction_labels",
    "pool_features",
    "temporal_perturbation_summary",
]
