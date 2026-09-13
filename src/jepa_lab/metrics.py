"""Small diagnostics for detecting collapsed latent representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions [..., D]")
    return features.reshape(-1, features.shape[-1]).float()


def feature_std(features: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension population standard deviation."""

    matrix = _matrix(features)
    return matrix.std(dim=0, unbiased=False).mean()


def mean_cosine_similarity(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean cosine similarity over every distinct pair without an NxN matrix."""

    matrix = _matrix(features)
    count = matrix.shape[0]
    if count < 2:
        return matrix.new_tensor(float("nan"))
    normalized = F.normalize(matrix, dim=-1, eps=eps)
    summed = normalized.sum(dim=0)
    ordered_pair_sum = summed.square().sum() - normalized.square().sum()
    return ordered_pair_sum / (count * (count - 1))


def effective_rank(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank of centered features using singular values."""

    matrix = _matrix(features)
    if matrix.shape[0] < 2:
        return matrix.new_tensor(0.0)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.svdvals(centered).square()
    total = eigenvalues.sum()
    if bool(total <= eps):
        return matrix.new_tensor(1.0)
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return entropy.exp()


@dataclass(frozen=True, slots=True)
class CollapseMetrics:
    feature_std: float
    mean_cosine_similarity: float
    effective_rank: float


@torch.no_grad()
def measure_collapse(features: torch.Tensor) -> CollapseMetrics:
    """Compute detached scalar diagnostics suitable for JSON/CSV logging."""

    return CollapseMetrics(
        feature_std=float(feature_std(features).cpu()),
        mean_cosine_similarity=float(mean_cosine_similarity(features).cpu()),
        effective_rank=float(effective_rank(features).cpu()),
    )
