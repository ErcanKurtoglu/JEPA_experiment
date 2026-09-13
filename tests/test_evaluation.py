from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_lab.evaluation import (
    evaluate_motion_direction,
    fit_linear_probe,
    knn_accuracy,
    knn_predict,
    linear_probe_accuracy,
    motion_direction_labels,
    pool_features,
    temporal_perturbation_summary,
)
from jepa_lab.runlog import export_features


def separable_features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.5], [2.0, 1.0], [1.0, 2.0], [1.5, 1.5]]
    )
    train_labels = torch.tensor([10, 10, 10, 30, 30, 30])
    evaluation = torch.tensor([[-1.8, -1.1], [-1.1, -1.8], [1.8, 1.1], [1.1, 1.8]])
    evaluation_labels = torch.tensor([10, 10, 30, 30])
    return train, train_labels, evaluation, evaluation_labels


def test_pool_features_means_every_token_axis() -> None:
    features = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    pooled = pool_features(features)
    assert pooled.shape == (2, 5)
    torch.testing.assert_close(pooled, features.mean(dim=(1, 2)))


def test_knn_is_deterministic_and_supports_non_contiguous_labels() -> None:
    train, train_labels, evaluation, evaluation_labels = separable_features()
    first = knn_predict(train, train_labels, evaluation, k=3)
    second = knn_predict(train, train_labels, evaluation, k=3)
    torch.testing.assert_close(first, second)
    assert knn_accuracy(train, train_labels, evaluation, evaluation_labels, k=3) == 1.0


def test_linear_probe_is_reproducible_and_learns_separable_features() -> None:
    train, train_labels, evaluation, evaluation_labels = separable_features()
    first = fit_linear_probe(train, train_labels, steps=80, learning_rate=0.1)
    second = fit_linear_probe(train, train_labels, steps=80, learning_rate=0.1)
    torch.testing.assert_close(first.weight, second.weight)
    assert first.loss_history[-1] < first.loss_history[0]
    assert linear_probe_accuracy(first, evaluation, evaluation_labels) == 1.0


def test_linear_probe_never_writes_feature_gradients() -> None:
    train, train_labels, _, _ = separable_features()
    train.requires_grad_(True)
    fit_linear_probe(train, train_labels, steps=2)
    assert train.grad is None


def test_motion_direction_labels_match_moving_shapes_order() -> None:
    displacement = torch.tensor([[2.0, 0.1], [-2.0, 0.1], [0.1, 2.0], [0.1, -2.0]])
    assert motion_direction_labels(displacement).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="static"):
        motion_direction_labels(torch.zeros(1, 2))


def test_motion_direction_probe_uses_trajectory_derived_labels() -> None:
    train, labels, evaluation, evaluation_labels = separable_features()
    direction = {
        10: torch.tensor([1.0, 0.0]),
        30: torch.tensor([-1.0, 0.0]),
    }
    train_positions = torch.stack(
        [torch.stack((torch.zeros(2), direction[int(label)])) for label in labels]
    )
    eval_positions = torch.stack(
        [torch.stack((torch.zeros(2), direction[int(label)])) for label in evaluation_labels]
    )
    scores = evaluate_motion_direction(
        train,
        train_positions,
        evaluation,
        eval_positions,
        k=3,
        probe_steps=80,
    )
    assert scores.knn_accuracy == 1.0
    assert scores.linear_accuracy == 1.0


def test_temporal_summary_reads_common_npz_and_sidecar(tmp_path: Path) -> None:
    # [variants, batch, tokens, dimensions]
    normal = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    reversed_video = np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    shuffled = np.array([[[0.5, 0.5], [0.5, 0.5]]], dtype=np.float32)
    static = normal.copy()
    features = np.stack((normal, reversed_video, shuffled, static))
    archive, _ = export_features(
        tmp_path / "temporal.npz",
        features,
        labels=np.arange(4),
        metadata={"variant_order": ["normal", "reversed", "shuffled", "static"]},
    )
    summary = temporal_perturbation_summary(archive)
    assert summary.feature_shape == features.shape
    assert summary.cosine_to_reference["normal"] == pytest.approx(1.0)
    assert summary.cosine_to_reference["reversed"] == pytest.approx(0.0)
    assert summary.mean_l1_to_reference["static"] == pytest.approx(0.0)
