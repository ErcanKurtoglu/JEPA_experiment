from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from jepa_lab.device import (
    cuda_device_compatible,
    seed_everything,
    seeded_generator,
    select_device,
)
from jepa_lab.ema import update_ema
from jepa_lab.image_jepa import PatchEmbed
from jepa_lab.masking import MultiBlockMasker, RandomPatchMasker
from jepa_lab.metrics import effective_rank, feature_std, mean_cosine_similarity
from jepa_lab.types import PlanResult, RawVideo, VideoBatch


def test_patch_embedding_has_196_tokens() -> None:
    embedding = PatchEmbed(image_size=224, patch_size=16, embed_dim=24)
    tokens = embedding(torch.randn(2, 3, 224, 224))
    assert embedding.grid_size == (14, 14)
    assert tokens.shape == (2, 196, 24)


def test_seed_everything_is_reproducible() -> None:
    seed_everything(7)
    first_torch = torch.rand(4)
    first_numpy = np.random.rand(4)
    seed_everything(7)
    assert torch.equal(first_torch, torch.rand(4))
    assert np.array_equal(first_numpy, np.random.rand(4))
    assert select_device("cpu") == torch.device("cpu")
    assert select_device(["cpu"]) == torch.device("cpu")


def test_cuda_architecture_must_be_present_in_torch_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=0: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "sm_80"])
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert not cuda_device_compatible()
    assert select_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="not available"):
        select_device("cuda")


def test_exact_ema_update_and_invalid_momentum() -> None:
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[3.0, 5.0]]))
        target.weight.copy_(torch.tensor([[1.0, 1.0]]))
    update_ema(target, source, 0.75)
    assert torch.equal(target.weight, torch.tensor([[1.5, 2.0]]))
    with pytest.raises(ValueError, match="momentum"):
        update_ema(target, source, 1.1)


def test_multiblock_context_is_disjoint_while_targets_can_overlap() -> None:
    masker = MultiBlockMasker(
        (14, 14),
        num_targets=8,
        target_scale=(0.08, 0.08),
        target_aspect_ratio=(1.0, 1.0),
        context_scale=(1.0, 1.0),
    )
    masks = masker(32, generator=seeded_generator(11))
    context, targets = masks.boolean()
    assert masker.targets_may_overlap
    assert not (context[:, None] & targets).any()
    unique_counts = torch.tensor(
        [torch.unique(sample.reshape(-1)).numel() for sample in masks.targets]
    )
    assert (unique_counts < masks.targets.shape[1] * masks.targets.shape[2]).any()


def test_random_patch_ablation_is_deterministic_and_disjoint() -> None:
    masker = RandomPatchMasker(
        (14, 14), num_targets=4, target_scale=(0.15, 0.15), context_scale=(0.85, 0.85)
    )
    first = masker(4, generator=seeded_generator(19))
    second = masker(4, generator=seeded_generator(19))
    assert torch.equal(first.context, second.context)
    assert torch.equal(first.targets, second.targets)
    context, targets = first.boolean()
    assert not bool((context[:, None] & targets).any())
    # Random patches should not accidentally be represented as one solid block
    # in every sample/target.
    assert first.targets.shape[-1] == round(196 * 0.15)


def test_collapse_metrics_distinguish_constant_and_diverse_features() -> None:
    collapsed = torch.ones(16, 8)
    diverse = torch.eye(8).repeat(2, 1)
    assert feature_std(collapsed) == 0
    assert torch.isclose(mean_cosine_similarity(collapsed), torch.tensor(1.0))
    assert effective_rank(collapsed) == 1
    assert feature_std(diverse) > 0
    assert effective_rank(diverse) > effective_rank(collapsed)


def test_shared_data_contracts_accept_torch_and_numpy() -> None:
    raw = RawVideo(np.zeros((2, 8, 16, 16, 3), dtype=np.uint8))
    converted = raw.to_video_batch()
    assert converted.frames.shape == (2, 8, 3, 16, 16)
    assert converted.frames.dtype == torch.float32

    batch = VideoBatch(
        frames=torch.zeros(2, 8, 3, 16, 16),
        actions=torch.zeros(2, 4, 7),
        states=torch.zeros(2, 4, 7),
    )
    assert batch.batch_size == 2
    assert batch.num_frames == 8

    result = PlanResult(
        first_action=np.zeros(7),
        sequence=np.zeros((4, 7)),
        energy=0.5,
        elite_mean=np.zeros((4, 7)),
        elite_std=np.ones((4, 7)),
        energy_history=(1.0, 0.7, 0.5),
    )
    assert result.sequence.shape == (4, 7)
    with pytest.raises(ValueError, match="frames"):
        VideoBatch(torch.zeros(2, 3, 16, 16))
    with pytest.raises(TypeError, match="float32"):
        VideoBatch(torch.zeros(2, 8, 3, 16, 16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="same horizon"):
        VideoBatch(
            torch.zeros(2, 8, 3, 16, 16),
            actions=torch.zeros(2, 4, 7),
            states=torch.zeros(2, 3, 7),
        )
