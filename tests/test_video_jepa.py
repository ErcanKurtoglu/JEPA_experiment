from __future__ import annotations

import torch

from jepa_lab.video_jepa import (
    TinyVideoJEPA,
    TubeletTokenizer,
    generate_future_mask,
    generate_tube_masks,
)


def test_tubelet_tokenizer_has_1568_tokens_for_paper_clip() -> None:
    tokenizer = TubeletTokenizer(embed_dim=4, tubelet_size=(2, 16, 16))
    video = torch.randn(1, 16, 3, 224, 224)
    tokens = tokenizer(video)

    assert tokenizer.output_grid(16, 224, 224) == (8, 14, 14)
    assert tokens.shape == (1, 1568, 4)


def test_short_and_long_blocks_are_full_time_tubes() -> None:
    masks = generate_tube_masks((8, 14, 14), seed=9)

    assert len(masks.short) == 8
    assert len(masks.long) == 2
    assert 0.0 < masks.coverage <= 0.90
    for mask in (masks.target, *masks.short, *masks.long):
        for time_index in range(1, mask.shape[0]):
            assert torch.equal(mask[0], mask[time_index])


def test_future_mask_exposes_past_and_targets_only_future() -> None:
    mask = generate_future_mask((4, 3, 2), context_temporal_tokens=2)
    assert mask.shape == (4, 3, 2)
    assert not bool(mask[:2].any())
    assert bool(mask[2:].all())


def test_video_jepa_teacher_sees_full_clip_and_has_no_gradients() -> None:
    torch.manual_seed(3)
    model = TinyVideoJEPA(
        num_frames=4,
        image_size=16,
        tubelet_size=(2, 8, 8),
        embed_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        predictor_dim=16,
        predictor_depth=1,
        predictor_heads=4,
    )
    masks = generate_tube_masks(
        model.grid_size,
        num_short=1,
        num_long=0,
        short_scale=0.25,
        min_context_ratio=0.25,
        seed=4,
    )
    observed_shapes: list[tuple[int, ...]] = []
    hook = model.target_encoder.register_forward_hook(
        lambda _module, _inputs, output: observed_shapes.append(tuple(output.shape))
    )
    output = model(torch.randn(2, 4, 3, 16, 16), masks.target)
    hook.remove()

    assert observed_shapes == [(2, 8, 16)]
    assert output.prediction.shape == output.target.shape
    assert output.prediction.shape[1] == int(masks.target.sum())
    assert torch.isfinite(output.loss)
    assert not output.target.requires_grad

    output.loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.context_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.target_encoder.parameters()
    )


def test_video_jepa_ema_matches_manual_update() -> None:
    model = TinyVideoJEPA(
        num_frames=2,
        image_size=8,
        tubelet_size=(1, 4, 4),
        embed_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        predictor_dim=8,
        predictor_depth=1,
        predictor_heads=2,
    )
    context = model.context_encoder.position_embedding
    teacher = model.target_encoder.position_embedding
    with torch.no_grad():
        context.fill_(2.0)
        teacher.fill_(0.0)
    model.update_teacher(0.75)
    assert torch.equal(teacher, torch.full_like(teacher, 0.5))
