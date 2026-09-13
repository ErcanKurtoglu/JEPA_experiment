from __future__ import annotations

import torch

from jepa_lab.device import seeded_generator
from jepa_lab.image_jepa import ImageJEPA, train_one_step
from jepa_lab.masking import MultiBlockMasker


def _model() -> ImageJEPA:
    return ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        predictor_dim=24,
        predictor_depth=1,
        predictor_heads=4,
        mlp_ratio=2.0,
    )


def _masks(batch_size: int = 2):
    return MultiBlockMasker(
        4,
        num_targets=2,
        target_scale=(0.20, 0.20),
        target_aspect_ratio=(1.0, 1.0),
        context_scale=(1.0, 1.0),
    )(batch_size, generator=seeded_generator(3))


def test_forward_is_finite_and_masks_teacher_output() -> None:
    model = _model()
    images = torch.randn(2, 3, 32, 32)
    masks = _masks()
    observed_shapes: list[tuple[int, ...]] = []
    hook = model.target_encoder.register_forward_hook(
        lambda _module, _inputs, output: observed_shapes.append(tuple(output.shape))
    )
    output = model(images, masks)
    hook.remove()

    assert observed_shapes == [(2, 16, 32)]  # teacher encoded every patch
    assert output.predicted.shape == (2, 2, masks.targets.shape[-1], 32)
    assert output.target.shape == output.predicted.shape
    assert output.context.shape[1] == masks.context.shape[1]
    assert torch.isfinite(output.loss)
    assert not output.target.requires_grad


def test_gradients_stop_at_target_encoder() -> None:
    model = _model()
    output = model(torch.randn(2, 3, 32, 32), _masks())
    output.loss.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.context_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.target_encoder.parameters()
    )


def test_input_masking_ablation_encodes_each_target_without_full_context() -> None:
    model = ImageJEPA(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        predictor_dim=24,
        predictor_depth=1,
        predictor_heads=4,
        target_masking="input",
    )
    masks = _masks()
    observed_shapes: list[tuple[int, ...]] = []
    hook = model.target_encoder.register_forward_hook(
        lambda _module, _inputs, output: observed_shapes.append(tuple(output.shape))
    )
    output = model(torch.randn(2, 3, 32, 32), masks)
    hook.remove()

    assert observed_shapes == [
        (2, masks.targets.shape[-1], 32),
        (2, masks.targets.shape[-1], 32),
    ]
    assert output.target.shape == output.predicted.shape
    assert not output.target.requires_grad


def test_one_step_reports_gradients_finite_loss_and_exact_ema() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    result = train_one_step(
        model,
        torch.randn(2, 3, 32, 32),
        _masks(),
        optimizer,
        ema_momentum=0.9,
    )

    assert result.loss > 0
    assert result.context_grad_norm > 0
    assert result.predictor_grad_norm > 0
    assert not result.target_has_grad
    assert result.all_finite
    assert result.ema_max_error <= 1e-7
