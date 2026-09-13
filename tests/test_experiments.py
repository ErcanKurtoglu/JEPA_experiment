import torch
from torch.utils.data import DataLoader, TensorDataset

from jepa_lab.action_world_model import ActionWorldModel
from jepa_lab.experiments import (
    linear_momentum,
    train_action_steps,
    train_image_steps,
    train_video_steps,
)
from jepa_lab.image_jepa import ImageJEPA
from jepa_lab.masking import MultiBlockMasker
from jepa_lab.video_jepa import TinyVideoJEPA, generate_future_mask


def test_linear_momentum_endpoints() -> None:
    assert linear_momentum(0.9, 1.0, 0, 3) == 0.9
    assert linear_momentum(0.9, 1.0, 2, 3) == 1.0


def test_short_image_training_loop_runs() -> None:
    images = torch.randn(4, 3, 16, 16)
    loader = DataLoader(TensorDataset(images, torch.zeros(4)), batch_size=2)
    model = ImageJEPA(
        image_size=16,
        patch_size=4,
        embed_dim=16,
        encoder_depth=1,
        encoder_heads=2,
        predictor_dim=16,
        predictor_depth=1,
        predictor_heads=2,
    )
    masker = MultiBlockMasker(
        (4, 4),
        num_targets=2,
        target_scale=(0.2, 0.2),
        context_scale=(0.8, 0.8),
    )
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    metrics = train_image_steps(
        model, loader, masker, optimizer, steps=2, device="cpu"
    )
    assert metrics.steps == 2
    assert metrics.first_loss > 0
    assert metrics.effective_rank > 0


def test_short_action_training_distinguishes_actions() -> None:
    model = ActionWorldModel(
        latent_dim=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    metrics = train_action_steps(
        model, optimizer, steps=20, batch_size=16, horizon=2, seed=3
    )
    assert metrics.final_loss < metrics.first_loss
    assert metrics.correct_action_l1 < metrics.zero_action_l1


def test_video_training_accepts_causal_ablation_mask() -> None:
    frames = torch.randn(4, 4, 3, 16, 16)
    loader = DataLoader(TensorDataset(frames), batch_size=2)
    model = TinyVideoJEPA(
        num_frames=4,
        image_size=16,
        tubelet_size=(2, 8, 8),
        embed_dim=16,
        encoder_depth=1,
        encoder_heads=2,
        predictor_dim=16,
        predictor_depth=1,
        predictor_heads=2,
    )
    optimizer = torch.optim.AdamW(
        [*model.context_encoder.parameters(), *model.predictor.parameters()], lr=1e-3
    )
    metrics = train_video_steps(
        model,
        loader,
        optimizer,
        steps=2,
        device="cpu",
        mask_factory=lambda grid, _seed: generate_future_mask(grid),
    )
    assert metrics.steps == 2
    assert metrics.first_loss > 0
