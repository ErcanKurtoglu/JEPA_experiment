from __future__ import annotations

import numpy as np
import torch

from jepa_lab.action_world_model import ActionWorldModel
from jepa_lab.types import VisualEncoder
from jepa_lab.visual_control import (
    FrozenVisualEncoder,
    VisualCEMPlanner,
    assert_frozen_encoder_excluded,
    make_tiny_video_jepa,
    pretrain_tiny_video_jepa,
    rgb_to_canonical_frames,
    sample_planar_rgb_trajectories,
    train_visual_action_steps,
    visual_latent_objective,
    visual_rollout_errors,
)


def test_rgb_trajectory_contract_and_conversion() -> None:
    batch = sample_planar_rgb_trajectories(
        2, 2, image_size=32, action_limit=0.1, seed=7
    )
    assert batch.frames.shape == (2, 3, 3, 32, 32)
    assert batch.frames.dtype == torch.float32
    assert batch.actions.shape == (2, 2, 7)
    assert batch.states.shape == (2, 2, 7)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    converted = rgb_to_canonical_frames(image)
    assert converted.shape == (1, 3, 32, 32)
    assert converted.dtype == torch.float32


def test_frozen_video_encoder_and_visual_action_pipeline() -> None:
    torch.manual_seed(3)
    backbone = make_tiny_video_jepa(image_size=32, num_frames=2, embed_dim=8)
    pretraining = pretrain_tiny_video_jepa(
        backbone,
        steps=1,
        batch_size=2,
        image_size=32,
        action_limit=0.1,
        learning_rate=1e-3,
        device="cpu",
        seed=3,
    )
    assert not pretraining.target_has_grad

    encoder = FrozenVisualEncoder(backbone)
    assert isinstance(encoder, VisualEncoder)
    frozen_weight = next(encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(backbone.target_encoder.parameters()).add_(1.0)
    torch.testing.assert_close(next(encoder.parameters()), frozen_weight)
    model = ActionWorldModel(
        latent_dim=encoder.latent_dim,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert_frozen_encoder_excluded(encoder, optimizer)
    metrics = train_visual_action_steps(
        model,
        encoder,
        optimizer,
        steps=1,
        batch_size=2,
        horizon=2,
        image_size=32,
        action_limit=0.1,
        device="cpu",
        seed=9,
    )
    assert np.isfinite(metrics.final_loss)
    assert all(parameter.grad is None for parameter in encoder.parameters())

    errors = visual_rollout_errors(
        model,
        encoder,
        horizons=(1, 2),
        batch_size=2,
        image_size=32,
        action_limit=0.1,
        device="cpu",
        seed=11,
    )
    assert set(errors) == {"1", "2"}
    assert all(np.isfinite(error) for error in errors.values())

    observations = sample_planar_rgb_trajectories(
        1, 1, image_size=32, action_limit=0.1, seed=12
    )
    latents = encoder.encode_observation_sequences(observations.frames)
    candidates = np.zeros((3, 1, 7), dtype=np.float32)
    energy = visual_latent_objective(
        model,
        latents[:, 0],
        latents[:, 1],
        observations.states[0, 0],
        candidates,
    )
    assert energy.shape == (3,)
    assert np.isfinite(energy).all()

    planner = VisualCEMPlanner(
        model, encoder, horizon=1, candidates=4, elites=1, refinements=1, seed=2
    )
    current_rgb = (observations.frames[0, 0].permute(1, 2, 0).numpy() * 255).astype(
        np.uint8
    )
    goal_rgb = (observations.frames[0, 1].permute(1, 2, 0).numpy() * 255).astype(
        np.uint8
    )
    result = planner.plan(
        current_rgb,
        goal_rgb,
        observations.states[0, 0],
        (
            np.array([-0.1, -0.1, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([0.1, 0.1, 0, 0, 0, 0, 0], dtype=np.float32),
        ),
    )
    assert result.first_action.shape == (7,)
    assert not model.training and not encoder.training

    bad_low = np.full(7, -0.1, dtype=np.float32)
    bad_high = np.full(7, 0.1, dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "fixed to zero"):
        planner.plan(current_rgb, goal_rgb, observations.states[0, 0], (bad_low, bad_high))

    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        planner.plan(
            np.stack((current_rgb, current_rgb)),
            goal_rgb,
            observations.states[0, 0],
            (
                np.array([-0.1, -0.1, 0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0.1, 0.1, 0, 0, 0, 0, 0], dtype=np.float32),
            ),
        )
