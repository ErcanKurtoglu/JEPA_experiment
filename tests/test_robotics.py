from __future__ import annotations

import numpy as np
import torch

from jepa_lab.action_world_model import ActionWorldModel, BlockCausalPredictor
from jepa_lab.cem import CEMPlanner, receding_horizon_control
from jepa_lab.simulator import PlanarReachEnv, canonical_action


def test_simulator_is_deterministic_bounded_and_uses_only_dx_dy() -> None:
    first = PlanarReachEnv(seed=17)
    second = PlanarReachEnv(seed=17)
    first_observation, first_info = first.reset(seed=23)
    second_observation, second_info = second.reset(seed=23)

    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_info["position"], second_info["position"])
    assert np.array_equal(first_info["goal"], second_info["goal"])

    first.reset(start=(0.98, 0.02), goal=(0.5, 0.5))
    second.reset(start=(0.98, 0.02), goal=(0.5, 0.5))
    large = np.full(7, 100.0, dtype=np.float32)
    planar_only = canonical_action(100.0, 100.0)
    first.step(large)
    second.step(planar_only)

    assert np.array_equal(first.position, second.position)
    assert bool(((first.position >= 0.0) & (first.position <= 1.0)).all())
    assert first.state.shape == (7,)
    assert first.render().shape == (64, 64, 3)


def test_simulator_vectorised_rollout_does_not_mutate_state() -> None:
    env = PlanarReachEnv(seed=1)
    env.reset(start=(0.2, 0.3), goal=(0.8, 0.7))
    before = env.position
    actions = np.zeros((5, 4, 7), dtype=np.float32)
    actions[..., :2] = 0.04
    trajectories = env.simulate_actions(actions)

    assert trajectories.shape == (5, 5, 2)
    assert np.array_equal(env.position, before)
    assert bool(((trajectories >= 0.0) & (trajectories <= 1.0)).all())


def test_action_world_model_shapes_causality_and_losses() -> None:
    torch.manual_seed(5)
    model = ActionWorldModel(
        latent_dim=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        max_horizon=6,
    )
    z0 = torch.randn(2, 3, 8)
    actions = torch.randn(2, 4, 7)
    states = torch.randn(2, 4, 7)
    targets = torch.randn(2, 4, 3, 8)

    causal_mask = BlockCausalPredictor.block_causal_mask(4)
    assert not bool(causal_mask.diagonal().any())
    assert bool(causal_mask[0, 1])
    assert not bool(causal_mask[3, 0])

    output = model.compute_loss(
        z0, actions, targets, states, rollout_steps=2
    )
    assert output.teacher_forcing_predictions.shape == targets.shape
    assert output.rollout_predictions.shape == (2, 2, 3, 8)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )


def test_cem_elite_energy_is_non_increasing_on_quadratic_objective() -> None:
    target = np.array(
        [[0.5, -0.4], [0.2, 0.1], [-0.3, 0.6], [0.0, -0.2]],
        dtype=np.float32,
    )
    planner = CEMPlanner(
        horizon=4,
        action_dim=2,
        candidates=256,
        elites=32,
        refinements=6,
        action_low=-1.0,
        action_high=1.0,
        seed=11,
    )

    result = planner.plan(
        lambda candidates: np.square(candidates - target).sum(axis=(1, 2))
    )

    assert result.first_action.shape == (2,)
    assert result.sequence.shape == (4, 2)
    assert len(result.energy_history) == 6
    assert bool((np.diff(result.energy_history) <= 1e-12).all())
    assert result.energy < 0.05


def test_cem_reports_raw_energy_instead_of_masking_stateful_objective() -> None:
    call = 0

    def stateful_objective(candidates: np.ndarray) -> np.ndarray:
        nonlocal call
        call += 1
        return np.square(candidates).sum(axis=(1, 2)) + 100.0 * call

    planner = CEMPlanner(
        horizon=2,
        action_dim=2,
        candidates=16,
        elites=4,
        refinements=3,
        seed=4,
    )
    result = planner.plan(stateful_objective)
    assert np.diff(result.energy_history).max() > 50.0


def test_receding_horizon_executes_one_action_then_replans() -> None:
    env = PlanarReachEnv(max_delta=0.10, success_radius=0.08, seed=0)
    env.reset(start=(0.10, 0.10), goal=(0.80, 0.80))
    initial_distance = float(np.linalg.norm(env.position - env.goal))
    planner = CEMPlanner(
        horizon=4,
        action_dim=7,
        candidates=128,
        elites=16,
        refinements=4,
        action_low=env.action_low,
        action_high=env.action_high,
        seed=4,
    )
    trace = receding_horizon_control(
        env,
        planner,
        lambda current: current.trajectory_cost,
        max_steps=12,
    )

    assert trace
    assert all(step.action.shape == (7,) for step in trace)
    assert float(np.linalg.norm(env.position - env.goal)) < initial_distance
    assert trace[-1].terminated
