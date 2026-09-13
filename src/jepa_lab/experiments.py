"""Reusable training loops for the manual image, video, and robotics labs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .action_world_model import ActionWorldModel
from .cem import CEMPlanner, receding_horizon_control
from .datasets import planar_states_from_actions, sample_planar_latent_batch
from .device import seeded_generator
from .image_jepa import ImageJEPA
from .masking import MultiBlockMasker
from .metrics import measure_collapse
from .simulator import PlanarReachEnv
from .video_jepa import TinyVideoJEPA, generate_tube_masks


def linear_momentum(start: float, end: float, step: int, total_steps: int) -> float:
    if not 0 <= start <= end <= 1:
        raise ValueError("EMA bounds must satisfy 0 <= start <= end <= 1")
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("step must be in [0, total_steps)")
    if total_steps == 1:
        return end
    return start + (end - start) * step / (total_steps - 1)


def _infinite_batches(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("data loader is empty")


def _images(batch: Any) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("image loader must yield [B,C,H,W] tensors, optionally with labels")
    return images


@dataclass(frozen=True)
class TrainMetrics:
    first_loss: float
    final_loss: float
    mean_loss: float
    steps: int
    feature_std: float
    mean_cosine: float
    effective_rank: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def train_image_steps(
    model: ImageJEPA,
    loader: Iterable[Any],
    masker: MultiBlockMasker,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: torch.device | str,
    gradient_accumulation: int = 1,
    ema_start: float = 0.996,
    ema_end: float = 1.0,
    seed: int = 42,
) -> TrainMetrics:
    """Train image JEPA for an exact number of optimizer steps."""

    if steps <= 0 or gradient_accumulation <= 0:
        raise ValueError("steps and gradient_accumulation must be positive")
    device = torch.device(device)
    model.to(device).train()
    batches = _infinite_batches(loader)
    generator = seeded_generator(seed)
    losses: list[float] = []
    last_images: torch.Tensor | None = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(gradient_accumulation):
            images = _images(next(batches)).to(device)
            masks = masker(images.shape[0], generator=generator, device=device)
            output = model(images, masks)
            (output.loss / gradient_accumulation).backward()
            accumulated += float(output.loss.detach()) / gradient_accumulation
            last_images = images
        optimizer.step()
        model.update_target(linear_momentum(ema_start, ema_end, step, steps))
        losses.append(accumulated)

    assert last_images is not None
    with torch.inference_mode():
        features = model.target_encoder(last_images)
    collapse = measure_collapse(features)
    return TrainMetrics(
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=float(np.mean(losses)),
        steps=steps,
        feature_std=collapse.feature_std,
        mean_cosine=collapse.mean_cosine_similarity,
        effective_rank=collapse.effective_rank,
    )


def train_video_steps(
    model: TinyVideoJEPA,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: torch.device | str,
    ema_start: float = 0.998,
    ema_end: float = 1.0,
    seed: int = 42,
    mask_factory: Callable[[tuple[int, int, int], int], torch.Tensor] | None = None,
) -> TrainMetrics:
    """Train tiny V-JEPA with resampled tube masks or an explicit ablation mask."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device)
    model.to(device).train()
    batches = _infinite_batches(loader)
    losses: list[float] = []
    last_frames: torch.Tensor | None = None
    for step in range(steps):
        batch = next(batches)
        frames = batch["frames"] if isinstance(batch, dict) else batch[0]
        frames = frames.to(device)
        mask = (
            generate_tube_masks(model.grid_size, seed=seed + step).target
            if mask_factory is None
            else mask_factory(model.grid_size, seed + step)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(frames, mask)
        output.loss.backward()
        optimizer.step()
        model.update_teacher(linear_momentum(ema_start, ema_end, step, steps))
        losses.append(float(output.loss.detach()))
        last_frames = frames

    assert last_frames is not None
    with torch.inference_mode():
        features = model.encode(last_frames)
    collapse = measure_collapse(features)
    return TrainMetrics(
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=float(np.mean(losses)),
        steps=steps,
        feature_std=collapse.feature_std,
        mean_cosine=collapse.mean_cosine_similarity,
        effective_rank=collapse.effective_rank,
    )


@dataclass(frozen=True)
class ActionTrainMetrics:
    first_loss: float
    final_loss: float
    correct_action_l1: float
    zero_action_l1: float
    shuffled_action_l1: float
    steps: int

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def train_action_steps(
    model: ActionWorldModel,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 500,
    batch_size: int = 128,
    horizon: int = 4,
    action_limit: float = 0.10,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> ActionTrainMetrics:
    if steps <= 0 or batch_size < 2:
        raise ValueError("steps must be positive and batch_size at least two")
    device = torch.device(device)
    model.to(device).train()
    losses: list[float] = []
    for step in range(steps):
        z0, actions, states, targets = sample_planar_latent_batch(
            batch_size,
            horizon,
            action_limit=action_limit,
            seed=seed + step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model.compute_loss(
            z0, actions, targets, states, rollout_steps=2, rollout_weight=1.0
        )
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    model.eval()
    z0, actions, states, targets = sample_planar_latent_batch(
        batch_size,
        horizon,
        action_limit=action_limit,
        seed=seed + 100_000,
        device=device,
    )
    with torch.inference_mode():
        correct = F.l1_loss(model.rollout(z0, actions, states), targets)
        zero_actions = torch.zeros_like(actions)
        zero_states = planar_states_from_actions(z0, zero_actions, max_delta=action_limit)
        zero = F.l1_loss(model.rollout(z0, zero_actions, zero_states), targets)
        shuffled_actions = torch.roll(actions, shifts=1, dims=0)
        shuffled_states = planar_states_from_actions(
            z0, shuffled_actions, max_delta=action_limit
        )
        shuffled = F.l1_loss(
            model.rollout(z0, shuffled_actions, shuffled_states), targets
        )
    return ActionTrainMetrics(
        first_loss=losses[0],
        final_loss=losses[-1],
        correct_action_l1=float(correct),
        zero_action_l1=float(zero),
        shuffled_action_l1=float(shuffled),
        steps=steps,
    )


def learned_planar_objective(
    model: ActionWorldModel,
    env: PlanarReachEnv,
    candidates: np.ndarray,
    *,
    device: torch.device | str,
) -> np.ndarray:
    """Evaluate CEM candidates with learned latent rollouts and known delta-state integration."""

    device = torch.device(device)
    actions = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    batch, horizon, _ = actions.shape
    z0 = torch.as_tensor(env.position, dtype=torch.float32, device=device).view(1, 1, 2)
    z0 = z0.expand(batch, -1, -1)
    goal = torch.as_tensor(env.goal, dtype=torch.float32, device=device).view(1, 1, 2)
    goal = goal.expand(batch, -1, -1)

    states = torch.zeros(batch, horizon, 7, device=device)
    current = z0[:, 0]
    for step in range(horizon):
        states[:, step, :2] = current
        current = (current + actions[:, step, :2]).clamp(0.0, 1.0)
    with torch.inference_mode():
        prediction = model.rollout(z0, actions, states)[:, -1]
        energy = (prediction - goal).abs().mean(dim=(1, 2))
    return energy.cpu().numpy().astype(np.float32)


def run_learned_planar_episode(
    model: ActionWorldModel,
    *,
    seed: int,
    device: torch.device | str,
    horizon: int = 4,
    candidates: int = 256,
    elites: int = 32,
    refinements: int = 5,
    max_steps: int = 20,
) -> dict[str, Any]:
    env = PlanarReachEnv(
        seed=seed,
        max_delta=0.10,
        success_radius=0.08,
        max_steps=max_steps,
    )
    env.reset(seed=seed)
    planner = CEMPlanner(
        horizon=horizon,
        action_dim=7,
        candidates=candidates,
        elites=elites,
        refinements=refinements,
        action_low=env.action_low,
        action_high=env.action_high,
        seed=seed,
    )
    trace = receding_horizon_control(
        env,
        planner,
        lambda current_env: lambda population: learned_planar_objective(
            model, current_env, population, device=device
        ),
        max_steps=max_steps,
    )
    final = trace[-1].info if trace else {
        "success": False,
        "distance": float(np.linalg.norm(env.position - env.goal)),
    }
    return {
        "success": bool(final["success"]),
        "final_distance": float(final["distance"]),
        "steps": len(trace),
        "energy_histories": [list(step.plan.energy_history) for step in trace],
    }


__all__ = [
    "ActionTrainMetrics",
    "TrainMetrics",
    "learned_planar_objective",
    "linear_momentum",
    "run_learned_planar_episode",
    "train_action_steps",
    "train_image_steps",
    "train_video_steps",
]
