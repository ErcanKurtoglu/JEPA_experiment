"""Experimental RGB-to-latent control bridge for the planar M6 laboratory.

The analytic ``xy`` experiment remains the fastest way to debug dynamics and CEM.
This module adds the missing visual path: a tiny V-JEPA is trained on rendered
clips, its EMA target encoder is frozen, action-conditioned dynamics are trained
entirely on those visual latents, and planning compares imagined terminal latents
with a latent encoded from the goal image.  It is deliberately small and is not a
claim of V-JEPA2-AC parity or real-robot readiness.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from .action_world_model import ActionWorldModel
from .cem import CEMPlanner, receding_horizon_control
from .datasets import planar_states_from_actions
from .simulator import PlanarReachEnv
from .types import PlanResult
from .video_jepa import TinyVideoJEPA, generate_tube_masks


@dataclass(frozen=True)
class VisualTrajectoryBatch:
    """Rendered planar transitions in the canonical project layouts."""

    frames: Tensor  # [B,H+1,C,Height,Width], float32 in [0,1]
    actions: Tensor  # [B,H,7]
    states: Tensor  # [B,H,7], state immediately before each action

    def __post_init__(self) -> None:
        if self.frames.ndim != 5 or self.frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,H+1,3,Height,Width]")
        batch, observations = self.frames.shape[:2]
        horizon = observations - 1
        if self.frames.dtype != torch.float32:
            raise ValueError("frames must be float32")
        if self.actions.shape != (batch, horizon, 7):
            raise ValueError("actions must have shape [B,H,7]")
        if self.states.shape != (batch, horizon, 7):
            raise ValueError("states must have shape [B,H,7]")


@dataclass(frozen=True)
class VisualPretrainMetrics:
    first_loss: float
    final_loss: float
    mean_loss: float
    steps: int
    target_has_grad: bool
    latent_tokens: int
    latent_dim: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class VisualActionMetrics:
    first_loss: float
    final_loss: float
    mean_loss: float
    correct_action_l1: float
    zero_action_l1: float
    shuffled_action_l1: float
    steps: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def rgb_to_canonical_frames(
    observations: NDArray[np.uint8] | Tensor,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Convert one/batched RGB observation from HWC/BHWC to float BCHW."""

    frames = torch.as_tensor(observations, device=device)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError("RGB observations must have shape [H,W,3] or [B,H,W,3]")
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] != 3:
        raise ValueError("RGB observations must have exactly three channels")
    frames = frames.contiguous()
    if frames.dtype == torch.uint8:
        frames = frames.float().div_(255.0)
    else:
        frames = frames.float()
        if not bool(torch.isfinite(frames).all()):
            raise ValueError("RGB observations contain NaN or Inf")
        if bool((frames.min() < 0.0) or (frames.max() > 1.0)):
            raise ValueError("floating-point RGB observations must be in [0,1]")
    return frames


def sample_planar_rgb_trajectories(
    batch_size: int,
    horizon: int,
    *,
    image_size: int = 64,
    action_limit: float = 0.10,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> VisualTrajectoryBatch:
    """Render deterministic random-action trajectories from ``PlanarReachEnv``."""

    if batch_size <= 0 or horizon <= 0:
        raise ValueError("batch_size and horizon must be positive")
    if not 0.0 < action_limit <= 1.0:
        raise ValueError("action_limit must be in (0,1]")
    frame_batches: list[Tensor] = []
    action_batches: list[Tensor] = []
    state_batches: list[Tensor] = []
    for index in range(batch_size):
        sample_seed = seed + index
        env = PlanarReachEnv(
            image_size=image_size,
            max_delta=action_limit,
            success_radius=0.01,
            max_steps=horizon + 1,
            seed=sample_seed,
        )
        initial_rgb, _ = env.reset(seed=sample_seed)
        rng = np.random.default_rng(sample_seed + 1_000_000_000)
        images = [initial_rgb]
        actions = np.zeros((horizon, 7), dtype=np.float32)
        states = np.zeros((horizon, 7), dtype=np.float32)
        for step in range(horizon):
            states[step] = env.state
            actions[step, :2] = rng.uniform(
                -action_limit, action_limit, size=2
            ).astype(np.float32)
            image, _, _, _, _ = env.step(actions[step])
            images.append(image)
        frame_batches.append(rgb_to_canonical_frames(np.stack(images))[0: horizon + 1])
        action_batches.append(torch.from_numpy(actions))
        state_batches.append(torch.from_numpy(states))
    return VisualTrajectoryBatch(
        frames=torch.stack(frame_batches).to(device),
        actions=torch.stack(action_batches).to(device),
        states=torch.stack(state_batches).to(device),
    )


class FrozenVisualEncoder(nn.Module):
    """Read-only adapter around a trained tiny V-JEPA EMA target encoder."""

    def __init__(self, source: TinyVideoJEPA) -> None:
        super().__init__()
        # The control stage owns an immutable snapshot. Sharing the source module
        # would let a later JEPA EMA update silently change the frozen features.
        self.encoder = copy.deepcopy(source.target_encoder)
        self.num_frames = self.encoder.grid_size[0] * self.encoder.tokenizer.tubelet_size[0]
        self.latent_tokens = self.encoder.num_tokens
        self.latent_dim = self.encoder.tokenizer.embed_dim
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> FrozenVisualEncoder:
        # Calling parent models' ``train`` must never switch the frozen teacher.
        super().train(False)
        self.encoder.eval()
        return self

    @torch.inference_mode()
    def encode_clips(self, clips: Tensor) -> Tensor:
        """Encode canonical clips ``[B,T,C,H,W]`` as ``[B,N,D]``."""

        if clips.ndim != 5 or clips.shape[1] != self.num_frames or clips.shape[2] != 3:
            raise ValueError(
                f"clips must have shape [B,{self.num_frames},3,Height,Width]"
            )
        return self.encoder(clips)

    @torch.inference_mode()
    def encode(self, frames: Tensor, mask: Tensor | None = None) -> Tensor:
        """Implement the common ``VisualEncoder.encode`` contract."""

        if mask is None:
            return self.encode_clips(frames)
        return self.encoder(frames, visible_mask=mask)

    @torch.inference_mode()
    def encode_rgb(self, observations: NDArray[np.uint8] | Tensor) -> Tensor:
        """Encode each RGB observation by repeating it into a static video clip."""

        try:
            device = next(self.encoder.parameters()).device
        except StopIteration:  # pragma: no cover - the encoder always has parameters
            device = torch.device("cpu")
        frames = rgb_to_canonical_frames(observations, device=device)
        clips = frames.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)
        return self.encode_clips(clips.contiguous())

    @torch.inference_mode()
    def encode_observation_sequences(self, sequences: Tensor) -> Tensor:
        """Encode ``[B,S,C,H,W]`` observations independently to ``[B,S,N,D]``."""

        if sequences.ndim != 5 or sequences.shape[2] != 3:
            raise ValueError("sequences must have shape [B,S,3,Height,Width]")
        batch, steps, channels, height, width = sequences.shape
        flattened = sequences.reshape(batch * steps, channels, height, width)
        latents = self.encode_rgb(flattened)
        return latents.reshape(batch, steps, self.latent_tokens, self.latent_dim)


def make_tiny_video_jepa(
    *,
    image_size: int = 64,
    num_frames: int = 2,
    embed_dim: int = 32,
) -> TinyVideoJEPA:
    """Construct the small visual backbone used by the control laboratory."""

    if image_size % 16:
        raise ValueError("image_size must be divisible by 16")
    if num_frames < 2 or num_frames % 2:
        raise ValueError("num_frames must be an even integer of at least two")
    encoder_heads = 4 if embed_dim % 4 == 0 else 2
    if embed_dim % encoder_heads:
        raise ValueError("embed_dim must be divisible by two")
    return TinyVideoJEPA(
        num_frames=num_frames,
        image_size=image_size,
        tubelet_size=(2, 16, 16),
        embed_dim=embed_dim,
        encoder_depth=2,
        encoder_heads=encoder_heads,
        predictor_dim=embed_dim,
        predictor_depth=2,
        predictor_heads=encoder_heads,
    )


def pretrain_tiny_video_jepa(
    model: TinyVideoJEPA,
    *,
    steps: int,
    batch_size: int,
    image_size: int,
    action_limit: float,
    learning_rate: float,
    device: torch.device | str,
    seed: int,
) -> VisualPretrainMetrics:
    """Run a deterministic V-JEPA pretraining stage on rendered motion clips."""

    if steps <= 0:
        raise ValueError("visual pretraining steps must be positive")
    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [*model.context_encoder.parameters(), *model.predictor.parameters()],
        lr=learning_rate,
    )
    losses: list[float] = []
    for step in range(steps):
        trajectories = sample_planar_rgb_trajectories(
            batch_size,
            model.target_encoder.grid_size[0]
            * model.target_encoder.tokenizer.tubelet_size[0]
            - 1,
            image_size=image_size,
            action_limit=action_limit,
            seed=seed + step * batch_size,
            device=device,
        )
        mask = generate_tube_masks(
            model.grid_size,
            num_short=1,
            num_long=1,
            short_scale=0.20,
            long_scale=0.50,
            min_context_ratio=0.25,
            seed=seed + step,
        ).target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(trajectories.frames, mask)
        output.loss.backward()
        optimizer.step()
        momentum = 1.0 if steps == 1 else 0.996 + (1.0 - 0.996) * step / (steps - 1)
        model.update_teacher(momentum)
        losses.append(float(output.loss.detach()))
    target_has_grad = any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    return VisualPretrainMetrics(
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=float(np.mean(losses)),
        steps=steps,
        target_has_grad=target_has_grad,
        latent_tokens=model.target_encoder.num_tokens,
        latent_dim=model.target_encoder.tokenizer.embed_dim,
    )


def assert_frozen_encoder_excluded(
    encoder: FrozenVisualEncoder,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail loudly if frozen visual weights could be changed by AC training."""

    encoder_parameters = tuple(encoder.parameters())
    if any(parameter.requires_grad for parameter in encoder_parameters):
        raise AssertionError("the visual encoder has trainable parameters")
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if encoder_ids & optimizer_ids:
        raise AssertionError("frozen visual encoder parameters are present in the AC optimizer")


def train_visual_action_steps(
    model: ActionWorldModel,
    encoder: FrozenVisualEncoder,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    batch_size: int,
    horizon: int,
    image_size: int,
    action_limit: float,
    device: torch.device | str,
    seed: int,
) -> VisualActionMetrics:
    """Fit AC dynamics using only target-encoder latents from rendered frames."""

    if steps <= 0 or batch_size < 2:
        raise ValueError("action training steps must be positive and batch_size at least two")
    assert_frozen_encoder_excluded(encoder, optimizer)
    device = torch.device(device)
    model.to(device).train()
    encoder.to(device).eval()
    losses: list[float] = []
    for step in range(steps):
        batch = sample_planar_rgb_trajectories(
            batch_size,
            horizon,
            image_size=image_size,
            action_limit=action_limit,
            seed=seed + 100_000 + step * batch_size,
            device=device,
        )
        with torch.inference_mode():
            latents = encoder.encode_observation_sequences(batch.frames)
        optimizer.zero_grad(set_to_none=True)
        output = model.compute_loss(
            latents[:, 0],
            batch.actions,
            latents[:, 1:],
            batch.states,
            rollout_steps=2,
            rollout_weight=1.0,
        )
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    evaluation = sample_planar_rgb_trajectories(
        batch_size,
        horizon,
        image_size=image_size,
        action_limit=action_limit,
        seed=seed + 700_000,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        latents = encoder.encode_observation_sequences(evaluation.frames)
        correct = F.l1_loss(
            model.rollout(latents[:, 0], evaluation.actions, evaluation.states),
            latents[:, 1:],
        )
        zero = F.l1_loss(
            model.rollout(
                latents[:, 0],
                torch.zeros_like(evaluation.actions),
                planar_states_from_actions(
                    evaluation.states[:, 0, :2],
                    torch.zeros_like(evaluation.actions),
                    max_delta=action_limit,
                ),
            ),
            latents[:, 1:],
        )
        shuffled_actions = torch.roll(evaluation.actions, shifts=1, dims=0)
        shuffled = F.l1_loss(
            model.rollout(
                latents[:, 0],
                shuffled_actions,
                planar_states_from_actions(
                    evaluation.states[:, 0, :2],
                    shuffled_actions,
                    max_delta=action_limit,
                ),
            ),
            latents[:, 1:],
        )
    return VisualActionMetrics(
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=float(np.mean(losses)),
        correct_action_l1=float(correct),
        zero_action_l1=float(zero),
        shuffled_action_l1=float(shuffled),
        steps=steps,
    )


def visual_rollout_errors(
    model: ActionWorldModel,
    encoder: FrozenVisualEncoder,
    *,
    horizons: tuple[int, ...] = (1, 2, 4),
    batch_size: int,
    image_size: int,
    action_limit: float,
    device: torch.device | str,
    seed: int,
) -> dict[str, float]:
    """Measure open-loop RGB-latent error without updating either model."""

    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons must contain positive values")
    maximum = max(horizons)
    batch = sample_planar_rgb_trajectories(
        batch_size,
        maximum,
        image_size=image_size,
        action_limit=action_limit,
        seed=seed,
        device=device,
    )
    model.eval()
    encoder.eval()
    with torch.inference_mode():
        latents = encoder.encode_observation_sequences(batch.frames)
        return {
            str(horizon): float(
                F.l1_loss(
                    model.rollout(
                        latents[:, 0],
                        batch.actions[:, :horizon],
                        batch.states[:, :horizon],
                    ),
                    latents[:, 1 : horizon + 1],
                )
            )
            for horizon in horizons
        }


def visual_latent_objective(
    model: ActionWorldModel,
    current_latent: Tensor,
    goal_image_latent: Tensor,
    current_state: Tensor,
    candidates: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Terminal latent energy with no goal coordinate or simulator reference.

    ``current_state`` may provide proprioception and is integrated under candidate
    actions.  The desired outcome enters exclusively as ``goal_image_latent``.
    """

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        device = current_latent.device
    actions = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    batch, horizon, _ = actions.shape
    initial = current_latent.to(device).expand(batch, -1, -1)
    goal = goal_image_latent.to(device).expand(batch, -1, -1)
    state = current_state.to(device).flatten()
    if state.shape != (7,):
        raise ValueError("current_state must have shape [7]")
    states = state.view(1, 1, 7).expand(batch, horizon, -1).clone()
    position = states[:, 0, :2].clone()
    for step in range(horizon):
        states[:, step, :2] = position
        position = (position + actions[:, step, :2]).clamp(0.0, 1.0)
    with torch.inference_mode():
        terminal = model.rollout(initial, actions, states)[:, -1]
        energy = (terminal - goal).abs().mean(dim=(1, 2))
    return energy.cpu().numpy().astype(np.float32)


def render_goal_observation(env: PlanarReachEnv) -> NDArray[np.uint8]:
    """Render the desired image with the controlled marker placed on the goal."""

    renderer = PlanarReachEnv(
        image_size=env.image_size,
        max_delta=env.max_delta,
        success_radius=env.success_radius,
        max_steps=env.max_steps,
        seed=0,
    )
    goal = env.goal
    image, _ = renderer.reset(start=goal, goal=goal)
    return image


class VisualCEMPlanner:
    """High-level planner implementing the project RGB planning contract."""

    def __init__(
        self,
        model: ActionWorldModel,
        encoder: FrozenVisualEncoder,
        *,
        horizon: int = 4,
        candidates: int = 256,
        elites: int = 32,
        refinements: int = 5,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.horizon = horizon
        self.candidates = candidates
        self.elites = elites
        self.refinements = refinements
        self.seed = seed

    def plan(
        self,
        current_rgb: NDArray[np.uint8] | Tensor,
        goal_rgb: NDArray[np.uint8] | Tensor,
        state: NDArray[np.floating] | Tensor,
        action_bounds: tuple[NDArray[np.floating], NDArray[np.floating]],
    ) -> PlanResult:
        """Return a CEM plan using only RGB goal information in the cost."""

        low, high = (np.asarray(bound, dtype=np.float32) for bound in action_bounds)
        if low.shape != (7,) or high.shape != (7,):
            raise ValueError("planar action bounds must each have shape [7]")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("planar action bounds must be finite")
        if not np.allclose(low[2:], 0.0) or not np.allclose(high[2:], 0.0):
            raise ValueError("planar action dimensions 2:7 must be fixed to zero")

        current_frames = rgb_to_canonical_frames(current_rgb)
        goal_frames = rgb_to_canonical_frames(goal_rgb)
        if current_frames.shape[0] != 1 or goal_frames.shape[0] != 1:
            raise ValueError("visual planning accepts exactly one current and one goal image")
        self.model.eval()
        self.encoder.eval()
        encoded = self.encoder.encode_rgb(torch.cat((current_frames, goal_frames), dim=0))
        current_latent, goal_latent = encoded[0:1], encoded[1:2]
        current_state = torch.as_tensor(state, dtype=torch.float32)
        planner = CEMPlanner(
            horizon=self.horizon,
            action_dim=7,
            candidates=self.candidates,
            elites=self.elites,
            refinements=self.refinements,
            action_low=low,
            action_high=high,
            seed=self.seed,
        )
        return planner.plan(
            lambda population: visual_latent_objective(
                self.model,
                current_latent,
                goal_latent,
                current_state,
                population,
            )
        )


def run_visual_cem_episode(
    model: ActionWorldModel,
    encoder: FrozenVisualEncoder,
    *,
    seed: int,
    horizon: int,
    candidates: int,
    elites: int,
    refinements: int,
    max_steps: int,
    action_limit: float,
    success_distance: float,
    image_size: int,
) -> dict[str, Any]:
    """Execute receding-horizon visual-latent planning for one held-out seed."""

    model.eval()
    encoder.eval()
    env = PlanarReachEnv(
        image_size=image_size,
        max_delta=action_limit,
        success_radius=success_distance,
        max_steps=max_steps,
        seed=seed,
    )
    _, initial_info = env.reset(seed=seed)
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
    encoded_observations = 0

    def objective_factory(current_env: PlanarReachEnv):
        nonlocal encoded_observations
        current_rgb = current_env.render()
        goal_rgb = render_goal_observation(current_env)
        latents = encoder.encode_rgb(np.stack((current_rgb, goal_rgb)))
        current_latent, goal_image_latent = latents[0:1], latents[1:2]
        current_state = torch.as_tensor(current_env.state, dtype=torch.float32)
        encoded_observations += 2
        return lambda population: visual_latent_objective(
            model,
            current_latent,
            goal_image_latent,
            current_state,
            population,
        )

    trace = receding_horizon_control(
        env,
        planner,
        objective_factory,
        max_steps=max_steps,
    )
    final_info = trace[-1].info if trace else initial_info
    energy_histories = [list(step.plan.energy_history) for step in trace]
    monotonic = all(
        all(b <= a + 1e-7 for a, b in pairwise(history))
        for history in energy_histories
    )
    return {
        "seed": seed,
        "initial_distance": float(initial_info["distance"]),
        "final_distance": float(final_info["distance"]),
        "success": bool(float(final_info["distance"]) <= success_distance),
        "steps": len(trace),
        "elite_energy_non_increasing": monotonic,
        "elite_energy_histories": energy_histories,
        "encoded_rgb_observations": encoded_observations,
        "goal_condition": "goal_image_latent_only",
    }


__all__ = [
    "FrozenVisualEncoder",
    "VisualActionMetrics",
    "VisualCEMPlanner",
    "VisualPretrainMetrics",
    "VisualTrajectoryBatch",
    "assert_frozen_encoder_excluded",
    "make_tiny_video_jepa",
    "pretrain_tiny_video_jepa",
    "render_goal_observation",
    "rgb_to_canonical_frames",
    "run_visual_cem_episode",
    "sample_planar_rgb_trajectories",
    "train_visual_action_steps",
    "visual_latent_objective",
    "visual_rollout_errors",
]
