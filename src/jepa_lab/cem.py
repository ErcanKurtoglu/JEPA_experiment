"""Cross-Entropy Method planning for analytic or learned latent objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .types import PlanResult

CandidateBatch = NDArray[np.float32]
Objective = Callable[[CandidateBatch], NDArray[np.floating] | Tensor]


class StepEnvironment(Protocol):
    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class RecedingHorizonStep:
    observation: NDArray[np.uint8]
    action: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    plan: PlanResult


class CEMPlanner:
    """Minimise a batched sequence objective with the Cross-Entropy Method.

    Previous elites are retained in the next candidate population. For a
    deterministic objective, ``PlanResult.energy_history`` (the *observed* mean
    energy of the elite set) should therefore be non-increasing. The raw values
    are retained so a stochastic or stateful objective cannot fake this invariant.
    """

    def __init__(
        self,
        *,
        horizon: int = 4,
        action_dim: int = 7,
        candidates: int = 256,
        elites: int = 32,
        refinements: int = 5,
        action_low: float | NDArray[np.floating] = -1.0,
        action_high: float | NDArray[np.floating] = 1.0,
        min_std: float = 1e-3,
        seed: int = 0,
    ) -> None:
        if min(horizon, action_dim, candidates, elites, refinements) <= 0:
            raise ValueError("planner dimensions and counts must be positive")
        if elites >= candidates:
            raise ValueError("elites must be strictly smaller than candidates")
        if min_std < 0.0:
            raise ValueError("min_std cannot be negative")

        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.candidates = int(candidates)
        self.elites = int(elites)
        self.refinements = int(refinements)
        self.min_std = float(min_std)
        self.action_low = self._bound(action_low, "action_low")
        self.action_high = self._bound(action_high, "action_high")
        if bool((self.action_low > self.action_high).any()):
            raise ValueError("action_low cannot exceed action_high")
        self._rng = np.random.default_rng(seed)

    def _bound(
        self, value: float | NDArray[np.floating], name: str
    ) -> NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full(self.action_dim, float(array), dtype=np.float32)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite scalar or shape [{self.action_dim}]")
        return array

    def reset_rng(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def plan(
        self,
        objective: Objective,
        *,
        initial_mean: NDArray[np.floating] | None = None,
        initial_std: NDArray[np.floating] | None = None,
    ) -> PlanResult:
        """Optimise and return the lowest-energy action sequence."""

        shape = (self.horizon, self.action_dim)
        span = self.action_high - self.action_low
        mean = (
            (self.action_low + self.action_high) / 2.0
            if initial_mean is None
            else self._initial(initial_mean, shape, "initial_mean")
        )
        std = (
            np.broadcast_to(span / 2.0, shape).copy()
            if initial_std is None
            else self._initial(initial_std, shape, "initial_std")
        )
        mean = np.clip(mean, self.action_low, self.action_high).astype(np.float32)
        std = np.maximum(std, 0.0).astype(np.float32)
        std[:, span == 0.0] = 0.0

        retained: CandidateBatch | None = None
        best_sequence: NDArray[np.float32] | None = None
        best_energy = float("inf")
        history: list[float] = []

        for _ in range(self.refinements):
            retained_count = 0 if retained is None else retained.shape[0]
            sample_count = self.candidates - retained_count
            samples = self._rng.normal(
                loc=mean,
                scale=std,
                size=(sample_count, self.horizon, self.action_dim),
            ).astype(np.float32)
            np.clip(samples, self.action_low, self.action_high, out=samples)
            population = samples if retained is None else np.concatenate((retained, samples))
            energies = self._energies(objective, population)
            elite_indices = np.argpartition(energies, self.elites - 1)[: self.elites]
            elite_order = elite_indices[np.argsort(energies[elite_indices])]
            retained = population[elite_order].copy()
            elite_energies = energies[elite_order]

            mean = retained.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = retained.std(axis=0, dtype=np.float64).astype(np.float32)
            movable = span > 0.0
            std[:, movable] = np.maximum(std[:, movable], self.min_std)
            std[:, ~movable] = 0.0
            elite_energy = float(elite_energies.mean(dtype=np.float64))
            history.append(elite_energy)

            if float(elite_energies[0]) < best_energy:
                best_energy = float(elite_energies[0])
                best_sequence = retained[0].copy()

        if retained is None or best_sequence is None:  # guarded by validation
            raise RuntimeError("CEM did not produce a candidate")
        return PlanResult(
            first_action=best_sequence[0].copy(),
            sequence=best_sequence,
            energy=best_energy,
            elite_mean=mean.copy(),
            elite_std=std.copy(),
            energy_history=tuple(history),
        )

    @staticmethod
    def _initial(
        value: NDArray[np.floating], shape: tuple[int, int], name: str
    ) -> NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite and have shape {shape}")
        return array.copy()

    def _energies(self, objective: Objective, population: CandidateBatch) -> NDArray[np.float32]:
        values = objective(population)
        if isinstance(values, Tensor):
            values = values.detach().cpu().numpy()
        energies = np.asarray(values, dtype=np.float32)
        if energies.shape != (population.shape[0],):
            raise ValueError(
                f"objective must return [{population.shape[0]}] energies, got {energies.shape}"
            )
        if not np.isfinite(energies).all():
            raise ValueError("objective returned NaN or infinite energy")
        return energies


def make_latent_objective(
    model: Any,
    z0: Tensor,
    goal_latent: Tensor,
    *,
    states: Tensor | None = None,
    device: torch.device | str | None = None,
) -> Objective:
    """Wrap ``ActionWorldModel.rollout`` as a NumPy-compatible CEM objective."""

    if z0.ndim == 2:
        z0 = z0.unsqueeze(0)
    if goal_latent.ndim == 2:
        goal_latent = goal_latent.unsqueeze(0)
    if z0.ndim != 3 or goal_latent.ndim != 3 or z0.shape[1:] != goal_latent.shape[1:]:
        raise ValueError("z0 and goal_latent must have compatible [B,N,D] shapes")
    if z0.shape[0] != 1 or goal_latent.shape[0] != 1:
        raise ValueError("planning objective expects one current and one goal latent")
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = z0.device
    z0 = z0.to(device)
    goal_latent = goal_latent.to(device)
    if states is not None:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3 or states.shape[0] != 1:
            raise ValueError("states must have shape [H,S] or [1,H,S]")
        states = states.to(device)

    def objective(candidates: CandidateBatch) -> NDArray[np.float32]:
        action_tensor = torch.as_tensor(candidates, device=device, dtype=z0.dtype)
        batch = action_tensor.shape[0]
        initial = z0.expand(batch, -1, -1)
        state_batch = None
        if states is not None:
            if states.shape[1] != action_tensor.shape[1]:
                raise ValueError("state horizon must match candidate action horizon")
            state_batch = states.expand(batch, -1, -1)
        with torch.no_grad():
            predictions = model.rollout(initial, action_tensor, state_batch)
            goal = goal_latent.expand(batch, -1, -1)
            energy = (predictions[:, -1] - goal).abs().mean(dim=(1, 2))
        return energy.float().cpu().numpy()

    return objective


def receding_horizon_control(
    env: StepEnvironment,
    planner: CEMPlanner,
    objective_factory: Callable[[StepEnvironment], Objective],
    *,
    max_steps: int = 50,
    warm_start: bool = True,
) -> list[RecedingHorizonStep]:
    """Plan, execute only the first action, observe, and replan until termination."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    trace: list[RecedingHorizonStep] = []
    mean: NDArray[np.float32] | None = None
    std: NDArray[np.float32] | None = None
    for _ in range(max_steps):
        result = planner.plan(
            objective_factory(env),
            initial_mean=mean if warm_start else None,
            initial_std=std if warm_start else None,
        )
        observation, reward, terminated, truncated, info = env.step(result.first_action)
        trace.append(
            RecedingHorizonStep(
                observation=observation,
                action=np.asarray(result.first_action, dtype=np.float32).copy(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
                plan=result,
            )
        )
        if terminated or truncated:
            break
        if warm_start:
            # Shift the distribution after executing its first element; repeat the
            # terminal estimate to keep the requested horizon fixed.
            mean = np.concatenate((result.elite_mean[1:], result.elite_mean[-1:]))
            std = np.concatenate((result.elite_std[1:], result.elite_std[-1:]))
    return trace


__all__ = [
    "CEMPlanner",
    "Objective",
    "RecedingHorizonStep",
    "make_latent_objective",
    "receding_horizon_control",
]
