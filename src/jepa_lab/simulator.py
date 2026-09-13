"""A deterministic image-based planar reaching environment.

The environment is intentionally dependency-free: NumPy supplies both dynamics
and rendering.  It gives the action-conditioned JEPA labs a controlled bridge from
latent prediction to planning without introducing a robotics framework first.
Coordinates are normalised to ``[0, 1]`` and actions use the canonical robot shape
``[dx, dy, dz, droll, dpitch, dyaw, dgripper]``.  Only ``dx`` and ``dy`` affect this
two-dimensional world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
UInt8Image = NDArray[np.uint8]


def canonical_action(dx: float, dy: float) -> NDArray[np.float32]:
    """Create the project's seven-dimensional delta-action representation."""

    action = np.zeros(7, dtype=np.float32)
    action[:2] = (dx, dy)
    return action


@dataclass(frozen=True)
class ReachInfo:
    position: NDArray[np.float32]
    goal: NDArray[np.float32]
    distance: float
    success: bool
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "distance": self.distance,
            "success": self.success,
            "step": self.step,
        }


class PlanarReachEnv:
    """A tiny Gym-like reaching task with deterministic seeding and rendering."""

    action_dim = 7
    state_dim = 7

    def __init__(
        self,
        *,
        image_size: int = 64,
        max_delta: float = 0.08,
        success_radius: float = 0.08,
        max_steps: int = 50,
        seed: int = 0,
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16 pixels")
        if not 0.0 < max_delta <= 1.0:
            raise ValueError("max_delta must be in (0, 1]")
        if not 0.0 < success_radius < 1.0:
            raise ValueError("success_radius must be in (0, 1)")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.image_size = int(image_size)
        self.max_delta = float(max_delta)
        self.success_radius = float(success_radius)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._position = np.zeros(2, dtype=np.float32)
        self._goal = np.ones(2, dtype=np.float32)
        self._step = 0
        self.reset()

    @property
    def action_low(self) -> NDArray[np.float32]:
        low = np.zeros(self.action_dim, dtype=np.float32)
        low[:2] = -self.max_delta
        return low

    @property
    def action_high(self) -> NDArray[np.float32]:
        high = np.zeros(self.action_dim, dtype=np.float32)
        high[:2] = self.max_delta
        return high

    @property
    def position(self) -> NDArray[np.float32]:
        return self._position.copy()

    @property
    def goal(self) -> NDArray[np.float32]:
        return self._goal.copy()

    @property
    def state(self) -> NDArray[np.float32]:
        """Return 7D proprioception; the first two entries are planar position."""

        state = np.zeros(self.state_dim, dtype=np.float32)
        state[:2] = self._position
        return state

    @property
    def goal_state(self) -> NDArray[np.float32]:
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[:2] = self._goal
        return state

    def _sample_point(self) -> NDArray[np.float32]:
        # A margin keeps both rendered disks completely visible.
        return self._rng.uniform(0.10, 0.90, size=2).astype(np.float32)

    @staticmethod
    def _point(value: FloatArray | tuple[float, float], name: str) -> NDArray[np.float32]:
        point = np.asarray(value, dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"{name} must be a finite point with shape [2]")
        if bool(((point < 0.0) | (point > 1.0)).any()):
            raise ValueError(f"{name} coordinates must lie in [0, 1]")
        return point.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        start: FloatArray | tuple[float, float] | None = None,
        goal: FloatArray | tuple[float, float] | None = None,
    ) -> tuple[UInt8Image, dict[str, Any]]:
        """Reset state and return ``(rgb_observation, info)``.

        Passing a seed restarts the local generator.  No module-level RNG state is
        read, so two environments given the same calls generate identical episodes.
        """

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        position = self._sample_point() if start is None else self._point(start, "start")
        target = self._sample_point() if goal is None else self._point(goal, "goal")
        if start is None and goal is None:
            # Avoid episodes that terminate before the first action.
            for _ in range(32):
                if np.linalg.norm(position - target) > 2.0 * self.success_radius:
                    break
                target = self._sample_point()
        self._position = position
        self._goal = target
        self._step = 0
        return self.render(), self._info().as_dict()

    def _info(self) -> ReachInfo:
        distance = float(np.linalg.norm(self._position - self._goal))
        return ReachInfo(
            position=self.position,
            goal=self.goal,
            distance=distance,
            success=distance <= self.success_radius,
            step=self._step,
        )

    def step(
        self, action: FloatArray
    ) -> tuple[UInt8Image, float, bool, bool, dict[str, Any]]:
        """Apply one 7D delta action using only its ``dx`` and ``dy`` entries."""

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"action must have shape [{self.action_dim}]")
        if not np.isfinite(action_array).all():
            raise ValueError("action must contain only finite values")

        delta = np.clip(action_array[:2], -self.max_delta, self.max_delta)
        self._position = np.clip(self._position + delta, 0.0, 1.0).astype(np.float32)
        self._step += 1
        info = self._info()
        terminated = info.success
        truncated = self._step >= self.max_steps and not terminated
        reward = -info.distance
        return self.render(), reward, terminated, truncated, info.as_dict()

    def render(self) -> UInt8Image:
        """Render an RGB array with a green goal and blue controlled point."""

        canvas = np.full(
            (self.image_size, self.image_size, 3), 245, dtype=np.uint8
        )
        canvas[[0, -1], :, :] = 40
        canvas[:, [0, -1], :] = 40
        self._draw_disk(canvas, self._goal, radius=4, colour=(45, 180, 80))
        self._draw_disk(canvas, self._position, radius=3, colour=(40, 90, 220))
        return canvas

    def _draw_disk(
        self,
        canvas: UInt8Image,
        point: NDArray[np.float32],
        *,
        radius: int,
        colour: tuple[int, int, int],
    ) -> None:
        centre_x = round(float(point[0]) * (self.image_size - 1))
        centre_y = round(float(point[1]) * (self.image_size - 1))
        yy, xx = np.ogrid[: self.image_size, : self.image_size]
        disk = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
        canvas[disk] = colour

    def simulate_actions(
        self,
        actions: FloatArray,
        *,
        start: FloatArray | None = None,
    ) -> NDArray[np.float32]:
        """Simulate action sequences without mutating the environment.

        ``actions`` may be ``[H, 7]`` or ``[K, H, 7]``.  The returned trajectory
        includes the initial point and therefore has shape ``[H+1, 2]`` or
        ``[K, H+1, 2]``.
        """

        action_array = np.asarray(actions, dtype=np.float32)
        single = action_array.ndim == 2
        if single:
            action_array = action_array[None, ...]
        if action_array.ndim != 3 or action_array.shape[-1] != self.action_dim:
            raise ValueError("actions must have shape [H, 7] or [K, H, 7]")
        if not np.isfinite(action_array).all():
            raise ValueError("actions must contain only finite values")

        initial = self._position if start is None else self._point(start, "start")
        batch_size, horizon, _ = action_array.shape
        trajectory = np.empty((batch_size, horizon + 1, 2), dtype=np.float32)
        trajectory[:, 0] = initial
        clipped = np.clip(action_array[..., :2], -self.max_delta, self.max_delta)
        for index in range(horizon):
            trajectory[:, index + 1] = np.clip(
                trajectory[:, index] + clipped[:, index], 0.0, 1.0
            )
        return trajectory[0] if single else trajectory

    def trajectory_cost(
        self,
        actions: FloatArray,
        *,
        action_weight: float = 0.01,
    ) -> NDArray[np.float32]:
        """Evaluate candidate sequences by terminal goal distance and action cost."""

        action_array = np.asarray(actions, dtype=np.float32)
        single = action_array.ndim == 2
        batched = action_array[None] if single else action_array
        trajectory = self.simulate_actions(batched)
        terminal_distance = np.linalg.norm(trajectory[:, -1] - self._goal, axis=-1)
        effort = np.square(batched[..., :2]).mean(axis=(1, 2))
        cost = terminal_distance + float(action_weight) * effort
        return cost[0] if single else cost.astype(np.float32)


__all__ = ["PlanarReachEnv", "ReachInfo", "canonical_action"]
