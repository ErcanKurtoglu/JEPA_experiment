"""Compact action-conditioned latent dynamics inspired by V-JEPA2-AC.

This is a teaching model, not a reimplementation of Meta's 300M-parameter action
predictor.  It retains the important structure: a frozen visual representation can
be supplied as latent patch maps, actions and proprioception condition a causal
transformer, teacher forcing trains every horizon position in parallel, and an
autoregressive rollout term exposes compounding prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActionWorldModelOutput:
    loss: Tensor
    teacher_forcing_loss: Tensor
    rollout_loss: Tensor
    teacher_forcing_predictions: Tensor
    rollout_predictions: Tensor


class BlockCausalPredictor(nn.Module):
    """Predict the next latent patch map using causal action/state history.

    One summary token represents each time block.  The transformer's upper
    triangular attention mask prevents block ``t`` from observing any future block.
    Its output conditions every spatial latent token at the corresponding step.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int = 7,
        state_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_horizon: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden_dim, num_layers, num_heads, max_horizon) <= 0:
            raise ValueError("model dimensions must be positive")
        if state_dim < 0:
            raise ValueError("state_dim cannot be negative")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = int(max_horizon)

        self.summary_projection = nn.Linear(latent_dim, hidden_dim)
        self.token_projection = nn.Linear(latent_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim, bias=False)
        self.state_projection = (
            nn.Linear(state_dim, hidden_dim, bias=False) if state_dim else None
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, max_horizon, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.causal_transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim),
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

    @staticmethod
    def block_causal_mask(length: int, device: torch.device | None = None) -> Tensor:
        """Return a mask where each block can attend only to itself and its past."""

        if length <= 0:
            raise ValueError("length must be positive")
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
        )

    def _validate(
        self, latents: Tensor, actions: Tensor, states: Tensor | None
    ) -> tuple[int, int, int]:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [B, H, N, D]")
        batch, horizon, tokens, dimension = latents.shape
        if dimension != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}, got {dimension}")
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.max_horizon}]")
        if actions.shape != (batch, horizon, self.action_dim):
            raise ValueError(
                f"actions must have shape [{batch}, {horizon}, {self.action_dim}]"
            )
        if states is not None and states.shape != (batch, horizon, self.state_dim):
            raise ValueError(
                f"states must have shape [{batch}, {horizon}, {self.state_dim}]"
            )
        if tokens <= 0:
            raise ValueError("at least one latent token is required")
        return batch, horizon, tokens

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
    ) -> Tensor:
        """Predict one next latent grid for every causal input block."""

        _, horizon, _ = self._validate(latents, actions, states)
        summaries = latents.mean(dim=2)
        steps = self.summary_projection(summaries) + self.action_projection(actions)
        if states is not None and self.state_projection is not None:
            steps = steps + self.state_projection(states)
        steps = steps + self.temporal_embedding[:, :horizon]
        causal_context = self.causal_transformer(
            steps, mask=self.block_causal_mask(horizon, latents.device)
        )
        token_features = self.token_projection(latents)
        conditioned = token_features + causal_context.unsqueeze(2)
        # Residual prediction makes the model learn action-conditioned change.
        return latents + self.prediction_head(conditioned)


class ActionWorldModel(nn.Module):
    """Teacher-forced and autoregressive interface around a causal predictor."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int = 7,
        state_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_horizon: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictor = BlockCausalPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_horizon=max_horizon,
            dropout=dropout,
        )

    @property
    def latent_dim(self) -> int:
        return self.predictor.latent_dim

    @property
    def action_dim(self) -> int:
        return self.predictor.action_dim

    @property
    def state_dim(self) -> int:
        return self.predictor.state_dim

    def predict_next(
        self,
        latent: Tensor,
        action: Tensor,
        state: Tensor | None = None,
    ) -> Tensor:
        """Convenience interface for one-step prediction.

        Inputs have shapes ``latent[B,N,D]``, ``action[B,A]`` and optionally
        ``state[B,S]``.
        """

        if latent.ndim != 3 or action.ndim != 2:
            raise ValueError("latent/action must have shapes [B,N,D] and [B,A]")
        state_sequence = state.unsqueeze(1) if state is not None else None
        return self.predictor(
            latent.unsqueeze(1), action.unsqueeze(1), state_sequence
        )[:, 0]

    def teacher_forcing(
        self,
        z0: Tensor,
        actions: Tensor,
        target_latents: Tensor,
        states: Tensor | None = None,
    ) -> Tensor:
        """Predict all target steps while feeding ground-truth previous latents."""

        self._validate_rollout_inputs(z0, actions, states)
        if target_latents.ndim != 4:
            raise ValueError("target_latents must have shape [B,H,N,D]")
        expected = (z0.shape[0], actions.shape[1], z0.shape[1], z0.shape[2])
        if target_latents.shape != expected:
            raise ValueError(f"target_latents must have shape {expected}")
        inputs = torch.cat((z0.unsqueeze(1), target_latents[:, :-1]), dim=1)
        return self.predictor(inputs, actions, states)

    def rollout(
        self,
        z0: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
    ) -> Tensor:
        """Autoregressively imagine a latent trajectory for an action sequence."""

        self._validate_rollout_inputs(z0, actions, states)
        history: list[Tensor] = [z0]
        predictions: list[Tensor] = []
        for step in range(actions.shape[1]):
            input_history = torch.stack(history, dim=1)
            state_history = states[:, : step + 1] if states is not None else None
            next_latent = self.predictor(
                input_history,
                actions[:, : step + 1],
                state_history,
            )[:, -1]
            predictions.append(next_latent)
            history.append(next_latent)
        return torch.stack(predictions, dim=1)

    def compute_loss(
        self,
        z0: Tensor,
        actions: Tensor,
        target_latents: Tensor,
        states: Tensor | None = None,
        *,
        rollout_steps: int = 2,
        rollout_weight: float = 1.0,
    ) -> ActionWorldModelOutput:
        """Combine teacher-forcing L1 with a short autoregressive rollout L1."""

        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if rollout_weight < 0.0:
            raise ValueError("rollout_weight cannot be negative")
        teacher_predictions = self.teacher_forcing(
            z0, actions, target_latents, states
        )
        teacher_loss = F.l1_loss(teacher_predictions, target_latents)

        steps = min(rollout_steps, actions.shape[1])
        rollout_predictions = self.rollout(
            z0,
            actions[:, :steps],
            states[:, :steps] if states is not None else None,
        )
        rollout_loss = F.l1_loss(
            rollout_predictions, target_latents[:, :steps]
        )
        loss = teacher_loss + float(rollout_weight) * rollout_loss
        return ActionWorldModelOutput(
            loss=loss,
            teacher_forcing_loss=teacher_loss,
            rollout_loss=rollout_loss,
            teacher_forcing_predictions=teacher_predictions,
            rollout_predictions=rollout_predictions,
        )

    def forward(
        self,
        z0: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
    ) -> Tensor:
        return self.rollout(z0, actions, states)

    def _validate_rollout_inputs(
        self, z0: Tensor, actions: Tensor, states: Tensor | None
    ) -> None:
        if z0.ndim != 3:
            raise ValueError("z0 must have shape [B,N,D]")
        batch, tokens, dimension = z0.shape
        if tokens <= 0 or dimension != self.latent_dim:
            raise ValueError(
                f"z0 must contain tokens with latent dimension {self.latent_dim}"
            )
        if actions.ndim != 3 or actions.shape[0] != batch:
            raise ValueError("actions must have shape [B,H,A]")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must end in dimension {self.action_dim}")
        if not 0 < actions.shape[1] <= self.predictor.max_horizon:
            raise ValueError(
                f"action horizon must be in [1, {self.predictor.max_horizon}]"
            )
        if states is not None and states.shape != (
            batch,
            actions.shape[1],
            self.state_dim,
        ):
            raise ValueError(
                f"states must have shape [{batch}, {actions.shape[1]}, {self.state_dim}]"
            )


__all__ = [
    "ActionWorldModel",
    "ActionWorldModelOutput",
    "BlockCausalPredictor",
]
