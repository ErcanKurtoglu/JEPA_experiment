"""Train and evaluate the teaching-scale action-conditioned planar world model.

The default analytic profile uses an exact 2-D position latent to isolate dynamics
and CEM/MPC mechanics.  ``--latent-source visual`` runs the complete teaching path:
rendered RGB → trained/frozen tiny video-JEPA → action-conditioned rollout →
goal-image latent CEM.  Neither mode is a V-JEPA2-AC reproduction or a paper-scale
robotics result.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from jepa_lab.action_world_model import ActionWorldModel
from jepa_lab.cem import CEMPlanner, receding_horizon_control
from jepa_lab.datasets import sample_planar_latent_batch
from jepa_lab.device import seed_everything, select_device
from jepa_lab.experiments import learned_planar_objective, train_action_steps
from jepa_lab.foundations import environment_report
from jepa_lab.runlog import config_digest
from jepa_lab.simulator import PlanarReachEnv
from jepa_lab.visual_control import (
    FrozenVisualEncoder,
    assert_frozen_encoder_excluded,
    make_tiny_video_jepa,
    pretrain_tiny_video_jepa,
    run_visual_cem_episode,
    train_visual_action_steps,
    visual_rollout_errors,
)


@dataclass(frozen=True)
class DemoConfig:
    """Resolved configuration; every value is included in the JSON result."""

    profile: str
    requested_device: str
    seed: int
    train_steps: int
    batch_size: int
    learning_rate: float
    horizon: int
    hidden_dim: int
    layers: int
    heads: int
    eval_seeds: int
    candidates: int
    elites: int
    refinements: int
    max_episode_steps: int
    action_limit: float
    success_distance: float
    min_success_rate: float
    min_random_margin: float
    latent_source: str = "analytic"
    visual_pretrain_steps: int = 5
    visual_embed_dim: int = 16
    image_size: int = 64


PROFILES = {
    # Fast structural check for a laptop/CI runner.  Its metrics are not expected
    # to satisfy the full M6 acceptance thresholds.
    "smoke": DemoConfig(
        profile="smoke",
        requested_device="auto",
        seed=42,
        train_steps=80,
        batch_size=64,
        learning_rate=3e-3,
        horizon=4,
        hidden_dim=64,
        layers=2,
        heads=4,
        eval_seeds=3,
        candidates=32,
        elites=8,
        refinements=2,
        max_episode_steps=10,
        action_limit=0.10,
        success_distance=0.08,
        min_success_rate=0.80,
        min_random_margin=0.30,
        visual_pretrain_steps=5,
        visual_embed_dim=16,
    ),
    # The exact M6 planner/model/evaluation counts from the study plan.  Although
    # substantially heavier, this remains a small synthetic teaching experiment.
    "full": DemoConfig(
        profile="full",
        requested_device="auto",
        seed=42,
        train_steps=5_000,
        batch_size=128,
        learning_rate=1e-3,
        horizon=4,
        hidden_dim=256,
        layers=4,
        heads=8,
        eval_seeds=100,
        candidates=256,
        elites=32,
        refinements=5,
        max_episode_steps=20,
        action_limit=0.10,
        success_distance=0.08,
        min_success_rate=0.80,
        min_random_margin=0.30,
        visual_pretrain_steps=500,
        visual_embed_dim=64,
    ),
}


def _validate(config: DemoConfig) -> None:
    positive_ints = {
        "train_steps": config.train_steps,
        "batch_size": config.batch_size,
        "horizon": config.horizon,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "heads": config.heads,
        "eval_seeds": config.eval_seeds,
        "candidates": config.candidates,
        "elites": config.elites,
        "refinements": config.refinements,
        "max_episode_steps": config.max_episode_steps,
        "visual_pretrain_steps": config.visual_pretrain_steps,
        "visual_embed_dim": config.visual_embed_dim,
        "image_size": config.image_size,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError(f"these values must be positive: {', '.join(invalid)}")
    if config.hidden_dim % config.heads:
        raise ValueError("hidden_dim must be divisible by heads")
    if config.elites >= config.candidates:
        raise ValueError("elites must be strictly smaller than candidates")
    if not 0.0 < config.learning_rate:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < config.action_limit <= 1.0:
        raise ValueError("action_limit must be in (0, 1]")
    if not 0.0 < config.success_distance < 1.0:
        raise ValueError("success_distance must be in (0, 1)")
    if not 0.0 <= config.min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be in [0, 1]")
    if not 0.0 <= config.min_random_margin <= 1.0:
        raise ValueError("min_random_margin must be in [0, 1]")
    if config.latent_source not in {"analytic", "visual"}:
        raise ValueError("latent_source must be 'analytic' or 'visual'")
    if config.image_size % 16:
        raise ValueError("image_size must be divisible by the visual patch size 16")


def _non_increasing(values: Sequence[float], *, atol: float = 1e-7) -> bool:
    return all(next_value <= value + atol for value, next_value in pairwise(values))


def _run_cem_episode(
    model: ActionWorldModel,
    config: DemoConfig,
    *,
    episode_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    env = PlanarReachEnv(
        max_delta=config.action_limit,
        success_radius=config.success_distance,
        max_steps=config.max_episode_steps,
        seed=episode_seed,
    )
    _, initial_info = env.reset(seed=episode_seed)
    planner = CEMPlanner(
        horizon=config.horizon,
        action_dim=7,
        candidates=config.candidates,
        elites=config.elites,
        refinements=config.refinements,
        action_low=env.action_low,
        action_high=env.action_high,
        seed=episode_seed,
    )
    trace = receding_horizon_control(
        env,
        planner,
        lambda current_env: lambda population: learned_planar_objective(
            model, current_env, population, device=device
        ),
        max_steps=config.max_episode_steps,
    )
    final_info = trace[-1].info if trace else initial_info
    energy_histories = [list(step.plan.energy_history) for step in trace]
    return {
        "seed": episode_seed,
        "initial_distance": float(initial_info["distance"]),
        "final_distance": float(final_info["distance"]),
        "success": bool(float(final_info["distance"]) <= config.success_distance),
        "steps": len(trace),
        "elite_energy_non_increasing": all(
            _non_increasing(history) for history in energy_histories
        ),
        "elite_energy_histories": energy_histories,
    }


def _run_random_episode(config: DemoConfig, *, episode_seed: int) -> dict[str, Any]:
    """Run a seeded random-action policy on the same start/goal distribution."""

    env = PlanarReachEnv(
        max_delta=config.action_limit,
        success_radius=config.success_distance,
        max_steps=config.max_episode_steps,
        seed=episode_seed,
    )
    _, initial_info = env.reset(seed=episode_seed)
    # A separate stream prevents environment sampling from changing the policy.
    rng = np.random.default_rng(episode_seed + 2_000_000_000)
    final_info = initial_info
    steps = 0
    for steps in range(1, config.max_episode_steps + 1):
        action = np.zeros(7, dtype=np.float32)
        action[:2] = rng.uniform(
            -config.action_limit, config.action_limit, size=2
        ).astype(np.float32)
        _, _, terminated, truncated, final_info = env.step(action)
        if terminated or truncated:
            break
    return {
        "seed": episode_seed,
        "initial_distance": float(initial_info["distance"]),
        "final_distance": float(final_info["distance"]),
        "success": bool(float(final_info["distance"]) <= config.success_distance),
        "steps": steps,
    }


def _summarize_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    distances = np.asarray([episode["final_distance"] for episode in episodes], dtype=np.float64)
    successes = np.asarray([episode["success"] for episode in episodes], dtype=np.float64)
    steps = np.asarray([episode["steps"] for episode in episodes], dtype=np.float64)
    # Wilson score interval is well behaved even near 0%/100% and makes the
    # 100-seed acceptance result more informative than a point estimate alone.
    count = len(episodes)
    successes_count = int(successes.sum())
    rate = float(successes.mean())
    z = 1.959963984540054
    denominator = 1.0 + z**2 / count
    centre = (rate + z**2 / (2.0 * count)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / count + z**2 / (4.0 * count**2))
        / denominator
    )
    return {
        "count": count,
        "success_count": successes_count,
        "success_rate": rate,
        "success_rate_wilson_95": [
            float(max(0.0, centre - half_width)),
            float(min(1.0, centre + half_width)),
        ],
        "final_distance_mean": float(distances.mean()),
        "final_distance_median": float(np.median(distances)),
        "final_distance_min": float(distances.min()),
        "final_distance_max": float(distances.max()),
        "steps_mean": float(steps.mean()),
    }


def _rollout_errors(
    model: ActionWorldModel,
    config: DemoConfig,
    device: torch.device,
) -> dict[str, float]:
    """Measure compounding error at the requested 1/2/4-step checkpoints."""

    z0, actions, states, targets = sample_planar_latent_batch(
        config.batch_size,
        config.horizon,
        action_limit=config.action_limit,
        seed=config.seed + 500_000,
        device=device,
    )
    errors: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for steps in (1, 2, 4):
            if steps <= config.horizon:
                predictions = model.rollout(z0, actions[:, :steps], states[:, :steps])
                errors[str(steps)] = float(F.l1_loss(predictions, targets[:, :steps]))
    return errors


def _run_analytic_experiment(config: DemoConfig) -> dict[str, Any]:
    """Run the exact-xy control experiment used to debug dynamics and CEM."""

    _validate(config)
    started = time.perf_counter()
    device = select_device(config.requested_device)
    seed_everything(config.seed)
    model = ActionWorldModel(
        latent_dim=2,
        action_dim=7,
        state_dim=7,
        hidden_dim=config.hidden_dim,
        num_layers=config.layers,
        num_heads=config.heads,
        max_horizon=config.horizon,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    training = train_action_steps(
        model,
        optimizer,
        steps=config.train_steps,
        batch_size=config.batch_size,
        horizon=config.horizon,
        action_limit=config.action_limit,
        device=device,
        seed=config.seed,
    )
    rollout_errors = _rollout_errors(model, config, device)

    held_out_seeds = [config.seed + 1_000_000 + index for index in range(config.eval_seeds)]
    cem_episodes = [
        _run_cem_episode(model, config, episode_seed=episode_seed, device=device)
        for episode_seed in held_out_seeds
    ]
    random_episodes = [
        _run_random_episode(config, episode_seed=episode_seed)
        for episode_seed in held_out_seeds
    ]
    cem_summary = _summarize_episodes(cem_episodes)
    random_summary = _summarize_episodes(random_episodes)
    success_margin = float(cem_summary["success_rate"] - random_summary["success_rate"])
    energy_monotonic = all(
        bool(episode["elite_energy_non_increasing"]) for episode in cem_episodes
    )
    training_dict = training.as_dict()
    checks = {
        "correct_action_better_than_zero": (
            training.correct_action_l1 < training.zero_action_l1
        ),
        "correct_action_better_than_shuffled": (
            training.correct_action_l1 < training.shuffled_action_l1
        ),
        "cem_success_rate": cem_summary["success_rate"] >= config.min_success_rate,
        "cem_margin_over_random": success_margin >= config.min_random_margin,
        "elite_energy_non_increasing": energy_monotonic,
    }
    resolved_config = asdict(config)
    return {
        "schema_version": 1,
        "experiment": "planar_action_conditioned_world_model",
        "scope": {
            "paper_scale": False,
            "kind": "teaching_lab",
            "latent_source": "exact_normalized_xy_proxy",
            "claim": (
                "Tests action-conditioned latent dynamics and CEM/MPC mechanics; "
                "it is not a V-JEPA2-AC reproduction or real-robot safety result."
            ),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "device": str(device),
        "environment": environment_report(),
        "config": resolved_config,
        "config_sha256": config_digest(resolved_config),
        "model_parameters": parameter_count,
        "metrics": {
            "training": training_dict,
            "rollout_l1_by_horizon": rollout_errors,
            "cem": cem_summary,
            "random_action": random_summary,
            "cem_success_margin_over_random": success_margin,
        },
        "evaluation": {
            "held_out_seed_rule": "seed + 1_000_000 + episode_index",
            "held_out_seeds": held_out_seeds,
            "cem_episodes": cem_episodes,
            "random_action_episodes": random_episodes,
        },
        "acceptance": {
            "enforced_only_when_requested": True,
            "thresholds": {
                "minimum_cem_success_rate": config.min_success_rate,
                "minimum_success_margin_over_random": config.min_random_margin,
                "success_distance": config.success_distance,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_visual_experiment(config: DemoConfig) -> dict[str, Any]:
    """Run the experimental RGB → frozen V-JEPA → AC model → CEM path."""

    _validate(config)
    started = time.perf_counter()
    device = select_device(config.requested_device)
    seed_everything(config.seed)

    video_jepa = make_tiny_video_jepa(
        image_size=config.image_size,
        num_frames=2,
        embed_dim=config.visual_embed_dim,
    ).to(device)
    visual_pretraining = pretrain_tiny_video_jepa(
        video_jepa,
        steps=config.visual_pretrain_steps,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_limit=config.action_limit,
        learning_rate=config.learning_rate,
        device=device,
        seed=config.seed,
    )
    encoder = FrozenVisualEncoder(video_jepa).to(device)
    model = ActionWorldModel(
        latent_dim=encoder.latent_dim,
        action_dim=7,
        state_dim=7,
        hidden_dim=config.hidden_dim,
        num_layers=config.layers,
        num_heads=config.heads,
        max_horizon=config.horizon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    assert_frozen_encoder_excluded(encoder, optimizer)
    training = train_visual_action_steps(
        model,
        encoder,
        optimizer,
        steps=config.train_steps,
        batch_size=config.batch_size,
        horizon=config.horizon,
        image_size=config.image_size,
        action_limit=config.action_limit,
        device=device,
        seed=config.seed,
    )
    requested_horizons = tuple(step for step in (1, 2, 4) if step <= config.horizon)
    rollout_errors = visual_rollout_errors(
        model,
        encoder,
        horizons=requested_horizons,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_limit=config.action_limit,
        device=device,
        seed=config.seed + 500_000,
    )

    held_out_seeds = [config.seed + 1_000_000 + index for index in range(config.eval_seeds)]
    cem_episodes = [
        run_visual_cem_episode(
            model,
            encoder,
            seed=episode_seed,
            horizon=config.horizon,
            candidates=config.candidates,
            elites=config.elites,
            refinements=config.refinements,
            max_steps=config.max_episode_steps,
            action_limit=config.action_limit,
            success_distance=config.success_distance,
            image_size=config.image_size,
        )
        for episode_seed in held_out_seeds
    ]
    random_episodes = [
        _run_random_episode(config, episode_seed=episode_seed)
        for episode_seed in held_out_seeds
    ]
    cem_summary = _summarize_episodes(cem_episodes)
    random_summary = _summarize_episodes(random_episodes)
    success_margin = float(cem_summary["success_rate"] - random_summary["success_rate"])
    checks = {
        "visual_teacher_has_no_grad": not visual_pretraining.target_has_grad,
        "visual_encoder_frozen": not any(
            parameter.requires_grad for parameter in encoder.parameters()
        ),
        "correct_action_better_than_zero": (
            training.correct_action_l1 < training.zero_action_l1
        ),
        "correct_action_better_than_shuffled": (
            training.correct_action_l1 < training.shuffled_action_l1
        ),
        "cem_success_rate": cem_summary["success_rate"] >= config.min_success_rate,
        "cem_margin_over_random": success_margin >= config.min_random_margin,
        "elite_energy_non_increasing": all(
            bool(episode["elite_energy_non_increasing"]) for episode in cem_episodes
        ),
    }
    resolved_config = asdict(config)
    return {
        "schema_version": 1,
        "experiment": "planar_visual_action_conditioned_world_model",
        "scope": {
            "paper_scale": False,
            "kind": "experimental teaching lab",
            "latent_source": "frozen_tiny_video_jepa_target_encoder_from_rgb",
            "goal_information_used_by_cost": "goal image latent only",
            "claim": (
                "Exercises the complete visual-latent control data flow at teaching scale; "
                "it is not V-JEPA2-AC parity or a real-robot safety result."
            ),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "device": str(device),
        "environment": environment_report(),
        "config": resolved_config,
        "config_sha256": config_digest(resolved_config),
        "model_parameters": {
            "visual_backbone": sum(parameter.numel() for parameter in encoder.parameters()),
            "action_predictor": sum(parameter.numel() for parameter in model.parameters()),
        },
        "metrics": {
            "visual_pretraining": visual_pretraining.as_dict(),
            "action_training": training.as_dict(),
            "rollout_l1_by_horizon": rollout_errors,
            "cem": cem_summary,
            "random_action": random_summary,
            "cem_success_margin_over_random": success_margin,
        },
        "evaluation": {
            "held_out_seed_rule": "seed + 1_000_000 + episode_index",
            "held_out_seeds": held_out_seeds,
            "cem_episodes": cem_episodes,
            "random_action_episodes": random_episodes,
        },
        "acceptance": {
            "enforced_only_when_requested": True,
            "experimental_visual_path": True,
            "thresholds": {
                "minimum_cem_success_rate": config.min_success_rate,
                "minimum_success_margin_over_random": config.min_random_margin,
                "success_distance": config.success_distance,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_experiment(config: DemoConfig) -> dict[str, Any]:
    """Dispatch to the explicit analytic control or complete visual pipeline."""

    if config.latent_source == "visual":
        return _run_visual_experiment(config)
    return _run_analytic_experiment(config)


def _add_override(parser: argparse.ArgumentParser, name: str, *, value_type: type) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=value_type, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teaching-scale action-conditioned latent dynamics plus CEM/MPC evaluation"
        )
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--latent-source",
        choices=("analytic", "visual"),
        default=None,
        help="exact xy control or the experimental RGB/frozen-video-JEPA path",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--output", type=Path, help="also write the exact stdout JSON here")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit non-zero unless every configured M6 acceptance check passes",
    )
    for name in (
        "seed",
        "train_steps",
        "batch_size",
        "horizon",
        "hidden_dim",
        "layers",
        "heads",
        "eval_seeds",
        "candidates",
        "elites",
        "refinements",
        "max_episode_steps",
        "visual_pretrain_steps",
        "visual_embed_dim",
        "image_size",
    ):
        _add_override(parser, name, value_type=int)
    for name in (
        "learning_rate",
        "action_limit",
        "success_distance",
        "min_success_rate",
        "min_random_margin",
    ):
        _add_override(parser, name, value_type=float)
    return parser


def _config_from_args(args: argparse.Namespace) -> DemoConfig:
    config = PROFILES[args.profile]
    overrides = {
        field: getattr(args, field)
        for field in asdict(config)
        if field != "profile" and getattr(args, field, None) is not None
    }
    if args.device is not None:
        overrides["requested_device"] = args.device
    return replace(config, **overrides)


def _write_json(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_experiment(_config_from_args(args))
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        _write_json(args.output, rendered)
    return int(args.enforce and not result["acceptance"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
