#!/usr/bin/env python3
"""Safely preflight or opt in to Meta's official V-JEPA2-AC replay.

The default path never imports or allocates the ~1.3B-parameter model.  Actual
inference requires both ``--run-replay`` and a locally verified checkpoint.
Run this script in its own process because Meta JEPA repositories share a
top-level ``src`` package name.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jepa_lab.device import seed_everything, select_device
from jepa_lab.foundations import environment_report
from jepa_lab.upstreams import verify_upstreams
from jepa_lab.vjepa2_ac_replay import (
    PINNED_VJEPA2_COMMIT,
    cartesian_action_grid,
    checkpoint_provenance,
    load_trajectory,
    replay_dependency_report,
    system_memory_gib,
    write_json_result,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "vjepa2"
DEFAULT_TRAJECTORY = UPSTREAM / "notebooks" / "franka_example_traj.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight by default; load the official ViT-g action world model only with "
            "--run-replay and a verified local checkpoint"
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        dest="run_replay",
        action="store_false",
        help="inspect pins, assets, dependencies and optional checkpoint without loading the model",
    )
    mode.add_argument(
        "--run-replay",
        dest="run_replay",
        action="store_true",
        help="explicitly opt in to heavyweight official encoder/predictor inference",
    )
    parser.set_defaults(run_replay=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--sha256",
        help="trusted checkpoint SHA-256; alternatively use an adjacent .json provenance sidecar",
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-samples", type=int, default=5)
    parser.add_argument("--grid-bound", type=float, default=0.075)
    parser.add_argument("--energy-chunk-size", type=int, default=1)
    parser.add_argument("--cem-horizon", type=int, default=2)
    parser.add_argument("--cem-candidates", type=int, default=25)
    parser.add_argument("--cem-elites", type=int, default=10)
    parser.add_argument("--cem-refinements", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.energy_chunk_size < 1:
        parser.error("--energy-chunk-size must be positive")
    if args.cem_horizon < 1 or args.cem_candidates < 2 or args.cem_refinements < 1:
        parser.error("CEM horizon/refinements must be positive and candidates must be at least 2")
    if not 1 <= args.cem_elites < args.cem_candidates:
        parser.error("--cem-elites must be in [1, cem-candidates)")
    try:
        cartesian_action_grid(args.grid_samples, args.grid_bound)
    except ValueError as error:
        parser.error(str(error))
    if args.run_replay and args.checkpoint is None:
        parser.error("--run-replay requires --checkpoint")


def _vjepa2_pin() -> dict[str, object]:
    matches = [entry for entry in verify_upstreams(ROOT) if entry["name"] == "vjepa2"]
    if len(matches) != 1:
        raise RuntimeError("could not resolve the pinned V-JEPA2 submodule")
    return matches[0]


def _base_payload(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    observations, states, trajectory = load_trajectory(args.trajectory)
    pin = _vjepa2_pin()
    dependencies = replay_dependency_report()
    checkpoint: dict[str, Any] = {
        "provided": args.checkpoint is not None,
        "required_for_replay": True,
    }
    if args.checkpoint is not None:
        checkpoint.update(checkpoint_provenance(args.checkpoint, expected_sha256=args.sha256))
    hardware = environment_report()
    hardware["system_memory_gib"] = system_memory_gib()
    inputs_ready = bool(
        pin["matches_pin"] and checkpoint["provided"] and all(dependencies.values())
    )
    cuda_vram_gib = float(hardware.get("cuda_vram_gib", 0.0))
    system_ram_gib = hardware["system_memory_gib"]
    recommended_resource_available = bool(
        cuda_vram_gib >= 24.0
        or (isinstance(system_ram_gib, (int, float)) and system_ram_gib >= 24.0)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "replay" if args.run_replay else "preflight-only",
        "paper_scale_claim": False,
        "source": {
            "repository": str(UPSTREAM),
            "expected_commit": PINNED_VJEPA2_COMMIT,
            "actual_commit": pin["actual_commit"],
            "matches_pin": pin["matches_pin"],
            "remote_code_used": False,
        },
        "hardware": hardware,
        "optional_dependencies": dependencies,
        "checkpoint": checkpoint,
        "trajectory": trajectory,
        "requested": {
            "device": args.device,
            "seed": args.seed,
            "energy_grid": {
                "samples_per_axis": args.grid_samples,
                "candidate_count": args.grid_samples**3,
                "xyz_bound": args.grid_bound,
                "chunk_size": args.energy_chunk_size,
            },
            "cem": {
                "horizon": args.cem_horizon,
                "candidates": args.cem_candidates,
                "elites": args.cem_elites,
                "refinements": args.cem_refinements,
                "xyz_bound": args.grid_bound,
            },
        },
        "inputs_ready": inputs_ready,
        "recommended_resource_available": recommended_resource_available,
        "safety": {
            "default_allocates_model": False,
            "estimated_parameters": {
                "encoder": "~1.01B",
                "predictor": "~0.31B",
                "combined": "~1.32B",
            },
            "note": (
                "The official float32 load can exceed 10 GiB at peak while model and checkpoint "
                "state coexist; preflight success is not a memory guarantee."
            ),
            "recommended_resource_policy": (
                "at least 24 GiB CUDA VRAM, or at least 24 GiB system RAM for the very slow "
                "reduced CPU replay"
            ),
        },
    }
    return payload, observations, states


def _clean_state_dict(state: object, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {label} must be a state-dict mapping")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint {label} must map string keys to tensors")
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def _load_official_models(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    # Import only after --run-replay has passed every cheap preflight guard.
    sys.path.insert(0, str(UPSTREAM))
    from src.hub.backbones import vjepa2_ac_vit_giant

    encoder, predictor = vjepa2_ac_vit_giant(pretrained=False)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("official checkpoint root must be a mapping")
    encoder_state = _clean_state_dict(checkpoint.get("encoder"), label="encoder")
    predictor_state = _clean_state_dict(checkpoint.get("predictor"), label="predictor")
    encoder_incompatible = encoder.load_state_dict(encoder_state, strict=False)
    predictor.load_state_dict(predictor_state, strict=True)
    load_report = {
        "encoder_missing": list(encoder_incompatible.missing_keys),
        "encoder_unexpected": list(encoder_incompatible.unexpected_keys),
        "predictor_strict": True,
    }
    del checkpoint, encoder_state, predictor_state
    return encoder, predictor, load_report


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_transition(
    encoder: torch.nn.Module,
    transform: Any,
    observations: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    # Match Meta's notebook exactly: encode each RGB frame as a duplicated
    # two-frame tubelet, then split the flattened tokens back by frame.
    clip = transform(observations[0, :2]).unsqueeze(0).to(device)
    batch, _, frames, _, _ = clip.shape
    encoder_input = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    encoded = encoder(encoder_input)
    encoded = encoded.view(batch, frames, -1, encoded.size(-1))
    encoded = F.layer_norm(encoded, (encoded.size(-1),))
    return encoded[:, 0], encoded[:, 1], tuple(encoder_input.shape)


def _action_energies(
    predictor: torch.nn.Module,
    current: torch.Tensor,
    goal: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    energies: list[torch.Tensor] = []
    for chunk in actions.split(chunk_size):
        count = chunk.size(0)
        predicted = predictor(
            current.repeat(count, 1, 1),
            chunk.unsqueeze(1),
            state.repeat(count, 1, 1),
        )
        predicted = F.layer_norm(predicted[:, -current.size(1) :], (predicted.size(-1),))
        reference = goal.repeat(count, 1, 1)
        energies.append(torch.mean(torch.abs(predicted - reference), dim=(1, 2)))
    return torch.cat(energies)


def _run_replay(
    args: argparse.Namespace,
    payload: dict[str, Any],
    observations: np.ndarray,
    states: np.ndarray,
) -> dict[str, Any]:
    if not payload["source"]["matches_pin"]:
        raise RuntimeError("V-JEPA2 submodule does not match the required pinned commit")
    missing_dependencies = [
        name for name, present in payload["optional_dependencies"].items() if not present
    ]
    if missing_dependencies:
        raise RuntimeError(
            "official replay dependencies are missing: " + ", ".join(missing_dependencies)
        )
    if not payload["checkpoint"]["provided"]:
        raise RuntimeError("a verified checkpoint is required")

    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type == "mps":
        raise RuntimeError(
            "the pinned official AC replay is not certified on MPS; rerun with --device cpu"
        )
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    encoder, predictor, load_report = _load_official_models(args.checkpoint)
    encoder.eval().to(device)
    predictor.eval().to(device)
    _synchronize(device)
    load_seconds = time.perf_counter() - load_start

    from app.vjepa_droid.transforms import make_transforms
    from notebooks.utils import mpc_utils
    from notebooks.utils.mpc_utils import poses_to_diff
    from notebooks.utils.world_model_wrapper import WorldModel

    # The upstream helper logs distribution updates unconditionally.  Keep
    # stdout valid JSON and preserve those values in the explicit config below.
    mpc_utils.logger.setLevel(logging.ERROR)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=256,
    )
    tokens_per_frame = (256 // int(encoder.patch_size)) ** 2
    state = torch.as_tensor(states[0, :1], device=device, dtype=torch.float32).unsqueeze(0)
    ground_truth_action = poses_to_diff(states[0, 0], states[0, 1]).to(
        device=device, dtype=torch.float32
    )

    encode_start = time.perf_counter()
    with torch.inference_mode():
        current, goal, encoder_input_shape = _encode_transition(
            encoder, transform, observations, device
        )
    _synchronize(device)
    encode_seconds = time.perf_counter() - encode_start

    comparison_actions = torch.stack(
        (ground_truth_action, torch.zeros_like(ground_truth_action)), dim=0
    )
    comparison_start = time.perf_counter()
    with torch.inference_mode():
        comparison_energy = _action_energies(
            predictor,
            current,
            goal,
            state,
            comparison_actions,
            chunk_size=args.energy_chunk_size,
        )
    _synchronize(device)
    comparison_seconds = time.perf_counter() - comparison_start

    grid_actions_np = cartesian_action_grid(args.grid_samples, args.grid_bound)
    grid_actions = torch.as_tensor(grid_actions_np, device=device)
    grid_start = time.perf_counter()
    with torch.inference_mode():
        grid_energy = _action_energies(
            predictor,
            current,
            goal,
            state,
            grid_actions,
            chunk_size=args.energy_chunk_size,
        )
    _synchronize(device)
    grid_seconds = time.perf_counter() - grid_start
    best_index = int(grid_energy.argmin().item())

    mpc_args = {
        "rollout": args.cem_horizon,
        "samples": args.cem_candidates,
        "topk": args.cem_elites,
        "cem_steps": args.cem_refinements,
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": args.grid_bound,
        "verbose": False,
    }
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args=mpc_args,
        normalize_reps=True,
        device=str(device),
    )
    cem_start = time.perf_counter()
    with torch.inference_mode():
        planned_action = world_model.infer_next_action(current, state, goal)
    _synchronize(device)
    cem_seconds = time.perf_counter() - cem_start

    gt_translation = ground_truth_action[:3].detach().cpu().numpy()
    ground_truth_in_grid = bool(np.all(np.abs(gt_translation) <= args.grid_bound))
    payload.update(
        {
            "device": str(device),
            "dtype": "float32",
            "model_load": load_report,
            "latents": {
                "encoder_input_shape": list(encoder_input_shape),
                "current_shape": list(current.shape),
                "goal_shape": list(goal.shape),
                "tokens_per_frame": tokens_per_frame,
            },
            "action_energy": {
                "objective": "mean absolute error between LayerNorm predicted and goal latents",
                "ground_truth_action": ground_truth_action.detach().cpu().tolist(),
                "ground_truth": float(comparison_energy[0].item()),
                "zero": float(comparison_energy[1].item()),
                "shuffled": payload["trajectory"]["shuffled_action"],
            },
            "rollout_error_by_horizon": {
                "1": {
                    "status": "computed",
                    "action_source": "ground_truth pose delta",
                    "latent_l1": float(comparison_energy[0].item()),
                },
                "2": {
                    "status": "unsupported",
                    "reason": (
                        "the bundled trajectory has two frames and therefore only one observed "
                        "transition; CEM horizon=2 is a model rollout, not a measured 2-step error"
                    ),
                },
            },
            "energy_grid": {
                **payload["requested"]["energy_grid"],
                "ground_truth_inside_xyz_bounds": ground_truth_in_grid,
                "best_action": grid_actions_np[best_index].tolist(),
                "minimum_energy": float(grid_energy[best_index].item()),
                "actions": grid_actions_np.tolist(),
                "energies": grid_energy.detach().cpu().tolist(),
            },
            "cem": {
                **payload["requested"]["cem"],
                "implementation": "pinned upstream notebooks.utils.mpc_utils.cem",
                "first_action": planned_action[0].detach().cpu().tolist(),
                "applied_to_robot_or_environment": False,
            },
            "latency_seconds": {
                "model_and_checkpoint_load": load_seconds,
                "current_goal_encoding": encode_seconds,
                "ground_truth_and_zero_energy": comparison_seconds,
                "energy_grid": grid_seconds,
                "reduced_cem": cem_seconds,
                "total": time.perf_counter() - total_start,
            },
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    payload, observations, states = _base_payload(args)
    if args.run_replay:
        payload = _run_replay(args, payload, observations, states)
    if args.output is not None:
        payload["output"] = str(args.output.resolve())
        write_json_result(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
