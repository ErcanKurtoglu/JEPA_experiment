from __future__ import annotations

import json
import runpy
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "planar_world_model_demo.py"
_NAMESPACE = runpy.run_path(str(_SCRIPT))
DemoConfig = _NAMESPACE["DemoConfig"]
_run_random_episode = _NAMESPACE["_run_random_episode"]
main = _NAMESPACE["main"]


def _tiny_config():
    return DemoConfig(
        profile="test",
        requested_device="cpu",
        seed=7,
        train_steps=1,
        batch_size=2,
        learning_rate=1e-3,
        horizon=1,
        hidden_dim=8,
        layers=1,
        heads=2,
        eval_seeds=1,
        candidates=4,
        elites=1,
        refinements=1,
        max_episode_steps=1,
        action_limit=0.10,
        success_distance=0.08,
        min_success_rate=0.80,
        min_random_margin=0.30,
    )


def test_random_baseline_is_seed_deterministic() -> None:
    config = _tiny_config()
    first = _run_random_episode(config, episode_seed=123)
    second = _run_random_episode(config, episode_seed=123)
    assert first == second


def test_tiny_cli_writes_same_machine_readable_json(tmp_path: Path, capsys) -> None:
    output = tmp_path / "nested" / "result.json"
    exit_code = main(
        [
            "--profile",
            "smoke",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--train-steps",
            "1",
            "--batch-size",
            "2",
            "--horizon",
            "1",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--eval-seeds",
            "1",
            "--candidates",
            "4",
            "--elites",
            "1",
            "--refinements",
            "1",
            "--max-episode-steps",
            "1",
            "--output",
            str(output),
        ]
    )
    stdout = capsys.readouterr().out
    stdout_payload = json.loads(stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert stdout_payload == file_payload
    assert stdout_payload["scope"]["paper_scale"] is False
    assert stdout_payload["device"] == "cpu"
    assert stdout_payload["config"]["train_steps"] == 1
    assert stdout_payload["evaluation"]["held_out_seeds"] == [1_000_007]
    assert set(stdout_payload["acceptance"]["checks"]) == {
        "correct_action_better_than_zero",
        "correct_action_better_than_shuffled",
        "cem_success_rate",
        "cem_margin_over_random",
        "elite_energy_non_increasing",
    }


def test_tiny_visual_cli_uses_frozen_rgb_encoder(capsys) -> None:
    exit_code = main(
        [
            "--profile",
            "smoke",
            "--latent-source",
            "visual",
            "--device",
            "cpu",
            "--train-steps",
            "1",
            "--visual-pretrain-steps",
            "1",
            "--visual-embed-dim",
            "8",
            "--image-size",
            "32",
            "--batch-size",
            "2",
            "--horizon",
            "1",
            "--hidden-dim",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--eval-seeds",
            "1",
            "--candidates",
            "4",
            "--elites",
            "1",
            "--refinements",
            "1",
            "--max-episode-steps",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["scope"]["latent_source"].startswith("frozen_tiny_video_jepa")
    assert payload["scope"]["goal_information_used_by_cost"] == "goal image latent only"
    assert payload["acceptance"]["checks"]["visual_encoder_frozen"]
