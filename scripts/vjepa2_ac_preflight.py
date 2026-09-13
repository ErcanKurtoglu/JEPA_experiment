#!/usr/bin/env python3
"""Check V-JEPA2-AC replay assets and select the safe hardware profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_lab.foundations import environment_report
from jepa_lab.upstreams import repository_root, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    root = repository_root()
    trajectory = root / "upstream/vjepa2/notebooks/franka_example_traj.npz"
    notebook = root / "upstream/vjepa2/notebooks/energy_landscape_example.ipynb"
    if not trajectory.exists() or not notebook.exists():
        raise SystemExit("Pinned V-JEPA2 notebook assets are missing")

    with np.load(trajectory, allow_pickle=False) as archive:
        trajectory_arrays = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }
    hardware = environment_report()
    vram = float(hardware.get("cuda_vram_gib", 0.0))
    mode = "gpu-replay" if vram >= 24.0 else "25-candidate/2-refinement reduced replay"
    checkpoint = None
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        checkpoint = {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        }
    print(
        json.dumps(
            {
                "source_commit": "9a061fffe395573a2ee61dadcd3cf5baa188e3f2",
                "hardware": hardware,
                "selected_mode": mode,
                "trajectory": str(trajectory),
                "trajectory_arrays": trajectory_arrays,
                "notebook": str(notebook),
                "checkpoint": checkpoint,
                "paper_scale_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
