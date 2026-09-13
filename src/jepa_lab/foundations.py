"""Foundational shape, gradient, EMA, and hardware checks used in M0."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .device import cuda_device_compatible


def image_token_grid(image_size: int = 224, patch_size: int = 16) -> tuple[int, int]:
    """Return the 2-D patch-token grid, rejecting partial patches."""
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
        raise ValueError("image_size must be positive and divisible by patch_size")
    side = image_size // patch_size
    return side, side


def image_token_count(image_size: int = 224, patch_size: int = 16) -> int:
    height, width = image_token_grid(image_size, patch_size)
    return height * width


def video_token_grid(
    frames: int = 16,
    image_size: int = 224,
    tubelet_size: int = 2,
    patch_size: int = 16,
) -> tuple[int, int, int]:
    """Return the temporal-height-width tubelet grid."""
    if frames <= 0 or tubelet_size <= 0 or frames % tubelet_size:
        raise ValueError("frames must be positive and divisible by tubelet_size")
    height, width = image_token_grid(image_size, patch_size)
    return frames // tubelet_size, height, width


def video_token_count(**kwargs: int) -> int:
    temporal, height, width = video_token_grid(**kwargs)
    return temporal * height * width


@dataclass(frozen=True)
class GradientDemo:
    online_grad_norm: float
    target_has_grad: bool
    loss: float


def stop_gradient_demo(seed: int = 42) -> GradientDemo:
    """Demonstrate that a detached target receives no optimizer gradient."""
    torch.manual_seed(seed)
    online = torch.nn.Linear(4, 4, bias=False)
    target = torch.nn.Linear(4, 4, bias=False)
    target.load_state_dict(online.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    inputs = torch.randn(3, 4)
    prediction = online(inputs)
    with torch.no_grad():
        reference = target(inputs)
    loss = torch.nn.functional.smooth_l1_loss(prediction, reference + 0.25)
    loss.backward()
    grad_norm = float(
        torch.sqrt(sum(parameter.grad.square().sum() for parameter in online.parameters()))
    )
    return GradientDemo(
        online_grad_norm=grad_norm,
        target_has_grad=any(parameter.grad is not None for parameter in target.parameters()),
        loss=float(loss.detach()),
    )


def ema_scalar(online: float, target: float, momentum: float) -> float:
    """One scalar EMA update used to check the model-level implementation."""
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    return momentum * target + (1.0 - momentum) * online


def environment_report() -> dict[str, Any]:
    """Return a JSON-serializable hardware and PyTorch report."""
    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    cuda_compatible = cuda_device_compatible(0) if cuda_available else False
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_arch_compatible": cuda_compatible,
        "mps_available": mps_available,
        "recommended_device": "cuda" if cuda_compatible else "mps" if mps_available else "cpu",
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        report["cuda_device"] = properties.name
        report["cuda_compute_capability"] = list(torch.cuda.get_device_capability(0))
        report["torch_cuda_arch_list"] = torch.cuda.get_arch_list()
        report["cuda_vram_gib"] = round(properties.total_memory / 1024**3, 2)
        report["vjepa2_ac_profile"] = (
            "gpu-replay" if properties.total_memory >= 24 * 1024**3 else "reduced-or-cpu-replay"
        )
    return report


def foundation_report() -> dict[str, Any]:
    gradient = stop_gradient_demo()
    return {
        "image_tokens_224_p16": image_token_count(),
        "video_grid_16x224_t2_p16": video_token_grid(),
        "video_tokens_16x224_t2_p16": video_token_count(),
        "gradient_demo": asdict(gradient),
        "ema_example": {
            "online": 2.0,
            "target_before": 1.0,
            "momentum": 0.9,
            "target_after": ema_scalar(2.0, 1.0, 0.9),
        },
        "environment": environment_report(),
    }


def foundation_report_json() -> str:
    return json.dumps(foundation_report(), indent=2, ensure_ascii=False)
