"""Deterministic stimuli shared by isolated official-model processes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

IMAGE_VIEW_ORDER = ("normal", "crop", "color", "occluded")
VIDEO_VARIANT_ORDER = ("normal", "reversed", "shuffled", "static")


def synthetic_image_scene(size: int = 320) -> Image.Image:
    """Create the single versioned RGB scene used by DINO and I-JEPA."""

    if size <= 0:
        raise ValueError("size must be positive")
    image = Image.new("RGB", (size, size), (42, 58, 83))
    draw = ImageDraw.Draw(image)
    draw.rectangle((size // 8, size // 5, size // 2, 4 * size // 5), fill=(229, 91, 73))
    draw.ellipse((size // 2, size // 6, 7 * size // 8, 3 * size // 5), fill=(74, 190, 149))
    draw.polygon(
        (
            (size // 2, 4 * size // 5),
            (7 * size // 8, 4 * size // 5),
            (3 * size // 4, size // 2),
        ),
        fill=(246, 198, 79),
    )
    return image


def image_views(path: str | Path | None, image_size: int = 224) -> dict[str, torch.Tensor]:
    """Return identical unnormalised image views for official model comparisons."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if path is None:
        image = synthetic_image_scene(max(320, image_size))
    else:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")

    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    base = image.crop((left, top, left + side, top + side)).resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    margin = max(1, image_size // 8)
    crop = base.crop((margin, margin, image_size - margin, image_size - margin)).resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    color = TF.adjust_saturation(TF.adjust_brightness(base, 0.65), 1.55)
    occluded = base.copy()
    draw = ImageDraw.Draw(occluded)
    radius = image_size // 6
    centre = image_size // 2
    draw.rectangle(
        (centre - radius, centre - radius, centre + radius, centre + radius),
        fill=(0, 0, 0),
    )
    pil_views = {"normal": base, "crop": crop, "color": color, "occluded": occluded}
    return {
        name: TF.pil_to_tensor(pil_views[name]).float().div_(255.0)
        for name in IMAGE_VIEW_ORDER
    }


def moving_square_video(frames: int, image_size: int) -> torch.Tensor:
    """Render a versioned normalised left-to-right path as ``[1,T,3,H,W]``."""

    if frames < 2 or image_size < 16:
        raise ValueError("frames must be at least two and image_size at least 16")
    video = torch.zeros(1, frames, 3, image_size, image_size)
    side = max(2, image_size // 14)
    top = image_size // 2 - side // 2
    start = 1
    travel = max(1, image_size - side - 2)
    for time in range(frames):
        left = start + round(time * travel / (frames - 1))
        video[:, time, 0, top : top + side, left : left + side] = 1.0
    return video


def video_clip_from_file(
    path: str | Path,
    *,
    frames: int = 16,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    """Uniformly sample and centre-crop a real video as ``[1,T,3,H,W]``.

    OpenCV is an optional dependency provided by the project's ``video`` extra.
    Returning the original frame indices keeps the exported feature archive
    traceable to the source clip.
    """

    if frames < 2 or image_size < 16:
        raise ValueError("frames must be at least two and image_size at least 16")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("video loading requires: pip install -e '.[video]'") from error

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < frames:
            raise ValueError(
                f"video contains {frame_count} frames; at least {frames} are required"
            )
        indices = np.rint(np.linspace(0, frame_count - 1, frames)).astype(np.int64)
        sampled: list[torch.Tensor] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"could not read frame {int(index)} from {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            side = min(height, width)
            top = (height - side) // 2
            left = (width - side) // 2
            image = Image.fromarray(rgb[top : top + side, left : left + side])
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            sampled.append(TF.pil_to_tensor(image).float().div_(255.0))
    finally:
        capture.release()
    return torch.stack(sampled).unsqueeze(0), indices


def temporal_frame_indices(frames: int) -> np.ndarray:
    """Frame-index manifest matching :func:`jepa_lab.adapters.temporal_variants`."""

    if frames < 2:
        raise ValueError("frames must be at least two")
    timeline = torch.arange(frames, dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(42)
    permutation = torch.randperm(frames, generator=generator)
    return torch.stack(
        (timeline, timeline.flip(0), permutation, torch.zeros_like(timeline))
    ).numpy()


__all__ = [
    "IMAGE_VIEW_ORDER",
    "VIDEO_VARIANT_ORDER",
    "image_views",
    "moving_square_video",
    "synthetic_image_scene",
    "temporal_frame_indices",
    "video_clip_from_file",
]
