import numpy as np
import pytest
import torch

from jepa_lab.adapters import temporal_variants
from jepa_lab.stimuli import (
    IMAGE_VIEW_ORDER,
    VIDEO_VARIANT_ORDER,
    image_views,
    moving_square_video,
    temporal_frame_indices,
    video_clip_from_file,
)


def test_shared_image_views_are_deterministic() -> None:
    first = image_views(None, 32)
    second = image_views(None, 32)
    assert tuple(first) == IMAGE_VIEW_ORDER
    for name in IMAGE_VIEW_ORDER:
        torch.testing.assert_close(first[name], second[name])
        assert first[name].shape == (3, 32, 32)


def test_video_frame_manifest_matches_temporal_variants() -> None:
    frames = moving_square_video(8, 32)
    variants = temporal_variants(frames)
    indices = temporal_frame_indices(8)
    assert tuple(variants) == VIDEO_VARIANT_ORDER
    for row, name in zip(indices, VIDEO_VARIANT_ORDER):
        torch.testing.assert_close(variants[name], frames.index_select(1, torch.from_numpy(row)))
    assert indices.dtype == np.int64


def test_real_video_clip_is_uniformly_sampled(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (24, 16)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    for value in range(8):
        writer.write(np.full((16, 24, 3), value * 30, dtype=np.uint8))
    writer.release()

    clip, indices = video_clip_from_file(path, frames=4, image_size=16)
    assert clip.shape == (1, 4, 3, 16, 16)
    assert indices.tolist() == [0, 2, 5, 7]
    assert bool((clip[:, 1:] >= clip[:, :-1]).all())
