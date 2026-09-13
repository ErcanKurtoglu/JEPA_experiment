import numpy as np
import torch

from jepa_lab.adapters import temporal_variants
from jepa_lab.stimuli import (
    IMAGE_VIEW_ORDER,
    VIDEO_VARIANT_ORDER,
    image_views,
    moving_square_video,
    temporal_frame_indices,
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
