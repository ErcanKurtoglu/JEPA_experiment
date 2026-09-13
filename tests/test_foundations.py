import pytest

from jepa_lab.foundations import (
    ema_scalar,
    image_token_count,
    stop_gradient_demo,
    video_token_count,
    video_token_grid,
)
from jepa_lab.upstreams import UPSTREAMS


def test_paper_token_counts() -> None:
    assert image_token_count(224, 16) == 196
    assert video_token_grid(16, 224, 2, 16) == (8, 14, 14)
    assert video_token_count(frames=16, image_size=224, tubelet_size=2, patch_size=16) == 1568


def test_invalid_patch_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        image_token_count(225, 16)


def test_stop_gradient_demo() -> None:
    result = stop_gradient_demo()
    assert result.online_grad_norm > 0
    assert not result.target_has_grad
    assert result.loss > 0


def test_scalar_ema() -> None:
    assert ema_scalar(2.0, 1.0, 0.9) == pytest.approx(1.1)
    with pytest.raises(ValueError):
        ema_scalar(1.0, 1.0, 1.1)


def test_upstream_specs_are_full_sha_pins() -> None:
    assert {spec.name for spec in UPSTREAMS} == {"ijepa", "vjepa", "vjepa2"}
    assert all(len(spec.commit) == 40 for spec in UPSTREAMS)
