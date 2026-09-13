import torch

from jepa_lab.adapters import (
    canonical_to_vjepa,
    select_checkpoint_state,
    strip_state_dict_prefixes,
    temporal_variants,
    vjepa_to_canonical,
)


def test_video_layout_round_trip() -> None:
    canonical = torch.randn(2, 8, 3, 16, 16)
    upstream = canonical_to_vjepa(canonical)
    assert upstream.shape == (2, 3, 8, 16, 16)
    torch.testing.assert_close(vjepa_to_canonical(upstream), canonical)


def test_prefix_stripping_and_checkpoint_selection() -> None:
    tensor = torch.ones(2)
    cleaned = strip_state_dict_prefixes({"module.backbone.layer.weight": tensor})
    assert list(cleaned) == ["layer.weight"]
    selected = select_checkpoint_state({"target_encoder": {"module.x": tensor}})
    assert list(selected) == ["x"]
    grouped = select_checkpoint_state(
        {
            "model": {
                "encoder.weight": torch.zeros(2),
                "target_encoder.weight": tensor,
                "predictor.weight": torch.zeros(2),
            }
        }
    )
    assert list(grouped) == ["weight"]
    assert grouped["weight"] is tensor


def test_temporal_variants_are_well_defined() -> None:
    frames = torch.arange(1 * 4 * 1 * 1 * 1).reshape(1, 4, 1, 1, 1)
    variants = temporal_variants(frames)
    assert set(variants) == {"normal", "reversed", "shuffled", "static"}
    assert variants["reversed"].flatten().tolist() == [3, 2, 1, 0]
    assert variants["static"].flatten().tolist() == [0, 0, 0, 0]
