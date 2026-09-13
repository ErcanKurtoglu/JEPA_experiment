from pathlib import Path

import numpy as np

from jepa_lab.runlog import build_run_summary, export_features, import_features


def test_run_summary_has_reproducibility_fields() -> None:
    summary = build_run_summary(
        {"experiment": "smoke", "seed": 7, "value": 1}, {"loss": 0.25}
    )
    assert summary.experiment == "smoke"
    assert summary.seed == 7
    assert len(summary.config_sha256) == 64
    assert summary.metrics == {"loss": 0.25}


def test_feature_round_trip(tmp_path: Path) -> None:
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    labels = np.array([1, 2])
    path, sidecar = export_features(
        tmp_path / "features.npz",
        features,
        labels=labels,
        sample_ids=np.array(["first", "second"]),
        frame_indices=np.array([[0, 2], [1, 3]]),
        metadata={"backend": "test"},
    )
    loaded_features, loaded_labels, metadata = import_features(path)
    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_labels, labels)
    assert sidecar.exists()
    assert metadata["metadata"]["backend"] == "test"
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sample_ids"], ["first", "second"])
        np.testing.assert_array_equal(archive["frame_indices"], [[0, 2], [1, 3]])


def test_feature_export_rejects_label_length_mismatch(tmp_path: Path) -> None:
    try:
        export_features(
            tmp_path / "bad.npz", np.zeros((2, 3), dtype=np.float32), labels=np.zeros(3)
        )
    except ValueError as error:
        assert "same first dimension" in str(error)
    else:
        raise AssertionError("Expected a label/feature length error")
