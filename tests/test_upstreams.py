import hashlib
import io
import json
from pathlib import Path
from typing import Self

import pytest

from jepa_lab import upstreams


class _Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_checkpoint_download_records_and_reuses_verified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"small-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    calls = 0

    def fake_urlopen(url: str) -> _Response:
        nonlocal calls
        calls += 1
        assert url == "https://example.test/model.pt"
        return _Response(payload)

    monkeypatch.setattr(upstreams.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "model.pt"
    first = upstreams.download_and_record(
        name="tiny", url="https://example.test/model.pt", destination=destination
    )
    second = upstreams.download_and_record(
        name="tiny", url="https://example.test/model.pt", destination=destination
    )

    assert calls == 1
    assert first["sha256"] == digest and first["reused"] is False
    assert second["sha256"] == digest and second["reused"] is True
    sidecar = json.loads((tmp_path / "model.pt.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == digest


def test_checkpoint_reuse_rejects_changed_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"changed")
    (tmp_path / "model.pt.json").write_text(
        json.dumps({"sha256": hashlib.sha256(b"original").hexdigest()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        upstreams.download_and_record(
            name="tiny", url="https://example.test/model.pt", destination=destination
        )


def test_unrecorded_existing_checkpoint_requires_explicit_digest(tmp_path: Path) -> None:
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="no trusted SHA-256"):
        upstreams.download_and_record(
            name="tiny", url="https://example.test/model.pt", destination=destination
        )
