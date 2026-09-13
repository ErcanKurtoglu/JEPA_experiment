"""Pinned upstream metadata and read-only verification helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    url: str
    relative_path: str
    commit: str


UPSTREAMS: tuple[UpstreamSpec, ...] = (
    UpstreamSpec(
        name="ijepa",
        url="https://github.com/facebookresearch/ijepa.git",
        relative_path="upstream/ijepa",
        commit="52c1ae95d05f743e000e8f10a1f3a79b10cff048",
    ),
    UpstreamSpec(
        name="vjepa",
        url="https://github.com/facebookresearch/jepa.git",
        relative_path="upstream/vjepa",
        commit="51c59d518fc63c08464af6de585f78ac0c7ed4d5",
    ),
    UpstreamSpec(
        name="vjepa2",
        url="https://github.com/facebookresearch/vjepa2.git",
        relative_path="upstream/vjepa2",
        commit="9a061fffe395573a2ee61dadcd3cf5baa188e3f2",
    ),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _head(path: Path) -> str | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def verify_upstreams(root: Path | None = None) -> list[dict[str, object]]:
    """Inspect submodules without modifying them."""
    root = root or repository_root()
    result: list[dict[str, object]] = []
    for spec in UPSTREAMS:
        actual = _head(root / spec.relative_path)
        result.append(
            {
                **asdict(spec),
                "actual_commit": actual,
                "present": actual is not None,
                "matches_pin": actual == spec.commit,
            }
        )
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_record(
    *,
    name: str,
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Download atomically, then enforce the recorded digest on every reuse.

    If ``destination`` already exists, no network call is made.  Its digest must
    match ``expected_sha256`` when supplied; otherwise it must match the adjacent
    provenance sidecar created by the first successful download.  This prevents a
    silently replaced checkpoint from entering a later experiment.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    if destination.exists():
        digest = sha256_file(destination)
        recorded: dict[str, object] = {}
        if sidecar.exists():
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid checkpoint provenance sidecar: {sidecar}")
            recorded = loaded
        trusted_digest = expected_sha256 or recorded.get("sha256")
        if not isinstance(trusted_digest, str):
            raise ValueError(
                f"Existing checkpoint {destination} has no trusted SHA-256; "
                "supply expected_sha256 or remove it and download once"
            )
        if digest.lower() != trusted_digest.lower():
            raise ValueError(f"SHA-256 mismatch for existing {name}: {digest}")
        return {
            "name": name,
            "url": str(recorded.get("url", url)),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "reused": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = sha256_file(temporary)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {name}: {digest}")
    temporary.replace(destination)
    metadata = {
        "name": name,
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }
    metadata["reused"] = False
    sidecar.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def all_pins_match(entries: Iterable[dict[str, object]]) -> bool:
    return all(bool(entry["matches_pin"]) for entry in entries)
