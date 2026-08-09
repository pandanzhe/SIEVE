from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceArtifact:
    dataset: str
    version: str
    license: str
    url: str
    relative_path: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SourceArtifact":
        values = {}
        for field_name in (
            "dataset",
            "version",
            "license",
            "url",
            "relative_path",
            "sha256",
        ):
            value = raw.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source artifact requires {field_name}")
            values[field_name] = value.strip()
        if len(values["sha256"]) != 64:
            raise ValueError("source artifact sha256 must contain 64 hex characters")
        try:
            bytes.fromhex(values["sha256"])
        except ValueError as error:
            raise ValueError("source artifact sha256 is not hexadecimal") from error
        return cls(**values)


@dataclass(frozen=True)
class FetchResult:
    artifact: SourceArtifact
    path: Path
    sha256: str
    size_bytes: int
    reused: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_path(cache_dir: Path, relative_path: str) -> Path:
    root = cache_dir.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("source artifact path escapes the cache directory") from error
    return target


def fetch_artifact(
    artifact: SourceArtifact,
    cache_dir: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> FetchResult:
    cache_root = Path(cache_dir)
    target = _target_path(cache_root, artifact.relative_path)
    if target.is_file():
        existing_hash = _sha256(target)
        if existing_hash == artifact.sha256:
            return FetchResult(
                artifact, target, existing_hash, target.stat().st_size, True
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part-{os.getpid()}")
    try:
        with urllib.request.urlopen(artifact.url, timeout=timeout_seconds) as response:
            with partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        actual_hash = _sha256(partial)
        if actual_hash != artifact.sha256:
            raise ValueError(
                f"checksum mismatch for {artifact.dataset}: "
                f"expected {artifact.sha256}, got {actual_hash}"
            )
        partial.replace(target)
        return FetchResult(
            artifact, target, actual_hash, target.stat().st_size, False
        )
    finally:
        if partial.exists():
            partial.unlink()


def load_acquisition_manifest(path: str | Path) -> list[SourceArtifact]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("artifacts") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("acquisition manifest requires a non-empty artifacts list")
    return [SourceArtifact.from_dict(item) for item in entries]


def fetch_manifest(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> list[FetchResult]:
    return [
        fetch_artifact(artifact, cache_dir, timeout_seconds=timeout_seconds)
        for artifact in load_acquisition_manifest(manifest_path)
    ]
