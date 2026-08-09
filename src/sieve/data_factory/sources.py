from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import GroundedSourceRecord, Provenance, ScenarioType


_REQUIRED_RECORD_FIELDS = (
    "id",
    "domain",
    "scenario_type",
    "entity",
    "field_id",
    "old_value",
    "new_value",
    "source",
    "observed_at",
    "goal",
    "source_text",
)


def _canonical_json(raw: dict[str, Any]) -> str:
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_records(path: Path, data_format: str) -> list[dict[str, Any]]:
    if data_format == "jsonl":
        result = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                result.append(raw)
        return result
    if data_format == "json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("records")
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise ValueError(f"{path} must contain a JSON array or an object with records")
        return raw
    raise ValueError(f"unsupported source format: {data_format}")


def _normalize_record(
    raw: dict[str, Any],
    *,
    dataset: str,
    version: str,
    license_name: str,
    source_path: Path,
) -> GroundedSourceRecord:
    missing = [name for name in _REQUIRED_RECORD_FIELDS if name not in raw]
    if missing:
        raise ValueError(
            f"source record from {dataset} is missing fields: {','.join(missing)}"
        )
    record_id = str(raw["id"])
    digest = hashlib.sha256(_canonical_json(raw).encode("utf-8")).hexdigest()
    known = set(_REQUIRED_RECORD_FIELDS) | {"valid_from", "aliases", "metadata"}
    metadata = dict(raw.get("metadata") or {})
    metadata.update({key: value for key, value in raw.items() if key not in known})
    return GroundedSourceRecord(
        provenance=Provenance(
            source_dataset=dataset,
            source_version=version,
            source_record_id=record_id,
            source_uri=f"{source_path.as_posix()}#{record_id}",
            source_sha256=digest,
            license=license_name,
        ),
        domain=str(raw["domain"]),
        scenario_type=ScenarioType(str(raw["scenario_type"])),
        entity=str(raw["entity"]),
        field_id=str(raw["field_id"]),
        old_value=raw["old_value"],
        new_value=raw["new_value"],
        source=str(raw["source"]),
        observed_at=int(raw["observed_at"]),
        valid_from=(
            int(raw["valid_from"]) if raw.get("valid_from") is not None else None
        ),
        goal=str(raw["goal"]),
        source_text=str(raw["source_text"]),
        aliases=tuple(str(item) for item in raw.get("aliases", ())),
        metadata=metadata,
    )


def load_source_manifest(path: str | Path) -> list[GroundedSourceRecord]:
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("sources") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest requires a non-empty sources list")

    records: list[GroundedSourceRecord] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"source manifest entry {index} must be an object")
        for field_name in ("dataset", "version", "license", "path", "format"):
            if not entry.get(field_name):
                raise ValueError(
                    f"source manifest entry {index} requires {field_name}"
                )
        source_path = (manifest_path.parent / str(entry["path"])).resolve()
        if not source_path.is_file():
            raise ValueError(f"source file does not exist: {source_path}")
        for source_record in _read_records(source_path, str(entry["format"])):
            records.append(
                _normalize_record(
                    source_record,
                    dataset=str(entry["dataset"]),
                    version=str(entry["version"]),
                    license_name=str(entry["license"]),
                    source_path=source_path,
                )
            )
    if not records:
        raise ValueError("source manifest produced no records")
    return records
