from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..policies.hf_data import SIEVE_SYSTEM_PROMPT


def _file_stats(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    return count, digest.hexdigest()


def _scenario_count(path: Path) -> int:
    scenarios: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            scenario_id = str(json.loads(line).get("scenario_id", "")).strip()
            if not scenario_id:
                raise ValueError(f"missing scenario_id in {path} at line {line_number}")
            scenarios.add(scenario_id)
    return len(scenarios)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def publish_canonical_sft(
    accepted_path: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    prompt_version: str,
    quality_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically publish audited source plus train/dev views; external tests stay separate."""
    accepted = Path(accepted_path).resolve()
    splits = Path(split_dir).resolve()
    output = Path(output_dir).resolve()
    train = splits / "train.jsonl"
    dev = splits / "dev.jsonl"
    quality_report = (
        Path(quality_report_path).resolve() if quality_report_path is not None else None
    )
    required = (accepted, train, dev) + ((quality_report,) if quality_report else ())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing publish inputs: {missing}")

    source_target = output / "source" / "records.jsonl"
    train_target = output / "clean" / "train.jsonl"
    dev_target = output / "clean" / "dev.jsonl"
    for source, target in (
        (accepted, source_target),
        (train, train_target),
        (dev, dev_target),
    ):
        _copy_atomic(source, target)
    quality_target = output / "quality_report.json"
    if quality_report is not None:
        _copy_atomic(quality_report, quality_target)

    source_count, source_sha = _file_stats(source_target)
    train_count, train_sha = _file_stats(train_target)
    dev_count, dev_sha = _file_stats(dev_target)
    prompt_sha = hashlib.sha256(SIEVE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "source": {
            "path": "source/records.jsonl",
            "records": source_count,
            "sha256": source_sha,
        },
        "train": {
            "path": "clean/train.jsonl",
            "records": train_count,
            "scenario_groups": _scenario_count(train_target),
            "sha256": train_sha,
        },
        "validation": {
            "path": "clean/dev.jsonl",
            "records": dev_count,
            "scenario_groups": _scenario_count(dev_target),
            "sha256": dev_sha,
        },
        "system_prompt": {
            "version": prompt_version,
            "position": "first chat message with role=system; context is role=user",
            "sha256": prompt_sha,
            "text": SIEVE_SYSTEM_PROMPT,
        },
        "quality_report": (
            {
                "path": "quality_report.json",
                "sha256": _file_stats(quality_target)[1],
            }
            if quality_report is not None
            else None
        ),
        "training_format": (
            "single-step SFT; fixed system prompt + context JSON + target JSON + native EOS"
        ),
        "split_policy": "sha256(seed:group_id), source export 80/10/10",
        "test_policy": (
            "Internal test export is not published or consumed by Stage-1; "
            "final testing uses external official benchmarks."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest
