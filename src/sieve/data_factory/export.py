from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


_SPLITS = ("train", "dev", "test")


def assign_split(group_id: str, *, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "dev"
    return "test"


def audit_to_sft_dict(raw: dict[str, Any], *, step_index: int) -> dict[str, Any]:
    if not raw.get("validation", {}).get("accepted", False):
        raise ValueError("only accepted audit records may be exported")
    record_id = str(raw.get("record_id", "")).strip()
    group_id = str(raw.get("group_id", "")).strip()
    observation_text = str(raw.get("observation_text", "")).strip()
    if not record_id or not group_id or not observation_text:
        raise ValueError("accepted audit record requires record_id, group_id, and observation_text")

    context = copy.deepcopy(raw["state"])
    context["observation"]["value"] = observation_text
    internal_fields = {"condition", "relevant", "perturbation"}
    for field_name in internal_fields:
        context["observation"].pop(field_name, None)
    for entry in context.get("ledger", {}).get("entries", []):
        ledger_observation = entry.get("observation", {})
        for field_name in internal_fields:
            ledger_observation.pop(field_name, None)

    raw_target = raw["target"]
    target = {
        "decision": raw_target["decision"],
        "affected_fields": copy.deepcopy(raw_target.get("affected_fields", [])),
        "patches": copy.deepcopy(raw_target.get("patches", [])),
        "verification": copy.deepcopy(raw_target.get("verification")),
    }
    return {
        "scenario_id": group_id,
        "step_index": int(step_index),
        "context": context,
        "target": target,
    }


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def export_training_splits(
    accepted_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, int]:
    source = Path(accepted_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in _SPLITS}
    group_steps: Counter[str] = Counter()
    seen_record_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            record_id = str(raw.get("record_id", ""))
            if record_id in seen_record_ids:
                raise ValueError(f"duplicate record_id at line {line_number}: {record_id}")
            seen_record_ids.add(record_id)
            group_id = str(raw.get("group_id", ""))
            step_index = group_steps[group_id]
            group_steps[group_id] += 1
            split = assign_split(group_id, seed=seed)
            rows[split].append(audit_to_sft_dict(raw, step_index=step_index))

    for split in _SPLITS:
        _write_jsonl_atomic(target / f"{split}.jsonl", rows[split])
    counts = {split: len(rows[split]) for split in _SPLITS}
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "split_policy": "sha256(seed:group_id), 80/10/10",
        "counts": counts,
        "group_counts": {
            split: len({item["scenario_id"] for item in rows[split]})
            for split in _SPLITS
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return counts
