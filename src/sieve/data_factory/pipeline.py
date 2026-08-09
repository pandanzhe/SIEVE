from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .glm import GlmClient
from .models import AuditRecord, QuotaCell
from .quality import validate_candidate
from .quotas import full_quota_cells, preview_quota_cells
from .rules import build_candidate
from .sources import load_source_manifest
from .trajectories import build_corpus_candidates


@dataclass(frozen=True)
class PipelineConfig:
    source_manifest: str | Path
    output_dir: str | Path
    total: int = 140
    batch_size: int = 5
    seed: int = 42
    max_attempts_per_record: int = 3
    max_requests: int = 200
    resume: bool = True
    confirm_full: int | None = None
    prompt_version: str = "v1"
    rule_version: str = "v1"
    trajectory_mode: bool = False


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    accepted: int
    rejected: int
    requests: int
    resumed: int


def _cell_key(cell: QuotaCell) -> tuple[str, str, str]:
    return (
        cell.observation_type.value,
        cell.decision.value,
        cell.verification_action.value,
    )


def _quota_units(total: int) -> list[QuotaCell]:
    if total <= 0:
        raise ValueError("total must be positive")
    cells = full_quota_cells() if total == 6000 else preview_quota_cells()
    units = [
        QuotaCell(cell.observation_type, cell.decision, 1, cell.verification_action)
        for cell in cells
        for _ in range(cell.count)
    ]
    if total <= len(units):
        return units[:total]
    raise ValueError("only preview totals up to 140 or the guarded 6000 total are supported")


def _source_targets(total: int) -> dict[str, int]:
    if total == 140:
        return {"tau2-bench": 70, "ToolBench": 35, "AgentBench": 21, "WebArena": 14}
    if total == 6000:
        return {"tau2-bench": 3000, "ToolBench": 1500, "AgentBench": 900, "WebArena": 600}
    tau = total // 2
    tool = total // 4
    agent = (total * 15) // 100
    web = total - tau - tool - agent
    return {"tau2-bench": tau, "ToolBench": tool, "AgentBench": agent, "WebArena": web}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _accepted_cell_counts(records: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            item["taxonomy"]["observation_type"],
            item["target"]["decision"],
            item["target"].get("verification_action", "NO_VERIFY"),
        )
        for item in records
    )


def _remaining_units(
    target_units: list[QuotaCell], existing: list[dict[str, Any]]
) -> list[QuotaCell]:
    remaining = Counter(_cell_key(cell) for cell in target_units)
    remaining.subtract(_accepted_cell_counts(existing))
    if any(value < 0 for value in remaining.values()):
        raise ValueError("existing accepted records exceed configured quota")
    result: list[QuotaCell] = []
    exemplar = {_cell_key(cell): cell for cell in target_units}
    for key, count in remaining.items():
        result.extend([exemplar[key]] * count)
    return result


def _remaining_sources(total: int, existing: list[dict[str, Any]]) -> list[str]:
    targets = Counter(_source_targets(total))
    targets.subtract(item["provenance"]["source_dataset"] for item in existing)
    if any(value < 0 for value in targets.values()):
        raise ValueError("existing accepted source counts exceed configured quota")
    return [dataset for dataset, count in targets.items() for _ in range(count)]


def _generator_name(client: object) -> str:
    inner = getattr(client, "inner", client)
    return str(getattr(inner, "model", type(inner).__name__))


def _report(records: list[dict[str, Any]], rejected: int, requests: int) -> dict[str, Any]:
    return {
        "accepted": len(records),
        "rejected": rejected,
        "requests": requests,
        "source_counts": dict(Counter(item["provenance"]["source_dataset"] for item in records)),
        "observation_counts": dict(Counter(item["taxonomy"]["observation_type"] for item in records)),
        "decision_counts": dict(Counter(item["target"]["decision"] for item in records)),
        "verification_counts": dict(Counter(item["target"].get("verification_action", "NO_VERIFY") for item in records)),
        "total_tokens": sum(
            int(item.get("generation", {}).get("usage", {}).get("total_tokens", 0))
            for item in records
        ),
    }


def _run_trajectory_pipeline(config: PipelineConfig, client: GlmClient) -> PipelineResult:
    output_dir = Path(config.output_dir).resolve()
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rejected_path.touch(exist_ok=True)
    if not config.resume and accepted_path.exists():
        raise ValueError("output already exists and resume is disabled")

    source_records = load_source_manifest(config.source_manifest)
    candidates = build_corpus_candidates(
        source_records,
        total=config.total,
        seed=config.seed,
    )
    existing = _read_jsonl(accepted_path)
    if len(existing) > len(candidates):
        raise ValueError("existing accepted records exceed configured total")
    expected_ids = [candidate.record_id for candidate in candidates[:len(existing)]]
    actual_ids = [str(item.get("record_id", "")) for item in existing]
    if actual_ids != expected_ids:
        raise ValueError("existing accepted records do not match the deterministic trajectory schedule")

    seen_texts = {
        " ".join(item["observation_text"].casefold().split())
        for item in existing
    }
    rejected_count = len(_read_jsonl(rejected_path))
    request_count = 0
    seen_raw_hashes: set[str] = set()
    remaining = deque(candidates[len(existing):])
    attempts: Counter[str] = Counter()

    while remaining:
        if request_count >= config.max_requests:
            raise RuntimeError("maximum request count reached before trajectory corpus was filled")
        batch = [remaining.popleft() for _ in range(min(config.batch_size, len(remaining)))]
        realizations = client.realize(batch)
        request_count += 1
        if len(realizations) != len(batch):
            raise RuntimeError("GLM client returned a different number of realizations")
        by_id = {item.record_id: item for item in realizations}
        for candidate in batch:
            realization = by_id.get(candidate.record_id)
            if realization is None:
                raise RuntimeError("GLM client omitted a requested record")
            if realization.raw_response and realization.request_hash not in seen_raw_hashes:
                with (raw_dir / "glm_responses.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "request_hash": realization.request_hash,
                        "raw_response": realization.raw_response,
                        "usage": realization.usage,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                seen_raw_hashes.add(realization.request_hash)

            validation = validate_candidate(candidate, realization, seen_texts=seen_texts)
            audit = AuditRecord(
                candidate,
                realization,
                validation,
                config.prompt_version,
                config.rule_version,
                _generator_name(client),
            )
            destination = accepted_path if validation.accepted else rejected_path
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            if validation.accepted:
                seen_texts.add(" ".join(realization.observation_text.casefold().split()))
                continue
            rejected_count += 1
            attempts[candidate.record_id] += 1
            if attempts[candidate.record_id] >= config.max_attempts_per_record:
                raise RuntimeError(
                    f"quality attempts exhausted for trajectory record {candidate.record_id}"
                )
            remaining.appendleft(candidate)

    final_records = _read_jsonl(accepted_path)
    manifest_path = Path(config.source_manifest)
    manifest = {
        "total": config.total,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "trajectory_mode": True,
        "domain_policy": {"commerce": 0.5, "service": 0.3, "workflow": 0.2},
        "decision_policy": {"UPDATE": 0.7, "HOLD": 0.15, "IGNORE": 0.15},
        "hold_successor": "authoritative verification UPDATE in the same group",
        "prompt_version": config.prompt_version,
        "rule_version": config.rule_version,
        "generator": _generator_name(client),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "full_generation_guard": config.confirm_full,
    }
    report = _report(final_records, rejected_count, request_count)
    report["macro_domain_counts"] = dict(Counter(
        str(item["group_id"]).split(":", 1)[0] for item in final_records
    ))
    report["paired_hold_groups"] = sum(
        1 for count in Counter(item["group_id"] for item in final_records).values()
        if count == 2
    )
    _write_json_atomic(output_dir / "manifest.json", manifest)
    _write_json_atomic(output_dir / "quality_report.json", report)
    (output_dir / "quality_report.md").write_text(
        "# SIEVE trajectory data quality report\n\n"
        f"- Accepted: {report['accepted']}\n"
        f"- Rejected: {report['rejected']}\n"
        f"- Requests in this run: {report['requests']}\n"
        f"- Paired HOLD groups: {report['paired_hold_groups']}\n"
        f"- Total recorded tokens: {report['total_tokens']}\n",
        encoding="utf-8",
    )
    return PipelineResult(
        output_dir,
        len(final_records),
        rejected_count,
        request_count,
        len(existing),
    )


def run_pipeline(config: PipelineConfig, client: GlmClient) -> PipelineResult:
    if config.total == 6000 and config.confirm_full != 6000:
        raise ValueError("confirm_full must equal 6000 for full generation")
    if config.batch_size <= 0 or config.max_requests <= 0:
        raise ValueError("batch_size and max_requests must be positive")
    if config.trajectory_mode:
        return _run_trajectory_pipeline(config, client)

    output_dir = Path(config.output_dir).resolve()
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rejected_path.touch(exist_ok=True)
    if not config.resume and accepted_path.exists():
        raise ValueError("output already exists and resume is disabled")

    existing = _read_jsonl(accepted_path)
    if len(existing) > config.total:
        raise ValueError("existing accepted records exceed configured total")
    target_units = _quota_units(config.total)
    units = _remaining_units(target_units, existing)
    sources_schedule = _remaining_sources(config.total, existing)
    if len(units) != len(sources_schedule):
        raise RuntimeError("quota and source schedules have different lengths")

    rng = random.Random(config.seed)
    rng.shuffle(units)
    rng.shuffle(sources_schedule)
    jobs = deque(zip(units, sources_schedule))

    source_records = load_source_manifest(config.source_manifest)
    by_dataset: dict[str, list[Any]] = {}
    for record in source_records:
        by_dataset.setdefault(record.provenance.source_dataset, []).append(record)
    for records in by_dataset.values():
        rng.shuffle(records)
    required_sources = set(_source_targets(config.total))
    missing_sources = required_sources - set(by_dataset)
    if missing_sources:
        raise ValueError(f"source manifest is missing datasets: {sorted(missing_sources)}")

    used_derivations = {
        (item["group_id"], item["provenance"]["transformation"])
        for item in existing
    }
    source_indexes: Counter[str] = Counter()
    attempts: Counter[tuple[str, str, str, str]] = Counter()
    seen_texts = {
        " ".join(item["observation_text"].casefold().split()) for item in existing
    }
    rejected_count = len(_read_jsonl(rejected_path))
    request_count = 0
    sequence = len(existing) + rejected_count
    seen_raw_hashes: set[str] = set()

    def next_source(dataset: str, cell: QuotaCell):
        records = by_dataset[dataset]
        start = source_indexes[dataset]
        transformation = f"{cell.observation_type.value}:{cell.decision.value.lower()}"
        for offset in range(len(records)):
            index = (start + offset) % len(records)
            record = records[index]
            group = f"{dataset}:{record.provenance.source_record_id}"
            derivation = (group, transformation)
            if derivation not in used_derivations:
                source_indexes[dataset] = index + 1
                used_derivations.add(derivation)
                return record
        raise ValueError(
            f"not enough distinct grounded derivations for source {dataset} "
            f"and transformation {transformation}"
        )

    while jobs:
        if request_count >= config.max_requests:
            raise RuntimeError("maximum request count reached before quotas were filled")
        batch_jobs = [jobs.popleft() for _ in range(min(config.batch_size, len(jobs)))]
        candidates = []
        sources_for_batch = []
        for cell, dataset in batch_jobs:
            source_record = next_source(dataset, cell)
            sequence += 1
            candidates.append(build_candidate(source_record, cell, sequence=sequence))
            sources_for_batch.append((cell, dataset))

        realizations = client.realize(candidates)
        request_count += 1
        if len(realizations) != len(candidates):
            raise RuntimeError("GLM client returned a different number of realizations")
        by_id = {item.record_id: item for item in realizations}
        for candidate, job in zip(candidates, sources_for_batch):
            realization = by_id.get(candidate.record_id)
            if realization is None:
                raise RuntimeError("GLM client omitted a requested record")
            if realization.raw_response and realization.request_hash not in seen_raw_hashes:
                with (raw_dir / "glm_responses.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "request_hash": realization.request_hash,
                        "raw_response": realization.raw_response,
                        "usage": realization.usage,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                seen_raw_hashes.add(realization.request_hash)
            validation = validate_candidate(candidate, realization, seen_texts=seen_texts)
            audit = AuditRecord(
                candidate,
                realization,
                validation,
                config.prompt_version,
                config.rule_version,
                _generator_name(client),
            )
            destination = accepted_path if validation.accepted else rejected_path
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            if not validation.accepted:
                rejected_count += 1
                key = (*_cell_key(job[0]), job[1])
                attempts[key] += 1
                if attempts[key] >= config.max_attempts_per_record:
                    raise RuntimeError(f"quality attempts exhausted for quota cell {key}")
                jobs.append(job)

    final_records = _read_jsonl(accepted_path)
    manifest_bytes = Path(config.source_manifest).read_bytes()
    manifest = {
        "total": config.total,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "prompt_version": config.prompt_version,
        "rule_version": config.rule_version,
        "generator": _generator_name(client),
        "source_manifest": str(Path(config.source_manifest).resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "full_generation_guard": config.confirm_full,
    }
    report = _report(final_records, rejected_count, request_count)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    _write_json_atomic(output_dir / "quality_report.json", report)
    (output_dir / "quality_report.md").write_text(
        "# SIEVE data quality report\n\n"
        f"- Accepted: {report['accepted']}\n"
        f"- Rejected: {report['rejected']}\n"
        f"- Requests in this run: {report['requests']}\n"
        f"- Total recorded tokens: {report['total_tokens']}\n",
        encoding="utf-8",
    )
    return PipelineResult(
        output_dir,
        len(final_records),
        rejected_count,
        request_count,
        len(existing),
    )

