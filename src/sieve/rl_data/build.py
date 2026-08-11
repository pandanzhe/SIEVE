from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    Observation,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
)
from .io import write_scenarios
from .schema import RLScenario, ScenarioEvent

SPLITS = ("train", "dev", "test")
MACRO_QUOTAS = {"commerce": 0.5, "service": 0.3, "workflow": 0.2}


def split_base_task(base_task_id: str, seed: int) -> str:
    """Assign a base task before augmentation so variants cannot cross splits."""
    digest = hashlib.sha256(f"{seed}:{base_task_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 75:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def _base_task_id(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance", {})
    dataset = str(provenance.get("source_dataset", "unknown"))
    # source_record_id names the upstream benchmark task/field. parent_record_id
    # can instead point to another generated SFT row in verification chains.
    source_id = str(
        provenance.get("source_record_id") or provenance.get("parent_record_id")
    )
    field_id = str(row.get("state", {}).get("observation", {}).get("field_id", ""))
    suffix = f":{field_id}"
    if field_id and source_id.endswith(suffix):
        source_id = source_id[: -len(suffix)]
    if source_id.lower().startswith(f"{dataset.lower()}:"):
        return source_id
    return f"{dataset}:{source_id}"


def _macro_domain(domain: str) -> str:
    normalized = domain.lower()
    if normalized in {"shopping", "shopping_admin", "retail", "commerce"}:
        return "commerce"
    if normalized in {"airline", "telecom", "service", "travel"}:
        return "service"
    return "workflow"


def _eligible(row: Mapping[str, Any]) -> bool:
    target = row.get("target", {})
    observation = row.get("state", {}).get("observation", {})
    validation = row.get("validation", {})
    return bool(
        validation.get("accepted")
        and target.get("decision") == "UPDATE"
        and target.get("patches")
        and observation.get("source_authority") == "primary_record"
        and observation.get("authenticated") is True
    )


def _gold_value(row: Mapping[str, Any], field_id: str) -> Any:
    patches = row["target"]["patches"]
    for patch in patches:
        if patch.get("field_id") == field_id and patch.get("op") == "SET_VALUE":
            return patch.get("value")
    raise ValueError(f"eligible source row has no SET_VALUE patch for {field_id!r}")


def _observation(
    *,
    entity: str,
    field_id: str,
    value: Any,
    source: str,
    timestamp: int,
    condition: str,
    relevant: bool,
    authority: str,
    authenticated: bool,
    perturbation: str,
) -> Observation:
    rendered = (
        f"{source} reports {field_id} for {entity} as {value}; "
        f"observed_at={timestamp}."
    )
    return Observation(
        field_id=field_id,
        value=rendered,
        source=source,
        observed_at=timestamp,
        valid_from=timestamp,
        entity=entity,
        condition=condition,
        relevant=relevant,
        perturbation=perturbation,
        source_authority=authority,
        authenticated=authenticated,
    )


def _distinct_field_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_field: dict[str, Mapping[str, Any]] = {}
    for candidate in rows:
        candidate_field = str(candidate["state"]["observation"]["field_id"])
        by_field.setdefault(candidate_field, candidate)
    return [by_field[field] for field in sorted(by_field)]


def _ensure_two_task_fields(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    fields = _distinct_field_rows(rows)
    if len(fields) >= 2:
        return sorted(
            fields,
            key=lambda item: (
                item.get("_stage2_stale_value") is not None,
                str(item["state"]["observation"]["field_id"]),
            ),
        )
    source = fields[0]
    derived = deepcopy(dict(source))
    domain = str(source.get("taxonomy", {}).get("domain", "other"))
    macro = _macro_domain(domain)
    field_id, value, stale_value = {
        "commerce": ("fulfillment_readiness", "ready", "not_ready"),
        "service": ("service_request_status", "confirmed", "unconfirmed"),
        "workflow": ("workflow_record_status", "validated", "invalid"),
    }[macro]
    observation = derived["state"]["observation"]
    observation["field_id"] = field_id
    observation["value"] = value
    observation["observed_at"] = int(observation["observed_at"]) + 5
    observation["valid_from"] = observation["observed_at"]
    derived["record_id"] = f"{source.get('record_id', 'unknown')}::stage2:{field_id}"
    provenance = derived.setdefault("provenance", {})
    provenance["source_record_id"] = (
        f"{provenance.get('source_record_id', 'unknown')}:{field_id}"
    )
    provenance["transformation"] = "stage2_derived_task_status:update"
    derived["target"] = {
        "decision": "UPDATE",
        "affected_fields": [field_id],
        "patches": [{"op": "SET_VALUE", "field_id": field_id, "value": value}],
        "verification": None,
    }
    derived["_stage2_stale_value"] = stale_value
    return [source, derived]


def _scenario_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    variant_id: int,
    alternative_entity: str,
    alternative_value: Any,
    stale_value: Any,
    distractor_base_task_ids: tuple[str, ...],
) -> RLScenario:
    if not rows:
        raise ValueError("a scenario requires at least one task field")
    field_rows = _ensure_two_task_fields(rows)
    row = field_rows[0]
    secondary_row = field_rows[1]
    state = row["state"]
    raw_observation = state["observation"]
    secondary_observation = secondary_row["state"]["observation"]
    domain = str(row.get("taxonomy", {}).get("domain", "other"))
    macro = _macro_domain(domain)
    entity = str(raw_observation["entity"])
    field_id = str(raw_observation["field_id"])
    timestamp = int(raw_observation["observed_at"]) + variant_id * 10
    value = _gold_value(row, field_id)
    secondary_field_id = str(secondary_observation["field_id"])
    secondary_entity = str(secondary_observation["entity"])
    secondary_value = _gold_value(secondary_row, secondary_field_id)
    goal = str(state["goal"])
    base_task_id = _base_task_id(row)
    identifier = hashlib.sha256(
        f"{base_task_id}:{split}:{variant_id}".encode("utf-8")
    ).hexdigest()[:16]

    verification = _observation(
        entity=entity,
        field_id=field_id,
        value=value,
        source="official_verification_api",
        timestamp=timestamp + 1,
        condition="verified_current_value",
        relevant=True,
        authority="primary_record",
        authenticated=True,
        perturbation="clean",
    )
    events = (
        ScenarioEvent(
            event_id=f"{identifier}:ambiguous",
            kind="ambiguous",
            observation=_observation(
                entity=entity,
                field_id=field_id,
                value=value,
                source="unverified_channel",
                timestamp=timestamp,
                condition="relevant_unverified_change",
                relevant=True,
                authority="secondary_report",
                authenticated=False,
                perturbation="source_ambiguity",
            ),
            expected_decision="HOLD",
            verification_tool=f"verify_{field_id}",
            verification_observation=verification,
        ),
        ScenarioEvent(
            event_id=f"{identifier}:wrong-entity",
            kind="wrong_entity",
            observation=_observation(
                entity=alternative_entity,
                field_id=field_id,
                value=alternative_value,
                source="official_api",
                timestamp=timestamp + 2,
                condition="outside_active_entity",
                relevant=False,
                authority="primary_record",
                authenticated=True,
                perturbation="entity_mismatch",
            ),
            expected_decision="IGNORE",
        ),
        ScenarioEvent(
            event_id=f"{identifier}:authoritative",
            kind="authoritative",
            observation=_observation(
                entity=secondary_entity,
                field_id=secondary_field_id,
                value=secondary_value,
                source="official_api",
                timestamp=timestamp + 3,
                condition="relevant_authoritative_change",
                relevant=True,
                authority="primary_record",
                authenticated=True,
                perturbation="clean",
            ),
            expected_decision="UPDATE",
        ),
        ScenarioEvent(
            event_id=f"{identifier}:stale",
            kind="stale_conflict",
            observation=_observation(
                entity=secondary_entity,
                field_id=secondary_field_id,
                value=stale_value,
                source="cached_replica",
                timestamp=max(0, timestamp - 100),
                condition="stale_conflict",
                relevant=True,
                authority="secondary_report",
                authenticated=True,
                perturbation="stale_value",
            ),
            expected_decision="IGNORE",
        ),
    )
    risk_raw = state.get("risk", {})
    risk_name = str(risk_raw.get("risk", "medium"))
    if risk_name not in {item.value for item in RiskLevel}:
        risk_name = "medium"
    dependent_rows = [row]
    if secondary_field_id != field_id:
        dependent_rows.append(secondary_row)
    dependent_fields = tuple(
        str(item["state"]["observation"]["field_id"]) for item in dependent_rows
    )
    dependent_slots = [
        BeliefSlot(
            id=str(item["state"]["observation"]["field_id"]),
            value=None,
            status=SlotStatus.EMPTY,
            source="initial_state",
            observed_at=max(0, timestamp - 1),
            valid_from=None,
            entity=str(item["state"]["observation"]["entity"]),
        )
        for item in dependent_rows
    ]
    initial_state = BeliefState(
        max_slots=8,
        slots=dependent_slots
        + [
            BeliefSlot(
                id="__task_scope",
                value={"domain": domain, "entity": entity},
                status=SlotStatus.TRUSTED,
                source="scenario_definition",
                observed_at=max(0, timestamp - 1),
                valid_from=max(0, timestamp - 1),
                entity=entity,
            ),
            BeliefSlot(
                id="__verification_policy",
                value={
                    "required_authority": "primary_record",
                    "tool_by_field": {field_id: f"verify_{field_id}"},
                },
                status=SlotStatus.TRUSTED,
                source="scenario_definition",
                observed_at=max(0, timestamp - 1),
                valid_from=max(0, timestamp - 1),
                entity=entity,
            ),
        ],
    )
    provenance = dict(row.get("provenance", {}))
    provenance["source_uri"] = (
        f"data/sft/source/records.jsonl#{row.get('record_id', 'unknown')}"
    )
    provenance["stage2_source_record_ids"] = [
        item.get("record_id") for item in dependent_rows
    ]
    provenance["derived_secondary_field"] = bool(
        secondary_row.get("_stage2_stale_value") is not None
    )
    provenance["distractor_base_task_ids"] = list(
        dict.fromkeys(distractor_base_task_ids)
    )
    return RLScenario(
        scenario_id=f"rl:{identifier}",
        base_task_id=base_task_id,
        variant_id=variant_id,
        split=split,
        domain=domain,
        macro_domain=macro,
        provenance=provenance,
        initial_state=initial_state,
        goal=goal,
        risk=RiskEnvelope(
            active_subgoal=goal,
            dependent_fields=dependent_fields,
            risk=RiskLevel(risk_name),
            reversible=bool(risk_raw.get("reversible", True)),
        ),
        budget=Budget(
            verification_remaining=2,
            tool_remaining=5,
            steps_remaining=8,
            tokens_remaining=4096,
        ),
        ledger_capacity=4,
        events=events,
        oracle_state={
            str(item["state"]["observation"]["field_id"]): _gold_value(
                item, str(item["state"]["observation"]["field_id"])
            )
            for item in dependent_rows
        },
    )


def _split_macro_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("split counts must be positive")
    raw = {name: total * ratio for name, ratio in MACRO_QUOTAS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alternative_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = str(row.get("taxonomy", {}).get("domain", "other"))
        field_id = str(row["state"]["observation"]["field_id"])
        macro = _macro_domain(domain)
        for key in (
            ("domain_field", domain, field_id),
            ("macro_field", macro, field_id),
            ("domain", domain, ""),
            ("macro", macro, ""),
            ("all", "", ""),
        ):
            index[key].append(row)
    for candidates in index.values():
        candidates.sort(key=lambda item: str(item.get("record_id", "")))
    return index


def _select_alternative(
    index: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    *,
    base_task_id: str,
    domain: str,
    field_id: str,
    entity: str,
    exclude_value: Any,
    salt: int,
) -> Mapping[str, Any]:
    macro = _macro_domain(domain)
    keys = (
        ("domain_field", domain, field_id),
        ("macro_field", macro, field_id),
        ("domain", domain, ""),
        ("macro", macro, ""),
        ("all", "", ""),
    )
    fallback: list[Mapping[str, Any]] | None = None
    for key in keys:
        candidates = [
            row
            for row in index.get(key, ())
            if _base_task_id(row) != base_task_id
            and str(row["state"]["observation"]["entity"]) != entity
        ]
        if not candidates:
            continue
        if fallback is None:
            fallback = candidates
        conflicting = []
        for candidate in candidates:
            candidate_field = str(candidate["state"]["observation"]["field_id"])
            if _gold_value(candidate, candidate_field) != exclude_value:
                conflicting.append(candidate)
        if conflicting:
            return conflicting[salt % len(conflicting)]
    if fallback:
        return fallback[salt % len(fallback)]
    raise ValueError(f"no distractor is available for base task {base_task_id}")


def build_rl_corpus(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build deterministic, task-disjoint Stage-2 episodes from accepted source rows."""
    source = Path(source_path)
    destination = Path(output_dir)
    requested = dict(split_counts or {"train": 1600, "dev": 200, "test": 300})
    if set(requested) != set(SPLITS):
        raise ValueError(f"split_counts must contain exactly {SPLITS}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_rows: list[dict[str, Any]] = []
    eligible_count = 0
    for row in _read_jsonl(source):
        if not _eligible(row):
            continue
        eligible_count += 1
        eligible_rows.append(row)
        grouped[_base_task_id(row)].append(row)
    if not grouped:
        raise ValueError("no accepted authoritative UPDATE rows found in source data")

    pools: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for base_id, rows in grouped.items():
        row = rows[0]
        split = split_base_task(base_id, seed)
        domain = str(row.get("taxonomy", {}).get("domain", "other"))
        pools[(split, _macro_domain(domain))].append(rows)
    for task_groups in pools.values():
        task_groups.sort(key=lambda rows: _base_task_id(rows[0]))

    eligible_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        eligible_by_split[split_base_task(_base_task_id(row), seed)].append(row)
    distractors_by_split = {
        split: _alternative_index(rows)
        for split, rows in eligible_by_split.items()
    }
    scenarios_by_split: dict[str, list[RLScenario]] = {name: [] for name in SPLITS}
    for split in SPLITS:
        macro_counts = _split_macro_counts(int(requested[split]))
        for macro, count in macro_counts.items():
            candidates = pools.get((split, macro), [])
            if not candidates:
                raise ValueError(
                    f"no base tasks available for split={split!r}, macro_domain={macro!r}"
                )
            for offset in range(count):
                rows = candidates[offset % len(candidates)]
                field_rows = _ensure_two_task_fields(rows)
                base_id = _base_task_id(field_rows[0])
                variant = offset // len(candidates)
                primary = field_rows[0]
                secondary = field_rows[1]
                primary_observation = primary["state"]["observation"]
                secondary_observation = secondary["state"]["observation"]
                domain = str(primary.get("taxonomy", {}).get("domain", "other"))
                alternative = _select_alternative(
                    distractors_by_split[split],
                    base_task_id=base_id,
                    domain=domain,
                    field_id=str(primary_observation["field_id"]),
                    entity=str(primary_observation["entity"]),
                    exclude_value=_gold_value(
                        primary, str(primary_observation["field_id"])
                    ),
                    salt=offset + variant,
                )
                alternative_observation = alternative["state"]["observation"]
                alternative_field = str(alternative_observation["field_id"])
                if secondary.get("_stage2_stale_value") is not None:
                    stale_value = secondary["_stage2_stale_value"]
                    stale_distractor_id = None
                else:
                    stale_alternative = _select_alternative(
                        distractors_by_split[split],
                        base_task_id=base_id,
                        domain=domain,
                        field_id=str(secondary_observation["field_id"]),
                        entity=str(secondary_observation["entity"]),
                        exclude_value=_gold_value(
                            secondary, str(secondary_observation["field_id"])
                        ),
                        salt=offset + variant + 1,
                    )
                    stale_distractor_id = _base_task_id(stale_alternative)
                    stale_field = str(
                        stale_alternative["state"]["observation"]["field_id"]
                    )
                    stale_value = _gold_value(stale_alternative, stale_field)
                scenario = _scenario_from_rows(
                    field_rows,
                    split=split,
                    variant_id=variant,
                    alternative_entity=str(alternative_observation["entity"]),
                    alternative_value=_gold_value(alternative, alternative_field),
                    stale_value=stale_value,
                    distractor_base_task_ids=tuple(
                        item
                        for item in (
                            _base_task_id(alternative),
                            stale_distractor_id,
                        )
                        if item is not None
                    ),
                )
                scenarios_by_split[split].append(scenario)
        scenarios_by_split[split].sort(key=lambda item: item.scenario_id)

    scenario_dir = destination / "scenarios"
    split_hashes: dict[str, str] = {}
    for split, scenarios in scenarios_by_split.items():
        split_path = scenario_dir / f"{split}.jsonl"
        write_scenarios(split_path, scenarios)
        split_hashes[split] = _file_sha256(split_path)

    task_sets = {
        split: {scenario.base_task_id for scenario in scenarios}
        for split, scenarios in scenarios_by_split.items()
    }
    overlaps = {
        "train_dev": len(task_sets["train"] & task_sets["dev"]),
        "train_test": len(task_sets["train"] & task_sets["test"]),
        "dev_test": len(task_sets["dev"] & task_sets["test"]),
    }
    quality = {
        "schema_version": "sieve.rl.scenario.v1",
        "eligible_source_rows": eligible_count,
        "unique_base_tasks": len(grouped),
        "multi_field_base_tasks": sum(
            len(_distinct_field_rows(rows)) > 1 for rows in grouped.values()
        ),
        "derived_secondary_scenarios": sum(
            bool(item.provenance.get("derived_secondary_field"))
            for scenarios in scenarios_by_split.values()
            for item in scenarios
        ),
        "split_base_tasks": {name: len(task_sets[name]) for name in SPLITS},
        "base_task_overlap": overlaps,
        "distractor_split_violations": sum(
            split_base_task(donor, seed) != item.split
            for scenarios in scenarios_by_split.values()
            for item in scenarios
            for donor in item.provenance["distractor_base_task_ids"]
        ),
        "macro_domain_counts": {
            split: dict(Counter(item.macro_domain for item in scenarios))
            for split, scenarios in scenarios_by_split.items()
        },
        "event_kind_counts": {
            split: dict(
                Counter(event.kind for item in scenarios for event in item.events)
            )
            for split, scenarios in scenarios_by_split.items()
        },
    }
    if any(overlaps.values()):
        raise AssertionError(f"base task leakage detected: {overlaps}")
    if quality["distractor_split_violations"]:
        raise AssertionError("distractor donors must stay inside the scenario split")

    manifest = {
        "schema_version": "sieve.rl.scenario.v1",
        "seed": seed,
        "source_file": source.name,
        "source_sha256": _file_sha256(source),
        "split_policy": "sha256(seed:base_task_id), 75/10/15 before augmentation",
        "macro_domain_targets": MACRO_QUOTAS,
        "split_counts": {name: len(scenarios_by_split[name]) for name in SPLITS},
        "split_sha256": split_hashes,
    }
    _write_json(destination / "quality_report.json", quality)
    _write_json(destination / "manifest.json", manifest)
    return manifest
