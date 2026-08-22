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
from ..data_factory.models import GroundedSourceRecord
from ..data_factory.normalize import (
    normalize_agentbench,
    normalize_tau2,
    normalize_toolbench,
    normalize_webarena,
)
from .io import write_scenarios
from .schema import RLScenario, ScenarioEvent

SPLITS = ("train", "dev", "test")
MACRO_QUOTAS = {"commerce": 0.5, "service": 0.3, "workflow": 0.2}
SCENARIO_TEMPLATES = (
    "no_auto_repair",
    "delayed_contamination",
    "multi_field_dependency",
    "verification_budget_choice",
    "complex_stale_conflict",
)
RAW_SOURCE_LIMITS = {
    "tau2-bench": 1500,
    "ToolBench": 800,
    "AgentBench": 800,
    "WebArena": 800,
}
RAW_SOURCE_VERSIONS = {
    "tau2-bench": "1d244f5dca42944b67a379b44bfeb9f5748f189d",
    "ToolBench": "d56fdd89faf8c91fa135090b212bb9057ee5cfc2",
    "AgentBench": "d1e4a10db08c87075c78972e48ecc182be03e2d5",
    "WebArena": "dce04686a56253aefba7b18a4fa0937cf1dc987b",
}
RAW_SOURCE_LICENSES = {
    "tau2-bench": "MIT",
    "ToolBench": "Apache-2.0",
    "AgentBench": "Apache-2.0",
    "WebArena": "Apache-2.0",
}


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


def _context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("state") or row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("source row must contain state or context")
    return context


def _observation_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = _context(row).get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("source row context must contain observation")
    return observation


def _goal_from_row(row: Mapping[str, Any]) -> str:
    return str(_context(row).get("goal", ""))


def _risk_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    risk = _context(row).get("risk", {})
    return risk if isinstance(risk, Mapping) else {}


def _domain_from_row(row: Mapping[str, Any]) -> str:
    taxonomy = row.get("taxonomy", {})
    if isinstance(taxonomy, Mapping) and taxonomy.get("domain"):
        return str(taxonomy["domain"])
    return str(row.get("domain", "other"))


def _base_task_id(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance", {})
    dataset = str(provenance.get("source_dataset", "unknown"))
    # source_record_id names the upstream benchmark task/field. parent_record_id
    # can instead point to another generated SFT row in verification chains.
    source_id = str(
        provenance.get("source_record_id") or provenance.get("parent_record_id")
    )
    field_id = str(_observation_from_row(row).get("field_id", ""))
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
    observation = _observation_from_row(row)
    validation = row.get("validation", {})
    accepted = validation.get("accepted", True) if validation else True
    return bool(
        accepted
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
    return Observation(
        field_id=field_id,
        value=value,
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
        candidate_field = str(_observation_from_row(candidate)["field_id"])
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
                str(_observation_from_row(item)["field_id"]),
            ),
        )
    source = fields[0]
    derived = deepcopy(dict(source))
    domain = _domain_from_row(source)
    macro = _macro_domain(domain)
    field_id, value, stale_value = {
        "commerce": ("fulfillment_readiness", "ready", "not_ready"),
        "service": ("service_request_status", "confirmed", "unconfirmed"),
        "workflow": ("workflow_record_status", "validated", "invalid"),
    }[macro]
    observation = _observation_from_row(derived)
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


def _raw_record_to_row(record: GroundedSourceRecord) -> dict[str, Any]:
    provenance = record.provenance.to_dict()
    provenance["parent_record_id"] = (
        provenance.get("parent_record_id")
        or f"{record.provenance.source_dataset}:{record.provenance.source_record_id}"
    )
    return {
        "record_id": f"raw:{record.provenance.source_dataset}:{record.provenance.source_record_id}",
        "group_id": f"{record.provenance.source_dataset}:{record.provenance.source_record_id}",
        "provenance": provenance,
        "taxonomy": {"domain": record.domain},
        "state": {
            "goal": record.goal,
            "observation": {
                "entity": record.entity,
                "field_id": record.field_id,
                "value": record.new_value,
                "source": record.source,
                "observed_at": record.observed_at,
                "valid_from": record.valid_from,
                "source_authority": "primary_record",
                "authenticated": True,
            },
            "risk": {"risk": "medium", "reversible": True},
        },
        "observation_text": record.source_text,
        "target": {
            "decision": "UPDATE",
            "affected_fields": [record.field_id],
            "patches": [
                {"op": "SET_VALUE", "field_id": record.field_id, "value": record.new_value}
            ],
            "verification": None,
        },
        "validation": {"accepted": True, "reason_codes": []},
    }


def _default_raw_dir(source: Path) -> Path | None:
    for parent in (source.parent, *source.parents):
        candidate = parent / "data" / "raw"
        if (candidate / "manifest.json").is_file():
            return candidate
    return None


def _load_raw_stage2_rows(raw_dir: str | Path | None) -> list[dict[str, Any]]:
    if raw_dir is None:
        return []
    root = Path(raw_dir).resolve()
    if not (root / "manifest.json").is_file():
        return []

    records: list[GroundedSourceRecord] = []
    tau_root = root / "tau2-bench" / "data" / "tau2" / "domains"
    for domain in ("retail", "airline", "telecom"):
        path = tau_root / domain / "tasks.json"
        if path.is_file():
            records.extend(
                normalize_tau2(
                    path,
                    RAW_SOURCE_VERSIONS["tau2-bench"],
                    RAW_SOURCE_LICENSES["tau2-bench"],
                    limit=max(1, RAW_SOURCE_LIMITS["tau2-bench"] // 3),
                )
            )

    tool_root = root / "ToolBench" / "data_example" / "answer"
    if tool_root.is_dir():
        records.extend(
            normalize_toolbench(
                sorted(tool_root.glob("**/*.json")),
                RAW_SOURCE_VERSIONS["ToolBench"],
                RAW_SOURCE_LICENSES["ToolBench"],
                limit=RAW_SOURCE_LIMITS["ToolBench"],
            )
        )

    agent_path = root / "AgentBench" / "data" / "dbbench" / "standard.jsonl"
    if agent_path.is_file():
        records.extend(
            normalize_agentbench(
                agent_path,
                RAW_SOURCE_VERSIONS["AgentBench"],
                RAW_SOURCE_LICENSES["AgentBench"],
                limit=RAW_SOURCE_LIMITS["AgentBench"],
            )
        )

    web_path = root / "webarena" / "config_files" / "test.raw.json"
    if web_path.is_file():
        records.extend(
            normalize_webarena(
                web_path,
                RAW_SOURCE_VERSIONS["WebArena"],
                RAW_SOURCE_LICENSES["WebArena"],
                limit=RAW_SOURCE_LIMITS["WebArena"],
            )
        )
    return [_raw_record_to_row(record) for record in records]


def _scenario_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    variant_id: int,
    scenario_template: str,
    alternative_entity: str,
    alternative_value: Any,
    stale_value: Any,
    distractor_base_task_ids: tuple[str, ...],
) -> RLScenario:
    if scenario_template not in SCENARIO_TEMPLATES:
        raise ValueError(f"unknown scenario template: {scenario_template}")
    if not rows:
        raise ValueError("a scenario requires at least one task field")
    field_rows = _ensure_two_task_fields(rows)
    row = field_rows[0]
    secondary_row = field_rows[1]
    raw_observation = _observation_from_row(row)
    secondary_observation = _observation_from_row(secondary_row)
    domain = _domain_from_row(row)
    macro = _macro_domain(domain)
    entity = str(raw_observation["entity"])
    field_id = str(raw_observation["field_id"])
    timestamp = int(raw_observation["observed_at"]) + variant_id * 10
    value = _gold_value(row, field_id)
    secondary_field_id = str(secondary_observation["field_id"])
    secondary_entity = str(secondary_observation["entity"])
    secondary_value = _gold_value(secondary_row, secondary_field_id)
    goal = _goal_from_row(row)
    base_task_id = _base_task_id(row)
    identifier = hashlib.sha256(
        f"{base_task_id}:{split}:{variant_id}:{scenario_template}".encode("utf-8")
    ).hexdigest()[:16]

    def verified(
        *,
        field: str,
        verified_entity: str,
        verified_value: Any,
        delta: int,
    ) -> Observation:
        return _observation(
            entity=verified_entity,
            field_id=field,
            value=verified_value,
            source="official_verification_api",
            timestamp=timestamp + delta,
            condition="verified_current_value",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
        )

    def event(
        suffix: str,
        kind: str,
        *,
        event_entity: str,
        field: str,
        event_value: Any,
        source: str,
        delta: int,
        condition: str,
        relevant: bool,
        authority: str,
        authenticated: bool,
        perturbation: str,
        expected: str,
        verification_value: Any | None = None,
    ) -> ScenarioEvent:
        verify_tool = None
        verify_observation = None
        if expected == "HOLD":
            verify_tool = f"verify_{field}"
            verify_observation = verified(
                field=field,
                verified_entity=event_entity,
                verified_value=verification_value if verification_value is not None else event_value,
                delta=delta + 1,
            )
        return ScenarioEvent(
            event_id=f"{identifier}:{suffix}",
            kind=kind,
            observation=_observation(
                entity=event_entity,
                field_id=field,
                value=event_value,
                source=source,
                timestamp=timestamp + delta,
                condition=condition,
                relevant=relevant,
                authority=authority,
                authenticated=authenticated,
                perturbation=perturbation,
            ),
            expected_decision=expected,
            verification_tool=verify_tool,
            verification_observation=verify_observation,
        )

    no_auto_repair = (
        event(
            "ambiguous",
            "ambiguous",
            event_entity=entity,
            field=field_id,
            event_value=value,
            source="unverified_channel",
            delta=0,
            condition="relevant_unverified_change_no_later_repair",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="source_ambiguity",
            expected="HOLD",
            verification_value=value,
        ),
        event(
            "wrong-entity",
            "wrong_entity",
            event_entity=alternative_entity,
            field=field_id,
            event_value=alternative_value,
            source="official_api",
            delta=2,
            condition="outside_active_entity",
            relevant=False,
            authority="primary_record",
            authenticated=True,
            perturbation="entity_mismatch",
            expected="IGNORE",
        ),
        event(
            "authoritative-secondary",
            "authoritative",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=secondary_value,
            source="official_api",
            delta=3,
            condition="relevant_authoritative_change",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
            expected="UPDATE",
        ),
        event(
            "stale-secondary",
            "stale_conflict",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=stale_value,
            source="cached_replica",
            delta=-100,
            condition="stale_conflict",
            relevant=True,
            authority="secondary_report",
            authenticated=True,
            perturbation="stale_value",
            expected="IGNORE",
        ),
    )
    delayed_contamination = (
        event(
            "wrong-ambiguous",
            "delayed_contamination",
            event_entity=entity,
            field=field_id,
            event_value=alternative_value,
            source="partner_feed",
            delta=0,
            condition="unverified_value_controls_later_precondition",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="wrong_value",
            expected="HOLD",
            verification_value=value,
        ),
        event(
            "downstream-precondition",
            "downstream_precondition",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=secondary_value,
            source="official_api",
            delta=3,
            condition="requires_clean_prior_belief",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
            expected="UPDATE",
        ),
        event(
            "wrong-entity",
            "wrong_entity",
            event_entity=alternative_entity,
            field=field_id,
            event_value=alternative_value,
            source="official_api",
            delta=4,
            condition="outside_active_entity",
            relevant=False,
            authority="primary_record",
            authenticated=True,
            perturbation="entity_mismatch",
            expected="IGNORE",
        ),
    )
    multi_field_dependency = (
        event(
            "authoritative-primary",
            "authoritative",
            event_entity=entity,
            field=field_id,
            event_value=value,
            source="official_api",
            delta=0,
            condition="relevant_authoritative_change",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
            expected="UPDATE",
        ),
        event(
            "ambiguous-secondary",
            "ambiguous",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=secondary_value,
            source="unverified_channel",
            delta=1,
            condition="second_required_field_is_unverified",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="source_ambiguity",
            expected="HOLD",
            verification_value=secondary_value,
        ),
        event(
            "wrong-entity-secondary",
            "wrong_entity",
            event_entity=alternative_entity,
            field=secondary_field_id,
            event_value=alternative_value,
            source="official_api",
            delta=4,
            condition="outside_active_entity",
            relevant=False,
            authority="primary_record",
            authenticated=True,
            perturbation="entity_mismatch",
            expected="IGNORE",
        ),
        event(
            "stale-primary",
            "stale_conflict",
            event_entity=entity,
            field=field_id,
            event_value=stale_value,
            source="cached_replica",
            delta=-50,
            condition="stale_conflict_after_primary_update",
            relevant=True,
            authority="secondary_report",
            authenticated=True,
            perturbation="stale_value",
            expected="IGNORE",
        ),
    )
    verification_budget_choice = (
        event(
            "critical-ambiguous",
            "verification_budget_choice",
            event_entity=entity,
            field=field_id,
            event_value=value,
            source="unverified_channel",
            delta=0,
            condition="critical_field_requires_verification",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="source_ambiguity",
            expected="HOLD",
            verification_value=value,
        ),
        event(
            "noncritical-ambiguous",
            "ambiguous_noncritical",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=stale_value,
            source="user_note",
            delta=2,
            condition="noncritical_ambiguous_field_with_later_authority",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="verification_budget_distractor",
            expected="IGNORE",
        ),
        event(
            "authoritative-secondary",
            "authoritative",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=secondary_value,
            source="official_api",
            delta=4,
            condition="later_authoritative_repair_for_noncritical_ambiguous",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
            expected="UPDATE",
        ),
    )
    complex_stale_conflict = (
        event(
            "authoritative-primary",
            "authoritative",
            event_entity=entity,
            field=field_id,
            event_value=value,
            source="official_api",
            delta=0,
            condition="current_primary_authoritative_value",
            relevant=True,
            authority="primary_record",
            authenticated=True,
            perturbation="clean",
            expected="UPDATE",
        ),
        event(
            "stale-primary",
            "complex_stale_conflict",
            event_entity=entity,
            field=field_id,
            event_value=stale_value,
            source="legacy_sync_cache",
            delta=-120,
            condition="older_secondary_conflict_same_entity",
            relevant=True,
            authority="secondary_report",
            authenticated=True,
            perturbation="stale_value",
            expected="IGNORE",
        ),
        event(
            "ambiguous-secondary",
            "ambiguous",
            event_entity=secondary_entity,
            field=secondary_field_id,
            event_value=secondary_value,
            source="unverified_channel",
            delta=2,
            condition="related_field_requires_confirmation",
            relevant=True,
            authority="secondary_report",
            authenticated=False,
            perturbation="source_ambiguity",
            expected="HOLD",
            verification_value=secondary_value,
        ),
        event(
            "wrong-entity",
            "wrong_entity",
            event_entity=alternative_entity,
            field=field_id,
            event_value=alternative_value,
            source="official_api",
            delta=5,
            condition="different_entity_authoritative_but_out_of_scope",
            relevant=False,
            authority="primary_record",
            authenticated=True,
            perturbation="entity_mismatch",
            expected="IGNORE",
        ),
    )
    events_by_template = {
        "no_auto_repair": no_auto_repair,
        "delayed_contamination": delayed_contamination,
        "multi_field_dependency": multi_field_dependency,
        "verification_budget_choice": verification_budget_choice,
        "complex_stale_conflict": complex_stale_conflict,
    }
    events = events_by_template[scenario_template]
    risk_raw = _risk_from_row(row)
    risk_name = str(risk_raw.get("risk", "medium"))
    if risk_name not in {item.value for item in RiskLevel}:
        risk_name = "medium"
    dependent_rows = [row]
    if secondary_field_id != field_id:
        dependent_rows.append(secondary_row)
    dependent_fields = tuple(
        str(_observation_from_row(item)["field_id"]) for item in dependent_rows
    )
    dependent_slots = [
        BeliefSlot(
            id=str(_observation_from_row(item)["field_id"]),
            value=None,
            status=SlotStatus.EMPTY,
            source="initial_state",
            observed_at=max(0, timestamp - 1),
            valid_from=None,
            entity=str(_observation_from_row(item)["entity"]),
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
    provenance["scenario_template"] = scenario_template
    verification_tools = {
        event.observation.field_id: event.verification_tool
        for event in events
        if event.verification_tool is not None
    }
    for slot in initial_state.slots:
        if slot.id == "__verification_policy" and isinstance(slot.value, dict):
            slot.value["tool_by_field"] = verification_tools
            slot.value["verification_budget"] = (
                1 if scenario_template == "verification_budget_choice" else 2
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
            verification_remaining=(
                1 if scenario_template == "verification_budget_choice" else 2
            ),
            tool_remaining=5,
            steps_remaining=8,
            tokens_remaining=4096,
        ),
        ledger_capacity=4,
        events=events,
        oracle_state={
            str(_observation_from_row(item)["field_id"]): _gold_value(
                item, str(_observation_from_row(item)["field_id"])
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
        domain = _domain_from_row(row)
        field_id = str(_observation_from_row(row)["field_id"])
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
            and str(_observation_from_row(row)["entity"]) != entity
        ]
        if not candidates:
            continue
        if fallback is None:
            fallback = candidates
        conflicting = []
        for candidate in candidates:
            candidate_field = str(_observation_from_row(candidate)["field_id"])
            if _gold_value(candidate, candidate_field) != exclude_value:
                conflicting.append(candidate)
        if conflicting:
            return conflicting[salt % len(conflicting)]
    if fallback:
        return fallback[salt % len(fallback)]
    raise ValueError(f"no distractor is available for base task {base_task_id}")


def _source_dataset_from_row(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance", {})
    if isinstance(provenance, Mapping):
        return str(provenance.get("source_dataset", "unknown"))
    return "unknown"


def _interleave_task_groups(
    task_groups: Sequence[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    by_source: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rows in task_groups:
        by_source[_source_dataset_from_row(rows[0])].append(rows)
    for groups in by_source.values():
        groups.sort(key=lambda rows: _base_task_id(rows[0]))
    result: list[list[dict[str, Any]]] = []
    source_names = sorted(by_source)
    index = 0
    while True:
        added = False
        for source_name in source_names:
            groups = by_source[source_name]
            if index < len(groups):
                result.append(groups[index])
                added = True
        if not added:
            return result
        index += 1


def build_rl_corpus(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    split_counts: Mapping[str, int] | None = None,
    raw_dir: str | Path | None = None,
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
    rows_from_sft = list(_read_jsonl(source))
    resolved_raw_dir = _default_raw_dir(source) if raw_dir is None else Path(raw_dir)
    rows_from_raw = _load_raw_stage2_rows(resolved_raw_dir)
    for row in rows_from_sft + rows_from_raw:
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
        domain = _domain_from_row(row)
        pools[(split, _macro_domain(domain))].append(rows)
    for key, task_groups in list(pools.items()):
        pools[key] = _interleave_task_groups(task_groups)

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
                primary_observation = _observation_from_row(primary)
                secondary_observation = _observation_from_row(secondary)
                domain = _domain_from_row(primary)
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
                alternative_observation = _observation_from_row(alternative)
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
                    stale_field = str(_observation_from_row(stale_alternative)["field_id"])
                    stale_value = _gold_value(stale_alternative, stale_field)
                scenario = _scenario_from_rows(
                    field_rows,
                    split=split,
                    variant_id=variant,
                    scenario_template=SCENARIO_TEMPLATES[offset % len(SCENARIO_TEMPLATES)],
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
        "sft_source_rows": len(rows_from_sft),
        "raw_enrichment_rows": len(rows_from_raw),
        "source_dataset_counts": dict(
            Counter(_source_dataset_from_row(row) for row in eligible_rows)
        ),
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
        "scenario_source_counts": {
            split: dict(
                Counter(
                    str(item.provenance.get("source_dataset", "unknown"))
                    for item in scenarios
                )
            )
            for split, scenarios in scenarios_by_split.items()
        },
        "event_kind_counts": {
            split: dict(
                Counter(event.kind for item in scenarios for event in item.events)
            )
            for split, scenarios in scenarios_by_split.items()
        },
        "scenario_template_counts": {
            split: dict(
                Counter(
                    str(item.provenance.get("scenario_template", "unknown"))
                    for item in scenarios
                )
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
        "raw_enrichment_dir": (
            str(resolved_raw_dir.resolve()) if resolved_raw_dir is not None else None
        ),
        "split_policy": "sha256(seed:base_task_id), 75/10/15 before augmentation",
        "macro_domain_targets": MACRO_QUOTAS,
        "scenario_templates": SCENARIO_TEMPLATES,
        "split_counts": {name: len(scenarios_by_split[name]) for name in SPLITS},
        "split_sha256": split_hashes,
    }
    _write_json(destination / "quality_report.json", quality)
    _write_json(destination / "manifest.json", manifest)
    return manifest
