from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import GroundedSourceRecord, Provenance, ScenarioType


def _digest(raw: Any) -> str:
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provenance(
    dataset: str,
    version: str,
    license_name: str,
    path: Path,
    record_id: str,
    raw: Any,
) -> Provenance:
    return Provenance(
        dataset,
        version,
        record_id,
        f"{path.as_posix()}#{record_id}",
        _digest(raw),
        license_name,
    )


def _first_scalar(value: Any) -> Any | None:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        for child in value.values():
            result = _first_scalar(child)
            if result is not None:
                return result
    if isinstance(value, list):
        for child in value:
            result = _first_scalar(child)
            if result is not None:
                return result
    return None


def _field_identifier(value: object, fallback: str) -> str:
    if value is None or not str(value).strip():
        return fallback
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or fallback

def normalize_tau2(
    path: str | Path,
    version: str,
    license_name: str,
    *,
    limit: int | None = None,
) -> list[GroundedSourceRecord]:
    source_path = Path(path).resolve()
    tasks = json.loads(source_path.read_text(encoding="utf-8"))
    records: list[GroundedSourceRecord] = []
    for task in tasks:
        instructions = task.get("user_scenario", {}).get("instructions", {})
        actions = task.get("evaluation_criteria", {}).get("actions", [])
        arguments: dict[str, Any] = {}
        tool_name = "tau2_environment"
        for action in actions:
            if isinstance(action, dict) and isinstance(action.get("arguments"), dict):
                arguments = action["arguments"]
                tool_name = str(action.get("name") or tool_name)
                if arguments:
                    break
        if not arguments:
            continue
        entity_keys = ("order_id", "reservation_id", "booking_id", "user_id", "phone_number")
        entity = next((str(arguments[key]) for key in entity_keys if key in arguments), None)
        if entity is None:
            entity = f"{instructions.get('domain', 'tau2')}-task-{task.get('id')}"
        value_keys = [key for key in arguments if key not in entity_keys]
        field_id = value_keys[0] if value_keys else next(iter(arguments))
        new_value = arguments[field_id]
        record_id = f"{instructions.get('domain', 'tau2')}:{task.get('id')}:{field_id}"
        source_text = " ".join(
            str(value)
            for value in (
                instructions.get("reason_for_call"),
                instructions.get("known_info"),
            )
            if value
        )
        timestamp = 1_700_000_000 + len(records)
        records.append(
            GroundedSourceRecord(
                _provenance("tau2-bench", version, license_name, source_path, record_id, task),
                str(instructions.get("domain") or "tau2"),
                ScenarioType.TRANSACTION,
                entity,
                str(field_id),
                arguments.get(f"old_{field_id}", "unknown"),
                new_value,
                tool_name,
                timestamp,
                timestamp,
                str(instructions.get("reason_for_call") or "complete the service task"),
                source_text or json.dumps(arguments, ensure_ascii=False),
                metadata={"raw_arguments": arguments},
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def _tool_observations(node: Any, action: str = "tool_api") -> Iterable[tuple[str, str]]:
    if not isinstance(node, dict):
        return
    current_action = action
    if node.get("node_type") == "Action" and node.get("description"):
        current_action = str(node["description"])
    observation = node.get("observation")
    if isinstance(observation, str) and observation.strip():
        yield current_action, observation
    for child in node.get("children", []):
        yield from _tool_observations(child, current_action)


def normalize_toolbench(
    paths: Iterable[str | Path],
    version: str,
    license_name: str,
    *,
    limit: int | None = None,
) -> list[GroundedSourceRecord]:
    records: list[GroundedSourceRecord] = []
    for path_value in paths:
        source_path = Path(path_value).resolve()
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        tree = raw.get("tree", {}).get("tree", raw)
        for index, (tool_name, observation) in enumerate(_tool_observations(tree)):
            try:
                parsed = json.loads(observation)
            except json.JSONDecodeError:
                parsed = {"response": observation}
            response = parsed.get("response", parsed) if isinstance(parsed, dict) else parsed
            new_value = response if not isinstance(response, (dict, list)) else json.dumps(
                response, ensure_ascii=False, sort_keys=True
            )
            record_id = f"{source_path.stem}:{index}"
            timestamp = 1_710_000_000 + len(records)
            records.append(
                GroundedSourceRecord(
                    _provenance("ToolBench", version, license_name, source_path, record_id, parsed),
                    "tool_api",
                    ScenarioType.INFORMATION_TOOL,
                    tool_name,
                    "tool_result",
                    "unknown",
                    new_value,
                    tool_name,
                    timestamp,
                    timestamp,
                    f"use {tool_name} result",
                    observation,
                    metadata={"tool_error": parsed.get("error") if isinstance(parsed, dict) else None},
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def normalize_agentbench(
    path: str | Path,
    version: str,
    license_name: str,
    *,
    limit: int | None = None,
) -> list[GroundedSourceRecord]:
    source_path = Path(path).resolve()
    records: list[GroundedSourceRecord] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            table = raw.get("table", {})
            table_info = table.get("table_info", {})
            columns = table_info.get("columns", [])
            rows = table_info.get("rows", [])
            labels = raw.get("label", [])
            if not columns or not rows or not labels:
                continue
            field_id = str(columns[0].get("name") or "answer")
            old_value = rows[0][0] if rows[0] else "unknown"
            new_value = labels[0]
            entity = str(table.get("table_name") or f"table-{index}")
            record_id = f"dbbench:{index}:{field_id}"
            timestamp = 1_720_000_000 + len(records)
            records.append(
                GroundedSourceRecord(
                    _provenance("AgentBench", version, license_name, source_path, record_id, raw),
                    "database",
                    ScenarioType.INFORMATION_TOOL,
                    entity,
                    field_id,
                    old_value,
                    new_value,
                    "database_query",
                    timestamp,
                    timestamp,
                    str(raw.get("description") or "query the database"),
                    str(raw.get("description") or ""),
                    metadata={"table_name": entity, "source": raw.get("source")},
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def normalize_webarena(
    path: str | Path,
    version: str,
    license_name: str,
    *,
    limit: int | None = None,
) -> list[GroundedSourceRecord]:
    source_path = Path(path).resolve()
    tasks = json.loads(source_path.read_text(encoding="utf-8"))
    records: list[GroundedSourceRecord] = []
    for task in tasks:
        answers = task.get("eval", {}).get("reference_answers", {})
        answer = _first_scalar(answers)
        if answer is None:
            continue
        task_id = str(task.get("task_id"))
        sites = task.get("sites") or ["web"]
        instantiation = task.get("instantiation_dict") or {}
        order_number = instantiation.get("order_number")
        entity = (
            f"order:{order_number}"
            if order_number not in (None, "")
            else f"{sites[0]}:{task_id}"
        )
        field_id = _field_identifier(instantiation.get("info"), "reference_answer")
        timestamp = 1_730_000_000 + len(records)
        records.append(
            GroundedSourceRecord(
                _provenance("WebArena", version, license_name, source_path, task_id, task),
                str(sites[0]),
                ScenarioType.INFORMATION_TOOL,
                entity,
                field_id,
                "unknown",
                answer,
                f"{sites[0]}_website",
                timestamp,
                timestamp,
                str(task.get("intent") or "complete the web task"),
                str(task.get("intent") or ""),
                metadata={
                    "sites": sites,
                    "intent_template_id": task.get("intent_template_id"),
                    "instantiation_dict": instantiation,
                },
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def write_normalized_jsonl(
    records: Iterable[GroundedSourceRecord], path: str | Path
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            raw = {
                "id": record.provenance.source_record_id,
                "domain": record.domain,
                "scenario_type": record.scenario_type.value,
                "entity": record.entity,
                "field_id": record.field_id,
                "old_value": record.old_value,
                "new_value": record.new_value,
                "source": record.source,
                "observed_at": record.observed_at,
                "valid_from": record.valid_from,
                "goal": record.goal,
                "source_text": record.source_text,
                "aliases": list(record.aliases),
                "metadata": record.metadata,
                "upstream_provenance": record.provenance.to_dict(),
            }
            handle.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
