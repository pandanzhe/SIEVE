from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from ..core.executor import StateExecutor
from ..core.types import Decision
from .io import read_jsonl
from .schema import SFTRecord


@dataclass(frozen=True)
class ValidationIssue:
    scenario_id: str
    code: str
    message: str


def validate_records(records: Sequence[SFTRecord]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    executor = StateExecutor()
    decisions_by_scenario: dict[str, set[Decision]] = {}
    steps_by_scenario: dict[str, list[int]] = {}
    for record in records:
        decisions_by_scenario.setdefault(record.scenario_id, set()).add(record.target.decision)
        steps_by_scenario.setdefault(record.scenario_id, []).append(record.step_index)
        context, target = record.context, record.target
        result = executor.apply(
            context.belief_state, context.ledger, target, context.observation
        )
        if not result.executed:
            issues.append(
                ValidationIssue(record.scenario_id, "invalid_target", result.error or "executor rejected target")
            )
        observation = context.observation
        current_slot = context.belief_state.get(observation.field_id)
        entity_matches = current_slot is not None and current_slot.entity == observation.entity
        known_fresh = (
            observation.valid_from is not None
            and observation.valid_from >= observation.observed_at
        )
        if target.decision is Decision.UPDATE:
            if not observation.relevant or not entity_matches or not known_fresh:
                issues.append(
                    ValidationIssue(
                        record.scenario_id,
                        "unsafe_update_label",
                        "UPDATE target lacks visible relevance, entity match, or freshness",
                    )
                )
        elif target.decision is Decision.HOLD:
            if observation.valid_from is not None and known_fresh and observation.source.startswith("official_"):
                issues.append(
                    ValidationIssue(
                        record.scenario_id,
                        "unnecessary_hold_label",
                        "HOLD target is already fresh and authoritative",
                    )
                )
        elif target.decision is Decision.IGNORE:
            visibly_invalid = (
                not observation.relevant
                or not entity_matches
                or (observation.valid_from is not None and not known_fresh)
            )
            if not visibly_invalid:
                issues.append(
                    ValidationIssue(
                        record.scenario_id,
                        "unsupported_ignore_label",
                        "IGNORE target has no visible invalidity cue",
                    )
                )

    expected = set(Decision)
    for scenario_id, decisions in decisions_by_scenario.items():
        if decisions != expected:
            missing = sorted(decision.value for decision in expected - decisions)
            issues.append(
                ValidationIssue(
                    scenario_id,
                    "incomplete_counterfactual_group",
                    f"missing decisions: {','.join(missing)}",
                )
            )
    for scenario_id, steps in steps_by_scenario.items():
        if sorted(steps) != list(range(len(steps))):
            issues.append(
                ValidationIssue(
                    scenario_id,
                    "invalid_step_index",
                    "step_index values must be unique and contiguous from zero",
                )
            )

    return issues


def validate_split_isolation(splits: Mapping[str, Sequence[SFTRecord]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    names = sorted(splits)
    scenarios = {name: {record.scenario_id for record in records} for name, records in splits.items()}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for scenario_id in sorted(scenarios[left] & scenarios[right]):
                issues.append(
                    ValidationIssue(
                        scenario_id,
                        "split_leakage",
                        f"scenario appears in both {left} and {right}",
                    )
                )
    return issues


def validate_jsonl_directory(data_dir: str | Path) -> list[ValidationIssue]:
    path = Path(data_dir)
    splits = {name: read_jsonl(path / f"{name}.jsonl") for name in ("train", "dev", "test")}
    issues = validate_split_isolation(splits)
    for records in splits.values():
        issues.extend(validate_records(records))
    return issues
