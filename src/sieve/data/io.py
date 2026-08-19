from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    Decision,
    EvidenceCandidate,
    EvidenceLedger,
    Observation,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
    VerificationRequest,
)
from .schema import SFTRecord


def _observation_from_dict(raw: dict[str, Any]) -> Observation:
    values = dict(raw)
    values.setdefault("condition", "runtime_observation")
    values.setdefault("relevant", True)
    values.setdefault("perturbation", "unlabelled")
    return Observation(**values)

def record_from_dict(raw: dict[str, Any]) -> SFTRecord:
    context = raw["context"]
    state_raw = context["belief_state"]
    state = BeliefState(
        int(state_raw["max_slots"]),
        [
            BeliefSlot(
                slot["id"], slot["value"], SlotStatus(slot["status"]), slot["source"],
                int(slot["observed_at"]), slot.get("valid_from"), slot["entity"],
            )
            for slot in state_raw["slots"]
        ],
    )
    obs = _observation_from_dict(context["observation"])
    risk_raw = context["risk"]
    risk = RiskEnvelope(
        risk_raw["active_subgoal"], tuple(risk_raw["dependent_fields"]),
        RiskLevel(risk_raw["risk"]), bool(risk_raw["reversible"]),
    )
    budget = Budget(**context["budget"])
    ledger_raw = context["ledger"]
    ledger = EvidenceLedger(int(ledger_raw["capacity"]))
    for entry in ledger_raw.get("entries", []):
        ledger.entries.append(EvidenceCandidate(_observation_from_dict(entry["observation"]), entry["reason"]))
    target_raw = raw["target"]
    patches = tuple(Patch(PatchOp(item["op"]), item["field_id"], item["value"]) for item in target_raw["patches"])
    verification_raw = target_raw.get("verification")
    verification = VerificationRequest(**verification_raw) if verification_raw else None
    target = RevisionOutput(
        Decision(target_raw["decision"]), tuple(target_raw["affected_fields"]), patches, verification
    )
    revision_context = RevisionContext(state, obs, context["goal"], risk, budget, ledger)
    # v3 records use "record_id"; v2 records use "scenario_id"
    scenario_id = raw.get("scenario_id") or raw.get("record_id", "")
    step_index = int(raw.get("step_index", 0))
    return SFTRecord(scenario_id, revision_context, target, step_index)


def read_jsonl(path: str | Path) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(record_from_dict(json.loads(line)))
    return records
