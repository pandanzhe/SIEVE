from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..core.types import BeliefState, Decision, RevisionContext, RevisionOutput


class ObservationType(str, Enum):
    NEW_CONSISTENT = "new_consistent"
    EXPLICIT_CONFLICT = "explicit_conflict"
    IMPLICIT_CONFLICT = "implicit_conflict"
    STALE = "stale"
    IRRELEVANT = "irrelevant"
    TOOL_ERROR_OR_LOW_TRUST = "tool_error_or_low_trust"
    INSUFFICIENT_OR_AMBIGUOUS = "insufficient_or_ambiguous"
    VERIFICATION_CONFIRMED = "verification_confirmed"


class ScenarioType(str, Enum):
    TRANSACTION = "transaction"
    INFORMATION_TOOL = "information_tool"
    LONG_TERM_STATE = "long_term_state"


class VerificationAction(str, Enum):
    NO_VERIFY = "NO_VERIFY"
    VERIFY = "VERIFY"
    DEFER = "DEFER"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class QuotaCell:
    observation_type: ObservationType
    decision: Decision
    count: int
    verification_action: VerificationAction = VerificationAction.NO_VERIFY


@dataclass(frozen=True)
class Provenance:
    source_dataset: str
    source_version: str
    source_record_id: str
    source_uri: str
    source_sha256: str
    license: str
    parent_record_id: str | None = None
    transformation: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedSourceRecord:
    provenance: Provenance
    domain: str
    scenario_type: ScenarioType
    entity: str
    field_id: str
    old_value: Any
    new_value: Any
    source: str
    observed_at: int
    valid_from: int | None
    goal: str
    source_text: str
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationCandidate:
    record_id: str
    group_id: str
    provenance: Provenance
    observation_type: ObservationType
    scenario_type: ScenarioType
    domain: str
    difficulty: Difficulty
    context: RevisionContext
    target: RevisionOutput
    oracle_state: BeliefState
    reason_code: str
    conflict_type: str | None
    verification_action: VerificationAction
    structured_event: dict[str, Any]


@dataclass(frozen=True)
class Realization:
    record_id: str
    observation_text: str
    request_hash: str
    raw_response: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditRecord:
    candidate: GenerationCandidate
    realization: Realization
    validation: ValidationResult
    prompt_version: str
    rule_version: str
    generator: str

    def to_dict(self) -> dict[str, Any]:
        context = asdict(self.candidate.context)
        context["risk"]["risk"] = self.candidate.context.risk.risk.value
        for slot_index, slot in enumerate(self.candidate.context.belief_state.slots):
            context["belief_state"]["slots"][slot_index]["status"] = slot.status.value
        for entry_index, entry in enumerate(self.candidate.context.ledger.entries):
            context["ledger"]["entries"][entry_index]["observation"] = asdict(entry.observation)

        target = asdict(self.candidate.target)
        target["decision"] = self.candidate.target.decision.value
        for patch_index, patch in enumerate(self.candidate.target.patches):
            target["patches"][patch_index]["op"] = patch.op.value
        target["reason_code"] = self.candidate.reason_code
        target["conflict_type"] = self.candidate.conflict_type
        target["verification_action"] = self.candidate.verification_action.value

        oracle = asdict(self.candidate.oracle_state)
        for slot_index, slot in enumerate(self.candidate.oracle_state.slots):
            oracle["slots"][slot_index]["status"] = slot.status.value

        return {
            "record_id": self.candidate.record_id,
            "group_id": self.candidate.group_id,
            "provenance": self.candidate.provenance.to_dict(),
            "taxonomy": {
                "observation_type": self.candidate.observation_type.value,
                "scenario_type": self.candidate.scenario_type.value,
                "domain": self.candidate.domain,
                "difficulty": self.candidate.difficulty.value,
            },
            "state": context,
            "observation_text": self.realization.observation_text,
            "target": target,
            "oracle": {"target_belief_state": oracle},
            "generation": {
                "generator": self.generator,
                "prompt_version": self.prompt_version,
                "request_hash": self.realization.request_hash,
                "rule_version": self.rule_version,
                "usage": dict(self.realization.usage),
            },
            "validation": {
                "accepted": self.validation.accepted,
                "reason_codes": list(self.validation.reason_codes),
            },
        }
