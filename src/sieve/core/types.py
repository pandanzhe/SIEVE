from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    UPDATE = "UPDATE"
    HOLD = "HOLD"
    IGNORE = "IGNORE"


class SlotStatus(str, Enum):
    TRUSTED = "trusted"
    PENDING = "pending_verification"
    EMPTY = "empty"


class PatchOp(str, Enum):
    ADD_FIELD = "ADD_FIELD"
    SET_VALUE = "SET_VALUE"
    SET_STATUS = "SET_STATUS"
    SET_PROVENANCE = "SET_PROVENANCE"
    SET_VALIDITY = "SET_VALIDITY"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BeliefSlot:
    id: str
    value: Any
    status: SlotStatus
    source: str
    observed_at: int
    valid_from: int | None
    entity: str


@dataclass
class BeliefState:
    max_slots: int
    slots: list[BeliefSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.slots) > self.max_slots:
            raise ValueError("belief state exceeds max_slots")
        ids = [slot.id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("belief slot ids must be unique")

    def get(self, field_id: str) -> BeliefSlot | None:
        return next((slot for slot in self.slots if slot.id == field_id), None)

    def clone(self) -> "BeliefState":
        return deepcopy(self)


@dataclass(frozen=True)
class Observation:
    field_id: str
    value: Any
    source: str
    observed_at: int
    valid_from: int | None
    entity: str
    condition: str
    relevant: bool
    perturbation: str = "clean"
    source_authority: str | None = None
    authenticated: bool | None = None


@dataclass(frozen=True)
class RiskEnvelope:
    active_subgoal: str
    dependent_fields: tuple[str, ...]
    risk: RiskLevel
    reversible: bool


@dataclass(frozen=True)
class Patch:
    op: PatchOp
    field_id: str
    value: Any


@dataclass(frozen=True)
class VerificationRequest:
    tool: str
    field_id: str


@dataclass(frozen=True)
class RevisionOutput:
    decision: Decision
    affected_fields: tuple[str, ...] = ()
    patches: tuple[Patch, ...] = ()
    verification: VerificationRequest | None = None


@dataclass(frozen=True)
class EvidenceCandidate:
    observation: Observation
    reason: str


@dataclass
class EvidenceLedger:
    capacity: int
    entries: list[EvidenceCandidate] = field(default_factory=list)

    def append(self, observation: Observation, reason: str) -> None:
        self.entries.append(EvidenceCandidate(observation, reason))
        if len(self.entries) > self.capacity:
            del self.entries[: len(self.entries) - self.capacity]

    def clone(self) -> "EvidenceLedger":
        return deepcopy(self)


@dataclass
class Budget:
    verification_remaining: int
    tool_remaining: int
    steps_remaining: int
    tokens_remaining: int = 0


@dataclass
class ExecutorResult:
    state: BeliefState
    ledger: EvidenceLedger
    executed: bool
    costs: dict[str, float]
    error: str | None = None


@dataclass(frozen=True)
class RevisionContext:
    belief_state: BeliefState
    observation: Observation
    goal: str
    risk: RiskEnvelope
    budget: Budget
    ledger: EvidenceLedger
