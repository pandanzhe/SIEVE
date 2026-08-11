from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    EvidenceLedger,
    Observation,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
)


def _observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "field_id": observation.field_id,
        "value": observation.value,
        "source": observation.source,
        "observed_at": observation.observed_at,
        "valid_from": observation.valid_from,
        "entity": observation.entity,
        "condition": observation.condition,
        "relevant": observation.relevant,
        "perturbation": observation.perturbation,
        "source_authority": observation.source_authority,
        "authenticated": observation.authenticated,
    }


def _observation_from_dict(raw: Mapping[str, Any]) -> Observation:
    return Observation(
        field_id=str(raw["field_id"]),
        value=raw.get("value"),
        source=str(raw["source"]),
        observed_at=int(raw["observed_at"]),
        valid_from=(None if raw.get("valid_from") is None else int(raw["valid_from"])),
        entity=str(raw["entity"]),
        condition=str(raw.get("condition", "unspecified")),
        relevant=bool(raw.get("relevant", True)),
        perturbation=str(raw.get("perturbation", "clean")),
        source_authority=(
            None if raw.get("source_authority") is None else str(raw["source_authority"])
        ),
        authenticated=(
            None if raw.get("authenticated") is None else bool(raw["authenticated"])
        ),
    )


def _belief_to_dict(state: BeliefState) -> dict[str, Any]:
    return {
        "max_slots": state.max_slots,
        "slots": [
            {
                "id": slot.id,
                "value": slot.value,
                "status": slot.status.value,
                "source": slot.source,
                "observed_at": slot.observed_at,
                "valid_from": slot.valid_from,
                "entity": slot.entity,
            }
            for slot in state.slots
        ],
    }


def _belief_from_dict(raw: Mapping[str, Any]) -> BeliefState:
    return BeliefState(
        max_slots=int(raw["max_slots"]),
        slots=[
            BeliefSlot(
                id=str(slot["id"]),
                value=slot.get("value"),
                status=SlotStatus(str(slot["status"])),
                source=str(slot["source"]),
                observed_at=int(slot["observed_at"]),
                valid_from=(
                    None if slot.get("valid_from") is None else int(slot["valid_from"])
                ),
                entity=str(slot["entity"]),
            )
            for slot in raw.get("slots", [])
        ],
    )


@dataclass(frozen=True)
class ScenarioEvent:
    """One private environment event and its expected semantic response."""

    event_id: str
    kind: str
    observation: Observation
    expected_decision: str
    verification_tool: str | None = None
    verification_observation: Observation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "observation": _observation_to_dict(self.observation),
            "expected_decision": self.expected_decision,
            "verification_tool": self.verification_tool,
            "verification_observation": (
                None
                if self.verification_observation is None
                else _observation_to_dict(self.verification_observation)
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioEvent":
        verification = raw.get("verification_observation")
        return cls(
            event_id=str(raw["event_id"]),
            kind=str(raw["kind"]),
            observation=_observation_from_dict(raw["observation"]),
            expected_decision=str(raw["expected_decision"]),
            verification_tool=(
                None
                if raw.get("verification_tool") is None
                else str(raw["verification_tool"])
            ),
            verification_observation=(
                None if verification is None else _observation_from_dict(verification)
            ),
        )


@dataclass(frozen=True)
class RLScenario:
    """An episode definition with policy-visible and environment-private parts."""

    scenario_id: str
    base_task_id: str
    variant_id: int
    split: str
    domain: str
    macro_domain: str
    provenance: dict[str, Any]
    initial_state: BeliefState
    goal: str
    risk: RiskEnvelope
    budget: Budget
    ledger_capacity: int
    events: tuple[ScenarioEvent, ...]
    oracle_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "base_task_id": self.base_task_id,
            "variant_id": self.variant_id,
            "split": self.split,
            "domain": self.domain,
            "macro_domain": self.macro_domain,
            "provenance": self.provenance,
            "initial_context": {
                "belief_state": _belief_to_dict(self.initial_state),
                "goal": self.goal,
                "risk": {
                    "active_subgoal": self.risk.active_subgoal,
                    "dependent_fields": list(self.risk.dependent_fields),
                    "risk": self.risk.risk.value,
                    "reversible": self.risk.reversible,
                },
                "budget": {
                    "verification_remaining": self.budget.verification_remaining,
                    "tool_remaining": self.budget.tool_remaining,
                    "steps_remaining": self.budget.steps_remaining,
                    "tokens_remaining": self.budget.tokens_remaining,
                },
                "ledger": {"capacity": self.ledger_capacity, "entries": []},
            },
            "environment_private": {
                "oracle_state": self.oracle_state,
                "events": [event.to_dict() for event in self.events],
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RLScenario":
        context = raw["initial_context"]
        risk = context["risk"]
        budget = context["budget"]
        private = raw["environment_private"]
        return cls(
            scenario_id=str(raw["scenario_id"]),
            base_task_id=str(raw["base_task_id"]),
            variant_id=int(raw["variant_id"]),
            split=str(raw["split"]),
            domain=str(raw["domain"]),
            macro_domain=str(raw["macro_domain"]),
            provenance=dict(raw.get("provenance", {})),
            initial_state=_belief_from_dict(context["belief_state"]),
            goal=str(context["goal"]),
            risk=RiskEnvelope(
                active_subgoal=str(risk["active_subgoal"]),
                dependent_fields=tuple(str(item) for item in risk["dependent_fields"]),
                risk=RiskLevel(str(risk["risk"])),
                reversible=bool(risk["reversible"]),
            ),
            budget=Budget(
                verification_remaining=int(budget["verification_remaining"]),
                tool_remaining=int(budget["tool_remaining"]),
                steps_remaining=int(budget["steps_remaining"]),
                tokens_remaining=int(budget.get("tokens_remaining", 0)),
            ),
            ledger_capacity=int(context["ledger"]["capacity"]),
            events=tuple(ScenarioEvent.from_dict(item) for item in private["events"]),
            oracle_state=dict(private["oracle_state"]),
        )

    def empty_ledger(self) -> EvidenceLedger:
        return EvidenceLedger(capacity=self.ledger_capacity)
