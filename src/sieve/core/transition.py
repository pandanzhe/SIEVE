from __future__ import annotations

from dataclasses import dataclass

from .executor import StateExecutor
from .types import BeliefState, Budget, EvidenceLedger, Observation, RevisionOutput


@dataclass(frozen=True)
class EnvironmentAction:
    kind: str
    field_id: str | None = None


class AgentEventLoop:
    def __init__(self, env, belief_state: BeliefState, ledger: EvidenceLedger, budget: Budget) -> None:
        self.env = env
        self.belief_state = belief_state
        self.ledger = ledger
        self.budget = budget
        self.revision_t = 0
        self.action_k = 0
        self.executor = StateExecutor()

    def revise(self, observation: Observation, output: RevisionOutput):
        result = self.executor.apply(self.belief_state, self.ledger, output, observation)
        self.belief_state = result.state
        self.ledger = result.ledger
        self.revision_t += 1
        self.budget.steps_remaining = max(0, self.budget.steps_remaining - 1)
        return result

    def act(self, action: EnvironmentAction) -> list[Observation]:
        if self.budget.tool_remaining <= 0:
            return []
        if action.kind == "verify" and self.budget.verification_remaining <= 0:
            return []
        self.budget.tool_remaining -= 1
        if action.kind == "verify":
            self.budget.verification_remaining -= 1
        self.action_k += 1
        return self.env.step(action)
