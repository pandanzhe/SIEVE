from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ..core.executor import StateExecutor
from ..core.types import (
    BeliefState,
    Budget,
    Decision,
    EvidenceLedger,
    RevisionContext,
    RevisionOutput,
    PatchOp,
)
from ..data.schema import SFTRecord
from ..training.rewards import belief_consistency, potential_difference, zero_costs


@dataclass(frozen=True)
class RevisionStep:
    next_context: RevisionContext | None
    reward: float
    costs: dict[str, float]
    terminated: bool
    info: dict[str, object]


class RevisionRLEnvironment(Protocol):
    @property
    def scenario_ids(self) -> tuple[str, ...]: ...

    def reset(self, scenario_id: str, seed: int) -> RevisionContext: ...

    def step(self, output: RevisionOutput) -> RevisionStep: ...


class ReplayRevisionEnvironment:
    """Event-level RL environment backed by ordered synthetic trajectories.

    This adapter is deterministic for a scenario and keeps oracle state internal.
    A live benchmark adapter can implement the same protocol with real env.step calls.
    """

    def __init__(self, records: Sequence[SFTRecord], gamma: float = 0.97) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        grouped: dict[str, list[SFTRecord]] = defaultdict(list)
        for record in records:
            grouped[record.scenario_id].append(record)
        self._scenarios = {
            scenario_id: tuple(sorted(items, key=lambda item: item.step_index))
            for scenario_id, items in grouped.items()
        }
        self._oracles: dict[str, dict[str, object]] = {}
        for scenario_id, items in self._scenarios.items():
            oracle = {
                patch.field_id: patch.value
                for item in items
                if item.target.decision is Decision.UPDATE
                for patch in item.target.patches
                if patch.op in {PatchOp.ADD_FIELD, PatchOp.SET_VALUE}
            }
            if not oracle:
                raise ValueError(f"scenario {scenario_id} has no oracle update target")
            self._oracles[scenario_id] = oracle
            steps = [item.step_index for item in items]
            if steps != list(range(len(items))):
                raise ValueError(f"scenario {scenario_id} must have contiguous step_index values")
        if not self._scenarios:
            raise ValueError("replay environment requires at least one scenario")
        self.gamma = gamma
        self.executor = StateExecutor()
        self._records: tuple[SFTRecord, ...] = ()
        self._index = 0
        self._state: BeliefState | None = None
        self._ledger: EvidenceLedger | None = None
        self._budget: Budget | None = None
        self._oracle: dict[str, object] = {}

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._scenarios))

    def reset(self, scenario_id: str, seed: int) -> RevisionContext:
        del seed  # replay is deterministic; live adapters may use it
        if scenario_id not in self._scenarios:
            raise KeyError(f"unknown scenario: {scenario_id}")
        self._records = self._scenarios[scenario_id]
        self._index = 0
        first = self._records[0]
        self._state = first.context.belief_state.clone()
        self._ledger = first.context.ledger.clone()
        self._budget = replace(first.context.budget)
        self._oracle = self._oracles[scenario_id]
        return self._current_context()

    def _current_context(self) -> RevisionContext:
        if self._state is None or self._ledger is None or self._budget is None:
            raise RuntimeError("environment must be reset before use")
        record = self._records[self._index]
        return replace(
            record.context,
            belief_state=self._state,
            ledger=self._ledger,
            budget=replace(self._budget),
        )

    def step(self, output: RevisionOutput) -> RevisionStep:
        context = self._current_context()
        record = self._records[self._index]
        previous = context.belief_state.clone()
        result = self.executor.apply(
            context.belief_state, context.ledger, output, context.observation
        )
        self._state, self._ledger = result.state, result.ledger
        reward = potential_difference(
            previous,
            self._state,
            self._oracle,
            context.risk.dependent_fields,
            self.gamma,
        )
        costs = zero_costs()
        costs.update(result.costs)
        invalid_admission = (
            record.target.decision is not Decision.UPDATE and output.decision is Decision.UPDATE
        )
        costs["false_update"] = float(invalid_admission)
        costs["unsafe_action"] = float(invalid_admission and not context.risk.reversible)
        if self._budget is None:
            raise RuntimeError("missing environment budget")
        requested_verification = output.verification is not None
        can_verify = (
            requested_verification
            and self._budget.verification_remaining > 0
            and self._budget.tool_remaining > 0
        )
        costs["verification"] = float(can_verify)
        costs["budget_violation"] = float(requested_verification and not can_verify)
        costs["stall"] = float(
            output.decision is Decision.HOLD and not can_verify
        )
        self._budget.steps_remaining = max(0, self._budget.steps_remaining - 1)
        if can_verify:
            self._budget.verification_remaining -= 1
            self._budget.tool_remaining -= 1

        self._index += 1
        terminated = (
            self._index >= len(self._records) or self._budget.steps_remaining <= 0
        )
        success = False
        next_context = None
        if terminated:
            success = belief_consistency(
                self._state,
                self._oracle,
                context.risk.dependent_fields,
            ) == 1.0
            reward += float(success)
        else:
            next_context = self._current_context()
        return RevisionStep(
            next_context=next_context,
            reward=reward,
            costs=costs,
            terminated=terminated,
            info={
                "success": success,
                "budget_exhausted": self._budget.steps_remaining <= 0,
                "executor_error": result.error,
            },
        )
