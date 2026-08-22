from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..core.executor import StateExecutor
from ..core.types import (
    BeliefState,
    Budget,
    Decision,
    EvidenceLedger,
    Observation,
    RevisionContext,
    RevisionOutput,
)
from ..rl_data.schema import RLScenario, ScenarioEvent
from ..training.rewards import (
    belief_consistency,
    oracle_gap,
    potential_difference,
    zero_costs,
)
from .revision_env import RevisionStep


class ScenarioRevisionEnvironment:
    """Action-conditioned Stage-2 environment over private scenario events.

    The policy receives only :class:`RevisionContext`. Expected decisions and the
    oracle remain private and are used solely for transition costs and rewards.
    """

    def __init__(self, scenarios: Sequence[RLScenario], gamma: float = 0.97) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not scenarios:
            raise ValueError("scenario environment requires at least one scenario")
        self._scenarios = {scenario.scenario_id: scenario for scenario in scenarios}
        if len(self._scenarios) != len(scenarios):
            raise ValueError("scenario ids must be unique")
        self.gamma = gamma
        self.executor = StateExecutor()
        self._scenario: RLScenario | None = None
        self._event_index = 0
        self._verification_observation: Observation | None = None
        self._state: BeliefState | None = None
        self._ledger: EvidenceLedger | None = None
        self._budget: Budget | None = None

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._scenarios))

    def reset(self, scenario_id: str, seed: int) -> RevisionContext:
        del seed  # Scenarios are deterministic; sampling happens in the policy.
        try:
            scenario = self._scenarios[scenario_id]
        except KeyError as error:
            raise KeyError(f"unknown scenario: {scenario_id}") from error
        self._scenario = scenario
        self._event_index = 0
        self._verification_observation = None
        self._state = scenario.initial_state.clone()
        self._ledger = scenario.empty_ledger()
        self._budget = replace(scenario.budget)
        return self._current_context()

    def _require_runtime(self) -> tuple[RLScenario, BeliefState, EvidenceLedger, Budget]:
        if (
            self._scenario is None
            or self._state is None
            or self._ledger is None
            or self._budget is None
        ):
            raise RuntimeError("environment must be reset before use")
        return self._scenario, self._state, self._ledger, self._budget

    def _base_event(self) -> ScenarioEvent:
        scenario, _, _, _ = self._require_runtime()
        if self._event_index >= len(scenario.events):
            raise RuntimeError("episode has already terminated")
        return scenario.events[self._event_index]

    def _current_observation(self) -> Observation:
        return self._verification_observation or self._base_event().observation

    def _expected_decision(self) -> str:
        if self._verification_observation is not None:
            return Decision.UPDATE.value
        return self._base_event().expected_decision

    def _current_context(self) -> RevisionContext:
        scenario, state, ledger, budget = self._require_runtime()
        return RevisionContext(
            belief_state=state.clone(),
            observation=self._current_observation(),
            goal=scenario.goal,
            risk=scenario.risk,
            budget=replace(budget),
            ledger=ledger.clone(),
        )

    def step(
        self,
        output: RevisionOutput,
        *,
        invalid_format: bool = False,
        token_cost: int = 0,
    ) -> RevisionStep:
        if (
            isinstance(token_cost, bool)
            or not isinstance(token_cost, int)
            or token_cost < 0
        ):
            raise ValueError("token_cost must be a non-negative integer")
        scenario, state, ledger, budget = self._require_runtime()
        context = self._current_context()
        event = self._base_event()
        was_verification = self._verification_observation is not None
        expected_decision = self._expected_decision()
        previous = state.clone()
        result = self.executor.apply(state, ledger, output, context.observation)
        self._state, self._ledger = result.state, result.ledger

        reward = potential_difference(
            previous,
            self._state,
            scenario.oracle_state,
            scenario.risk.dependent_fields,
            self.gamma,
        )
        costs = zero_costs()
        costs.update(result.costs)
        costs["invalid_format"] = float(invalid_format)
        false_update = (
            output.decision is Decision.UPDATE
            and expected_decision != Decision.UPDATE.value
        )
        costs["false_update"] = float(false_update)
        costs["unsafe_action"] = float(false_update and not scenario.risk.reversible)

        requested_verification = (
            output.decision is Decision.HOLD and output.verification is not None
        )
        available_evidence = (
            not was_verification and event.verification_observation is not None
        )
        request_matches_evidence = bool(
            requested_verification
            and available_evidence
            and output.verification is not None
            and event.verification_tool is not None
            and output.verification.tool == event.verification_tool
            and output.verification.field_id
            == event.verification_observation.field_id
            and output.verification.field_id == context.observation.field_id
        )
        can_verify = (
            requested_verification
            and available_evidence
            and request_matches_evidence
            and budget.verification_remaining > 0
            and budget.tool_remaining > 0
            and result.executed
        )
        costs["verification"] = float(can_verify)
        costs["budget_violation"] = float(
            (
                requested_verification
                and (
                    budget.verification_remaining <= 0
                    or budget.tool_remaining <= 0
                )
            )
            or token_cost > budget.tokens_remaining
        )
        costs["stall"] = float(output.decision is Decision.HOLD and not can_verify)
        invalid_tool = bool(
            requested_verification
            and available_evidence
            and not request_matches_evidence
        )
        if invalid_tool:
            costs["invalid_patch"] = 1.0

        if not invalid_format and result.executed:
            if output.decision.value == expected_decision:
                reward += 0.10
            else:
                reward -= 0.10
            if expected_decision == Decision.HOLD.value:
                if can_verify:
                    reward += 0.30
                elif output.decision is Decision.IGNORE:
                    reward -= 0.25
                elif output.decision is Decision.UPDATE:
                    reward -= 0.35
                elif output.decision is Decision.HOLD:
                    reward -= 0.20
            elif expected_decision == Decision.IGNORE.value and output.decision is Decision.HOLD:
                reward -= 0.15
            elif expected_decision == Decision.UPDATE.value and output.decision is Decision.IGNORE:
                reward -= 0.20

        budget.steps_remaining = max(0, budget.steps_remaining - 1)
        budget.tokens_remaining = max(0, budget.tokens_remaining - token_cost)
        transition = "next_event"
        if can_verify:
            budget.verification_remaining -= 1
            budget.tool_remaining -= 1
            self._verification_observation = event.verification_observation
            transition = "verification"
        else:
            if was_verification:
                self._verification_observation = None
            self._event_index += 1

        terminated = (
            self._event_index >= len(scenario.events)
            or budget.steps_remaining <= 0
            or budget.tokens_remaining <= 0
        )
        success = False
        next_context = None
        if terminated:
            success = belief_consistency(
                self._state,
                scenario.oracle_state,
                scenario.risk.dependent_fields,
            ) == 1.0
            reward += float(success)
            reward -= oracle_gap(
                self._state,
                scenario.oracle_state,
                scenario.risk.dependent_fields,
            )
            transition = "terminal"
        else:
            next_context = self._current_context()
        return RevisionStep(
            next_context=next_context,
            reward=reward,
            costs=costs,
            terminated=terminated,
            info={
                "scenario_id": scenario.scenario_id,
                "event_kind": "verification" if was_verification else event.kind,
                "transition": transition,
                "success": success,
                "budget_exhausted": (
                    budget.steps_remaining <= 0 or budget.tokens_remaining <= 0
                ),
                "executor_error": result.error,
                "verification_error": (
                    "unknown verification tool" if invalid_tool else None
                ),
            },
        )
