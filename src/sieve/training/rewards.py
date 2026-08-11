from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.types import BeliefState, SlotStatus


def belief_consistency(
    state: BeliefState, oracle: Mapping[str, object], dependent_fields: Sequence[str]
) -> float:
    if not dependent_fields:
        return 1.0
    correct = 0
    for field_id in dependent_fields:
        slot = state.get(field_id)
        correct += int(
            slot is not None
            and slot.status is SlotStatus.TRUSTED
            and field_id in oracle
            and slot.value == oracle[field_id]
        )
    return correct / len(dependent_fields)


def potential_difference(
    previous: BeliefState,
    current: BeliefState,
    oracle: Mapping[str, object],
    dependent_fields: Sequence[str],
    gamma: float,
) -> float:
    return gamma * belief_consistency(current, oracle, dependent_fields) - belief_consistency(
        previous, oracle, dependent_fields
    )


def zero_costs() -> dict[str, float]:
    return {
        "false_update": 0.0,
        "unsafe_action": 0.0,
        "verification": 0.0,
        "stall": 0.0,
        "invalid_format": 0.0,
        "invalid_patch": 0.0,
        "collateral_edit": 0.0,
        "budget_violation": 0.0,
    }
