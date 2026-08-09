from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ..core.types import RevisionContext, RevisionOutput


@dataclass(frozen=True)
class SFTRecord:
    scenario_id: str
    context: RevisionContext
    target: RevisionOutput
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target"]["decision"] = self.target.decision.value
        for index, patch in enumerate(self.target.patches):
            result["target"]["patches"][index]["op"] = patch.op.value
        result["context"]["risk"]["risk"] = self.context.risk.risk.value
        for slot in result["context"]["belief_state"]["slots"]:
            status = slot["status"]
            slot["status"] = status.value if hasattr(status, "value") else status
        return result


_INTERNAL_OBSERVATION_FIELDS = {"condition", "relevant", "perturbation"}


def _visible_observation(value: Any) -> dict[str, Any]:
    raw = asdict(value)
    return {
        key: item
        for key, item in raw.items()
        if key not in _INTERNAL_OBSERVATION_FIELDS and item is not None
    }


def context_prompt_payload(context: RevisionContext) -> dict[str, Any]:
    """Return only bounded fields available to the policy at deployment time."""
    return {
        "belief_state": {
            "max_slots": context.belief_state.max_slots,
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
                for slot in context.belief_state.slots
            ],
        },
        "goal": context.goal,
        "observation": _visible_observation(context.observation),
        "risk": {
            "active_subgoal": context.risk.active_subgoal,
            "dependent_fields": list(context.risk.dependent_fields),
            "risk": context.risk.risk.value,
            "reversible": context.risk.reversible,
        },
        "ledger": {
            "capacity": context.ledger.capacity,
            "entries": [
                {
                    "observation": _visible_observation(entry.observation),
                    "reason": entry.reason,
                }
                for entry in context.ledger.entries
            ],
        },
        "budget": asdict(context.budget),
    }


def context_summary(context: RevisionContext) -> str:
    """Serialize the deployment-visible policy context as deterministic JSON."""
    return json.dumps(
        context_prompt_payload(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )