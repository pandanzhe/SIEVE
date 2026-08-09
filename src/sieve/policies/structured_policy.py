from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    SlotStatus,
    VerificationRequest,
)
from .features import HashedFeaturizer


DECISIONS = (Decision.UPDATE, Decision.HOLD, Decision.IGNORE)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class StructuredPolicy:
    """Factorized NumPy policy used by the portable research harness."""

    def __init__(self, feature_dim: int, max_slots: int, seed: int = 0, temperature: float = 1.0) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feature_dim = feature_dim
        self.max_slots = max_slots
        self.temperature = temperature
        self.featurizer = HashedFeaturizer(feature_dim)
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.w_decision = rng.normal(0.0, scale, (3, feature_dim))
        self.b_decision = np.zeros(3)
        self.w_field = rng.normal(0.0, scale, (max_slots, feature_dim))
        self.b_field = np.zeros(max_slots)
        self.w_verify = rng.normal(0.0, scale, (2, feature_dim))
        self.b_verify = np.zeros(2)

    def clone(self) -> "StructuredPolicy":
        return deepcopy(self)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w_decision": self.w_decision,
            "b_decision": self.b_decision,
            "w_field": self.w_field,
            "b_field": self.b_field,
            "w_verify": self.w_verify,
            "b_verify": self.b_verify,
        }

    def decision_logits(self, context: RevisionContext) -> np.ndarray:
        x = self.featurizer.encode(context)
        return (self.w_decision @ x + self.b_decision) / self.temperature

    def field_logits(self, context: RevisionContext) -> np.ndarray:
        x = self.featurizer.encode(context)
        logits = self.w_field @ x + self.b_field
        valid = len(context.belief_state.slots) + int(len(context.belief_state.slots) < self.max_slots)
        logits[valid:] = -1e9
        return logits

    def verify_logits(self, context: RevisionContext) -> np.ndarray:
        x = self.featurizer.encode(context)
        logits = self.w_verify @ x + self.b_verify
        if context.budget.verification_remaining <= 0 or context.budget.tool_remaining <= 0:
            logits[1] = -1e9
        return logits

    def predict(self, context: RevisionContext, greedy: bool = True) -> RevisionOutput:
        output, _ = self.sample_with_trace(context, greedy=greedy)
        return output

    def sample_with_trace(self, context: RevisionContext, greedy: bool = False) -> tuple[RevisionOutput, dict[str, Any]]:
        x = self.featurizer.encode(context)
        decision_probs = softmax(self.decision_logits(context))
        decision_index = int(np.argmax(decision_probs) if greedy else np.random.choice(3, p=decision_probs))
        decision = DECISIONS[decision_index]
        trace: dict[str, Any] = {"x": x, "decision": decision_index, "decision_probs": decision_probs}
        if decision is Decision.IGNORE:
            return RevisionOutput(decision), trace

        field_probs = softmax(self.field_logits(context))
        field_index = int(np.argmax(field_probs) if greedy else np.random.choice(self.max_slots, p=field_probs))
        trace.update({"field": field_index, "field_probs": field_probs})
        slots = context.belief_state.slots
        field_id = slots[field_index].id if field_index < len(slots) else context.observation.field_id
        affected = (field_id,)

        if decision is Decision.UPDATE:
            op = PatchOp.SET_VALUE if context.belief_state.get(field_id) else PatchOp.ADD_FIELD
            patch = Patch(op, field_id, context.observation.value)
            return RevisionOutput(decision, affected, (patch,)), trace

        patches = ()
        if context.belief_state.get(field_id):
            patches = (Patch(PatchOp.SET_STATUS, field_id, SlotStatus.PENDING.value),)
        verify_probs = softmax(self.verify_logits(context))
        verify_index = int(np.argmax(verify_probs) if greedy else np.random.choice(2, p=verify_probs))
        trace.update({"verify": verify_index, "verify_probs": verify_probs})
        request = None
        if verify_index == 1:
            request = VerificationRequest("official_payment_api", field_id)
        return RevisionOutput(decision, affected, patches, request), trace

    def output_indices(self, context: RevisionContext, output: RevisionOutput) -> dict[str, int]:
        indices = {"decision": DECISIONS.index(output.decision)}
        if output.decision is not Decision.IGNORE:
            field_id = output.affected_fields[0]
            field_index = next(
                (i for i, slot in enumerate(context.belief_state.slots) if slot.id == field_id),
                min(len(context.belief_state.slots), self.max_slots - 1),
            )
            indices["field"] = field_index
        if output.decision is Decision.HOLD:
            indices["verify"] = int(output.verification is not None)
        return indices

    def log_probability(self, context: RevisionContext, output: RevisionOutput) -> float:
        indices = self.output_indices(context, output)
        total = np.log(softmax(self.decision_logits(context))[indices["decision"]] + 1e-12)
        if "field" in indices:
            total += np.log(softmax(self.field_logits(context))[indices["field"]] + 1e-12)
        if "verify" in indices:
            total += np.log(softmax(self.verify_logits(context))[indices["verify"]] + 1e-12)
        return float(total)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            feature_dim=np.array(self.feature_dim),
            max_slots=np.array(self.max_slots),
            temperature=np.array(self.temperature),
            **self.parameters(),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StructuredPolicy":
        data = np.load(Path(path), allow_pickle=False)
        policy = cls(int(data["feature_dim"]), int(data["max_slots"]), temperature=float(data["temperature"]))
        for name, parameter in policy.parameters().items():
            parameter[...] = data[name]
        return policy
