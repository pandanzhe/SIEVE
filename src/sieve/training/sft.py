from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.types import Decision
from ..data.schema import SFTRecord
from ..policies.structured_policy import DECISIONS, StructuredPolicy, softmax


def _clip(gradient: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(gradient))
    return gradient * (maximum / norm) if norm > maximum else gradient


def _update_head(
    weights: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    probabilities: np.ndarray,
    target: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    sample_weight: float = 1.0,
) -> float:
    gradient = probabilities.copy()
    gradient[target] -= 1.0
    gradient *= sample_weight
    gradient = _clip(gradient, gradient_clip)
    weights -= learning_rate * (np.outer(gradient, x) + weight_decay * weights)
    bias -= learning_rate * gradient
    return float(-sample_weight * np.log(probabilities[target] + 1e-12))


def train_sft(
    policy: StructuredPolicy,
    records: Sequence[SFTRecord],
    epochs: int,
    learning_rate: float,
    seed: int,
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    weight_decay: float = 0.0,
    gradient_clip: float = 5.0,
) -> list[float]:
    if not records:
        raise ValueError("SFT records must not be empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if len(class_weights) != len(DECISIONS) or any(value <= 0 for value in class_weights):
        raise ValueError("class_weights must contain three positive values")
    if weight_decay < 0 or gradient_clip <= 0:
        raise ValueError("weight_decay must be non-negative and gradient_clip positive")
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(records))
        total = 0.0
        terms = 0
        for record_index in order:
            record = records[int(record_index)]
            context, target = record.context, record.target
            x = policy.featurizer.encode(context)
            indices = policy.output_indices(context, target)
            decision_index = indices["decision"]
            total += _update_head(
                policy.w_decision, policy.b_decision, x,
                softmax(policy.decision_logits(context)), decision_index,
                learning_rate, weight_decay, gradient_clip, class_weights[decision_index],
            )
            terms += 1
            if target.decision is not Decision.IGNORE:
                total += _update_head(
                    policy.w_field, policy.b_field, x,
                    softmax(policy.field_logits(context)), indices["field"],
                    learning_rate, weight_decay, gradient_clip,
                )
                terms += 1
            if target.decision is Decision.HOLD:
                total += _update_head(
                    policy.w_verify, policy.b_verify, x,
                    softmax(policy.verify_logits(context)), indices["verify"],
                    learning_rate, weight_decay, gradient_clip,
                )
                terms += 1
        history.append(total / max(1, terms))
    return history


def evaluate_sft(policy: StructuredPolicy, records: Sequence[SFTRecord]) -> dict[str, float]:
    if not records:
        raise ValueError("evaluation records must not be empty")
    total_loss = 0.0
    terms = 0
    correct = 0
    for record in records:
        indices = policy.output_indices(record.context, record.target)
        decision_probs = softmax(policy.decision_logits(record.context))
        total_loss -= float(np.log(decision_probs[indices["decision"]] + 1e-12))
        terms += 1
        correct += int(int(np.argmax(decision_probs)) == indices["decision"])
        if record.target.decision is not Decision.IGNORE:
            field_probs = softmax(policy.field_logits(record.context))
            total_loss -= float(np.log(field_probs[indices["field"]] + 1e-12))
            terms += 1
        if record.target.decision is Decision.HOLD:
            verify_probs = softmax(policy.verify_logits(record.context))
            total_loss -= float(np.log(verify_probs[indices["verify"]] + 1e-12))
            terms += 1
    return {
        "loss": total_loss / max(1, terms),
        "decision_accuracy": correct / max(1, len(records)),
    }
