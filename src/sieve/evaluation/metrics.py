from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..core.types import Decision


def _safe_divide(numerator: float, denominator: float, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def decision_metrics(truth: Sequence[Decision], predicted: Sequence[Decision]) -> dict[str, float]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth and predicted decisions must have equal non-zero length")
    f1_values: list[float] = []
    result: dict[str, float] = {}
    for label in Decision:
        true_positive = sum(t is label and p is label for t, p in zip(truth, predicted, strict=True))
        false_positive = sum(t is not label and p is label for t, p in zip(truth, predicted, strict=True))
        false_negative = sum(t is label and p is not label for t, p in zip(truth, predicted, strict=True))
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        prefix = label.value.lower()
        result[f"{prefix}_precision"] = precision
        result[f"{prefix}_recall"] = recall
        result[f"{prefix}_f1"] = f1
        f1_values.append(f1)
    invalid_count = sum(t is not Decision.UPDATE for t in truth)
    update_count = sum(t is Decision.UPDATE for t in truth)
    result["macro_f1"] = float(np.mean(f1_values))
    result["decision_accuracy"] = float(np.mean([t is p for t, p in zip(truth, predicted, strict=True)]))
    result["false_update_rate"] = _safe_divide(
        sum(t is not Decision.UPDATE and p is Decision.UPDATE for t, p in zip(truth, predicted, strict=True)),
        invalid_count,
    )
    result["missed_update_rate"] = _safe_divide(
        sum(t is Decision.UPDATE and p is not Decision.UPDATE for t, p in zip(truth, predicted, strict=True)),
        update_count,
    )
    return result


def affected_set_f1(truth: set[str], predicted: set[str]) -> float:
    if not truth and not predicted:
        return 1.0
    precision = _safe_divide(len(truth & predicted), len(predicted))
    recall = _safe_divide(len(truth & predicted), len(truth))
    return _safe_divide(2 * precision * recall, precision + recall)


def contamination_area(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float(values[0]) if values else 0.0
    data = np.asarray(values, dtype=np.float64)
    return float(np.sum((data[:-1] + data[1:]) * 0.5))


def recovery_latency(values: Sequence[float]) -> int:
    first = next((index for index, value in enumerate(values) if value > 0), None)
    if first is None:
        return 0
    recovered = next((index for index in range(first + 1, len(values)) if values[index] <= 0), None)
    return len(values) if recovered is None else recovered - first


def pass_at_k(task_trials: Sequence[Sequence[bool]], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    eligible = [trials for trials in task_trials if len(trials) >= k]
    if not eligible:
        return 0.0
    return float(np.mean([all(trials[:k]) for trials in eligible]))


def bootstrap_mean_interval(
    values: Sequence[float], seed: int, samples: int = 2000, confidence: float = 0.95
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("bootstrap data must not be empty")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, data.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(data.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))
