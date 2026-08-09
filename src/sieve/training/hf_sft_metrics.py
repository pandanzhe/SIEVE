from __future__ import annotations

from collections.abc import Sequence


def _binary_f1(predictions: Sequence[int], targets: Sequence[int]) -> float:
    true_positive = sum(p == 1 and t == 1 for p, t in zip(predictions, targets))
    false_positive = sum(p == 1 and t == 0 for p, t in zip(predictions, targets))
    false_negative = sum(p == 0 and t == 1 for p, t in zip(predictions, targets))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def _macro_f1(predictions: Sequence[int], targets: Sequence[int], classes: int) -> float:
    scores = []
    for label in range(classes):
        scores.append(
            _binary_f1(
                [int(value == label) for value in predictions],
                [int(value == label) for value in targets],
            )
        )
    return sum(scores) / classes


def compute_structured_metrics(
    *,
    decision_predictions: Sequence[int],
    decision_targets: Sequence[int],
    affected_predictions: Sequence[Sequence[int]],
    affected_targets: Sequence[Sequence[int]],
    affected_masks: Sequence[Sequence[int]],
    verification_predictions: Sequence[int],
    verification_targets: Sequence[int],
    patch_predictions: Sequence[Sequence[Sequence[int]]],
    patch_targets: Sequence[Sequence[Sequence[int]]],
    patch_masks: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Compute Stage-1 metrics while excluding conditionally inactive branches."""
    if not decision_targets or len(decision_predictions) != len(decision_targets):
        raise ValueError("decision predictions and targets must have equal non-zero length")

    affected_pred_flat: list[int] = []
    affected_target_flat: list[int] = []
    patch_pred_flat: list[int] = []
    patch_target_flat: list[int] = []
    for pred_row, target_row, mask_row in zip(
        affected_predictions, affected_targets, affected_masks
    ):
        for prediction, target, active in zip(pred_row, target_row, mask_row):
            if active:
                affected_pred_flat.append(int(prediction))
                affected_target_flat.append(int(target))
    for pred_slots, target_slots, mask_row in zip(
        patch_predictions, patch_targets, patch_masks
    ):
        for pred_ops, target_ops, active in zip(pred_slots, target_slots, mask_row):
            if active:
                patch_pred_flat.extend(int(value) for value in pred_ops)
                patch_target_flat.extend(int(value) for value in target_ops)

    verification_pairs = [
        (int(prediction), int(target))
        for prediction, target in zip(verification_predictions, verification_targets)
        if target != -100
    ]
    non_update = [index for index, target in enumerate(decision_targets) if target != 0]
    false_updates = sum(decision_predictions[index] == 0 for index in non_update)
    return {
        "decision_accuracy": sum(
            prediction == target
            for prediction, target in zip(decision_predictions, decision_targets)
        )
        / len(decision_targets),
        "decision_macro_f1": _macro_f1(decision_predictions, decision_targets, 3),
        "false_update_rate": 0.0 if not non_update else false_updates / len(non_update),
        "affected_micro_f1": _binary_f1(affected_pred_flat, affected_target_flat),
        "verification_f1": _binary_f1(
            [pair[0] for pair in verification_pairs],
            [pair[1] for pair in verification_pairs],
        ),
        "patch_operation_micro_f1": _binary_f1(patch_pred_flat, patch_target_flat),
    }
