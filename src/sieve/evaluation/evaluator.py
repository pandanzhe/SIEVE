from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..core.executor import StateExecutor
from ..core.types import Decision
from ..data.schema import SFTRecord
from ..environments.revision_env import ReplayRevisionEnvironment, RevisionRLEnvironment
from ..policies.protocol import RevisionPolicy
from .metrics import affected_set_f1, bootstrap_mean_interval, decision_metrics


def evaluate_environment(
    policy: RevisionPolicy,
    environment: RevisionRLEnvironment,
    seed: int = 0,
) -> list[float]:
    """Return one terminal success value per scenario without updating the policy."""
    successes: list[float] = []
    for scenario_offset, scenario_id in enumerate(environment.scenario_ids):
        context = environment.reset(scenario_id, seed + scenario_offset)
        while True:
            output = policy.predict(context, greedy=True)
            step = environment.step(output)
            if step.terminated:
                successes.append(float(bool(step.info.get("success", False))))
                break
            if step.next_context is None:
                raise RuntimeError("non-terminal environment step must return next_context")
            context = step.next_context
    return successes


def evaluate_policy(
    policy: RevisionPolicy,
    records: Sequence[SFTRecord],
    seed: int = 0,
) -> dict[str, float]:
    if not records:
        raise ValueError("evaluation records must not be empty")
    truth: list[Decision] = []
    predicted: list[Decision] = []
    affected_scores: list[float] = []
    patch_validity: list[float] = []
    verification_calls: list[float] = []
    executor = StateExecutor()

    for record in records:
        output = policy.predict(record.context, greedy=True)
        result = executor.apply(
            record.context.belief_state,
            record.context.ledger,
            output,
            record.context.observation,
        )
        truth.append(record.target.decision)
        predicted.append(output.decision)
        affected_scores.append(
            affected_set_f1(set(record.target.affected_fields), set(output.affected_fields))
        )
        patch_validity.append(float(result.executed))
        verification_calls.append(float(output.verification is not None))

    metrics = decision_metrics(truth, predicted)
    metrics.update(
        {
            "affected_field_f1": float(np.mean(affected_scores)),
            "patch_validity": float(np.mean(patch_validity)),
            "verification_calls_per_step": float(np.mean(verification_calls)),
        }
    )

    successes = evaluate_environment(
        policy,
        ReplayRevisionEnvironment(records),
        seed=seed,
    )
    metrics["task_success_rate"] = float(np.mean(successes))
    mean, low, high = bootstrap_mean_interval(successes, seed=seed, samples=1000)
    metrics.update(
        {
            "task_success_mean": mean,
            "task_success_ci_low": low,
            "task_success_ci_high": high,
            "pass_at_1": metrics["task_success_rate"],
        }
    )
    return metrics


def write_evaluation(metrics: dict[str, float], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# SIEVE evaluation", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.6f} |" for key, value in sorted(metrics.items()))
    (path / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
