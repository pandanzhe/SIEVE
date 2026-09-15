from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sieve.core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    SlotStatus,
    VerificationRequest,
)
from sieve.environments.scenario_revision_env import ScenarioRevisionEnvironment
from sieve.rl_data.io import read_scenarios
from sieve.rl_data.schema import RLScenario
from sieve.training.rewards import zero_costs

_CATEGORIES = ("all", "commerce", "service", "workflow")


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _update_current_observation(context: RevisionContext) -> RevisionOutput:
    observation = context.observation
    slot = context.belief_state.get(observation.field_id)
    operation = PatchOp.SET_VALUE if slot is not None else PatchOp.ADD_FIELD
    return RevisionOutput(
        decision=Decision.UPDATE,
        affected_fields=(observation.field_id,),
        patches=(Patch(operation, observation.field_id, observation.value),),
    )


def _ignore() -> RevisionOutput:
    return RevisionOutput(decision=Decision.IGNORE)


def ignore_all_policy(context: RevisionContext) -> RevisionOutput:
    """Conservative baseline: never admit observations into executable state."""
    return _ignore()


def _hold_for_verification(context: RevisionContext) -> RevisionOutput:
    observation = context.observation
    patches: tuple[Patch, ...] = ()
    if context.belief_state.get(observation.field_id) is not None:
        patches = (
            Patch(PatchOp.SET_STATUS, observation.field_id, SlotStatus.PENDING.value),
        )
    verification = None
    if context.budget.verification_remaining > 0 and context.budget.tool_remaining > 0:
        verification = VerificationRequest(
            tool=f"verify_{observation.field_id}",
            field_id=observation.field_id,
        )
    return RevisionOutput(
        decision=Decision.HOLD,
        affected_fields=(observation.field_id,),
        patches=patches,
        verification=verification,
    )


def hold_all_policy(context: RevisionContext) -> RevisionOutput:
    """Conservative baseline: defer every observation for verification."""
    return _hold_for_verification(context)


def rule_gate_policy(context: RevisionContext) -> RevisionOutput:
    """Diagnostic heuristic gate with generator-internal labels.

    This is a near-oracle diagnostic upper bound because it uses fields such as
    ``condition`` and ``relevant`` that are intentionally hidden from model
    prompts.
    """
    observation = context.observation
    current = context.belief_state.get(observation.field_id)
    if not observation.relevant:
        return _ignore()
    if current is not None and observation.entity != current.entity:
        return _ignore()
    if current is not None and observation.observed_at < current.observed_at:
        return _ignore()
    if observation.condition in {"outside_active_entity", "stale_conflict"}:
        return _ignore()
    if observation.authenticated is True and observation.source_authority == "primary_record":
        return _update_current_observation(context)
    if observation.source == "official_verification_api":
        return _update_current_observation(context)
    return _hold_for_verification(context)


def visible_rule_gate_policy(context: RevisionContext) -> RevisionOutput:
    """Fair heuristic gate restricted to deployment-visible observation fields."""
    observation = context.observation
    current = context.belief_state.get(observation.field_id)
    if current is not None and observation.entity != current.entity:
        return _ignore()
    if current is not None and observation.observed_at < current.observed_at:
        return _ignore()
    if observation.source == "official_verification_api":
        return _update_current_observation(context)
    if observation.authenticated is True and observation.source_authority == "primary_record":
        return _update_current_observation(context)
    if context.budget.verification_remaining > 0 and context.budget.tool_remaining > 0:
        return _hold_for_verification(context)
    return _ignore()


def auto_write_policy(context: RevisionContext) -> RevisionOutput:
    """Naive baseline: admit every observed value into executable state."""
    return _update_current_observation(context)


def totals_to_metrics(totals: Any) -> dict[str, float]:
    values = np.asarray(totals, dtype=np.float64)
    if values.shape != (10,):
        raise ValueError("baseline evaluation totals must contain ten values")
    episodes = max(values[0], 1.0)
    actions = max(values[3], 1.0)
    return {
        "episodes": float(values[0]),
        "success_rate": float(values[1] / episodes),
        "mean_return": float(values[2] / episodes),
        "parse_rate": float(values[4] / actions),
        "false_update_rate": float(values[5] / actions),
        "invalid_format_rate": float(values[6] / actions),
        "verification_rate": float(values[7] / actions),
        "stall_rate": float(values[8] / actions),
        "invalid_patch_rate": float(values[9] / actions),
    }


def _evaluate_scenario(
    scenario: RLScenario,
    *,
    policy_fn: Callable[[RevisionContext], RevisionOutput],
    gamma: float,
    max_episode_steps: int,
) -> np.ndarray:
    environment = ScenarioRevisionEnvironment([scenario], gamma=gamma)
    context = environment.reset(scenario.scenario_id, seed=0)
    values = np.zeros(10, dtype=np.float64)
    episode_return = 0.0
    discount = 1.0
    for _step_index in range(max_episode_steps):
        output = policy_fn(context)
        transition = environment.step(output, invalid_format=False, token_cost=0)
        episode_return += discount * transition.reward
        discount *= gamma
        values[3] += 1.0
        values[4] += 1.0
        values[5] += transition.costs.get("false_update", 0.0)
        values[6] += transition.costs.get("invalid_format", 0.0)
        values[7] += transition.costs.get("verification", 0.0)
        values[8] += transition.costs.get("stall", 0.0)
        values[9] += transition.costs.get("invalid_patch", 0.0)
        if transition.terminated:
            values[1] = float(transition.info["success"])
            break
        if transition.next_context is None:
            raise RuntimeError("non-terminal baseline step has no context")
        context = transition.next_context
    values[0] = 1.0
    values[2] = episode_return
    return values


def evaluate_baseline(
    scenarios: list[RLScenario],
    *,
    baseline: str,
    gamma: float,
    max_episode_steps: int,
) -> dict[str, Any]:
    policies: dict[str, Callable[[RevisionContext], RevisionOutput]] = {
        "rule_gate": rule_gate_policy,
        "visible_rule_gate": visible_rule_gate_policy,
        "auto_write": auto_write_policy,
        "update_all": auto_write_policy,
        "ignore_all": ignore_all_policy,
        "always_ignore": ignore_all_policy,
        "hold_all": hold_all_policy,
        "always_hold": hold_all_policy,
    }
    if baseline not in policies:
        raise ValueError(f"unknown baseline: {baseline}")
    policy_fn = policies[baseline]
    local_totals = np.zeros((len(_CATEGORIES), 10), dtype=np.float64)
    category_index = {name: index for index, name in enumerate(_CATEGORIES)}
    started_at = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        values = _evaluate_scenario(
            scenario,
            policy_fn=policy_fn,
            gamma=gamma,
            max_episode_steps=max_episode_steps,
        )
        local_totals[category_index["all"]] += values
        local_totals[category_index[scenario.macro_domain]] += values
        if index == 1 or index == len(scenarios) or index % 50 == 0:
            elapsed = time.time() - started_at
            progress = index / max(len(scenarios), 1)
            eta = elapsed / progress - elapsed if progress > 0 else 0.0
            print(
                "[baseline-eval] progress "
                f"{index}/{len(scenarios)} "
                f"elapsed={_format_seconds(elapsed)} "
                f"eta={_format_seconds(eta)}",
                flush=True,
            )
    return {
        "baseline": baseline,
        "metrics": {
            category: totals_to_metrics(local_totals[index])
            for index, category in enumerate(_CATEGORIES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 rule baselines.")
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument(
        "--baseline",
        choices=(
            "rule_gate",
            "visible_rule_gate",
            "auto_write",
            "update_all",
            "ignore_all",
            "always_ignore",
            "hold_all",
            "always_hold",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--max-episode-steps", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    scenarios = read_scenarios(args.scenario_file)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    report = {
        "scenario_file": str(Path(args.scenario_file).resolve()),
        "scenario_count": len(scenarios),
        "gamma": args.gamma,
        "max_episode_steps": args.max_episode_steps,
        **evaluate_baseline(
            scenarios,
            baseline=args.baseline,
            gamma=args.gamma,
            max_episode_steps=args.max_episode_steps,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = report["metrics"]["all"]
    print(
        "[baseline-eval] done "
        f"baseline={args.baseline} "
        f"success={metrics['success_rate']:.4f} "
        f"parse={metrics['parse_rate']:.4f} "
        f"return={metrics['mean_return']:.4f} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
