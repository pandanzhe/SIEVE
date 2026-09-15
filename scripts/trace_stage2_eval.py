from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from sieve.config import load_config, resolve_repo_path
from sieve.core.types import Decision, RevisionContext, RevisionOutput
from sieve.environments.scenario_revision_env import ScenarioRevisionEnvironment
from sieve.policies.hf_revision_io import (
    render_context_prompt,
    revision_output_payload,
    safe_parse_revision_output,
)
from sieve.rl_data.io import read_scenarios
from sieve.rl_data.schema import RLScenario
from sieve.training.hf_grpo import _generate_completion, load_stage1_policy
from sieve.training.hf_grpo_config import parse_hf_grpo_config

try:
    from evaluate_stage2_baselines import (
        auto_write_policy,
        rule_gate_policy,
        visible_rule_gate_policy,
    )
except ImportError:  # pragma: no cover - supports direct execution from repo root.
    from scripts.evaluate_stage2_baselines import (
        auto_write_policy,
        rule_gate_policy,
        visible_rule_gate_policy,
    )


def _slot_map(slots: list[dict[str, Any]]) -> dict[str, Any]:
    return {slot["id"]: slot.get("value") for slot in slots}


def _belief_slots(state: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": slot.id,
            "value": slot.value,
            "status": slot.status.value,
            "source": slot.source,
            "observed_at": slot.observed_at,
            "entity": slot.entity,
        }
        for slot in state.slots
    ]


def _missing_or_wrong(final_slots: list[dict[str, Any]], oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = _slot_map(final_slots)
    result: dict[str, dict[str, Any]] = {}
    for field_id, expected in oracle.items():
        actual = values.get(field_id)
        if actual != expected:
            result[field_id] = {"expected": expected, "actual": actual}
    return result


def _trace_scenario(
    scenario: RLScenario,
    *,
    policy_fn: Callable[[RevisionContext, int], tuple[RevisionOutput, bool, str | None, str]],
    gamma: float,
    max_episode_steps: int,
    seed: int,
) -> dict[str, Any]:
    environment = ScenarioRevisionEnvironment([scenario], gamma=gamma)
    context = environment.reset(scenario.scenario_id, seed)
    steps: list[dict[str, Any]] = []
    costs = Counter()
    episode_return = 0.0
    discount = 1.0
    success = False
    for step_index in range(max_episode_steps):
        output, parsed, parse_error, raw_completion = policy_fn(context, seed + step_index)
        transition = environment.step(output, invalid_format=not parsed, token_cost=0)
        episode_return += discount * transition.reward
        discount *= gamma
        costs.update(transition.costs)
        steps.append(
            {
                "step_index": step_index,
                "event_kind": transition.info["event_kind"],
                "observation": {
                    "field_id": context.observation.field_id,
                    "value": context.observation.value,
                    "source": context.observation.source,
                    "entity": context.observation.entity,
                    "source_authority": context.observation.source_authority,
                    "authenticated": context.observation.authenticated,
                    "relevant": context.observation.relevant,
                    "condition": context.observation.condition,
                    "perturbation": context.observation.perturbation,
                    "observed_at": context.observation.observed_at,
                },
                "raw_completion": raw_completion,
                "parsed": parsed,
                "parse_error": parse_error,
                "action": revision_output_payload(output),
                "reward": transition.reward,
                "costs": dict(transition.costs),
                "transition": transition.info["transition"],
                "executor_error": transition.info["executor_error"],
                "verification_error": transition.info["verification_error"],
            }
        )
        if transition.terminated:
            success = bool(transition.info["success"])
            break
        if transition.next_context is None:
            raise RuntimeError("non-terminal trace step has no next context")
        context = transition.next_context
    final_slots = _belief_slots(environment._state)
    mismatches = _missing_or_wrong(final_slots, scenario.oracle_state)
    return {
        "scenario_id": scenario.scenario_id,
        "base_task_id": scenario.base_task_id,
        "macro_domain": scenario.macro_domain,
        "domain": scenario.domain,
        "variant_id": scenario.variant_id,
        "scenario_template": scenario.provenance.get("scenario_template"),
        "success": success,
        "task_return": episode_return,
        "action_count": len(steps),
        "costs": dict(costs),
        "oracle_state": scenario.oracle_state,
        "final_belief_slots": final_slots,
        "mismatches": mismatches,
        "steps": steps,
    }


def _baseline_policy(name: str) -> Callable[[RevisionContext, int], tuple[RevisionOutput, bool, str | None, str]]:
    raw: dict[str, Callable[[RevisionContext], RevisionOutput]] = {
        "rule_gate": rule_gate_policy,
        "visible_rule_gate": visible_rule_gate_policy,
        "auto_write": auto_write_policy,
    }
    if name not in raw:
        raise ValueError(f"unknown baseline mode: {name}")

    def wrapped(context: RevisionContext, _seed: int) -> tuple[RevisionOutput, bool, str | None, str]:
        output = raw[name](context)
        return output, True, None, json.dumps(revision_output_payload(output), ensure_ascii=False)

    return wrapped


def _hf_policy(
    *,
    root: Path,
    config_path: str,
    checkpoint: str,
    device_name: str,
) -> tuple[Any, Callable[[RevisionContext, int], tuple[RevisionOutput, bool, str | None, str]]]:
    import torch

    raw = load_config(resolve_repo_path(root, config_path))
    config = parse_hf_grpo_config(raw, root)
    checkpoint_path = resolve_repo_path(root, checkpoint)
    evaluation_config = replace(
        config,
        paths=replace(config.paths, sft_checkpoint=checkpoint_path),
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    policy, tokenizer = load_stage1_policy(evaluation_config, is_trainable=False)
    policy.to(device)
    policy.eval()

    def wrapped(context: RevisionContext, seed: int) -> tuple[RevisionOutput, bool, str | None, str]:
        prompt = render_context_prompt(
            tokenizer,
            context,
            enable_thinking=evaluation_config.model.enable_thinking,
        )
        _ids, _start, text = _generate_completion(
            policy,
            tokenizer,
            prompt,
            evaluation_config,
            device,
            seed,
            greedy=True,
        )
        output, parsed, parse_error = safe_parse_revision_output(text)
        return output, parsed, parse_error, text

    return config, wrapped


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["success"]]
    by_domain = defaultdict(lambda: {"episodes": 0, "failures": 0})
    failure_templates = Counter()
    failure_reasons = Counter()
    for row in rows:
        item = by_domain[row["macro_domain"]]
        item["episodes"] += 1
        item["failures"] += int(not row["success"])
        if not row["success"]:
            failure_templates[str(row.get("scenario_template"))] += 1
            if any(step["parse_error"] for step in row["steps"]):
                failure_reasons["parse_error"] += 1
            if row["mismatches"]:
                failure_reasons["state_mismatch"] += 1
            if sum(float(value) for value in row["costs"].values()) == 0:
                failure_reasons["no_explicit_cost"] += 1
    return {
        "episodes": len(rows),
        "success_rate": (len(rows) - len(failures)) / max(len(rows), 1),
        "failure_count": len(failures),
        "by_domain": {
            key: {
                **value,
                "success_rate": (value["episodes"] - value["failures"]) / max(value["episodes"], 1),
            }
            for key, value in sorted(by_domain.items())
        },
        "failure_templates": dict(failure_templates.most_common()),
        "failure_reasons": dict(failure_reasons.most_common()),
        "failure_examples": [
            {
                "scenario_id": row["scenario_id"],
                "macro_domain": row["macro_domain"],
                "scenario_template": row.get("scenario_template"),
                "mismatches": row["mismatches"],
                "costs": row["costs"],
            }
            for row in failures[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Stage-2 evaluation cases.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--mode",
        choices=("hf", "rule_gate", "visible_rule_gate", "auto_write"),
        required=True,
    )
    parser.add_argument("--config", default="configs/rl_qwen3_4b_8xh100_grpo.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--domain")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--max-episode-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    scenarios = read_scenarios(resolve_repo_path(root, args.scenario_file))
    if args.domain:
        scenarios = [scenario for scenario in scenarios if scenario.macro_domain == args.domain]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    if args.mode == "hf":
        if not args.checkpoint:
            parser.error("--checkpoint is required for --mode hf")
        config, policy_fn = _hf_policy(
            root=root,
            config_path=args.config,
            checkpoint=args.checkpoint,
            device_name=args.device,
        )
        gamma = config.train.gamma
        max_episode_steps = config.rollout.max_episode_steps
    else:
        policy_fn = _baseline_policy(args.mode)
        gamma = args.gamma
        max_episode_steps = args.max_episode_steps
    rows = [
        _trace_scenario(
            scenario,
            policy_fn=policy_fn,
            gamma=gamma,
            max_episode_steps=max_episode_steps,
            seed=42 + index * 100,
        )
        for index, scenario in enumerate(scenarios)
    ]
    output_jsonl = Path(args.output_jsonl)
    summary_path = Path(args.summary)
    _write_jsonl(output_jsonl, rows)
    summary = _summarize(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[trace-stage2] done "
        f"mode={args.mode} scenarios={len(rows)} "
        f"success={summary['success_rate']:.4f} "
        f"failures={summary['failure_count']} "
        f"jsonl={output_jsonl} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
