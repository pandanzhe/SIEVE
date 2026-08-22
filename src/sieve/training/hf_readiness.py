from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.executor import StateExecutor
from ..core.types import Decision
from ..data.io import read_jsonl
from ..data.schema import SFTRecord
from ..environments.scenario_revision_env import ScenarioRevisionEnvironment
from ..evaluation.hf_stage2 import _evaluate_local_scenario, totals_to_metrics
from ..evaluation.metrics import decision_metrics
from ..policies.hf_revision_io import (
    assess_stage1_readiness,
    render_context_prompt,
    revision_output_payload,
    safe_parse_revision_output,
)
from ..rl_data.io import read_scenarios
from .hf_grpo import (
    _adapter_fingerprint,
    _generate_completion,
    _model_asset_fingerprint,
    _sha256,
    load_stage1_policy,
)
from .hf_grpo_config import HFGRPOConfig


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _progress(prefix: str, current: int, total: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    progress = current / max(total, 1)
    eta = elapsed / progress - elapsed if progress > 0 else 0.0
    print(
        f"[readiness] {prefix} {current}/{total} "
        f"progress={progress:.2%} elapsed={_format_seconds(elapsed)} "
        f"eta={_format_seconds(eta)}",
        flush=True,
    )


def _select_stratified_sft_records(
    records: Sequence[SFTRecord], max_samples: int
) -> list[SFTRecord]:
    """Keep small readiness runs representative across UPDATE/HOLD/IGNORE."""
    if max_samples >= len(records):
        return list(records)
    buckets: dict[Decision, list[SFTRecord]] = {decision: [] for decision in Decision}
    for record in records:
        buckets[record.target.decision].append(record)
    non_empty = [decision for decision, bucket in buckets.items() if bucket]
    if not non_empty:
        return list(records[:max_samples])

    selected: list[SFTRecord] = []
    positions = {decision: 0 for decision in non_empty}
    while len(selected) < max_samples:
        made_progress = False
        for decision in non_empty:
            bucket = buckets[decision]
            position = positions[decision]
            if position < len(bucket):
                selected.append(bucket[position])
                positions[decision] = position + 1
                made_progress = True
                if len(selected) >= max_samples:
                    break
        if not made_progress:
            break
    return selected


def _decision_counts(records: Sequence[SFTRecord]) -> dict[str, int]:
    counts = {decision.value: 0 for decision in Decision}
    for record in records:
        counts[record.target.decision.value] += 1
    return counts


def evaluate_generated_actions(
    completions: Sequence[str], records: Sequence[SFTRecord]
) -> dict[str, float]:
    """Score free JSON generations against gold actions and executor constraints."""
    if len(completions) != len(records) or not records:
        raise ValueError("completions and records must have equal non-zero length")
    executor = StateExecutor()
    truth: list[Decision] = []
    predictions: list[Decision] = []
    parsed = 0
    executable = 0
    exact = 0
    update_count = 0
    patch_value_exact = 0
    for text, record in zip(completions, records, strict=True):
        output, valid, _error = safe_parse_revision_output(text)
        result = executor.apply(
            record.context.belief_state,
            record.context.ledger,
            output,
            record.context.observation,
        )
        parsed += int(valid)
        executable += int(valid and result.executed)
        exact += int(
            valid
            and revision_output_payload(output)
            == revision_output_payload(record.target)
        )
        if record.target.decision is Decision.UPDATE:
            update_count += 1
            gold_values = {
                patch.field_id: patch.value for patch in record.target.patches
            }
            predicted_values = {
                patch.field_id: patch.value for patch in output.patches
            }
            patch_value_exact += int(valid and predicted_values == gold_values)
        truth.append(record.target.decision)
        predictions.append(output.decision)
    decisions = decision_metrics(truth, predictions)
    count = len(records)
    return {
        "parse_rate": parsed / count,
        "executable_rate": executable / count,
        "decision_macro_f1": decisions["macro_f1"],
        "decision_accuracy": decisions["decision_accuracy"],
        "false_update_rate": decisions["false_update_rate"],
        "full_action_exact_match": exact / count,
        "patch_value_exact_match": patch_value_exact / max(update_count, 1),
    }


def _greedy_closed_loop_metrics(
    policy: Any,
    tokenizer: Any,
    config: HFGRPOConfig,
    device: Any,
    *,
    scenario_count: int,
    trace_limit: int = 3,
) -> dict[str, Any]:
    scenarios = read_scenarios(config.paths.dev_file)
    if not scenarios:
        raise ValueError("Stage-2 dev file contains no scenarios")
    count = min(scenario_count, len(scenarios))
    indices = [index * len(scenarios) // count for index in range(count)]
    totals = np.zeros(10, dtype=np.float64)
    failure_traces: list[dict[str, Any]] = []
    print(
        f"[readiness] stage2 closed-loop start scenarios={count}",
        flush=True,
    )
    started_at = time.time()
    for order, index in enumerate(indices):
        scenario = scenarios[index]
        values, trace = _evaluate_local_scenario_with_trace(
            policy,
            tokenizer,
            scenario,
            config,
            device,
            config.seed + order * 100,
        )
        totals += values
        if not trace["success"] and len(failure_traces) < trace_limit:
            failure_traces.append(trace)
        if order == 0 or (order + 1) % 10 == 0 or order + 1 == count:
            _progress("stage2 closed-loop", order + 1, count, started_at)
    metrics = totals_to_metrics(totals)
    print(
        "[readiness] stage2 closed-loop done "
        f"success_rate={metrics['success_rate']:.4f} "
        f"parse_rate={metrics['parse_rate']:.4f} "
        f"false_update_rate={metrics['false_update_rate']:.4f}",
        flush=True,
    )
    return {
        "closed_loop_success_rate": metrics["success_rate"],
        "closed_loop_parse_rate": metrics["parse_rate"],
        "closed_loop_false_update_rate": metrics["false_update_rate"],
        "closed_loop_verification_rate": metrics["verification_rate"],
        "closed_loop_stall_rate": metrics["stall_rate"],
        "closed_loop_invalid_patch_rate": metrics["invalid_patch_rate"],
        "closed_loop_mean_return": metrics["mean_return"],
        "closed_loop_failure_traces": failure_traces,
    }


def _slot_payload(slot: Any) -> dict[str, Any]:
    return {
        "id": slot.id,
        "value": slot.value,
        "status": slot.status.value,
        "source": slot.source,
        "observed_at": slot.observed_at,
        "valid_from": slot.valid_from,
        "entity": slot.entity,
    }


def _output_payload(output: Any) -> dict[str, Any]:
    return revision_output_payload(output)


def _evaluate_local_scenario_with_trace(
    policy: Any,
    tokenizer: Any,
    scenario: Any,
    config: HFGRPOConfig,
    device: Any,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    environment = ScenarioRevisionEnvironment([scenario], gamma=config.train.gamma)
    context = environment.reset(scenario.scenario_id, seed)
    values = np.zeros(10, dtype=np.float64)
    episode_return = 0.0
    discount = 1.0
    steps: list[dict[str, Any]] = []
    success = False
    for step_index in range(config.rollout.max_episode_steps):
        prompt = render_context_prompt(
            tokenizer, context, enable_thinking=config.model.enable_thinking
        )
        _ids, _start, text = _generate_completion(
            policy,
            tokenizer,
            prompt,
            config,
            device,
            seed + step_index,
            greedy=True,
        )
        output, valid, error = safe_parse_revision_output(text)
        transition = environment.step(
            output,
            invalid_format=not valid,
            token_cost=int(_ids.numel()) - _start,
        )
        episode_return += discount * transition.reward
        discount *= config.train.gamma
        values[3] += 1.0
        values[4] += float(valid)
        values[5] += transition.costs.get("false_update", 0.0)
        values[6] += transition.costs.get("invalid_format", 0.0)
        values[7] += transition.costs.get("verification", 0.0)
        values[8] += transition.costs.get("stall", 0.0)
        values[9] += transition.costs.get("invalid_patch", 0.0)
        steps.append(
            {
                "step_index": step_index,
                "event_kind": transition.info.get("event_kind"),
                "observation": {
                    "field_id": context.observation.field_id,
                    "value": context.observation.value,
                    "source": context.observation.source,
                    "entity": context.observation.entity,
                    "source_authority": context.observation.source_authority,
                    "authenticated": context.observation.authenticated,
                },
                "raw_completion": text,
                "parsed": valid,
                "parse_error": error,
                "action": _output_payload(output),
                "reward": transition.reward,
                "costs": transition.costs,
                "transition": transition.info.get("transition"),
                "executor_error": transition.info.get("executor_error"),
                "verification_error": transition.info.get("verification_error"),
            }
        )
        if transition.terminated:
            success = bool(transition.info["success"])
            values[1] = float(success)
            break
        if transition.next_context is None:
            raise RuntimeError("non-terminal Stage-2 evaluation step has no context")
        context = transition.next_context
    values[0] = 1.0
    values[2] = episode_return
    final_state = [
        _slot_payload(slot)
        for slot in (environment._state.slots if environment._state is not None else [])
    ]
    return values, {
        "scenario_id": scenario.scenario_id,
        "base_task_id": scenario.base_task_id,
        "macro_domain": scenario.macro_domain,
        "success": success,
        "episode_return": episode_return,
        "oracle_state": scenario.oracle_state,
        "final_belief_slots": final_state,
        "steps": steps,
    }


def _probe_group_variance(
    policy: Any,
    tokenizer: Any,
    config: HFGRPOConfig,
    device: Any,
    *,
    scenario_count: int,
) -> float:
    scenarios = read_scenarios(config.paths.dev_file)[:scenario_count]
    variances: list[float] = []
    print(
        f"[readiness] GRPO group probe start scenarios={len(scenarios)} "
        f"group_size={config.rollout.group_size}",
        flush=True,
    )
    started_at = time.time()
    for scenario_index, scenario in enumerate(scenarios):
        scores: list[float] = []
        for sample_index in range(config.rollout.group_size):
            environment = ScenarioRevisionEnvironment(
                [scenario], gamma=config.train.gamma
            )
            context = environment.reset(
                scenario.scenario_id,
                config.seed + scenario_index * 100 + sample_index,
            )
            score = 0.0
            discount = 1.0
            for step_index in range(config.rollout.max_episode_steps):
                prompt = render_context_prompt(
                    tokenizer,
                    context,
                    enable_thinking=config.model.enable_thinking,
                )
                _ids, _start, text = _generate_completion(
                    policy,
                    tokenizer,
                    prompt,
                    config,
                    device,
                    config.seed
                    + scenario_index * 10_000
                    + sample_index * 100
                    + step_index,
                    greedy=False,
                )
                output, valid, _error = safe_parse_revision_output(text)
                transition = environment.step(
                    output,
                    invalid_format=not valid,
                    token_cost=int(_ids.numel()) - _start,
                )
                score += discount * transition.reward
                discount *= config.train.gamma
                score -= sum(
                    transition.costs.get(name, 0.0)
                    for name in ("false_update", "unsafe_action", "invalid_format")
                )
                if transition.terminated:
                    break
                if transition.next_context is None:
                    raise RuntimeError("readiness probe has no next context")
                context = transition.next_context
            scores.append(score)
        variances.append(float(np.var(scores)))
        if (
            scenario_index == 0
            or (scenario_index + 1) % 5 == 0
            or scenario_index + 1 == len(scenarios)
        ):
            _progress(
                "GRPO group probe",
                scenario_index + 1,
                len(scenarios),
                started_at,
            )
    value = float(np.mean(variances)) if variances else 0.0
    print(
        f"[readiness] GRPO group probe done group_reward_variance={value:.6f}",
        flush=True,
    )
    return value


def run_stage1_readiness(
    config: HFGRPOConfig,
    *,
    max_samples: int = 300,
    probe_scenarios: int = 8,
    closed_loop_scenarios: int = 100,
) -> dict[str, Any]:
    """Run the server-side pre-RL free-generation gate on one CUDA device."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install requirements/rl.txt before readiness evaluation") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 readiness generation requires a CUDA GPU")
    if max_samples <= 0 or probe_scenarios <= 0 or closed_loop_scenarios <= 0:
        raise ValueError("readiness sample counts must be positive")

    all_records = read_jsonl(config.paths.stage1_dev_file)
    if not all_records:
        raise ValueError("Stage-1 dev file contains no records")
    records = _select_stratified_sft_records(all_records, max_samples)
    decision_counts = _decision_counts(records)
    print(
        "[readiness] load policy start "
        f"model={config.paths.model_path} checkpoint={config.paths.sft_checkpoint}",
        flush=True,
    )
    policy, tokenizer = load_stage1_policy(config, is_trainable=False)
    device = torch.device("cuda:0")
    policy.to(device)
    policy.eval()
    print(
        f"[readiness] load policy done device={device} samples={len(records)} "
        f"decision_counts={decision_counts}",
        flush=True,
    )
    completions: list[str] = []
    print(
        f"[readiness] SFT dev free-generation start samples={len(records)}",
        flush=True,
    )
    started_at = time.time()
    for index, record in enumerate(records):
        prompt = render_context_prompt(
            tokenizer,
            record.context,
            enable_thinking=config.model.enable_thinking,
        )
        _ids, _start, text = _generate_completion(
            policy,
            tokenizer,
            prompt,
            config,
            device,
            config.seed + index,
            greedy=True,
        )
        completions.append(text)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(records):
            _progress("SFT dev free-generation", index + 1, len(records), started_at)
    metrics = evaluate_generated_actions(completions, records)
    print(
        "[readiness] SFT dev free-generation done "
        f"parse_rate={metrics['parse_rate']:.4f} "
        f"executable_rate={metrics['executable_rate']:.4f} "
        f"decision_macro_f1={metrics['decision_macro_f1']:.4f} "
        f"full_action_exact_match={metrics['full_action_exact_match']:.4f}",
        flush=True,
    )
    metrics["group_reward_variance"] = _probe_group_variance(
        policy,
        tokenizer,
        config,
        device,
        scenario_count=probe_scenarios,
    )
    metrics.update(
        _greedy_closed_loop_metrics(
            policy,
            tokenizer,
            config,
            device,
            scenario_count=closed_loop_scenarios,
        )
    )
    ready, failures = assess_stage1_readiness(metrics)
    report: dict[str, Any] = {
        "ready_for_stage2": ready,
        "failures": failures,
        "sample_count": len(records),
        "sample_decision_counts": decision_counts,
        "probe_scenarios": probe_scenarios,
        "closed_loop_scenarios": min(
            closed_loop_scenarios, len(read_scenarios(config.paths.dev_file))
        ),
        "metrics": metrics,
        "artifacts": {
            "sft_adapter_sha256": _adapter_fingerprint(
                config.paths.sft_checkpoint
            ),
            "base_model_fingerprint": _model_asset_fingerprint(
                config.paths.model_path
            ),
            "stage1_dev_sha256": _sha256(config.paths.stage1_dev_file),
            "stage2_dev_sha256": _sha256(config.paths.dev_file),
        },
    }
    config.paths.readiness_report.parent.mkdir(parents=True, exist_ok=True)
    config.paths.readiness_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
