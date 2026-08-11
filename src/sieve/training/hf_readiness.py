from __future__ import annotations

import json
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
) -> dict[str, float]:
    scenarios = read_scenarios(config.paths.dev_file)
    if not scenarios:
        raise ValueError("Stage-2 dev file contains no scenarios")
    count = min(scenario_count, len(scenarios))
    indices = [index * len(scenarios) // count for index in range(count)]
    totals = np.zeros(10, dtype=np.float64)
    for order, index in enumerate(indices):
        totals += _evaluate_local_scenario(
            policy,
            tokenizer,
            scenarios[index],
            config,
            device,
            config.seed + order * 100,
        )
    metrics = totals_to_metrics(totals)
    return {
        "closed_loop_success_rate": metrics["success_rate"],
        "closed_loop_parse_rate": metrics["parse_rate"],
        "closed_loop_false_update_rate": metrics["false_update_rate"],
        "closed_loop_mean_return": metrics["mean_return"],
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
    return float(np.mean(variances)) if variances else 0.0


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

    records = read_jsonl(config.paths.stage1_dev_file)[:max_samples]
    if not records:
        raise ValueError("Stage-1 dev file contains no records")
    policy, tokenizer = load_stage1_policy(config, is_trainable=False)
    device = torch.device("cuda:0")
    policy.to(device)
    policy.eval()
    completions: list[str] = []
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
    metrics = evaluate_generated_actions(completions, records)
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
