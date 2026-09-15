from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..environments.scenario_revision_env import ScenarioRevisionEnvironment
from ..policies.hf_revision_io import render_context_prompt, safe_parse_revision_output
from ..rl_data.io import read_scenarios
from ..rl_data.schema import RLScenario
from ..training.hf_grpo import (
    _adapter_fingerprint,
    _generate_completion,
    _sha256,
    load_stage1_policy,
)
from ..training.hf_grpo_config import HFGRPOConfig

_CATEGORIES = ("all", "commerce", "service", "workflow")


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def totals_to_metrics(totals: Any) -> dict[str, float]:
    values = np.asarray(totals, dtype=np.float64)
    if values.shape != (10,):
        raise ValueError("Stage-2 evaluation totals must contain ten values")
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


def _evaluate_local_scenario(
    policy: Any,
    tokenizer: Any,
    scenario: RLScenario,
    config: HFGRPOConfig,
    device: Any,
    seed: int,
) -> np.ndarray:
    environment = ScenarioRevisionEnvironment([scenario], gamma=config.train.gamma)
    context = environment.reset(scenario.scenario_id, seed)
    values = np.zeros(10, dtype=np.float64)
    episode_return = 0.0
    discount = 1.0
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
        output, valid, _error = safe_parse_revision_output(text)
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
        if transition.terminated:
            values[1] = float(transition.info["success"])
            break
        if transition.next_context is None:
            raise RuntimeError("non-terminal Stage-2 evaluation step has no context")
        context = transition.next_context
    values[0] = 1.0
    values[2] = episode_return
    return values


def evaluate_stage2_checkpoint(
    config: HFGRPOConfig,
    *,
    checkpoint: Path,
    split: str,
    output_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run deterministic closed-loop evaluation once per scenario and report domains."""
    try:
        import torch
        from accelerate import Accelerator
    except ImportError as error:
        raise RuntimeError("install requirements/rl.txt before Stage-2 evaluation") from error
    if split not in {"dev", "test"}:
        raise ValueError("Stage-2 evaluation split must be dev or test")
    if limit is not None and limit <= 0:
        raise ValueError("evaluation limit must be positive")
    split_path = config.paths.dev_file if split == "dev" else config.paths.test_file
    scenarios: Sequence[RLScenario] = read_scenarios(split_path)
    if limit is not None:
        scenarios = scenarios[:limit]
    evaluation_config = replace(
        config,
        paths=replace(config.paths, sft_checkpoint=checkpoint),
    )
    accelerator = Accelerator(mixed_precision=config.hardware.mixed_precision)
    if accelerator.num_processes != config.hardware.num_processes:
        raise RuntimeError(
            "launched process count does not match hardware.num_processes: "
            f"{accelerator.num_processes} != {config.hardware.num_processes}"
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Stage-2 checkpoint evaluation requires CUDA")
    if accelerator.is_main_process:
        print(
            "[stage2-eval] start "
            f"split={split} scenarios={len(scenarios)} "
            f"checkpoint={checkpoint} output={output_path}",
            flush=True,
        )
    policy, tokenizer = load_stage1_policy(
        evaluation_config, is_trainable=False
    )
    policy.to(accelerator.device)
    policy.eval()

    local_totals = np.zeros((len(_CATEGORIES), 10), dtype=np.float64)
    category_index = {name: index for index, name in enumerate(_CATEGORIES)}
    indexed = list(enumerate(scenarios))
    local_indexed = indexed[accelerator.process_index :: accelerator.num_processes]
    started_at = time.time()
    for local_order, (global_index, scenario) in enumerate(local_indexed, start=1):
        values = _evaluate_local_scenario(
            policy,
            tokenizer,
            scenario,
            config,
            accelerator.device,
            config.seed + global_index * 100,
        )
        local_totals[category_index["all"]] += values
        local_totals[category_index[scenario.macro_domain]] += values
        if accelerator.is_main_process and (
            local_order == 1
            or local_order == len(local_indexed)
            or local_order % 10 == 0
        ):
            elapsed = time.time() - started_at
            progress = local_order / max(len(local_indexed), 1)
            eta = elapsed / progress - elapsed if progress > 0 else 0.0
            print(
                "[stage2-eval] progress "
                f"rank0={local_order}/{len(local_indexed)} "
                f"global_seen~={min(local_order * accelerator.num_processes, len(scenarios))}/{len(scenarios)} "
                f"elapsed={_format_seconds(elapsed)} "
                f"eta={_format_seconds(eta)}",
                flush=True,
            )
    gathered = accelerator.gather(
        torch.tensor(
            local_totals,
            dtype=torch.float64,
            device=accelerator.device,
        ).unsqueeze(0)
    ).reshape(-1, len(_CATEGORIES), 10)
    totals = gathered.sum(dim=0).cpu().numpy()
    report: dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _adapter_fingerprint(checkpoint),
        "scenario_file_sha256": _sha256(split_path),
        "metrics": {
            category: totals_to_metrics(totals[index])
            for index, category in enumerate(_CATEGORIES)
        },
    }
    if accelerator.is_main_process:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "[stage2-eval] done "
            f"split={split} "
            f"success={report['metrics']['all']['success_rate']:.4f} "
            f"parse={report['metrics']['all']['parse_rate']:.4f} "
            f"return={report['metrics']['all']['mean_return']:.4f} "
            f"output={output_path}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    return report
