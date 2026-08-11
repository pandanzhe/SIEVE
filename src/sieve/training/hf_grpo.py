from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..environments.scenario_revision_env import ScenarioRevisionEnvironment
from ..policies.hf_lora_policy import local_model_ready
from ..policies.hf_revision_io import render_context_prompt, safe_parse_revision_output
from ..rl_data.build import split_base_task
from ..rl_data.io import read_scenarios
from ..rl_data.schema import RLScenario
from .hf_grpo_config import HFGRPOConfig
from .hf_grpo_math import (
    cosine_iteration_multiplier,
    ddp_token_loss_scale,
    grouped_advantages,
)
from .hf_sft import dependency_report
from .lagrangian import LagrangeController


@dataclass(frozen=True)
class HFRolloutExperience:
    input_ids: tuple[int, ...]
    completion_start: int
    old_log_probabilities: tuple[float, ...]
    reference_log_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class HFRolloutTrajectory:
    experiences: tuple[HFRolloutExperience, ...]
    task_return: float
    costs: dict[str, float]
    valid_actions: int
    action_count: int
    success: bool


@dataclass(frozen=True)
class HFTrainingSample:
    experience: HFRolloutExperience
    advantage: float
    weight: float = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_dir(checkpoint: Path) -> Path:
    nested = checkpoint / "adapter"
    return nested if nested.is_dir() else checkpoint


def _adapter_ready(checkpoint: Path) -> bool:
    adapter = _adapter_dir(checkpoint)
    has_config = (adapter / "adapter_config.json").is_file()
    has_weights = (adapter / "adapter_model.safetensors").is_file() or (
        adapter / "adapter_model.bin"
    ).is_file()
    return has_config and has_weights


def _adapter_fingerprint(checkpoint: Path) -> str | None:
    adapter = _adapter_dir(checkpoint)
    files = [
        path
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
        if (path := adapter / name).is_file()
    ]
    if len(files) < 2:
        return None
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_asset_fingerprint(model_path: Path) -> str | None:
    """Fingerprint model/tokenizer identity without hashing multi-gigabyte weights."""
    if not local_model_ready(model_path):
        return None
    digest = hashlib.sha256()
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    for name in metadata_names:
        path = model_path / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    weights = sorted(
        [*model_path.glob("*.safetensors"), *model_path.glob("pytorch_model*.bin")],
        key=lambda item: item.name,
    )
    for path in weights:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _raw_private_contract(path: Path) -> tuple[int, int]:
    target_leaks = 0
    missing_private = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            target_leaks += int("target" in raw)
            missing_private += int("environment_private" not in raw)
    return target_leaks, missing_private


def _verification_contract_valid(scenario: RLScenario) -> bool:
    if not scenario.events:
        return False
    event = scenario.events[0]
    evidence = event.verification_observation
    policy_slot = scenario.initial_state.get("__verification_policy")
    policy_value = None if policy_slot is None else policy_slot.value
    tool_mapping = (
        policy_value.get("tool_by_field")
        if isinstance(policy_value, dict)
        else None
    )
    return bool(
        evidence is not None
        and event.verification_tool
        and event.verification_tool == f"verify_{event.observation.field_id}"
        and evidence.field_id == event.observation.field_id
        and evidence.entity == event.observation.entity
        and isinstance(tool_mapping, dict)
        and tool_mapping.get(event.observation.field_id) == event.verification_tool
        and all(
            item.verification_tool is None
            and item.verification_observation is None
            for item in scenario.events[1:]
        )
    )


def audit_hf_grpo_inputs(config: HFGRPOConfig) -> dict[str, Any]:
    """Audit Stage-2 data and local assets without importing torch or loading a model."""
    split_paths = {
        "train": config.paths.train_file,
        "dev": config.paths.dev_file,
        "test": config.paths.test_file,
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if not config.paths.stage1_dev_file.is_file():
        missing.append(str(config.paths.stage1_dev_file))
    if missing:
        raise FileNotFoundError(f"missing Stage-2 scenario files: {missing}")

    split_scenarios = {name: read_scenarios(path) for name, path in split_paths.items()}
    actual_split_counts = {
        split: len(scenarios) for split, scenarios in split_scenarios.items()
    }
    split_count_mismatches = sum(
        actual_split_counts[split] != expected
        for split, expected in config.expected_split_counts.items()
    )
    wrong_split = sum(
        scenario.split != split
        for split, scenarios in split_scenarios.items()
        for scenario in scenarios
    )
    task_sets = {
        split: {scenario.base_task_id for scenario in scenarios}
        for split, scenarios in split_scenarios.items()
    }
    overlap = (
        len(task_sets["train"] & task_sets["dev"])
        + len(task_sets["train"] & task_sets["test"])
        + len(task_sets["dev"] & task_sets["test"])
    )
    target_leaks = 0
    missing_private = 0
    for path in split_paths.values():
        leaks, missing_blocks = _raw_private_contract(path)
        target_leaks += leaks
        missing_private += missing_blocks
    event_contract_errors = sum(
        len(scenario.events) != 4
        or {event.kind for event in scenario.events}
        != {"ambiguous", "wrong_entity", "authoritative", "stale_conflict"}
        or len(scenario.risk.dependent_fields) != 2
        or not _verification_contract_valid(scenario)
        or any(
            event.expected_decision not in {"UPDATE", "HOLD", "IGNORE"}
            for event in scenario.events
        )
        or any(
            field_id not in scenario.oracle_state
            for field_id in scenario.risk.dependent_fields
        )
        for scenarios in split_scenarios.values()
        for scenario in scenarios
    )
    duplicate_scenario_ids = sum(
        len(scenarios)
        - len({scenario.scenario_id for scenario in scenarios})
        for scenarios in split_scenarios.values()
    )
    distractor_split_violations = sum(
        split_base_task(donor, config.seed) != scenario.split
        for scenarios in split_scenarios.values()
        for scenario in scenarios
        for donor in scenario.provenance.get("distractor_base_task_ids", ())
    )
    missing_distractor_provenance = sum(
        not scenario.provenance.get("distractor_base_task_ids")
        for scenarios in split_scenarios.values()
        for scenario in scenarios
    )
    split_hashes = {
        split: _sha256(path) for split, path in split_paths.items()
    }
    manifest_path = config.paths.train_file.parent.parent / "manifest.json"
    manifest_matches = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_matches = (
                manifest.get("split_counts")
                == {
                    split: len(scenarios)
                    for split, scenarios in split_scenarios.items()
                }
                and manifest.get("split_sha256") == split_hashes
            )
        except (OSError, json.JSONDecodeError):
            manifest_matches = False
    readiness: dict[str, Any] | None = None
    if config.paths.readiness_report.is_file():
        try:
            loaded = json.loads(
                config.paths.readiness_report.read_text(encoding="utf-8")
            )
            readiness = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            readiness = None
    adapter_ready = _adapter_ready(config.paths.sft_checkpoint)
    adapter_sha = _adapter_fingerprint(config.paths.sft_checkpoint)
    model_ready = local_model_ready(config.paths.model_path)
    expected_artifacts = {
        "sft_adapter_sha256": adapter_sha,
        "base_model_fingerprint": _model_asset_fingerprint(config.paths.model_path),
        "stage1_dev_sha256": _sha256(config.paths.stage1_dev_file),
        "stage2_dev_sha256": _sha256(config.paths.dev_file),
    }
    readiness_matches = bool(
        readiness is not None
        and readiness.get("artifacts") == expected_artifacts
    )
    stage1_gate_passed = bool(
        readiness is not None
        and readiness.get("ready_for_stage2") is True
        and readiness_matches
        and adapter_ready
    )
    dependencies = dependency_report()
    data_ready = bool(
        overlap == 0
        and wrong_split == 0
        and target_leaks == 0
        and missing_private == 0
        and event_contract_errors == 0
        and duplicate_scenario_ids == 0
        and distractor_split_violations == 0
        and missing_distractor_provenance == 0
        and manifest_matches
        and split_count_mismatches == 0
    )
    return {
        "split_counts": {
            split: len(scenarios) for split, scenarios in split_scenarios.items()
        },
        "expected_split_counts": config.expected_split_counts,
        "split_count_mismatches": split_count_mismatches,
        "split_sha256": split_hashes,
        "base_task_overlap": overlap,
        "duplicate_scenario_ids": duplicate_scenario_ids,
        "distractor_split_violations": distractor_split_violations,
        "missing_distractor_provenance": missing_distractor_provenance,
        "wrong_split_labels": wrong_split,
        "target_leaks": target_leaks,
        "missing_private_blocks": missing_private,
        "event_contract_errors": event_contract_errors,
        "manifest_path": str(manifest_path),
        "manifest_matches_files": manifest_matches,
        "model_path": str(config.paths.model_path),
        "model_ready": model_ready,
        "sft_checkpoint": str(config.paths.sft_checkpoint),
        "sft_adapter_ready": adapter_ready,
        "sft_adapter_sha256": adapter_sha,
        "readiness_report": str(config.paths.readiness_report),
        "readiness_report_present": readiness is not None,
        "readiness_report_matches_artifacts": readiness_matches,
        "stage1_readiness_gate_passed": stage1_gate_passed,
        "ready_for_stage2": bool(
            data_ready
            and model_ready
            and stage1_gate_passed
            and all(item["installed"] for item in dependencies.values())
        ),
        "dependencies": dependencies,
    }


def _require_training_assets(config: HFGRPOConfig, audit: dict[str, Any]) -> None:
    errors: list[str] = []
    for key in (
        "base_task_overlap",
        "wrong_split_labels",
        "target_leaks",
        "missing_private_blocks",
        "event_contract_errors",
        "duplicate_scenario_ids",
        "distractor_split_violations",
        "missing_distractor_provenance",
        "split_count_mismatches",
    ):
        if audit[key]:
            errors.append(f"{key}={audit[key]}")
    if not audit["manifest_matches_files"]:
        errors.append("Stage-2 manifest counts or hashes do not match scenario files")
    if not audit["model_ready"]:
        errors.append(f"incomplete local model: {config.paths.model_path}")
    if not audit["sft_adapter_ready"]:
        errors.append(f"incomplete Stage-1 adapter: {config.paths.sft_checkpoint}")
    if not audit["stage1_readiness_gate_passed"]:
        errors.append(
            "Stage-1 free-generation readiness gate has not passed: "
            f"{config.paths.readiness_report}"
        )
    missing_packages = [
        name for name, item in audit["dependencies"].items() if not item["installed"]
    ]
    if missing_packages:
        errors.append("missing packages: " + ", ".join(missing_packages))
    if errors:
        raise RuntimeError("Stage-2 preflight failed: " + "; ".join(errors))


def load_stage1_policy(
    config: HFGRPOConfig, *, is_trainable: bool
) -> tuple[Any, Any]:
    """Load one local Qwen2.5 base plus the Stage-1 PEFT adapter."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements/rl.txt before Stage-2 training") from error

    dtype = torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float32
    common: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": dtype,
    }
    if config.model.attention_implementation:
        common["attn_implementation"] = config.model.attention_implementation
    adapter = _adapter_dir(config.paths.sft_checkpoint)
    tokenizer_root = config.paths.sft_checkpoint / "tokenizer"
    if not tokenizer_root.is_dir():
        tokenizer_root = config.paths.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(config.paths.model_path, **common)
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=is_trainable)
    if is_trainable:
        policy.config.use_cache = False
        if config.model.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            if hasattr(policy, "enable_input_require_grads"):
                policy.enable_input_require_grads()
    else:
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
    _disable_policy_dropout(policy)
    return policy, tokenizer


def _load_policy_and_reference(config: HFGRPOConfig) -> tuple[Any, Any, Any]:
    policy, tokenizer = load_stage1_policy(config, is_trainable=True)
    reference, _reference_tokenizer = load_stage1_policy(
        config, is_trainable=False
    )
    return policy, reference, tokenizer


def _completion_log_probabilities(
    model: Any,
    input_ids: Any,
    completion_start: int,
) -> Any:
    import torch

    if completion_start <= 0 or completion_start >= input_ids.numel():
        raise ValueError("completion must contain at least one token after a prompt")
    outputs = model(input_ids=input_ids.unsqueeze(0), return_dict=True)
    logits = outputs.logits[0, completion_start - 1 : -1].float()
    targets = input_ids[completion_start:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _disable_policy_dropout(policy: Any) -> int:
    """Set standard Dropout modules to p=0 while preserving train mode/checkpointing."""
    dropout_names = {
        "Dropout",
        "Dropout1d",
        "Dropout2d",
        "Dropout3d",
        "AlphaDropout",
        "FeatureAlphaDropout",
    }
    disabled = 0
    for module in policy.modules():
        if module.__class__.__name__ in dropout_names and hasattr(module, "p"):
            module.p = 0.0
            disabled += 1
    return disabled


def _sampling_generation_args(
    config: HFGRPOConfig, tokenizer: Any, *, greedy: bool = False
) -> dict[str, Any]:
    """Build generation kwargs whose sampled logits equal the raw policy logits."""
    arguments: dict[str, Any] = {
        "max_new_tokens": config.model.max_completion_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "sequence_bias": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
    }
    if greedy:
        arguments["do_sample"] = False
    else:
        arguments.update(
            {
                "do_sample": True,
                "temperature": config.rollout.temperature,
                "top_p": config.rollout.top_p,
                "top_k": 0,
                "typical_p": 1.0,
                "min_p": None,
                "epsilon_cutoff": 0.0,
                "eta_cutoff": 0.0,
            }
        )
    return arguments


def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: HFGRPOConfig,
    device: Any,
    seed: int,
    *,
    greedy: bool,
) -> tuple[Any, int, str]:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.model.max_prompt_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_length = int(input_ids.shape[1])
    generation_args = _sampling_generation_args(config, tokenizer, greedy=greedy)
    cuda_devices = [device] if getattr(device, "type", None) == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_args,
        )[0]
    if generated.numel() <= prompt_length:
        raise RuntimeError("model generated an empty cognitive action")
    completion_text = tokenizer.decode(
        generated[prompt_length:], skip_special_tokens=True
    ).strip()
    return generated, prompt_length, completion_text


def _collect_trajectory(
    policy: Any,
    reference: Any,
    tokenizer: Any,
    scenario: RLScenario,
    config: HFGRPOConfig,
    device: Any,
    seed: int,
) -> HFRolloutTrajectory:
    import torch

    environment = ScenarioRevisionEnvironment([scenario], gamma=config.train.gamma)
    context = environment.reset(scenario.scenario_id, seed)
    experiences: list[HFRolloutExperience] = []
    total_return = 0.0
    discount = 1.0
    total_costs = {name: 0.0 for name in config.constraints}
    valid_actions = 0
    success = False

    policy.eval()
    reference.eval()
    for step_index in range(config.rollout.max_episode_steps):
        prompt = render_context_prompt(
            tokenizer,
            context,
            enable_thinking=config.model.enable_thinking,
        )
        full_ids, completion_start, text = _generate_completion(
            policy,
            tokenizer,
            prompt,
            config,
            device,
            seed + step_index,
            greedy=False,
        )
        output, valid, _error = safe_parse_revision_output(text)
        valid_actions += int(valid)
        with torch.no_grad():
            old = _completion_log_probabilities(
                policy, full_ids, completion_start
            ).detach()
            reference_log_probs = _completion_log_probabilities(
                reference, full_ids, completion_start
            ).detach()
        experiences.append(
            HFRolloutExperience(
                input_ids=tuple(int(value) for value in full_ids.detach().cpu().tolist()),
                completion_start=completion_start,
                old_log_probabilities=tuple(float(value) for value in old.cpu().tolist()),
                reference_log_probabilities=tuple(
                    float(value) for value in reference_log_probs.cpu().tolist()
                ),
            )
        )
        transition = environment.step(
            output,
            invalid_format=not valid,
            token_cost=int(full_ids.numel()) - completion_start,
        )
        total_return += discount * transition.reward
        discount *= config.train.gamma
        for name in total_costs:
            total_costs[name] += float(transition.costs.get(name, 0.0))
        if transition.terminated:
            success = bool(transition.info["success"])
            break
        if transition.next_context is None:
            raise RuntimeError("non-terminal Stage-2 step has no next context")
        context = transition.next_context

    action_count = len(experiences)
    if action_count == 0:
        raise RuntimeError("Stage-2 rollout produced no actions")
    return HFRolloutTrajectory(
        experiences=tuple(experiences),
        task_return=total_return,
        costs=total_costs,
        valid_actions=valid_actions,
        action_count=action_count,
        success=success,
    )


def _training_batch(
    samples: Sequence[HFTrainingSample], tokenizer: Any, device: Any
) -> dict[str, Any]:
    import torch

    if not samples:
        raise ValueError("GRPO update batch must not be empty")
    maximum = max(len(sample.experience.input_ids) for sample in samples)
    input_ids = torch.full(
        (len(samples), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    old = torch.zeros((len(samples), maximum - 1), dtype=torch.float32, device=device)
    reference = torch.zeros_like(old)
    advantages = torch.zeros_like(old)
    loss_mask = torch.zeros_like(old)
    for row, sample in enumerate(samples):
        experience = sample.experience
        length = len(experience.input_ids)
        completion_length = len(experience.old_log_probabilities)
        if completion_length != len(experience.reference_log_probabilities):
            raise ValueError("old and reference completion log probabilities disagree")
        if experience.completion_start + completion_length != length:
            raise ValueError("completion log probabilities do not cover completion tokens")
        input_ids[row, :length] = torch.tensor(
            experience.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
        start = experience.completion_start - 1
        end = start + completion_length
        old[row, start:end] = torch.tensor(
            experience.old_log_probabilities, dtype=torch.float32, device=device
        )
        reference[row, start:end] = torch.tensor(
            experience.reference_log_probabilities, dtype=torch.float32, device=device
        )
        advantages[row, start:end] = float(sample.advantage)
        loss_mask[row, start:end] = float(sample.weight)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "old_log_probabilities": old,
        "reference_log_probabilities": reference,
        "advantages": advantages,
        "loss_mask": loss_mask,
    }


def _grpo_loss(
    model: Any,
    batch: dict[str, Any],
    config: HFGRPOConfig,
    *,
    loss_scale: float,
) -> dict[str, Any]:
    import torch

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
    )
    logits = outputs.logits[:, :-1].float()
    targets = batch["input_ids"][:, 1:]
    new_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    old = batch["old_log_probabilities"]
    advantage = batch["advantages"]
    mask = batch["loss_mask"]
    ratio = torch.exp((new_log_probs - old).clamp(-20.0, 20.0))
    clipped_ratio = ratio.clamp(
        1.0 - config.train.clip_ratio, 1.0 + config.train.clip_ratio
    )
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_sum = -(surrogate * mask).sum()
    log_ratio = (batch["reference_log_probabilities"] - new_log_probs).clamp(
        -20.0, 20.0
    )
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
    kl_sum = (kl_per_token * mask).sum()
    return {
        "loss": loss_scale
        * (policy_sum + config.train.kl_coefficient * kl_sum),
        "policy_sum": policy_sum.detach(),
        "kl_sum": kl_sum.detach(),
        "token_count": mask.sum().detach(),
    }


def _global_max_sample_count(accelerator: Any, count: int) -> int:
    import torch

    local = torch.tensor([count], dtype=torch.long, device=accelerator.device)
    return int(accelerator.gather(local).max().item())


def _pad_training_samples(
    samples: list[HFTrainingSample], target_count: int
) -> list[HFTrainingSample]:
    if not samples:
        raise ValueError("each process must collect at least one GRPO sample")
    padded = list(samples)
    while len(padded) < target_count:
        padded.append(replace(samples[-1], weight=0.0))
    return padded


def _aggregate_iteration(
    accelerator: Any,
    trajectories: Sequence[HFRolloutTrajectory],
    scores: Sequence[float],
    group_ids: Sequence[int],
    constraint_names: Sequence[str],
) -> dict[str, Any]:
    import torch

    cost_rows = [
        [trajectory.costs.get(name, 0.0) for name in constraint_names]
        for trajectory in trajectories
    ]
    cost_tensor = torch.tensor(cost_rows, dtype=torch.float64, device=accelerator.device)
    all_costs = accelerator.gather(cost_tensor).cpu().numpy()
    scalar_rows = torch.tensor(
        [
            [
                trajectory.task_return,
                float(score),
                trajectory.valid_actions,
                trajectory.action_count,
                float(trajectory.success),
            ]
            for trajectory, score in zip(trajectories, scores, strict=True)
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    all_scalars = accelerator.gather(scalar_rows).cpu().numpy()
    local_group_variances = [
        float(np.var([score for score, item in zip(scores, group_ids) if item == group]))
        for group in sorted(set(group_ids))
    ]
    variance_tensor = torch.tensor(
        local_group_variances, dtype=torch.float64, device=accelerator.device
    )
    all_variances = accelerator.gather(variance_tensor).cpu().numpy()
    return {
        "mean_return": float(all_scalars[:, 0].mean()),
        "mean_penalized_score": float(all_scalars[:, 1].mean()),
        "parse_rate": float(all_scalars[:, 2].sum() / max(all_scalars[:, 3].sum(), 1.0)),
        "success_rate": float(all_scalars[:, 4].mean()),
        "group_reward_variance": float(all_variances.mean()),
        "mean_costs": {
            name: float(all_costs[:, index].mean())
            for index, name in enumerate(constraint_names)
        },
    }


def _evaluate_scenarios(
    accelerator: Any,
    policy: Any,
    tokenizer: Any,
    scenarios: Sequence[RLScenario],
    config: HFGRPOConfig,
    *,
    limit: int | None = None,
) -> dict[str, float]:
    import torch

    selected = list(scenarios if limit is None else scenarios[:limit])
    local = selected[accelerator.process_index :: accelerator.num_processes]
    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.eval()
    totals = np.zeros(7, dtype=np.float64)
    for scenario_index, scenario in enumerate(local):
        environment = ScenarioRevisionEnvironment([scenario], gamma=config.train.gamma)
        context = environment.reset(scenario.scenario_id, config.seed + scenario_index)
        episode_return = 0.0
        discount = 1.0
        for step_index in range(config.rollout.max_episode_steps):
            prompt = render_context_prompt(
                tokenizer, context, enable_thinking=config.model.enable_thinking
            )
            _ids, _start, text = _generate_completion(
                unwrapped,
                tokenizer,
                prompt,
                config,
                accelerator.device,
                config.seed + scenario_index * 100 + step_index,
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
            totals[2] += 1.0
            totals[3] += float(valid)
            totals[5] += transition.costs.get("false_update", 0.0)
            totals[6] += transition.costs.get("invalid_format", 0.0)
            if transition.terminated:
                totals[1] += float(transition.info["success"])
                break
            if transition.next_context is None:
                raise RuntimeError("non-terminal evaluation step has no context")
            context = transition.next_context
        totals[0] += 1.0
        totals[4] += episode_return
    gathered = accelerator.gather(
        torch.tensor(totals, dtype=torch.float64, device=accelerator.device)
    ).reshape(-1, len(totals)).sum(dim=0).cpu().numpy()
    episode_count = max(gathered[0], 1.0)
    action_count = max(gathered[2], 1.0)
    policy.train()
    return {
        "episodes": float(gathered[0]),
        "success_rate": float(gathered[1] / episode_count),
        "parse_rate": float(gathered[3] / action_count),
        "mean_return": float(gathered[4] / episode_count),
        "false_update_rate": float(gathered[5] / action_count),
        "invalid_format_rate": float(gathered[6] / action_count),
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _save_adapter(
    accelerator: Any, policy: Any, tokenizer: Any, destination: Path
) -> None:
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(policy).save_pretrained(destination / "adapter")
        tokenizer.save_pretrained(destination / "tokenizer")
    accelerator.wait_for_everyone()


def _prune_states(output_dir: Path, keep: int) -> None:
    states = sorted(
        output_dir.glob("state-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in states[keep:]:
        if stale.is_dir() and stale.parent == output_dir:
            shutil.rmtree(stale)


def _validate_accelerator_profile(
    accelerator: Any, *, expected_processes: int
) -> None:
    if accelerator.num_processes != expected_processes:
        raise RuntimeError(
            "launched process count does not match hardware.num_processes: "
            f"{accelerator.num_processes} != {expected_processes}"
        )
    if getattr(accelerator.device, "type", None) != "cuda":
        raise RuntimeError("Stage-2 constrained GRPO requires CUDA")


def train_hf_constrained_grpo(config: HFGRPOConfig) -> dict[str, Any]:
    """Fine-tune the Stage-1 Qwen2.5 LoRA adapter with constrained GRPO."""
    audit = audit_hf_grpo_inputs(config)
    _require_training_assets(config, audit)
    try:
        import torch
        from accelerate import Accelerator
        from transformers import set_seed
    except ImportError as error:
        raise RuntimeError("install requirements/rl.txt before Stage-2 training") from error

    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.hardware.mixed_precision,
    )
    _validate_accelerator_profile(
        accelerator, expected_processes=config.hardware.num_processes
    )
    policy, reference, tokenizer = _load_policy_and_reference(config)
    train_scenarios = read_scenarios(config.paths.train_file)
    dev_scenarios = read_scenarios(config.paths.dev_file)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Stage-2 policy exposes no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    # Rollout lengths are policy-dependent. Indexing LR by outer iterations keeps
    # one- and two-GPU profiles identical and avoids Accelerate's DataLoader-aware
    # distributed scheduler adjustment for our custom local batches.
    policy, optimizer = accelerator.prepare(policy, optimizer)
    reference.to(accelerator.device)
    controller = LagrangeController(
        config.constraints, config.train.multiplier_learning_rate
    )

    output_dir = config.paths.output_dir
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    start_iteration = 0
    best_dev_score = float("-inf")
    if config.train.resume_from is not None:
        controller_path = config.train.resume_from / "controller.json"
        if not controller_path.is_file():
            raise FileNotFoundError(
                f"resume state is missing controller metadata: {controller_path}"
            )
        saved = json.loads(controller_path.read_text(encoding="utf-8"))
        saved_multipliers = {
            key: float(value) for key, value in saved["multipliers"].items()
        }
        if set(saved_multipliers) != set(controller.multipliers):
            raise ValueError("resume constraint multipliers do not match the config")
        start_iteration = int(saved["iteration"])
        if not 0 < start_iteration <= config.train.iterations:
            raise ValueError("resume iteration is outside the configured horizon")
        saved_best = saved.get("best_dev_score")
        best_dev_score = (
            float("-inf") if saved_best is None else float(saved_best)
        )
        accelerator.load_state(str(config.train.resume_from))
        controller.multipliers.update(saved_multipliers)

    metrics_path = output_dir / "metrics.jsonl"
    randomizer = random.Random(config.seed + accelerator.process_index * 100_003)
    for _ in range(start_iteration * config.rollout.groups_per_iteration):
        randomizer.randrange(len(train_scenarios))
    optimizer.zero_grad(set_to_none=True)
    for iteration in range(start_iteration, config.train.iterations):
        lr_multiplier = cosine_iteration_multiplier(
            iteration, config.train.iterations, config.train.warmup_ratio
        )
        current_learning_rate = config.train.learning_rate * lr_multiplier
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        unwrapped = accelerator.unwrap_model(policy)
        trajectories: list[HFRolloutTrajectory] = []
        group_ids: list[int] = []
        for group_index in range(config.rollout.groups_per_iteration):
            scenario = train_scenarios[randomizer.randrange(len(train_scenarios))]
            for sample_index in range(config.rollout.group_size):
                rollout_seed = (
                    config.seed
                    + accelerator.process_index * 10_000_019
                    + iteration * 100_003
                    + group_index * config.rollout.group_size
                    + sample_index
                )
                trajectories.append(
                    _collect_trajectory(
                        unwrapped,
                        reference,
                        tokenizer,
                        scenario,
                        config,
                        accelerator.device,
                        rollout_seed,
                    )
                )
                group_ids.append(group_index)

        scores = [
            trajectory.task_return - controller.penalty(trajectory.costs)
            for trajectory in trajectories
        ]
        advantages = grouped_advantages(np.asarray(scores), np.asarray(group_ids))
        samples = [
            HFTrainingSample(experience, float(advantage))
            for trajectory, advantage in zip(trajectories, advantages, strict=True)
            for experience in trajectory.experiences
        ]
        maximum_sample_count = _global_max_sample_count(accelerator, len(samples))
        accumulation_block = (
            config.train.micro_batch_size
            * config.train.gradient_accumulation_steps
        )
        maximum_sample_count = (
            math.ceil(maximum_sample_count / accumulation_block) * accumulation_block
        )
        samples = _pad_training_samples(samples, maximum_sample_count)

        policy.train()
        accumulated_policy_sum = 0.0
        accumulated_kl_sum = 0.0
        accumulated_token_count = 0.0
        for update_epoch in range(config.train.update_epochs):
            order = list(range(len(samples)))
            random.Random(
                config.seed
                + iteration * 97
                + update_epoch
                + accelerator.process_index * 10_007
            ).shuffle(order)
            for window_offset in range(0, len(order), accumulation_block):
                window_indices = order[
                    window_offset : window_offset + accumulation_block
                ]
                local_token_count = sum(
                    len(samples[index].experience.old_log_probabilities)
                    * samples[index].weight
                    for index in window_indices
                )
                global_token_count = int(
                    accelerator.gather(
                        torch.tensor(
                            [local_token_count],
                            dtype=torch.float64,
                            device=accelerator.device,
                        )
                    )
                    .sum()
                    .item()
                )
                loss_scale = ddp_token_loss_scale(
                    global_token_count=global_token_count,
                    world_size=accelerator.num_processes,
                    gradient_accumulation_steps=(
                        config.train.gradient_accumulation_steps
                    ),
                )
                for batch_offset in range(
                    0, len(window_indices), config.train.micro_batch_size
                ):
                    batch_indices = window_indices[
                        batch_offset : batch_offset + config.train.micro_batch_size
                    ]
                    selected = [samples[index] for index in batch_indices]
                    batch = _training_batch(
                        selected, tokenizer, accelerator.device
                    )
                    with accelerator.accumulate(policy):
                        losses = _grpo_loss(
                            policy,
                            batch,
                            config,
                            loss_scale=loss_scale,
                        )
                        accelerator.backward(losses["loss"])
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                policy.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    accumulated_policy_sum += float(
                        losses["policy_sum"].cpu()
                    )
                    accumulated_kl_sum += float(losses["kl_sum"].cpu())
                    accumulated_token_count += float(
                        losses["token_count"].cpu()
                    )

        global_loss_totals = accelerator.gather(
            torch.tensor(
                [
                    accumulated_policy_sum,
                    accumulated_kl_sum,
                    accumulated_token_count,
                ],
                dtype=torch.float64,
                device=accelerator.device,
            ).unsqueeze(0)
        ).reshape(-1, 3).sum(dim=0)
        reported_token_count = max(float(global_loss_totals[2].item()), 1.0)

        names = tuple(config.constraints)
        aggregate = _aggregate_iteration(
            accelerator, trajectories, scores, group_ids, names
        )
        controller.update(aggregate["mean_costs"])
        row: dict[str, Any] = {
            "iteration": iteration + 1,
            "learning_rate": current_learning_rate,
            "policy_loss": float(global_loss_totals[0].item())
            / reported_token_count,
            "sampled_kl": float(global_loss_totals[1].item())
            / reported_token_count,
            **{key: value for key, value in aggregate.items() if key != "mean_costs"},
        }
        row.update(
            {f"cost_{key}": value for key, value in aggregate["mean_costs"].items()}
        )
        row.update(
            {f"lambda_{key}": value for key, value in controller.multipliers.items()}
        )

        should_evaluate = (iteration + 1) % config.train.eval_steps == 0
        if should_evaluate:
            dev_metrics = _evaluate_scenarios(
                accelerator, policy, tokenizer, dev_scenarios, config
            )
            row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
            dev_score = (
                dev_metrics["success_rate"]
                - dev_metrics["false_update_rate"]
                - dev_metrics["invalid_format_rate"]
            )
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                _save_adapter(accelerator, policy, tokenizer, output_dir / "best")
        if accelerator.is_main_process:
            _append_jsonl(metrics_path, row)

        if (iteration + 1) % config.train.save_steps == 0:
            state_dir = output_dir / f"state-{iteration + 1:06d}"
            accelerator.save_state(str(state_dir))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                (state_dir / "controller.json").write_text(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "multipliers": controller.multipliers,
                            "best_dev_score": (
                                None
                                if not math.isfinite(best_dev_score)
                                else best_dev_score
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _prune_states(output_dir, config.train.save_total_limit)
            accelerator.wait_for_everyone()

    _save_adapter(accelerator, policy, tokenizer, output_dir / "final")
    return {
        "iterations": config.train.iterations,
        "best_dev_score": best_dev_score,
        "output_dir": str(output_dir),
        "audit": audit,
    }
