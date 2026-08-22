from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import resolve_repo_path

REQUIRED_CONSTRAINTS = {
    "false_update",
    "unsafe_action",
    "verification",
    "stall",
    "invalid_format",
    "invalid_patch",
    "collateral_edit",
    "budget_violation",
}


@dataclass(frozen=True)
class HFGRPOPaths:
    stage1_dev_file: Path
    readiness_report: Path
    train_file: Path
    dev_file: Path
    test_file: Path
    model_path: Path
    sft_checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class HFGRPOModelConfig:
    dtype: str
    enable_thinking: bool
    max_prompt_length: int
    max_completion_length: int
    gradient_checkpointing: bool
    attention_implementation: str | None


@dataclass(frozen=True)
class HFGRPORolloutConfig:
    group_size: int
    groups_per_iteration: int
    max_episode_steps: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class HFGRPOTrainConfig:
    iterations: int
    update_epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    clip_ratio: float
    kl_coefficient: float
    gamma: float
    multiplier_learning_rate: float
    max_grad_norm: float
    eval_steps: int
    eval_scenario_limit: int | None
    save_steps: int
    save_total_limit: int
    resume_from: Path | None


@dataclass(frozen=True)
class HFGRPOHardwareConfig:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class HFGRPOConfig:
    seed: int
    expected_split_counts: dict[str, int]
    paths: HFGRPOPaths
    model: HFGRPOModelConfig
    rollout: HFGRPORolloutConfig
    train: HFGRPOTrainConfig
    hardware: HFGRPOHardwareConfig
    constraints: dict[str, float]


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = int(raw.get(key, 0))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _positive_float(raw: Mapping[str, Any], key: str) -> float:
    value = float(raw.get(key, 0.0))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _unit_interval(value: Any, name: str, *, include_one: bool = True) -> float:
    number = float(value)
    upper_ok = number <= 1.0 if include_one else number < 1.0
    if number < 0.0 or not upper_ok:
        closing = "]" if include_one else ")"
        raise ValueError(f"{name} must be in [0, 1{closing}")
    return number


def parse_hf_grpo_config(raw: Mapping[str, Any], root: str | Path) -> HFGRPOConfig:
    """Validate the Qwen2.5 Stage-2 configuration without importing GPU libraries."""
    if raw.get("backend") != "hf_constrained_grpo":
        raise ValueError("backend must be 'hf_constrained_grpo'")
    path_raw = _mapping(raw, "paths")
    model_raw = _mapping(raw, "model")
    rollout_raw = _mapping(raw, "rollout")
    train_raw = _mapping(raw, "train")
    hardware_raw = _mapping(raw, "hardware")
    constraint_raw = _mapping(raw, "constraints")
    expected_raw = _mapping(raw, "expected_split_counts")
    if set(expected_raw) != {"train", "dev", "test"}:
        raise ValueError(
            "expected_split_counts must contain exactly train, dev, and test"
        )
    expected_split_counts = {
        str(split): int(count) for split, count in expected_raw.items()
    }
    if any(count <= 0 for count in expected_split_counts.values()):
        raise ValueError("expected split counts must be positive")

    dtype = str(model_raw.get("dtype", "")).lower()
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("model.dtype must be bfloat16 or float32")
    max_prompt = _positive_int(model_raw, "max_prompt_length")
    max_completion = _positive_int(model_raw, "max_completion_length")
    if max_prompt + max_completion < 64:
        raise ValueError("combined model sequence length is implausibly small")

    group_size = _positive_int(rollout_raw, "group_size")
    if group_size < 2:
        raise ValueError("rollout.group_size must be at least two for GRPO")
    top_p = float(rollout_raw.get("top_p", 0.0))
    if not 0.0 < top_p <= 1.0:
        raise ValueError("rollout.top_p must be in (0, 1]")
    temperature = _positive_float(rollout_raw, "temperature")

    clip_ratio = _unit_interval(
        train_raw.get("clip_ratio", -1), "train.clip_ratio", include_one=False
    )
    gamma = _unit_interval(train_raw.get("gamma", -1), "train.gamma")
    warmup_ratio = _unit_interval(
        train_raw.get("warmup_ratio", -1), "train.warmup_ratio", include_one=False
    )
    constraints = {str(key): float(value) for key, value in constraint_raw.items()}
    missing = REQUIRED_CONSTRAINTS - set(constraints)
    extra = set(constraints) - REQUIRED_CONSTRAINTS
    if missing or extra:
        raise ValueError(
            f"constraints must contain the canonical cost set; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if any(value < 0.0 for value in constraints.values()):
        raise ValueError("constraint limits must be non-negative")
    if float(train_raw.get("weight_decay", 0.0)) < 0.0:
        raise ValueError("train.weight_decay must be non-negative")
    if float(train_raw.get("kl_coefficient", 0.0)) < 0.0:
        raise ValueError("train.kl_coefficient must be non-negative")
    if float(train_raw.get("multiplier_learning_rate", 0.0)) < 0.0:
        raise ValueError("train.multiplier_learning_rate must be non-negative")
    num_processes = _positive_int(hardware_raw, "num_processes")

    eval_scenario_limit = train_raw.get("eval_scenario_limit")
    if eval_scenario_limit in ("", None):
        parsed_eval_scenario_limit = None
    else:
        parsed_eval_scenario_limit = int(eval_scenario_limit)
        if parsed_eval_scenario_limit <= 0:
            raise ValueError("train.eval_scenario_limit must be positive when set")

    return HFGRPOConfig(
        seed=int(raw.get("seed", 42)),
        expected_split_counts=expected_split_counts,
        paths=HFGRPOPaths(
            stage1_dev_file=resolve_repo_path(root, path_raw["stage1_dev_file"]),
            readiness_report=resolve_repo_path(root, path_raw["readiness_report"]),
            train_file=resolve_repo_path(root, path_raw["train_file"]),
            dev_file=resolve_repo_path(root, path_raw["dev_file"]),
            test_file=resolve_repo_path(root, path_raw["test_file"]),
            model_path=resolve_repo_path(root, path_raw.get("model_path", "model")),
            sft_checkpoint=resolve_repo_path(root, path_raw["sft_checkpoint"]),
            output_dir=resolve_repo_path(root, path_raw["output_dir"]),
        ),
        model=HFGRPOModelConfig(
            dtype=dtype,
            enable_thinking=bool(model_raw.get("enable_thinking", False)),
            max_prompt_length=max_prompt,
            max_completion_length=max_completion,
            gradient_checkpointing=bool(model_raw.get("gradient_checkpointing", True)),
            attention_implementation=(
                str(model_raw["attention_implementation"])
                if model_raw.get("attention_implementation")
                else None
            ),
        ),
        rollout=HFGRPORolloutConfig(
            group_size=group_size,
            groups_per_iteration=_positive_int(rollout_raw, "groups_per_iteration"),
            max_episode_steps=_positive_int(rollout_raw, "max_episode_steps"),
            temperature=temperature,
            top_p=top_p,
        ),
        train=HFGRPOTrainConfig(
            iterations=_positive_int(train_raw, "iterations"),
            update_epochs=_positive_int(train_raw, "update_epochs"),
            micro_batch_size=_positive_int(train_raw, "micro_batch_size"),
            gradient_accumulation_steps=_positive_int(
                train_raw, "gradient_accumulation_steps"
            ),
            learning_rate=_positive_float(train_raw, "learning_rate"),
            weight_decay=float(train_raw.get("weight_decay", 0.0)),
            warmup_ratio=warmup_ratio,
            clip_ratio=clip_ratio,
            kl_coefficient=float(train_raw.get("kl_coefficient", 0.0)),
            gamma=gamma,
            multiplier_learning_rate=float(
                train_raw.get("multiplier_learning_rate", 0.0)
            ),
            max_grad_norm=_positive_float(train_raw, "max_grad_norm"),
            eval_steps=_positive_int(train_raw, "eval_steps"),
            eval_scenario_limit=parsed_eval_scenario_limit,
            save_steps=_positive_int(train_raw, "save_steps"),
            save_total_limit=_positive_int(train_raw, "save_total_limit"),
            resume_from=(
                resolve_repo_path(root, train_raw["resume_from"])
                if train_raw.get("resume_from")
                else None
            ),
        ),
        hardware=HFGRPOHardwareConfig(
            num_processes=num_processes,
            mixed_precision="bf16" if dtype == "bfloat16" else "no",
        ),
        constraints=constraints,
    )
