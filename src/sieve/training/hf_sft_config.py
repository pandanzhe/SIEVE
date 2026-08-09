from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import resolve_repo_path


@dataclass(frozen=True)
class HFSFTPaths:
    source_file: Path
    train_file: Path
    dev_file: Path
    output_dir: Path


@dataclass(frozen=True)
class HFModelConfig:
    path: Path
    dtype: str
    enable_thinking: bool
    max_slots: int
    max_length: int
    max_completion_length: int
    gradient_checkpointing: bool
    attention_implementation: str | None


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class HFSFTTrainConfig:
    epochs: int
    micro_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    learning_rate_lora: float
    learning_rate_heads: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    num_workers: int
    eval_steps: int
    save_total_limit: int
    resume_from: str | None


@dataclass(frozen=True)
class HFHardwareConfig:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class HFSFTConfig:
    seed: int
    paths: HFSFTPaths
    model: HFModelConfig
    lora: LoRAConfig
    train: HFSFTTrainConfig
    hardware: HFHardwareConfig
    loss_weights: dict[str, float]


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


def parse_hf_sft_config(raw: Mapping[str, Any], root: str | Path) -> HFSFTConfig:
    """Validate and resolve the Stage-1 HF configuration."""
    if raw.get("backend") != "hf":
        raise ValueError("backend must be 'hf' for the Qwen Stage-1 trainer")

    path_raw = _mapping(raw, "paths")
    model_raw = _mapping(raw, "model")
    lora_raw = _mapping(raw, "lora")
    train_raw = _mapping(raw, "train")
    hardware_raw = _mapping(raw, "hardware")

    dtype = str(model_raw.get("dtype", "")).lower()
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("model.dtype must be bfloat16 or float32")
    num_processes = _positive_int(hardware_raw, "num_processes")
    micro_batch_size = _positive_int(train_raw, "micro_batch_size")
    effective_batch_size = _positive_int(train_raw, "effective_batch_size")
    distributed_micro_batch = micro_batch_size * num_processes
    if effective_batch_size % distributed_micro_batch:
        raise ValueError(
            "effective_batch_size must be divisible by "
            "micro_batch_size * hardware.num_processes"
        )

    max_length = _positive_int(model_raw, "max_length")
    max_completion_length = _positive_int(model_raw, "max_completion_length")
    if max_completion_length >= max_length:
        raise ValueError("max_completion_length must be smaller than max_length")

    dropout = float(lora_raw.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("lora.dropout must be in [0, 1)")
    warmup_ratio = float(train_raw.get("warmup_ratio", 0.0))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    target_modules = tuple(str(value) for value in lora_raw.get("target_modules", ()))
    if not target_modules:
        raise ValueError("lora.target_modules must not be empty")

    loss_weights_raw = _mapping(raw, "loss_weights")
    loss_weights = {str(key): float(value) for key, value in loss_weights_raw.items()}
    if any(value < 0 for value in loss_weights.values()):
        raise ValueError("loss weights must be non-negative")

    paths = HFSFTPaths(
        source_file=resolve_repo_path(root, path_raw["source_file"]),
        train_file=resolve_repo_path(root, path_raw["train_file"]),
        dev_file=resolve_repo_path(root, path_raw["dev_file"]),
        output_dir=resolve_repo_path(root, path_raw["output_dir"]),
    )
    model = HFModelConfig(
        path=resolve_repo_path(root, model_raw.get("path", "model")),
        dtype=dtype,
        enable_thinking=bool(model_raw.get("enable_thinking", False)),
        max_slots=_positive_int(model_raw, "max_slots"),
        max_length=max_length,
        max_completion_length=max_completion_length,
        gradient_checkpointing=bool(model_raw.get("gradient_checkpointing", True)),
        attention_implementation=(
            str(model_raw["attention_implementation"])
            if model_raw.get("attention_implementation")
            else None
        ),
    )

    return HFSFTConfig(
        seed=int(raw.get("seed", 17)),
        paths=paths,
        model=model,
        lora=LoRAConfig(
            rank=_positive_int(lora_raw, "rank"),
            alpha=_positive_int(lora_raw, "alpha"),
            dropout=dropout,
            target_modules=target_modules,
        ),
        train=HFSFTTrainConfig(
            epochs=_positive_int(train_raw, "epochs"),
            micro_batch_size=micro_batch_size,
            effective_batch_size=effective_batch_size,
            gradient_accumulation_steps=effective_batch_size // distributed_micro_batch,
            learning_rate_lora=_positive_float(train_raw, "learning_rate_lora"),
            learning_rate_heads=_positive_float(train_raw, "learning_rate_heads"),
            weight_decay=float(train_raw.get("weight_decay", 0.0)),
            warmup_ratio=warmup_ratio,
            max_grad_norm=_positive_float(train_raw, "max_grad_norm"),
            num_workers=int(train_raw.get("num_workers", 0)),
            eval_steps=_positive_int(train_raw, "eval_steps"),
            save_total_limit=_positive_int(train_raw, "save_total_limit"),
            resume_from=(
                str(train_raw["resume_from"]) if train_raw.get("resume_from") else None
            ),
        ),
        hardware=HFHardwareConfig(
            num_processes=num_processes,
            mixed_precision="bf16" if dtype == "bfloat16" else "no",
        ),
        loss_weights=loss_weights,
    )
