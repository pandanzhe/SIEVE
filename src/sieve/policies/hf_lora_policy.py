from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def local_model_missing_files(model_path: str | Path) -> tuple[str, ...]:
    """Return missing local Hugging Face model assets without importing HF."""
    root = Path(model_path).resolve()
    if not root.is_dir():
        return ("directory",)
    missing: list[str] = []
    if not (root / "config.json").is_file():
        missing.append("config.json")
    if not ((root / "tokenizer.json").is_file() or (root / "tokenizer_config.json").is_file()):
        missing.append("tokenizer.json|tokenizer_config.json")
    has_weights = any(root.glob("*.safetensors")) or any(root.glob("pytorch_model*.bin"))
    if not has_weights:
        missing.append("*.safetensors|pytorch_model*.bin")
    return tuple(missing)


def local_model_ready(model_path: str | Path) -> bool:
    return not local_model_missing_files(model_path)

def create_hf_lora_policy(
    model_path: str | Path,
    max_slots: int,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    attention_implementation: str | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[Any, Any]:
    """Create the local Qwen LoRA policy and its structured revision heads."""
    resolved_model = Path(model_path).resolve()
    missing_model_files = local_model_missing_files(resolved_model)
    if missing_model_files:
        raise FileNotFoundError(
            f"local model directory is incomplete: {resolved_model}; "
            f"missing {', '.join(missing_model_files)}"
        )
    if max_slots <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("max_slots, lora_rank, and lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not target_modules:
        raise ValueError("target_modules must not be empty")
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install the 'hf' optional dependencies to use the LoRA backend"
        ) from exc

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    common_load_args = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True, **common_load_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_args: dict[str, Any] = {
        **common_load_args,
        "torch_dtype": torch_dtype,
    }
    if attention_implementation:
        model_args["attn_implementation"] = attention_implementation
    backbone = AutoModelForCausalLM.from_pretrained(resolved_model, **model_args)
    backbone.config.use_cache = False
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora)
    hidden_size = int(backbone.config.hidden_size)
    operation_count = 5
    weights = {
        "decision": 1.0,
        "affected": 1.0,
        "verification": 1.0,
        "patch_operation": 1.0,
        "patch_value": 1.0,
        **dict(loss_weights or {}),
    }

    class HFStructuredRevisionPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.decision_head = nn.Linear(hidden_size, 3)
            self.affected_head = nn.Linear(hidden_size, max_slots)
            self.verification_head = nn.Linear(hidden_size, 2)
            self.patch_operation_head = nn.Linear(
                hidden_size, max_slots * operation_count
            )

        def _head_input(self, pooled: Any) -> Any:
            return pooled.to(self.decision_head.weight.dtype)

        def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
            outputs = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("lm_labels"),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            row_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[row_indices, batch["pool_indices"]]
            head_input = self._head_input(pooled)
            result = {
                "decision_logits": self.decision_head(head_input),
                "affected_logits": self.affected_head(head_input),
                "verification_logits": self.verification_head(head_input),
                "patch_operation_logits": self.patch_operation_head(head_input).view(
                    -1, max_slots, operation_count
                ),
                "lm_logits": outputs.logits,
            }
            weighted_losses = []
            if "decision_labels" in batch:
                loss = functional.cross_entropy(
                    result["decision_logits"], batch["decision_labels"]
                )
                result["decision_loss"] = loss
                weighted_losses.append(weights["decision"] * loss)
            if "affected_labels" in batch:
                per_slot = functional.binary_cross_entropy_with_logits(
                    result["affected_logits"],
                    batch["affected_labels"].to(result["affected_logits"].dtype),
                    reduction="none",
                )
                active = batch["affected_mask"].bool()
                if active.any():
                    loss = per_slot[active].mean()
                    result["affected_loss"] = loss
                    weighted_losses.append(weights["affected"] * loss)
            if "verification_labels" in batch:
                active = batch["verification_labels"] != -100
                if active.any():
                    loss = functional.cross_entropy(
                        result["verification_logits"][active],
                        batch["verification_labels"][active],
                    )
                    result["verification_loss"] = loss
                    weighted_losses.append(weights["verification"] * loss)
            if "patch_operation_labels" in batch:
                per_operation = functional.binary_cross_entropy_with_logits(
                    result["patch_operation_logits"],
                    batch["patch_operation_labels"].to(
                        result["patch_operation_logits"].dtype
                    ),
                    reduction="none",
                )
                active = batch["patch_operation_mask"].bool()
                if active.any():
                    loss = per_operation[active].mean()
                    result["patch_operation_loss"] = loss
                    weighted_losses.append(weights["patch_operation"] * loss)
            if outputs.loss is not None:
                result["patch_value_loss"] = outputs.loss
                weighted_losses.append(weights["patch_value"] * outputs.loss)
            if not weighted_losses:
                raise ValueError("training batch contains no supervised targets")
            result["loss"] = torch.stack(weighted_losses).sum()
            return result

        def structured_head_parameters(self) -> tuple[Any, ...]:
            modules = (
                self.decision_head,
                self.affected_head,
                self.verification_head,
                self.patch_operation_head,
            )
            return tuple(parameter for module in modules for parameter in module.parameters())

        def adapter_parameters(self) -> tuple[Any, ...]:
            head_ids = {id(parameter) for parameter in self.structured_head_parameters()}
            return tuple(
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) not in head_ids
            )

        def trainable_parameter_names(self) -> tuple[str, ...]:
            return tuple(
                name for name, parameter in self.named_parameters() if parameter.requires_grad
            )

        def save_adapter_bundle(self, output_dir: str | Path) -> None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            self.backbone.save_pretrained(target / "adapter")
            tokenizer.save_pretrained(target / "tokenizer")
            torch.save(
                {
                    "decision": self.decision_head.state_dict(),
                    "affected": self.affected_head.state_dict(),
                    "verification": self.verification_head.state_dict(),
                    "patch_operation": self.patch_operation_head.state_dict(),
                },
                target / "structured_heads.pt",
            )
            metadata = {
                "schema_version": 1,
                "base_model_path": str(resolved_model),
                "max_slots": max_slots,
                "patch_operation_count": operation_count,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": list(target_modules),
                "dtype": dtype,
                "enable_thinking": False,
            }
            (target / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    return HFStructuredRevisionPolicy(), tokenizer


create_hf_lora_modules = create_hf_lora_policy


