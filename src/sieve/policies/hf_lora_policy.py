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


def _enable_non_reentrant_gradient_checkpointing(backbone: Any) -> None:
    """Enable checkpointing without reentrant autograd hooks that conflict with DDP."""
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()


def _ddp_zero_loss_anchors(result: Mapping[str, Any]) -> list[Any]:
    """Keep every conditional structured head in the DDP graph at zero cost."""
    anchors = [result["decision_logits"].sum() * 0.0]
    if "structure_loss" in result:
        anchors.append(result["structure_loss"] * 0.0)
    if "value_loss" in result:
        anchors.append(result["value_loss"] * 0.0)
    return anchors


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
        _enable_non_reentrant_gradient_checkpointing(backbone)
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
        "structure": 1.0,
        "patch_value": 1.0,
        **dict(loss_weights or {}),
    }

    class HFStructuredRevisionPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.decision_head = nn.Linear(hidden_size, 3)

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
                "lm_logits": outputs.logits,
            }
            weighted_losses = _ddp_zero_loss_anchors(result)
            # --- Three-objective loss: decision, structure, patch_value ---
            if "decision_labels" in batch:
                loss = functional.cross_entropy(
                    result["decision_logits"], batch["decision_labels"]
                )
                result["decision_loss"] = loss
                weighted_losses.append(weights["decision"] * loss)
            # Structure loss: causal-LM on JSON structure tokens
            if "structure_labels" in batch:
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = batch["structure_labels"][:, 1:].contiguous()
                active = shift_labels != -100
                if active.any():
                    loss = functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    result["structure_loss"] = loss
                    weighted_losses.append(weights["structure"] * loss)
            # Patch-value loss: causal-LM on writable patch-value tokens
            if "value_labels" in batch:
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = batch["value_labels"][:, 1:].contiguous()
                active = shift_labels != -100
                if active.any():
                    loss = functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    result["value_loss"] = loss
                    weighted_losses.append(weights["patch_value"] * loss)
            if not weighted_losses:
                raise ValueError("training batch contains no supervised targets")
            result["loss"] = torch.stack(weighted_losses).sum()
            return result

        def structured_head_parameters(self) -> tuple[Any, ...]:
            return tuple(self.decision_head.parameters())

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
                },
                target / "structured_heads.pt",
            )
            metadata = {
                "schema_version": 2,
                "base_model_path": str(resolved_model),
                "max_slots": max_slots,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": list(target_modules),
                "dtype": dtype,
                "enable_thinking": False,
                "loss_objectives": ["decision", "structure", "patch_value"],
            }
            (target / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    return HFStructuredRevisionPolicy(), tokenizer


create_hf_lora_modules = create_hf_lora_policy


