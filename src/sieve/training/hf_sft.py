from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl
from ..policies.hf_data import HFRevisionCollator, build_structured_label_plan
from ..policies.hf_lora_policy import create_hf_lora_policy, local_model_ready
from .hf_sft_config import HFSFTConfig


_OPTIONAL_PACKAGES = ("torch", "transformers", "peft", "accelerate")


def dependency_report() -> dict[str, dict[str, str | bool | None]]:
    """Inspect optional training dependencies without importing CUDA libraries."""
    report: dict[str, dict[str, str | bool | None]] = {}
    for package in _OPTIONAL_PACKAGES:
        installed = importlib.util.find_spec(package) is not None
        try:
            version = importlib.metadata.version(package) if installed else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        report[package] = {"installed": installed, "version": version}
    return report


def _line_count_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    return count, digest.hexdigest()


def audit_sft_inputs(config: HFSFTConfig) -> dict[str, Any]:
    """Validate canonical Stage-1 files without loading the model or torch."""
    required = (config.paths.source_file, config.paths.train_file, config.paths.dev_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage-1 data files: {missing}")

    source_count, source_sha = _line_count_and_sha256(config.paths.source_file)
    train_count, train_sha = _line_count_and_sha256(config.paths.train_file)
    dev_count, dev_sha = _line_count_and_sha256(config.paths.dev_file)
    train_records = read_jsonl(config.paths.train_file)
    dev_records = read_jsonl(config.paths.dev_file)
    train_scenarios = {record.scenario_id for record in train_records}
    dev_scenarios = {record.scenario_id for record in dev_records}
    label_contract_errors = 0
    for record in (*train_records, *dev_records):
        try:
            build_structured_label_plan(record, config.model.max_slots)
        except ValueError:
            label_contract_errors += 1
    return {
        "source_records": source_count,
        "train_records": train_count,
        "dev_records": dev_count,
        "train_scenarios": len(train_scenarios),
        "dev_scenarios": len(dev_scenarios),
        "scenario_overlap": len(train_scenarios & dev_scenarios),
        "label_contract_errors": label_contract_errors,
        "source_sha256": source_sha,
        "train_sha256": train_sha,
        "dev_sha256": dev_sha,
        "model_path": str(config.model.path),
        "model_ready": local_model_ready(config.model.path),
        "dependencies": dependency_report(),
    }


def checkpoint_score(metrics: dict[str, float]) -> float:
    """Select a useful checkpoint while strongly penalizing false state updates."""
    return (
        metrics.get("decision_macro_f1", 0.0)
        - metrics.get("loss", 0.0)
        - 2.0 * metrics.get("false_update_rate", 0.0)
    )


def _require_training_dependencies() -> None:
    missing = [name for name, item in dependency_report().items() if not item["installed"]]
    if missing:
        raise RuntimeError(
            "missing Stage-1 training dependencies: "
            + ", ".join(missing)
            + ". Install the locked training requirements first."
        )


def _evaluate(model: Any, batches: Any, accelerator: Any) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    decision_losses: list[float] = []
    structure_losses: list[float] = []
    value_losses: list[float] = []
    decisions_pred: list[int] = []
    decisions_gold: list[int] = []

    with torch.no_grad():
        for batch in batches:
            output = model(batch)
            gathered_loss = accelerator.gather_for_metrics(output["loss"].detach().reshape(1))
            losses.extend(gathered_loss.float().cpu().tolist())

            tensors = {
                "decision_pred": output["decision_logits"].argmax(dim=-1),
                "decision_gold": batch["decision_labels"],
            }
            gathered = {
                name: accelerator.gather_for_metrics(tensor).cpu().tolist()
                for name, tensor in tensors.items()
            }
            decisions_pred.extend(gathered["decision_pred"])
            decisions_gold.extend(gathered["decision_gold"])

            for name, sink in (
                ("decision_loss", decision_losses),
                ("structure_loss", structure_losses),
                ("value_loss", value_losses),
            ):
                if name in output:
                    gathered_branch_loss = accelerator.gather_for_metrics(
                        output[name].detach().reshape(1)
                    )
                    sink.extend(gathered_branch_loss.float().cpu().tolist())

    if not decisions_gold:
        raise ValueError("SFT evaluation received no batches")
    decision_correct = sum(
        pred == gold for pred, gold in zip(decisions_pred, decisions_gold, strict=True)
    )
    f1_values = []
    for label in range(3):
        true_positive = sum(
            pred == label and gold == label
            for pred, gold in zip(decisions_pred, decisions_gold, strict=True)
        )
        false_positive = sum(
            pred == label and gold != label
            for pred, gold in zip(decisions_pred, decisions_gold, strict=True)
        )
        false_negative = sum(
            pred != label and gold == label
            for pred, gold in zip(decisions_pred, decisions_gold, strict=True)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    non_update_indices = [
        index for index, gold in enumerate(decisions_gold) if gold != 0
    ]
    metrics = {
        "decision_accuracy": decision_correct / len(decisions_gold),
        "decision_macro_f1": sum(f1_values) / len(f1_values),
        "false_update_rate": (
            sum(decisions_pred[index] == 0 for index in non_update_indices)
            / len(non_update_indices)
            if non_update_indices
            else 0.0
        ),
    }
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    if decision_losses:
        metrics["decision_loss"] = sum(decision_losses) / len(decision_losses)
    if structure_losses:
        metrics["structure_loss"] = sum(structure_losses) / len(structure_losses)
    if value_losses:
        metrics["value_loss"] = sum(value_losses) / len(value_losses)
    model.train()
    return metrics


def _append_metrics(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _trim_for_distributed_batches(
    records: list[Any],
    divisor: int,
    split_name: str,
    accelerator: Any,
) -> list[Any]:
    """Keep all ranks on the same number of collectives in DDP runs."""
    if divisor <= 1:
        return records
    keep = (len(records) // divisor) * divisor
    if keep <= 0:
        raise ValueError(
            f"{split_name} split has too few records for distributed batch divisor "
            f"{divisor}: {len(records)}"
        )
    dropped = len(records) - keep
    if dropped and accelerator.is_main_process:
        print(
            f"[sft-data] split={split_name} records={len(records)} "
            f"using={keep} dropped_tail={dropped} divisor={divisor}",
            flush=True,
        )
    return records[:keep]


def _prune_states(output_dir: Path, keep: int) -> None:
    states = sorted(
        output_dir.glob("state-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in states[keep:]:
        if stale.is_dir() and stale.parent == output_dir:
            shutil.rmtree(stale)


def train_hf_sft(config: HFSFTConfig) -> dict[str, Any]:
    """Run Stage-1 LoRA SFT through Accelerate on one or more GPUs."""
    _require_training_dependencies()
    if not config.model.path.is_dir():
        raise FileNotFoundError(
            f"local model directory is missing: {config.model.path}"
        )

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup, set_seed

    audit = audit_sft_inputs(config)
    if audit["scenario_overlap"]:
        raise ValueError("train/dev scenario overlap must be zero")
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.hardware.mixed_precision,
    )
    train_records = read_jsonl(config.paths.train_file)
    dev_records = read_jsonl(config.paths.dev_file)
    train_records = _trim_for_distributed_batches(
        train_records,
        config.train.micro_batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps,
        "train",
        accelerator,
    )
    dev_records = _trim_for_distributed_batches(
        dev_records,
        config.train.micro_batch_size * accelerator.num_processes,
        "dev",
        accelerator,
    )
    model, tokenizer = create_hf_lora_policy(
        config.model.path,
        config.model.max_slots,
        lora_rank=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        dtype=config.model.dtype,
        gradient_checkpointing=config.model.gradient_checkpointing,
        attention_implementation=config.model.attention_implementation,
        loss_weights=config.loss_weights,
    )
    collator = HFRevisionCollator(
        tokenizer,
        config.model.max_slots,
        config.model.max_length,
        config.model.max_completion_length,
        enable_thinking=config.model.enable_thinking,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_records,
        batch_size=config.train.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_records,
        batch_size=config.train.micro_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.adapter_parameters(),
                "lr": config.train.learning_rate_lora,
                "weight_decay": config.train.weight_decay,
            },
            {
                "params": model.structured_head_parameters(),
                "lr": config.train.learning_rate_heads,
                "weight_decay": config.train.weight_decay,
            },
        ]
    )
    global_batch_size = (
        config.train.micro_batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )
    updates_per_epoch = len(train_records) // global_batch_size
    total_updates = updates_per_epoch * config.train.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config.train.warmup_ratio),
        num_training_steps=total_updates,
    )
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )

    output_dir = config.paths.output_dir
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    if config.train.resume_from:
        accelerator.load_state(config.train.resume_from)

    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("-inf")
    global_step = 0
    log_every = int(os.environ.get("SIEVE_LOG_STEPS", "10"))
    if log_every <= 0:
        log_every = 10
    train_start_time = time.time()
    if accelerator.is_main_process:
        print(
            "[sft-start] "
            f"epochs={config.train.epochs} "
            f"train_records={len(train_records)} "
            f"dev_records={len(dev_records)} "
            f"micro_batch_size={config.train.micro_batch_size} "
            f"gradient_accumulation_steps={config.train.gradient_accumulation_steps} "
            f"world_size={accelerator.num_processes} "
            f"updates_per_epoch={updates_per_epoch} "
            f"total_updates={total_updates} "
            f"log_every={log_every} "
            f"eval_steps={config.train.eval_steps}",
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.train.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                output = model(batch)
                accelerator.backward(output["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.train.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step == 1 or global_step % log_every == 0:
                train_metrics: dict[str, float] = {}
                for name in ("loss", "decision_loss", "structure_loss", "value_loss"):
                    if name in output:
                        gathered = accelerator.gather_for_metrics(
                            output[name].detach().reshape(1)
                        )
                        train_metrics[name] = float(gathered.float().mean().cpu())
                if accelerator.is_main_process:
                    elapsed = time.time() - train_start_time
                    progress = global_step / max(total_updates, 1)
                    eta = elapsed / progress - elapsed if progress > 0 else 0.0
                    lr_values = [group["lr"] for group in optimizer.param_groups]
                    lr_text = ",".join(f"{lr:.3e}" for lr in lr_values)
                    metric_text = " ".join(
                        f"train_{name}={value:.6f}"
                        for name, value in train_metrics.items()
                    )
                    print(
                        "[sft-train] "
                        f"epoch={epoch + 1}/{config.train.epochs} "
                        f"step={global_step}/{total_updates} "
                        f"progress={progress:.2%} "
                        f"{metric_text} "
                        f"lr={lr_text} "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={_format_seconds(eta)}",
                        flush=True,
                    )
            if global_step % config.train.eval_steps == 0:
                metrics = _evaluate(model, dev_loader, accelerator)
                score = checkpoint_score(metrics)
                if accelerator.is_main_process:
                    _append_metrics(
                        metrics_path,
                        {"epoch": epoch + 1, "step": global_step, "score": score, **metrics},
                    )
                    print(
                        "[sft-eval] "
                        + json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "score": score,
                                **metrics,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if score > best_score:
                        best_score = score
                        accelerator.unwrap_model(model).save_adapter_bundle(
                            output_dir / "best"
                        )
                accelerator.wait_for_everyone()

        metrics = _evaluate(model, dev_loader, accelerator)
        score = checkpoint_score(metrics)
        if accelerator.is_main_process:
            _append_metrics(
                metrics_path,
                {"epoch": epoch + 1, "step": global_step, "score": score, **metrics},
            )
            print(
                "[sft-eval] "
                + json.dumps(
                    {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "score": score,
                        **metrics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            if score > best_score:
                best_score = score
                accelerator.unwrap_model(model).save_adapter_bundle(output_dir / "best")
        state_dir = output_dir / f"state-{epoch + 1:03d}"
        accelerator.save_state(state_dir)
        if accelerator.is_main_process:
            _prune_states(output_dir, config.train.save_total_limit)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_adapter_bundle(output_dir / "final")
    accelerator.wait_for_everyone()
    return {"global_step": global_step, "best_score": best_score, "audit": audit}


# Backward-compatible one-epoch helper retained for small integration tests.
def train_hf_sft_epoch(
    model: Any,
    batches: Any,
    optimizer: Any,
    accelerator: Any | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    import torch

    model.train()
    total_loss = 0.0
    steps = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        loss = result["loss"]
        if accelerator is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        else:
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    if steps == 0:
        raise ValueError("HF SFT epoch received no batches")
    return {"loss": total_loss / steps, "steps": float(steps)}


