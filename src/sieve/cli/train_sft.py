from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ensure_output_dirs, load_config, resolve_repo_path
from ..data.io import read_jsonl
from ..policies.structured_policy import StructuredPolicy
from ..training.hf_sft import audit_sft_inputs, train_hf_sft
from ..training.hf_sft_config import parse_hf_sft_config
from ..training.sft import evaluate_sft, train_sft


def _train_numpy(config: dict, root: Path, validate_only: bool) -> dict[str, object]:
    paths = ensure_output_dirs(root, config)
    train_records = read_jsonl(paths["data_dir"] / "train.jsonl")
    dev_records = read_jsonl(paths["data_dir"] / "dev.jsonl")
    if validate_only:
        return {
            "backend": "numpy",
            "train_records": len(train_records),
            "dev_records": len(dev_records),
        }
    policy_cfg, train_cfg = config["policy"], config["train"]
    policy = StructuredPolicy(
        int(policy_cfg["feature_dim"]),
        int(policy_cfg["max_slots"]),
        int(config["seed"]),
        float(policy_cfg.get("temperature", 1.0)),
    )
    history = train_sft(
        policy,
        train_records,
        int(train_cfg["epochs"]),
        float(train_cfg["learning_rate"]),
        int(config["seed"]),
        tuple(float(value) for value in train_cfg["class_weights"]),
        float(train_cfg["weight_decay"]),
        float(train_cfg["gradient_clip"]),
    )
    policy.save(paths["checkpoint_dir"] / "sft_policy.npz")
    metrics = evaluate_sft(policy, dev_records)
    metrics.update({"initial_epoch_loss": history[0], "final_epoch_loss": history[-1]})
    (paths["checkpoint_dir"] / "sft_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage-1 SIEVE policy with SFT")
    parser.add_argument("--config", default="configs/sft_qwen25_3b.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit paths, records, split isolation, model presence, and dependencies only.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw = load_config(resolve_repo_path(root, args.config))
    if raw.get("backend") == "hf":
        if args.num_processes is not None:
            if args.num_processes <= 0:
                parser.error("--num-processes must be positive")
            raw.setdefault("hardware", {})["num_processes"] = args.num_processes
        config = parse_hf_sft_config(raw, root)
        result = audit_sft_inputs(config) if args.validate_only else train_hf_sft(config)
    else:
        result = _train_numpy(raw, root, args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
