from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ensure_output_dirs, load_config, resolve_repo_path
from ..data.io import read_jsonl
from ..policies.structured_policy import StructuredPolicy
from ..training.constrained_grpo import train_constrained_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained GRPO for the SIEVE revision policy")
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(resolve_repo_path(root, args.config))
    paths = ensure_output_dirs(root, config)
    records = read_jsonl(paths["data_dir"] / "train.jsonl")
    policy = StructuredPolicy.load(paths["checkpoint_dir"] / "sft_policy.npz")
    rl = config["rl"]
    if rl.get("algorithm") != "constrained_grpo":
        raise ValueError("rl.algorithm must be constrained_grpo")
    history = train_constrained_grpo(
        policy=policy,
        records=records,
        iterations=int(rl["iterations"]),
        groups_per_iteration=int(rl["groups_per_iteration"]),
        group_size=int(rl["group_size"]),
        learning_rate=float(rl["learning_rate"]),
        seed=int(config["seed"]),
        constraints={key: float(value) for key, value in rl["constraints"].items()},
        clip_ratio=float(rl["clip_ratio"]),
        kl_coefficient=float(rl["kl_coefficient"]),
        gamma=float(rl["gamma"]),
        multiplier_learning_rate=float(rl["multiplier_learning_rate"]),
    )
    policy.save(paths["checkpoint_dir"] / "grpo_policy.npz")
    (paths["checkpoint_dir"] / "grpo_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(history[-1])


if __name__ == "__main__":
    main()
