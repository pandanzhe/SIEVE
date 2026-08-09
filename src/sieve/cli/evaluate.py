from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ensure_output_dirs, load_config, resolve_repo_path
from ..data.io import read_jsonl
from ..evaluation.evaluator import evaluate_policy, write_evaluation
from ..policies.structured_policy import StructuredPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SIEVE revision policy")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--checkpoint", default="outputs/toy/checkpoints/grpo_policy.npz")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(resolve_repo_path(root, args.config))
    paths = ensure_output_dirs(root, config)
    policy = StructuredPolicy.load(resolve_repo_path(root, args.checkpoint))
    records = read_jsonl(paths["data_dir"] / "test.jsonl")
    metrics = evaluate_policy(policy, records, seed=int(config["seed"]))
    write_evaluation(metrics, paths["evaluation_dir"])
    print(metrics)


if __name__ == "__main__":
    main()
