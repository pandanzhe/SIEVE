from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import resolve_repo_path
from ..rl_data.build import build_rl_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic action-conditioned Stage-2 scenarios"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="data/sft/source/records.jsonl")
    parser.add_argument("--output-dir", default="data/rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--dev-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = build_rl_corpus(
        resolve_repo_path(root, args.source),
        resolve_repo_path(root, args.output_dir),
        seed=args.seed,
        split_counts={
            "train": args.train_count,
            "dev": args.dev_count,
            "test": args.test_count,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
