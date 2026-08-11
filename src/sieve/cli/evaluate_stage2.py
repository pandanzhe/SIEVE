from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, resolve_repo_path
from ..evaluation.hf_stage2 import evaluate_stage2_checkpoint
from ..training.hf_grpo_config import parse_hf_grpo_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen2.5 Stage-2 adapter in closed loop"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/rl_qwen25_3b.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/stage2-qwen25-3b/best"
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-processes", type=int)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = load_config(resolve_repo_path(root, args.config))
    if args.num_processes is not None:
        if args.num_processes <= 0:
            parser.error("--num-processes must be positive")
        raw.setdefault("hardware", {})["num_processes"] = args.num_processes
    config = parse_hf_grpo_config(raw, root)
    output_path = (
        resolve_repo_path(root, args.output)
        if args.output
        else config.paths.output_dir / f"evaluation_{args.split}.json"
    )
    report = evaluate_stage2_checkpoint(
        config,
        checkpoint=resolve_repo_path(root, args.checkpoint),
        split=args.split,
        output_path=output_path,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
