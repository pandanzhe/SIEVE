from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, resolve_repo_path
from ..training.hf_grpo_config import parse_hf_grpo_config
from ..training.hf_readiness import run_stage1_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage-1 free-generation gate before constrained GRPO"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/rl_qwen25_3b.yaml")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--probe-scenarios", type=int, default=8)
    parser.add_argument("--closed-loop-scenarios", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = load_config(resolve_repo_path(root, args.config))
    config = parse_hf_grpo_config(raw, root)
    result = run_stage1_readiness(
        config,
        max_samples=args.max_samples,
        probe_scenarios=args.probe_scenarios,
        closed_loop_scenarios=args.closed_loop_scenarios,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready_for_stage2"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
