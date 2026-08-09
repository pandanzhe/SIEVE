from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data_factory.glm import CachedGlmClient, FakeGlmClient, JsonlResponseCache, ZhipuGlmClient
from ..data_factory.pipeline import PipelineConfig, run_pipeline


def _load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate source-grounded SIEVE JSONL")
    parser.add_argument("--config", default="configs/grounded_preview.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-manifest")
    parser.add_argument("--total", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--confirm-full", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    raw = _load_config(root / args.config)
    output_dir = (root / (args.output_dir or raw["output_dir"])).resolve()
    source_manifest = (root / (args.source_manifest or raw["source_manifest"])).resolve()
    config = PipelineConfig(
        source_manifest=source_manifest,
        output_dir=output_dir,
        total=args.total if args.total is not None else int(raw.get("total", 140)),
        batch_size=args.batch_size if args.batch_size is not None else int(raw.get("batch_size", 5)),
        seed=args.seed if args.seed is not None else int(raw.get("seed", 42)),
        max_attempts_per_record=(
            args.max_attempts if args.max_attempts is not None else int(raw.get("max_attempts", 3))
        ),
        max_requests=(
            args.max_requests if args.max_requests is not None else int(raw.get("max_requests", 200))
        ),
        resume=not args.no_resume,
        confirm_full=args.confirm_full,
        prompt_version=str(raw.get("prompt_version", "v1")),
        rule_version=str(raw.get("rule_version", "v1")),
        trajectory_mode=bool(raw.get("trajectory_mode", False)),
    )
    inner = FakeGlmClient() if args.dry_run else ZhipuGlmClient(
        temperature=float(raw.get("temperature", 0.6)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        max_retries=int(raw.get("glm_max_retries", 2)),
    )
    client = CachedGlmClient(
        inner,
        JsonlResponseCache(output_dir / "raw" / "response_cache.jsonl"),
    )
    result = run_pipeline(config, client)
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "accepted": result.accepted,
        "rejected": result.rejected,
        "requests": result.requests,
        "resumed": result.resumed,
        "mode": "dry-run" if args.dry_run else "live",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
