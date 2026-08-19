"""Stage-2 RL v3 build: derive RL episodes from accepted SFT v3 records.

Usage:
  # Build RL episodes from existing SFT accepted records
  python scripts/build_rl_v3.py

  # Custom paths
  python scripts/build_rl_v3.py --sft-source tmp/sft-v3-build/full/accepted.jsonl --output-dir data/rl

  # Preview (check source stats only, no write)
  python scripts/build_rl_v3.py --preview
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sieve.rl_data.build import build_rl_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2 RL v3 Build")
    parser.add_argument(
        "--sft-source",
        type=str,
        default=str(ROOT / "tmp" / "sft-v3-build" / "full" / "accepted.jsonl"),
        help="Path to accepted SFT records JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "rl"),
        help="Output directory for RL scenarios",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic splits"
    )
    parser.add_argument(
        "--train", type=int, default=1600, help="Train split count"
    )
    parser.add_argument(
        "--dev", type=int, default=200, help="Dev split count"
    )
    parser.add_argument(
        "--test", type=int, default=300, help="Test split count"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Preview mode (check stats only)"
    )
    args = parser.parse_args()

    source_path = Path(args.sft_source)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Stage-2 RL v3 Episode Build")
    print("=" * 60)

    # Validate source
    if not source_path.exists():
        # Try alternative paths
        alt_paths = [
            ROOT / "tmp" / "sft-v3-build" / "full" / "accepted.jsonl",
            ROOT / "tmp" / "sft-v3-build" / "preview" / "accepted.jsonl",
            ROOT / "data" / "sft" / "source" / "records.jsonl",
        ]
        for alt in alt_paths:
            if alt.exists():
                source_path = alt
                print(f"  Using alternative source: {alt}")
                break
        else:
            print(f"ERROR: SFT source not found at {args.sft_source}")
            print("Run build_sft_v3.py first to generate SFT records.")
            return 1

    # Count source records
    with source_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"\n[1] SFT source: {len(rows)} records from {source_path.name}")

    # Count eligible (authoritative UPDATE with patches)
    eligible = 0
    for row in rows:
        if isinstance(row, dict):
            target = row.get("target", {})
            obs = row.get("state", {}).get("observation", {})
            val = row.get("validation", {})
            if (val.get("accepted", True)
                and target.get("decision") == "UPDATE"
                and target.get("patches")
                and obs.get("source_authority") == "primary_record"
                and obs.get("authenticated") is True):
                eligible += 1
    print(f"    Eligible for RL (auth UPDATE): {eligible}")

    if eligible < 50:
        print(f"    WARNING: Only {eligible} eligible rows. RL build may produce few episodes.")

    if args.preview:
        print("\n[Preview] Stats only, no write.")
        from collections import Counter
        decisions = Counter(row.get("target", {}).get("decision", "?") for row in rows if isinstance(row, dict))
        print(f"    Decision distribution: {dict(decisions)}")
        return 0

    # Build RL corpus
    print(f"\n[2] Building RL episodes (train={args.train}, dev={args.dev}, test={args.test})...")
    t0 = time.time()

    manifest = build_rl_corpus(
        source_path=source_path,
        output_dir=output_dir,
        seed=args.seed,
        split_counts={"train": args.train, "dev": args.dev, "test": args.test},
    )

    elapsed = time.time() - t0
    print(f"    Build time: {elapsed:.1f}s")

    # Report
    print(f"\n[3] Results:")
    print(f"    Schema: {manifest.get('schema_version', 'unknown')}")
    print(f"    Split counts: {manifest.get('split_counts', {})}")
    print(f"    Source SHA256: {manifest.get('source_sha256', 'unknown')[:16]}...")

    # Read quality report
    quality_path = output_dir / "quality_report.json"
    if quality_path.exists():
        with quality_path.open("r", encoding="utf-8") as f:
            quality = json.load(f)
        print(f"\n[4] Quality report:")
        print(f"    Eligible source rows: {quality.get('eligible_source_rows', 0)}")
        print(f"    Unique base tasks: {quality.get('unique_base_tasks', 0)}")
        print(f"    Base task overlap: {quality.get('base_task_overlap', {})}")
        print(f"    Distractor split violations: {quality.get('distractor_split_violations', 0)}")
        print(f"    Macro domain counts: {quality.get('macro_domain_counts', {})}")
        print(f"    Event kind counts: {quality.get('event_kind_counts', {})}")

    # Atomic publish: data/rl
    print(f"\n[5] RL scenarios written to: {output_dir}")

    print(f"\n{'=' * 60}")
    print(f"RL v3 build complete!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
