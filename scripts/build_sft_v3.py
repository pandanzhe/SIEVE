"""Stage-1 SFT v3 full build: sources -> candidates -> generate -> audit -> export.

Usage:
  # Preview (50 records, dry-run)
  python scripts/build_sft_v3.py --preview

  # Preview with live GLM (50 records)
  ZAI_API_KEY=... python scripts/build_sft_v3.py --preview --live

  # Full generation (6000 records, requires ZAI_API_KEY)
  ZAI_API_KEY=... python scripts/build_sft_v3.py --full

  # Resume interrupted full generation
  ZAI_API_KEY=... python scripts/build_sft_v3.py --full --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sieve.data_factory.v3_sources import (
    load_tau3_tasks,
    load_car_tasks,
    source_task_to_grounded_record,
    audit_source_snapshot,
)
from sieve.data_factory.v3_candidates import build_v3_candidates
from sieve.data_factory.v3_generation import create_v3_client
from sieve.data_factory.quality import validate_candidate
from sieve.data_factory.models import AuditRecord
from sieve.core.types import Decision


def load_and_convert(raw_dir: Path):
    """Load source tasks and convert to GroundedSourceRecord."""
    tau3_tasks = load_tau3_tasks(raw_dir / "tau2-bench")
    car_tasks = load_car_tasks(raw_dir / "car-bench")

    tau3_records = [
        source_task_to_grounded_record(t, "tau3-bench", "v3", "MIT", raw_dir / "tau2-bench", i)
        for i, t in enumerate(tau3_tasks)
    ]
    car_records = [
        source_task_to_grounded_record(t, "car-bench", "v3", "Apache-2.0", raw_dir / "car-bench", len(tau3_records) + i)
        for i, t in enumerate(car_tasks)
    ]
    return tau3_records, car_records


def generate_batch(client, candidates, accepted, rejected, batch_stats):
    """Generate realizations for a batch of candidates."""
    try:
        realizations = client.realize(candidates)
        by_id = {r.record_id: r for r in realizations}
    except Exception as e:
        print(f"    BATCH ERROR: {e}")
        return

    for candidate in candidates:
        realization = by_id.get(candidate.record_id)
        if realization is None:
            batch_stats["missing"] += 1
            continue

        try:
            validation = validate_candidate(candidate, realization)
            audit = AuditRecord(
                candidate, realization, validation,
                "v1", "v1", getattr(client, "model", "unknown"),
            )
            if validation.accepted:
                accepted.append(audit)
            else:
                rejected.append(audit)
                batch_stats["rejected"] += 1
        except Exception as e:
            batch_stats["validation_errors"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1 SFT v3 Build")
    parser.add_argument("--preview", action="store_true", help="Preview mode (50 records)")
    parser.add_argument("--live", action="store_true", help="Use live GLM (requires ZAI_API_KEY)")
    parser.add_argument("--full", action="store_true", help="Full generation (6000 records)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--total", type=int, default=6000, help="Total records to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Generation batch size")
    args = parser.parse_args()

    is_preview = args.preview or not args.full
    target_total = 50 if is_preview else args.total
    output_dir = ROOT / "tmp" / "sft-v3-build" / ("preview" if is_preview else "full")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Stage-1 SFT v3 {'Preview' if is_preview else 'Full'} Generation")
    print(f"Target: {target_total} records")
    print("=" * 60)

    # Step 1: Load sources
    raw_dir = ROOT / "data" / "raw"
    print("\n[1] Loading and converting sources...")
    tau3_records, car_records = load_and_convert(raw_dir)
    print(f"    tau3: {len(tau3_records)}, CAR: {len(car_records)}")

    report = audit_source_snapshot(
        load_tau3_tasks(raw_dir / "tau2-bench") + load_car_tasks(raw_dir / "car-bench"),
        raw_dir / "manifest_v3.json",
    )
    print(f"    Source audit: total={report['total']}, test={report['test_count']}")

    # Step 2: Build candidates
    print(f"\n[2] Building v3 candidates (total={args.total})...")
    all_candidates = build_v3_candidates(tau3_records, car_records, total=args.total, seed=42)
    print(f"    Candidates: {len(all_candidates)}")
    dec = Counter(c.target.decision for c in all_candidates)
    print(f"    UPDATE={dec[Decision.UPDATE]}, HOLD={dec[Decision.HOLD]}, IGNORE={dec[Decision.IGNORE]}")

    candidates = all_candidates[:target_total]
    print(f"    Selected: {len(candidates)} for generation")

    # Step 3: Resume from checkpoint
    accepted_path = output_dir / "accepted.jsonl"
    accepted = []
    start_idx = 0

    if args.resume and accepted_path.exists():
        print(f"\n[3] Resuming from checkpoint: {accepted_path}")
        with accepted_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    accepted.append(json.loads(line))
        seen_ids = {a["record_id"] if isinstance(a, dict) else a.candidate.record_id for a in accepted}
        # Skip candidates already processed
        remaining = [c for c in candidates if c.record_id not in seen_ids]
        print(f"    Already processed: {len(accepted)}, remaining: {len(remaining)}")
        candidates = remaining
        # Keep accepted as AuditRecord-like dicts for JSON writing

    # Step 4: Generate
    print(f"\n[4] Generating observations...")
    api_key = os.environ.get("ZAI_API_KEY")

    if args.live and api_key:
        print(f"    Using live GLM-4.7 (key length={len(api_key)})")
        cache_path = output_dir / "response_cache.jsonl"
        client = create_v3_client(
            dry_run=False,
            cache_path=cache_path,
            allow_preview=is_preview,
        )
    else:
        if args.live and not api_key:
            print("    WARNING: ZAI_API_KEY not set, falling back to dry-run")
        print("    Using dry-run (FakeGlmClient)")
        client = create_v3_client(dry_run=True)

    rejected = []
    batch_size = args.batch_size
    total_batches = (len(candidates) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(candidates), batch_size)):
        batch = candidates[i:i + batch_size]
        batch_stats = {"missing": 0, "rejected": 0, "validation_errors": 0}

        t0 = time.time()
        generate_batch(client, batch, accepted, rejected, batch_stats)
        elapsed = time.time() - t0

        done = batch_idx + 1
        pct = done / total_batches * 100
        print(f"    Batch {done}/{total_batches} ({pct:.0f}%): "
              f"+{len(batch) - batch_stats['missing'] - batch_stats['rejected'] - batch_stats['validation_errors']} accepted, "
              f"{batch_stats['rejected']} rejected, "
              f"{elapsed:.1f}s")

        # Checkpoint every 10 batches
        if done % 10 == 0:
            _write_checkpoint(accepted_path, accepted)

    # Final write
    _write_checkpoint(accepted_path, accepted)

    # Step 5: Summary
    dec_final = Counter(
        a.candidate.target.decision.value if hasattr(a, 'candidate') else a.get("target", {}).get("decision", "?")
        for a in accepted
    )
    src_final = Counter(
        a.candidate.provenance.source_dataset if hasattr(a, 'candidate') else a.get("provenance", {}).get("source_dataset", "?")
        for a in accepted
    )

    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    print(f"  Decisions: {dict(dec_final)}")
    print(f"  Sources: {dict(src_final)}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")
    return 0


def _write_checkpoint(path, accepted):
    """Write accepted records to JSONL checkpoint."""
    with path.open("w", encoding="utf-8") as f:
        for a in accepted:
            if hasattr(a, 'to_dict'):
                f.write(json.dumps(a.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            else:
                f.write(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
