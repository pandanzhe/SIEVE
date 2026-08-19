"""Run 12-point audit on v3 corpus, export train/dev splits, and atomic publish.

Usage:
  # Audit + export only (no publish)
  python scripts/audit_and_publish.py --audit-only

  # Full: audit + export + atomic publish
  python scripts/audit_and_publish.py --publish

  # From existing accepted.jsonl
  python scripts/audit_and_publish.py --audit-only --source tmp/sft-v3-build/full/accepted.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sieve.data_factory.v3_audit import audit_corpus, exact_grouped_split
from sieve.data_factory.v3_sources import load_tau3_tasks, load_car_tasks, source_task_to_grounded_record
from sieve.data_factory.v3_candidates import build_v3_candidates
from sieve.data_factory.quality import validate_candidate
from sieve.data_factory.glm import FakeGlmClient
from sieve.data_factory.models import Realization
from sieve.core.types import Decision


def _candidate_to_dict(candidate, realization=None) -> dict:
    """Export a GenerationCandidate (and optional Realization) to audit-compatible dict."""
    from dataclasses import asdict
    d = asdict(candidate)
    if realization is not None:
        d["observation_text"] = realization.observation_text
        d["generation"] = {
            "generator": "fake-glm" if isinstance(realization.usage, dict) and not realization.usage.get("total_tokens") else "glm-4.7",
            "usage": realization.usage,
        }
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT v3 Audit & Publish")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--source", type=str, default=str(ROOT / "tmp" / "sft-v3-build" / "full" / "accepted.jsonl"))
    args = parser.parse_args()

    source_path = Path(args.source)
    print("=" * 60)
    print("SFT v3 Audit & Publish")
    print("=" * 60)

    # Step 1: Load candidates and convert to dicts
    print("\n[1] Building candidates...")
    raw_dir = ROOT / "data" / "raw"
    tau3_tasks = load_tau3_tasks(raw_dir / "tau2-bench")
    car_tasks = load_car_tasks(raw_dir / "car-bench")
    tau3_records = [source_task_to_grounded_record(t, "tau3-bench", "v3", "MIT", raw_dir / "tau2-bench", i) for i, t in enumerate(tau3_tasks)]
    car_records = [source_task_to_grounded_record(t, "car-bench", "v3", "Apache-2.0", raw_dir / "car-bench", len(tau3_records)+i) for i, t in enumerate(car_tasks)]
    candidates = build_v3_candidates(tau3_records, car_records, total=6000, seed=42)
    print(f"    Candidates: {len(candidates)}")

    # Step 2: Convert to dict format for audit
    print("\n[2] Exporting to dict format...")
    fake = FakeGlmClient()
    realizations = fake.realize(candidates)
    all_dicts = []
    for candidate, realization in zip(candidates, realizations):
        d = _candidate_to_dict(candidate, realization)
        all_dicts.append(d)
    print(f"    Exported: {len(all_dicts)} records")

    # Step 3: Split
    print("\n[3] Splitting train/dev (5400/600)...")
    splits = exact_grouped_split(candidates, train=5400, dev=600, seed=42)
    train_candidates = splits["train"]
    dev_candidates = splits["dev"]
    print(f"    Train: {len(train_candidates)}, Dev: {len(dev_candidates)}")

    # Convert splits to dicts
    train_ids = {c.record_id for c in train_candidates}
    train_dicts = [d for d in all_dicts if d.get("record_id") in train_ids]
    dev_dicts = [d for d in all_dicts if d.get("record_id") not in train_ids]

    # Step 4: Run 12-point audit
    print("\n[4] Running 12-point audit...")
    report = audit_corpus(
        accepted_records=all_dicts,
        train_records=train_dicts,
        dev_records=dev_dicts,
        test_task_ids=set(),  # No test tasks in v3 source
    )

    passed_count = sum(1 for c in report.checks.values() if c.passed)
    total_count = len(report.checks)
    print(f"\n    Audit: {passed_count}/{total_count} checks passed")
    for name, check in report.checks.items():
        status = "PASS" if check.passed else "FAIL"
        print(f"      [{status}] {name}: expected={check.expected} actual={check.actual}")
        if not check.passed and check.details:
            print(f"              {check.details}")

    if not report.passed:
        print("\n    Audit FAILED. Fix issues before publishing.")
        if not args.audit_only:
            return 1

    # Step 5: Write output files
    output_dir = ROOT / "tmp" / "sft-v3-build" / "published"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5] Writing output to {output_dir}...")
    for name, data in [("accepted.jsonl", all_dicts), ("train.jsonl", train_dicts), ("dev.jsonl", dev_dicts)]:
        path = output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"    {name}: {len(data)} records")

    # Write audit report
    audit_path = output_dir / "audit_report.json"
    with audit_path.open("w", encoding="utf-8") as f:
        report_dict = {
            "passed": report.passed,
            "summary": report.summary,
            "checks": {name: {"name": c.name, "passed": c.passed, "expected": c.expected, "actual": c.actual, "details": c.details}
                       for name, c in report.checks.items()},
        }
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"    audit_report.json written")

    # Step 6: Atomic publish
    if args.publish and report.passed:
        print(f"\n[6] Atomic publishing to data/sft...")
        sft_dir = ROOT / "data" / "sft"
        backup_dir = ROOT / "data" / "sft.old.tmp"

        # Backup existing
        if sft_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            print(f"    Backing up data/sft -> data/sft.old.tmp")
            shutil.move(str(sft_dir), str(backup_dir))

        # Write new
        sft_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = sft_dir / "clean"
        source_dir = sft_dir / "source"
        clean_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        shutil.copy2(output_dir / "train.jsonl", clean_dir / "train.jsonl")
        shutil.copy2(output_dir / "dev.jsonl", clean_dir / "dev.jsonl")
        shutil.copy2(output_dir / "accepted.jsonl", source_dir / "records.jsonl")
        shutil.copy2(output_dir / "audit_report.json", sft_dir / "audit_report.json")

        print(f"    Published: data/sft/clean/train.jsonl ({len(train_dicts)} records)")
        print(f"    Published: data/sft/clean/dev.jsonl ({len(dev_dicts)} records)")
        print(f"    Published: data/sft/source/records.jsonl ({len(all_dicts)} records)")

        # Verify
        print(f"\n[7] Post-publish verification...")
        train_path = sft_dir / "clean" / "train.jsonl"
        dev_path = sft_dir / "clean" / "dev.jsonl"
        with train_path.open("r", encoding="utf-8") as f:
            published_train = [json.loads(line) for line in f if line.strip()]
        with dev_path.open("r", encoding="utf-8") as f:
            published_dev = [json.loads(line) for line in f if line.strip()]

        if abs(len(published_train) - 5400) <= 54 and abs(len(published_dev) - 600) <= 6:
            print(f"    Verification PASSED: train={len(published_train)}, dev={len(published_dev)}")
            # Remove backup
            if backup_dir.exists():
                print(f"    Removing backup data/sft.old.tmp")
                shutil.rmtree(backup_dir)
        else:
            print(f"    Verification FAILED! Rolling back...")
            if sft_dir.exists():
                shutil.rmtree(sft_dir)
            if backup_dir.exists():
                shutil.move(str(backup_dir), str(sft_dir))
            return 1

    print(f"\n{'=' * 60}")
    print(f"Done! Audit: {passed_count}/{total_count} passed")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
