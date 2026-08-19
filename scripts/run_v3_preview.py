"""End-to-end v3 preview: load sources -> build candidates -> generate -> audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sieve.data_factory.v3_sources import (
    SourceTask,
    load_tau3_tasks,
    load_car_tasks,
    source_task_to_grounded_record,
    audit_source_snapshot,
)
from sieve.data_factory.v3_candidates import build_v3_candidates
from sieve.data_factory.v3_generation import (
    CanonicalGlmClient,
    capture_generation_metadata,
    create_v3_client,
)
from sieve.data_factory.quality import validate_candidate
from sieve.data_factory.models import AuditRecord, Realization

PREVIEW_TOTAL = 50
OUTPUT_DIR = ROOT / "tmp" / "sft-v3-build" / "preview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> int:
    print("=" * 60)
    print("Stage-1 SFT v3 Preview Generation")
    print("=" * 60)

    # Step 1: Load sources
    print("\n[1] Loading tau3-bench training tasks...")
    raw_dir = ROOT / "data" / "raw"
    tau3_tasks = load_tau3_tasks(raw_dir / "tau2-bench")
    print(f"    tau3 tasks loaded: {len(tau3_tasks)}")

    print("[1] Loading CAR-bench training tasks...")
    car_tasks = load_car_tasks(raw_dir / "car-bench")
    print(f"    CAR tasks loaded: {len(car_tasks)}")

    # Audit
    report = audit_source_snapshot(tau3_tasks + car_tasks, raw_dir / "manifest_v3.json")
    print(f"    Audit: total={report['total']}, test_count={report['test_count']}")
    if report["test_count"] > 0:
        print("    WARNING: test tasks found in source data!")

    # Step 2: Convert to GroundedSourceRecord
    print("\n[2] Converting to GroundedSourceRecord...")
    tau3_records = []
    for i, task in enumerate(tau3_tasks):
        record = source_task_to_grounded_record(
            task, "tau3-bench", "v3", "MIT",
            raw_dir / "tau2-bench", i,
        )
        tau3_records.append(record)

    car_records = []
    for i, task in enumerate(car_tasks):
        record = source_task_to_grounded_record(
            task, "car-bench", "v3", "Apache-2.0",
            raw_dir / "car-bench", len(tau3_records) + i,
        )
        car_records.append(record)

    print(f"    tau3 records: {len(tau3_records)}, CAR records: {len(car_records)}")

    # Step 3: Build candidates (just a small subset for preview)
    print(f"\n[3] Building v3 candidates (full schedule, will sample {PREVIEW_TOTAL})...")
    all_candidates = build_v3_candidates(tau3_records, car_records, total=6000, seed=42)
    print(f"    Total candidates: {len(all_candidates)}")

    # Sample first N for preview
    preview_candidates = all_candidates[:PREVIEW_TOTAL]
    print(f"    Preview candidates: {len(preview_candidates)}")

    # Step 4: Generate observations with GLM
    print(f"\n[4] Generating observations with GLM-4.7...")
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        print("    ERROR: ZAI_API_KEY not set. Using dry-run (FakeGlmClient).")
        client = create_v3_client(dry_run=True)
    else:
        print(f"    ZAI_API_KEY found (length={len(api_key)})")
        cache_path = OUTPUT_DIR / "response_cache.jsonl"
        client = create_v3_client(
            dry_run=False,
            cache_path=cache_path,
            allow_preview=True,
        )

    # Generate in batches
    accepted = []
    rejected = []
    batch_size = 10
    for i in range(0, len(preview_candidates), batch_size):
        batch = preview_candidates[i:i+batch_size]
        try:
            realizations = client.realize(batch)
            by_id = {r.record_id: r for r in realizations}

            for candidate in batch:
                realization = by_id.get(candidate.record_id)
                if realization is None:
                    print(f"    MISSING: {candidate.record_id}")
                    continue

                validation = validate_candidate(candidate, realization)
                audit = AuditRecord(
                    candidate, realization, validation,
                    "v1", "v1", getattr(client, "model", "unknown"),
                )

                if validation.accepted:
                    accepted.append(audit)
                else:
                    rejected.append(audit)
                    print(f"    REJECTED: {candidate.record_id} reasons={validation.reason_codes}")

        except Exception as e:
            print(f"    BATCH ERROR [{i}]: {e}")
            # Fall back to dry-run for this batch
            from sieve.data_factory.glm import FakeGlmClient
            fake = FakeGlmClient()
            realizations = fake.realize(batch)
            by_id = {r.record_id: r for r in realizations}
            for candidate in batch:
                realization = by_id.get(candidate.record_id)
                if realization:
                    validation = validate_candidate(candidate, realization)
                    audit = AuditRecord(candidate, realization, validation, "v1", "v1", "fake-glm")
                    if validation.accepted:
                        accepted.append(audit)

    print(f"\n    Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    # Step 5: Write preview output
    print(f"\n[5] Writing preview output to {OUTPUT_DIR}...")
    accepted_path = OUTPUT_DIR / "accepted.jsonl"
    with accepted_path.open("w", encoding="utf-8") as f:
        for audit in accepted:
            f.write(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    # Decision distribution
    from collections import Counter
    decisions = Counter(a.candidate.target.decision.value for a in accepted)
    print(f"    Decision distribution: {dict(decisions)}")

    # Source distribution
    sources = Counter(a.candidate.provenance.source_dataset for a in accepted)
    print(f"    Source distribution: {dict(sources)}")

    # Token usage
    total_tokens = sum(
        int(a.realization.usage.get("total_tokens", 0))
        for a in accepted
    )
    print(f"    Total tokens: {total_tokens}")

    print(f"\n{'=' * 60}")
    print(f"Preview complete! {len(accepted)}/{PREVIEW_TOTAL} accepted.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
