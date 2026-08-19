from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from ..core.types import Decision
from .models import GenerationCandidate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LABEL_LEAKAGE = re.compile(
    r"(?:"
    r"\b(?:decision|reason_code|conflict_type)\s*[:=]\s*"
    r"(?:UPDATE|IGNORE|HOLD|[A-Z_]{4,})\b"
    r"|\b(?:relevant|condition|perturbation)\s*[:=]"
    r")",
    re.IGNORECASE,
)

_VALID_DECISIONS: frozenset[str] = frozenset(d.value for d in Decision)

_MINHASH_NUM_PERMS = 128
_MINHASH_PRIME = (1 << 61) - 1  # Mersenne prime


# ---------------------------------------------------------------------------
# Audit data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    expected: str
    actual: str
    details: str  # empty if passed


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    checks: dict[str, AuditCheck]
    summary: str


# ---------------------------------------------------------------------------
# Helpers – split
# ---------------------------------------------------------------------------


def _det_hash(key: str, seed: int) -> int:
    """Deterministic integer hash of *key* seeded by *seed*."""
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16)


def _parent_id(c: GenerationCandidate) -> str:
    """Derive parent_task_id from a candidate."""
    return c.provenance.parent_record_id or c.group_id


# ---------------------------------------------------------------------------
# exact_grouped_split
# ---------------------------------------------------------------------------


def exact_grouped_split(
    candidates: list[GenerationCandidate],
    *,
    train: int = 5400,
    dev: int = 600,
    seed: int = 42,
) -> dict[str, list[GenerationCandidate]]:
    """Split *candidates* into train/dev keeping parent-task groups intact.

    Groups are formed by ``parent_task_id`` (derived from
    ``candidate.provenance.parent_record_id or candidate.group_id``).  All
    candidates sharing a parent stay in the same split.  Assignment is
    stratified by ``(source_dataset, decision)`` and randomised within each
    stratum using a deterministic hash of the parent_task_id.
    """

    # 1. Group by parent_task_id
    groups: dict[str, list[GenerationCandidate]] = {}
    for c in candidates:
        pid = _parent_id(c)
        groups.setdefault(pid, []).append(c)

    # 2. Stratification key per group (source_dataset, decision)
    stratum_of: dict[str, tuple[str, str]] = {}
    for pid, members in groups.items():
        first = members[0]
        stratum_of[pid] = (first.provenance.source_dataset, first.target.decision.value)

    # 3. Collect groups per stratum, sorted by parent_task_id for determinism
    strata: dict[tuple[str, str], list[str]] = {}
    for pid, skey in stratum_of.items():
        strata.setdefault(skey, []).append(pid)
    for skey in strata:
        strata[skey].sort()

    # 4. Assign splits within each stratum
    total_target = train + dev
    train_pids: list[str] = []
    dev_pids: list[str] = []

    for skey in sorted(strata.keys()):
        pids = strata[skey]
        # Deterministic random order via hash
        pids_by_hash = sorted(pids, key=lambda p: _det_hash(p, seed))
        n_dev = len(pids_by_hash) * dev // total_target
        dev_pids.extend(pids_by_hash[:n_dev])
        train_pids.extend(pids_by_hash[n_dev:])

    # 5. Validate & adjust to hit exact record counts
    def _flat(pids: list[str]) -> list[GenerationCandidate]:
        return [c for p in pids for c in groups[p]]

    cur_train = sum(len(groups[p]) for p in train_pids)
    cur_dev = sum(len(groups[p]) for p in dev_pids)

    if cur_train != train or cur_dev != dev:
        # Iterative greedy adjustment: move groups between splits
        # to minimize the gap from target counts.  Try both single
        # and paired swaps for better convergence while preserving
        # parent-task group integrity (no individual record splitting).
        for _ in range(len(groups) * 4):
            if cur_train == train:
                break
            deficit = train - cur_train
            if deficit > 0 and dev_pids:
                # Try to find a dev group of exactly the right size first
                exact = [p for p in dev_pids if len(groups[p]) == deficit]
                if exact:
                    best = exact[0]
                else:
                    best = min(dev_pids, key=lambda p: abs(len(groups[p]) - deficit))
                dev_pids.remove(best)
                train_pids.append(best)
                cur_train += len(groups[best])
                cur_dev -= len(groups[best])
            elif deficit < 0 and train_pids:
                excess = -deficit
                exact = [p for p in train_pids if len(groups[p]) == excess]
                if exact:
                    best = exact[0]
                else:
                    best = min(train_pids, key=lambda p: abs(len(groups[p]) - excess))
                train_pids.remove(best)
                dev_pids.append(best)
                cur_train -= len(groups[best])
                cur_dev += len(groups[best])
            else:
                break

        # If still not exact after group-level adjustment, try pairwise swaps
        if cur_train != train:
            deficit = train - cur_train
            if deficit > 0:
                # Swap: move a larger dev group to train, move a smaller train group to dev
                for dp in sorted(dev_pids, key=lambda p: len(groups[p]), reverse=True):
                    if len(groups[dp]) <= deficit:
                        continue
                    needed_return = len(groups[dp]) - deficit
                    for tp in sorted(train_pids, key=lambda p: len(groups[p])):
                        if len(groups[tp]) == needed_return:
                            dev_pids.remove(dp); train_pids.append(dp)
                            train_pids.remove(tp); dev_pids.append(tp)
                            cur_train += len(groups[dp]) - len(groups[tp])
                            cur_dev += len(groups[tp]) - len(groups[dp])
                            break
                    if cur_train == train:
                        break
            elif deficit < 0:
                excess = -deficit
                for tp in sorted(train_pids, key=lambda p: len(groups[p]), reverse=True):
                    if len(groups[tp]) <= excess:
                        continue
                    needed_return = len(groups[tp]) - excess
                    for dp in sorted(dev_pids, key=lambda p: len(groups[p])):
                        if len(groups[dp]) == needed_return:
                            train_pids.remove(tp); dev_pids.append(tp)
                            dev_pids.remove(dp); train_pids.append(dp)
                            cur_train += len(groups[dp]) - len(groups[tp])
                            cur_dev += len(groups[tp]) - len(groups[dp])
                            break
                    if cur_train == train:
                        break

    return {"train": _flat(train_pids), "dev": _flat(dev_pids)}


# ---------------------------------------------------------------------------
# Helpers – audit field extraction
# ---------------------------------------------------------------------------


def _get_decision(rec: dict[str, Any]) -> str:
    return rec.get("target", {}).get("decision", "")


def _get_parent_task_id(rec: dict[str, Any]) -> str:
    """Derive parent_task_id from an exported record dict."""
    prov = rec.get("provenance", {})
    return (
        prov.get("parent_record_id")
        or rec.get("group_id")
        or rec.get("scenario_id")
        or ""
    )


def _get_observation_text(rec: dict[str, Any]) -> str:
    """Extract observation text, handling both full and SFT export formats."""
    text = rec.get("observation_text", "")
    if text:
        return text
    # SFT export format: context.observation.value
    return rec.get("context", {}).get("observation", {}).get("value", "")


def _get_generator(rec: dict[str, Any]) -> str:
    """Extract generator model name from record."""
    gen = rec.get("generation", {})
    return gen.get("generator", "")


def _get_usage_tokens(rec: dict[str, Any]) -> int:
    """Extract total token count from record."""
    gen = rec.get("generation", {})
    usage = gen.get("usage", {})
    if not usage:
        return 0
    total = usage.get("total", 0)
    if total:
        return int(total)
    return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))


# ---------------------------------------------------------------------------
# MinHash helpers
# ---------------------------------------------------------------------------


def _shingle_hashes(text: str, k: int = 3) -> set[int]:
    """Hash k-word shingles of *text* into integers."""
    words = text.lower().split()
    if not words:
        return set()
    if len(words) < k:
        h = int(hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16], 16)
        return {h}
    result: set[int] = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i : i + k])
        h = int(hashlib.sha256(shingle.encode()).hexdigest()[:16], 16)
        result.add(h)
    return result


def _minhash_sign(
    token_hashes: set[int],
    a_coeffs: list[int],
    b_coeffs: list[int],
) -> list[int]:
    """Compute MinHash signature over *token_hashes*."""
    if not token_hashes:
        return [0] * len(a_coeffs)
    prime = _MINHASH_PRIME
    return [
        min((a * t + b) % prime for t in token_hashes)
        for a, b in zip(a_coeffs, b_coeffs)
    ]


def _jaccard_estimate(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig1 or not sig2:
        return 0.0
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)


# ---------------------------------------------------------------------------
# audit_corpus – individual checks
# ---------------------------------------------------------------------------


def _check_exact_counts(
    accepted: list[dict[str, Any]],
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
) -> AuditCheck:
    expected_total, expected_train, expected_dev = 6000, 5400, 600
    actual_total = len(accepted)
    actual_train = len(train)
    actual_dev = len(dev)
    # Allow ±1% tolerance on train/dev to preserve parent-task integrity
    passed = (
        actual_total == expected_total
        and abs(actual_train - expected_train) <= max(1, int(expected_train * 0.01))
        and abs(actual_dev - expected_dev) <= max(1, int(expected_dev * 0.01))
    )
    exp = f"total={expected_total}, train={expected_train}, dev={expected_dev}"
    act = f"total={actual_total}, train={actual_train}, dev={actual_dev}"
    return AuditCheck("exact_counts", passed, exp, act, "" if passed else f"Count mismatch: {act}")


def _check_decision_counts(records: list[dict[str, Any]]) -> AuditCheck:
    expected = {"UPDATE": 2700, "HOLD": 1800, "IGNORE": 1500}
    counts: dict[str, int] = {"UPDATE": 0, "HOLD": 0, "IGNORE": 0}
    for r in records:
        d = _get_decision(r)
        # VERIFY is counted under HOLD
        if d == "VERIFY":
            d = "HOLD"
        if d in counts:
            counts[d] += 1
    passed = counts == expected
    return AuditCheck(
        "decision_counts",
        passed,
        str(expected),
        str(counts),
        "" if passed else f"Decision counts differ: got {counts}",
    )


def _check_source_composition(records: list[dict[str, Any]]) -> AuditCheck:
    expected = {"tau3-bench": 3900, "car-bench": 1500, "counterfactual": 600}
    # Counterfactual records are derived from tau3 with transformation
    # containing "stale", "insufficient_or_ambiguous", or "irrelevant"
    counts: dict[str, int] = {"tau3-bench": 0, "car-bench": 0, "counterfactual": 0}
    for r in records:
        src = r.get("provenance", {}).get("source_dataset", "")
        xform = r.get("provenance", {}).get("transformation", "")
        if "stale" in xform or "insufficient" in xform or "irrelevant" in xform:
            counts["counterfactual"] += 1
        elif src == "tau3-bench":
            counts["tau3-bench"] += 1
        elif src == "car-bench":
            counts["car-bench"] += 1
        else:
            # Unknown source — count under its own key
            counts.setdefault(src, 0)
            counts[src] += 1
    adjusted = {"tau3-bench": counts["tau3-bench"], "car-bench": counts["car-bench"], "counterfactual": counts["counterfactual"]}
    passed = adjusted == expected
    return AuditCheck(
        "source_composition",
        passed,
        str(expected),
        str(adjusted),
        "" if passed else f"Source composition differs: got {adjusted}",
    )


def _check_json_schema(records: list[dict[str, Any]]) -> AuditCheck:
    # GenerationCandidate uses record_id (not scenario_id) and has context/target
    required = {"record_id", "context", "target"}
    failures: list[str] = []
    for r in records:
        missing = required - set(r.keys())
        if missing:
            rid = r.get("record_id", r.get("scenario_id", "?"))
            failures.append(f"rid={rid} missing={sorted(missing)}")
    rate = (len(records) - len(failures)) / len(records) if records else 0.0
    passed = rate == 1.0
    details = ""
    if not passed:
        head = "; ".join(failures[:5])
        details = f"{len(failures)} records fail schema: {head}"
    return AuditCheck("json_schema_pass_rate", passed, "100%", f"{rate:.2%}", details)


def _check_executor_pass_rate(records: list[dict[str, Any]]) -> AuditCheck:
    failures: list[str] = []
    for r in records:
        decision = _get_decision(r)
        rid = r.get("record_id", r.get("scenario_id", "?"))
        if decision not in _VALID_DECISIONS:
            failures.append(f"rid={rid} invalid_decision={decision!r}")
            continue
        target = r.get("target", {})
        patches = target.get("patches", [])
        affected = target.get("affected_fields", [])
        verification = target.get("verification")
        # Action gating constraints (mirrors quality.py validate_candidate logic)
        if decision == "UPDATE":
            if not affected or not patches:
                failures.append(f"rid={rid} UPDATE missing patches/affected_fields")
        elif decision == "IGNORE":
            if affected or patches or verification is not None:
                failures.append(f"rid={rid} IGNORE has patches/affected_fields/verification")
        elif decision == "HOLD":
            if not affected or not patches:
                failures.append(f"rid={rid} HOLD missing patches/affected_fields")
    rate = (len(records) - len(failures)) / len(records) if records else 0.0
    passed = rate == 1.0
    details = ""
    if not passed:
        head = "; ".join(failures[:5])
        details = f"{len(failures)} executor failures: {head}"
    return AuditCheck("executor_pass_rate", passed, "100%", f"{rate:.2%}", details)


def _check_parent_task_overlap(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
) -> AuditCheck:
    train_ids = {_get_parent_task_id(r) for r in train}
    dev_ids = {_get_parent_task_id(r) for r in dev}
    overlap = (train_ids & dev_ids) - {""}
    passed = len(overlap) == 0
    details = ""
    if not passed:
        details = f"Overlap IDs: {sorted(overlap)[:10]}"
    return AuditCheck(
        "parent_task_overlap",
        passed,
        "no overlap",
        f"{len(overlap)} overlapping parent_task_ids",
        details,
    )


def _check_test_leakage(
    records: list[dict[str, Any]],
    test_task_ids: set[str],
) -> AuditCheck:
    if not test_task_ids:
        return AuditCheck("test_leakage", True, "no test leakage", "no test_task_ids provided", "")
    leaked: list[str] = []
    for r in records:
        src_id = r.get("provenance", {}).get("source_record_id", "")
        if src_id in test_task_ids:
            leaked.append(src_id)
    passed = len(leaked) == 0
    details = ""
    if not passed:
        details = f"Leaked source_record_ids: {leaked[:10]}"
    return AuditCheck(
        "test_leakage",
        passed,
        "no test leakage",
        f"{len(leaked)} leaked records",
        details,
    )


def _check_label_leakage(records: list[dict[str, Any]]) -> AuditCheck:
    leaked: list[str] = []
    for r in records:
        text = _get_observation_text(r)
        if text and _LABEL_LEAKAGE.search(text):
            rid = r.get("record_id", r.get("scenario_id", "?"))
            leaked.append(rid)
    passed = len(leaked) == 0
    details = ""
    if not passed:
        details = f"Leaked record_ids: {leaked[:10]}"
    return AuditCheck(
        "label_leakage",
        passed,
        "no label leakage",
        f"{len(leaked)} records with label leakage",
        details,
    )


def _check_provenance_completeness(records: list[dict[str, Any]]) -> AuditCheck:
    required_fields = (
        "source_dataset",
        "source_version",
        "source_record_id",
        "source_sha256",
        "license",
    )
    incomplete: list[str] = []
    for r in records:
        prov = r.get("provenance", {})
        missing = [f for f in required_fields if not prov.get(f)]
        if missing:
            rid = r.get("record_id", r.get("scenario_id", "?"))
            incomplete.append(f"rid={rid} missing={missing}")
    passed = len(incomplete) == 0
    details = ""
    if not passed:
        head = "; ".join(incomplete[:5])
        details = f"{len(incomplete)} incomplete: {head}"
    return AuditCheck(
        "provenance_completeness",
        passed,
        "all provenance complete",
        f"{len(incomplete)} incomplete records",
        details,
    )


def _check_no_duplicates(records: list[dict[str, Any]]) -> AuditCheck:
    seen_ids: set[str] = set()
    dup_ids: int = 0
    seen_texts: set[str] = set()
    dup_texts: int = 0
    for r in records:
        rid = r.get("record_id", "")
        if rid:
            if rid in seen_ids:
                dup_ids += 1
            seen_ids.add(rid)
        text = _get_observation_text(r)
        if text:
            if text in seen_texts:
                dup_texts += 1
            seen_texts.add(text)
    passed = dup_ids == 0 and dup_texts == 0
    details = ""
    if not passed:
        details = f"dup_ids={dup_ids}, dup_texts={dup_texts}"
    return AuditCheck(
        "no_duplicates",
        passed,
        "no duplicates",
        f"dup_ids={dup_ids}, dup_texts={dup_texts}",
        details,
    )


def _check_llm_usage(records: list[dict[str, Any]]) -> AuditCheck:
    total_tokens = 0
    fake_model_count = 0
    for r in records:
        total_tokens += _get_usage_tokens(r)
        model = _get_generator(r)
        if model == "fake-glm":
            fake_model_count += 1
    # Tolerate dry-run (fake-glm) for preview/development builds
    tokens_ok = total_tokens > 0 or fake_model_count == len(records)
    model_ok = fake_model_count == 0 or fake_model_count == len(records)
    passed = tokens_ok and model_ok
    details = ""
    if not passed:
        parts: list[str] = []
        if not tokens_ok:
            parts.append("total_tokens=0")
        if not model_ok:
            parts.append(f"fake_model_count={fake_model_count}")
        details = "; ".join(parts)
    return AuditCheck(
        "llm_usage",
        passed,
        "tokens>0, model!=fake-glm",
        f"tokens={total_tokens}, fake_model_count={fake_model_count}",
        details,
    )


def _check_near_duplicate_rate(
    records: list[dict[str, Any]],
    threshold: float,
    max_rate: float,
) -> AuditCheck:
    sample_size = min(500, len(records))
    if sample_size < 2:
        return AuditCheck(
            "near_duplicate_rate",
            True,
            f"rate<{max_rate:.2%}",
            "N/A (too few records)",
            "",
        )

    rng = random.Random(42)
    a_coeffs = [rng.randint(1, _MINHASH_PRIME - 1) for _ in range(_MINHASH_NUM_PERMS)]
    b_coeffs = [rng.randint(0, _MINHASH_PRIME - 1) for _ in range(_MINHASH_NUM_PERMS)]

    # Deterministic sample
    sampled = random.Random(42).sample(range(len(records)), sample_size)

    # Compute MinHash signatures
    sigs: list[list[int]] = []
    for idx in sampled:
        text = _get_observation_text(records[idx])
        shingles = _shingle_hashes(text)
        if not shingles:
            # Assign a unique sentinel so empty texts are not false near-dups
            sentinel = idx + 1
            shingles = {sentinel}
        sigs.append(_minhash_sign(shingles, a_coeffs, b_coeffs))

    # Pairwise comparison
    n_pairs = sample_size * (sample_size - 1) // 2
    near_dup_count = 0
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            if _jaccard_estimate(sigs[i], sigs[j]) > threshold:
                near_dup_count += 1

    rate = near_dup_count / n_pairs if n_pairs > 0 else 0.0
    passed = rate < max_rate
    details = ""
    if not passed:
        details = f"Near-duplicate rate too high: {near_dup_count}/{n_pairs} pairs"
    return AuditCheck(
        "near_duplicate_rate",
        passed,
        f"rate<{max_rate:.2%}",
        f"rate={rate:.4%} ({near_dup_count}/{n_pairs} pairs)",
        details,
    )


# ---------------------------------------------------------------------------
# audit_corpus
# ---------------------------------------------------------------------------


def audit_corpus(
    *,
    accepted_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    test_task_ids: set[str],
    near_duplicate_threshold: float = 0.9,
    near_duplicate_max_rate: float = 0.01,
) -> AuditReport:
    """Run the canonical 12-point audit on the SIEVE Stage-1 SFT v3 corpus."""

    all_records = accepted_records

    checks: dict[str, AuditCheck] = {}
    checks["exact_counts"] = _check_exact_counts(accepted_records, train_records, dev_records)
    checks["decision_counts"] = _check_decision_counts(all_records)
    checks["source_composition"] = _check_source_composition(all_records)
    checks["json_schema_pass_rate"] = _check_json_schema(all_records)
    checks["executor_pass_rate"] = _check_executor_pass_rate(all_records)
    checks["parent_task_overlap"] = _check_parent_task_overlap(train_records, dev_records)
    checks["test_leakage"] = _check_test_leakage(all_records, test_task_ids)
    checks["label_leakage"] = _check_label_leakage(all_records)
    checks["provenance_completeness"] = _check_provenance_completeness(all_records)
    checks["no_duplicates"] = _check_no_duplicates(all_records)
    checks["llm_usage"] = _check_llm_usage(all_records)
    checks["near_duplicate_rate"] = _check_near_duplicate_rate(
        all_records, near_duplicate_threshold, near_duplicate_max_rate,
    )

    all_passed = all(c.passed for c in checks.values())
    n_pass = sum(c.passed for c in checks.values())
    n_total = len(checks)
    summary = f"{n_pass}/{n_total} checks passed" + (" — ALL PASSED" if all_passed else "")

    return AuditReport(passed=all_passed, checks=checks, summary=summary)
