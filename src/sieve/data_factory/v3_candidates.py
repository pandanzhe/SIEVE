"""Deterministic candidate builder for Stage-1 SFT v3 with exact action allocation."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from .models import (
    Difficulty,
    GenerationCandidate,
    GroundedSourceRecord,
    ObservationType,
    Provenance,
    QuotaCell,
    VerificationAction,
)
from .rules import build_candidate, authoritative_source, _atomic_grounding_terms
from .trajectories import build_verification_successor
from ..core.types import Decision


# ── Exact allocation tables ──────────────────────────────────────────────
# Each source block uses 45/30/25 split: UPDATE/HOLD+VERIFY/IGNORE
# Safety counterfactuals are derived FROM tau3 tasks (not a separate source).

_TAU3_TOTAL = 3900
_TAU3_UPDATE = 1755
_TAU3_HOLD = 1170
_TAU3_IGNORE = 975

_CAR_TOTAL = 1500
_CAR_UPDATE = 675
_CAR_HOLD = 450
_CAR_IGNORE = 375

_COUNTER_TOTAL = 600
_COUNTER_UPDATE = 270
_COUNTER_HOLD = 180
_COUNTER_IGNORE = 150

assert _TAU3_UPDATE + _TAU3_HOLD + _TAU3_IGNORE == _TAU3_TOTAL
assert _CAR_UPDATE + _CAR_HOLD + _CAR_IGNORE == _CAR_TOTAL
assert _COUNTER_UPDATE + _COUNTER_HOLD + _COUNTER_IGNORE == _COUNTER_TOTAL
assert _TAU3_TOTAL + _CAR_TOTAL + _COUNTER_TOTAL == 6000

# With HOLD+VERIFY pairs each contributing one output candidate (the HOLD,
# oracle-enriched with the successor state), the expected decision split is:
#   UPDATE  = 1755 + 675 + 270 = 2700
#   HOLD    = 1170 + 450 + 180 = 1800
#   IGNORE  =  975 + 375 + 150 = 1500
_EXPECTED_UPDATE = _TAU3_UPDATE + _CAR_UPDATE + _COUNTER_UPDATE
_EXPECTED_HOLD = _TAU3_HOLD + _CAR_HOLD + _COUNTER_HOLD
_EXPECTED_IGNORE = _TAU3_IGNORE + _CAR_IGNORE + _COUNTER_IGNORE


# ── Distractor selection ─────────────────────────────────────────────────

def get_distractor(
    sources: list[GroundedSourceRecord],
    source: GroundedSourceRecord,
    rng: random.Random,
) -> GroundedSourceRecord | None:
    """Find a record with the same domain but a different entity.

    Returns ``None`` when no such record exists; the caller should
    handle this by passing ``distractor=None`` to :func:`build_candidate`,
    which falls back to a synthetic distractor entity.
    """
    candidates = [
        s for s in sources
        if s.domain == source.domain and s.entity != source.entity
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


# ── Record-id helper ─────────────────────────────────────────────────────

def _with_v3_ids(candidate: GenerationCandidate, seq: int) -> GenerationCandidate:
    """Stamp a candidate with the v3 record_id format ``sieve-v3-{seq:06d}``."""
    return replace(candidate, record_id=f"sieve-v3-{seq:06d}")


# ── Main entry point ─────────────────────────────────────────────────────

def build_v3_candidates(
    tau3_records: list[GroundedSourceRecord],
    car_records: list[GroundedSourceRecord],
    *,
    total: int = 6000,
    seed: int = 42,
) -> list[GenerationCandidate]:
    """Create exactly *total* candidates with the v3 action allocation.

    Allocation (default total = 6 000):
        tau3-derived:          3 900 (1 755 UPDATE, 1 170 HOLD+VERIFY,   975 IGNORE)
        CAR-derived:           1 500 (  675 UPDATE,   450 HOLD+VERIFY,   375 IGNORE)
        Safety counterfactual:   600 (  270 UPDATE,   180 HOLD+VERIFY,   150 IGNORE)

    HOLD+VERIFY pairs contribute a single output candidate whose
    ``oracle_state`` is enriched with the post-verification belief
    state.  This keeps the output cardinality at exactly *total* while
    still exercising the full HOLD -> VERIFY -> UPDATE trajectory
    internally.

    Parameters
    ----------
    tau3_records:
        Grounded source records derived from tau3 task traces.
    car_records:
        Grounded source records derived from CAR (contrastive
        attribution record) traces.
    total:
        Expected output count (default 6 000).
    seed:
        Seed for the deterministic :class:`random.Random` instance.

    Returns
    -------
    list[GenerationCandidate]
        Exactly *total* candidates with unique ``record_id`` values in
        the ``sieve-v3-{sequence:06d}`` format.

    Raises
    ------
    ValueError
        If either source pool is empty.
    RuntimeError
        If the produced candidate count does not equal *total*.
    """
    if not tau3_records:
        raise ValueError("tau3_records must not be empty")
    if not car_records:
        raise ValueError("car_records must not be empty")

    rng = random.Random(seed)

    # Shuffle source records within each pool for variety while
    # remaining fully deterministic.
    tau3_pool = list(tau3_records)
    car_pool = list(car_records)
    rng.shuffle(tau3_pool)
    rng.shuffle(car_pool)

    seq = 0
    result: list[GenerationCandidate] = []

    # ── tau3-derived block (3 900) ──────────────────────────────────────
    tau3_idx = 0

    # 1 755 standalone UPDATE (NEW_CONSISTENT)
    for _ in range(_TAU3_UPDATE):
        source = tau3_pool[tau3_idx % len(tau3_pool)]
        tau3_idx += 1
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1),
            sequence=seq,
        )
        result.append(_with_v3_ids(candidate, seq))

    # 1 170 HOLD + VERIFY pairs
    for _ in range(_TAU3_HOLD):
        source = tau3_pool[tau3_idx % len(tau3_pool)]
        tau3_idx += 1
        seq += 1
        hold = build_candidate(
            source,
            QuotaCell(
                ObservationType.TOOL_ERROR_OR_LOW_TRUST,
                Decision.HOLD,
                1,
                VerificationAction.VERIFY,
            ),
            sequence=seq,
        )
        hold = _with_v3_ids(hold, seq)
        # Build the verification successor to exercise the full
        # HOLD -> VERIFY -> UPDATE trajectory.  Advance the sequence
        # counter so record_ids remain unique, but only emit the HOLD
        # candidate whose oracle_state is enriched with the
        # post-verification belief state.
        seq += 1
        successor = build_verification_successor(hold, sequence=seq)
        hold = replace(hold, oracle_state=successor.oracle_state)
        result.append(hold)

    # 975 wrong-entity IGNORE (EXPLICIT_CONFLICT)
    for _ in range(_TAU3_IGNORE):
        source = tau3_pool[tau3_idx % len(tau3_pool)]
        tau3_idx += 1
        distractor = get_distractor(tau3_pool, source, rng)
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.EXPLICIT_CONFLICT, Decision.IGNORE, 1),
            sequence=seq,
            distractor=distractor,
        )
        result.append(_with_v3_ids(candidate, seq))

    # ── CAR-derived block (1 500) ───────────────────────────────────────
    car_idx = 0

    # 675 standalone UPDATE (NEW_CONSISTENT)
    for _ in range(_CAR_UPDATE):
        source = car_pool[car_idx % len(car_pool)]
        car_idx += 1
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1),
            sequence=seq,
        )
        result.append(_with_v3_ids(candidate, seq))

    # 450 HOLD + VERIFY pairs
    for _ in range(_CAR_HOLD):
        source = car_pool[car_idx % len(car_pool)]
        car_idx += 1
        seq += 1
        hold = build_candidate(
            source,
            QuotaCell(
                ObservationType.TOOL_ERROR_OR_LOW_TRUST,
                Decision.HOLD,
                1,
                VerificationAction.VERIFY,
            ),
            sequence=seq,
        )
        hold = _with_v3_ids(hold, seq)
        seq += 1
        successor = build_verification_successor(hold, sequence=seq)
        hold = replace(hold, oracle_state=successor.oracle_state)
        result.append(hold)

    # 375 wrong-entity IGNORE (EXPLICIT_CONFLICT)
    for _ in range(_CAR_IGNORE):
        source = car_pool[car_idx % len(car_pool)]
        car_idx += 1
        distractor = get_distractor(car_pool, source, rng)
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.EXPLICIT_CONFLICT, Decision.IGNORE, 1),
            sequence=seq,
            distractor=distractor,
        )
        result.append(_with_v3_ids(candidate, seq))

    # ── Safety counterfactual block (600, derived from tau3 tasks) ──────
    safety_idx = 0

    # 270 UPDATE from stale observations
    for _ in range(_COUNTER_UPDATE):
        source = tau3_pool[safety_idx % len(tau3_pool)]
        safety_idx += 1
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.STALE, Decision.UPDATE, 1),
            sequence=seq,
        )
        result.append(_with_v3_ids(candidate, seq))

    # 180 HOLD + VERIFY from insufficient / ambiguous evidence
    for _ in range(_COUNTER_HOLD):
        source = tau3_pool[safety_idx % len(tau3_pool)]
        safety_idx += 1
        seq += 1
        hold = build_candidate(
            source,
            QuotaCell(
                ObservationType.INSUFFICIENT_OR_AMBIGUOUS,
                Decision.HOLD,
                1,
                VerificationAction.VERIFY,
            ),
            sequence=seq,
        )
        hold = _with_v3_ids(hold, seq)
        seq += 1
        successor = build_verification_successor(hold, sequence=seq)
        hold = replace(hold, oracle_state=successor.oracle_state)
        result.append(hold)

    # 150 IGNORE from irrelevant observations
    for _ in range(_COUNTER_IGNORE):
        source = tau3_pool[safety_idx % len(tau3_pool)]
        safety_idx += 1
        seq += 1
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.IRRELEVANT, Decision.IGNORE, 1),
            sequence=seq,
        )
        result.append(_with_v3_ids(candidate, seq))

    # ── Validation ──────────────────────────────────────────────────────
    if len(result) != total:
        raise RuntimeError(
            f"v3 candidate schedule produced {len(result)} records, "
            f"expected {total}"
        )

    decision_counts = Counter(c.target.decision for c in result)
    actual_update = decision_counts.get(Decision.UPDATE, 0)
    actual_hold = decision_counts.get(Decision.HOLD, 0)
    actual_ignore = decision_counts.get(Decision.IGNORE, 0)
    if (actual_update, actual_hold, actual_ignore) != (
        _EXPECTED_UPDATE,
        _EXPECTED_HOLD,
        _EXPECTED_IGNORE,
    ):
        raise RuntimeError(
            f"v3 decision distribution ({actual_update} UPDATE, "
            f"{actual_hold} HOLD, {actual_ignore} IGNORE) "
            f"does not match ({_EXPECTED_UPDATE}, {_EXPECTED_HOLD}, {_EXPECTED_IGNORE})"
        )

    return result
