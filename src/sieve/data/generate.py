from __future__ import annotations

import json
import random
from pathlib import Path

from ..core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    Decision,
    EvidenceLedger,
    Observation,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
    VerificationRequest,
)
from .schema import SFTRecord


def _context(index: int, observation: Observation, max_slots: int) -> RevisionContext:
    entity = f"order-{index}"
    state = BeliefState(
        max_slots,
        [BeliefSlot("payment_status", "unpaid", SlotStatus.TRUSTED, "official_payment_api", 0, 0, entity)],
    )
    risk = RiskEnvelope("ship_order", ("payment_status",), RiskLevel.HIGH, False)
    return RevisionContext(state, observation, "ship paid orders", risk, Budget(2, 5, 8), EvidenceLedger(4))


def generate_toy_trajectories(scenarios: int, seed: int, max_slots: int) -> list[SFTRecord]:
    """Build three-event toy trajectories: invalid, uncertain, then authoritative."""
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    rng = random.Random(seed)
    records: list[SFTRecord] = []
    for index in range(scenarios):
        entity = f"order-{index}"
        now = 100 + index
        clean = Observation("payment_status", "paid", "official_payment_api", now, now, entity, "current_order", True, "clean")
        weak = Observation("payment_status", "paid", "cached_page", now, None, entity, "current_order", True, "weak_source")
        invalid_kind = rng.choice(["stale", "wrong_entity", "irrelevant"])
        if invalid_kind == "stale":
            invalid = Observation("payment_status", "paid", "archive", now, 0, entity, "current_order", True, invalid_kind)
        elif invalid_kind == "wrong_entity":
            invalid = Observation("payment_status", "paid", "official_payment_api", now, now, f"other-{entity}", "current_order", True, invalid_kind)
        else:
            invalid = Observation("weather", "rain", "weather_api", now, now, entity, "current_order", False, invalid_kind)

        update = RevisionOutput(
            Decision.UPDATE,
            ("payment_status",),
            (Patch(PatchOp.SET_VALUE, "payment_status", "paid"),),
        )
        hold = RevisionOutput(
            Decision.HOLD,
            ("payment_status",),
            (Patch(PatchOp.SET_STATUS, "payment_status", SlotStatus.PENDING.value),),
            VerificationRequest("official_payment_api", "payment_status"),
        )
        ignore = RevisionOutput(Decision.IGNORE)
        scenario = f"scenario-{index:05d}"
        records.extend(
            [
                SFTRecord(scenario, _context(index, invalid, max_slots), ignore, step_index=0),
                SFTRecord(scenario, _context(index, weak, max_slots), hold, step_index=1),
                SFTRecord(scenario, _context(index, clean, max_slots), update, step_index=2),
            ]
        )
    return records


def split_by_scenario(
    records: list[SFTRecord], train_fraction: float, dev_fraction: float, seed: int
) -> dict[str, list[SFTRecord]]:
    if train_fraction <= 0 or dev_fraction < 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("invalid split fractions")
    groups = sorted({record.scenario_id for record in records})
    if len(groups) < 3:
        raise ValueError("at least three scenarios are required for non-empty splits")
    random.Random(seed).shuffle(groups)
    train_end = min(max(1, int(len(groups) * train_fraction)), len(groups) - 2)
    dev_count = max(1, int(len(groups) * dev_fraction))
    dev_end = min(train_end + dev_count, len(groups) - 1)
    assignment = {
        group: "train" if i < train_end else "dev" if i < dev_end else "test"
        for i, group in enumerate(groups)
    }
    result = {"train": [], "dev": [], "test": []}
    for record in records:
        result[assignment[record.scenario_id]].append(record)
    return result


def write_splits(splits: dict[str, list[SFTRecord]], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (path / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
