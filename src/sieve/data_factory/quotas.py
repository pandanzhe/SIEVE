from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..core.types import Decision
from .models import ObservationType, QuotaCell, VerificationAction


_PREVIEW_ROWS: dict[ObservationType, tuple[int, int, int]] = {
    ObservationType.NEW_CONSISTENT: (16, 3, 2),
    ObservationType.EXPLICIT_CONFLICT: (9, 6, 6),
    ObservationType.IMPLICIT_CONFLICT: (8, 6, 7),
    ObservationType.STALE: (8, 10, 3),
    ObservationType.IRRELEVANT: (4, 15, 2),
    ObservationType.TOOL_ERROR_OR_LOW_TRUST: (8, 7, 6),
    ObservationType.INSUFFICIENT_OR_AMBIGUOUS: (3, 2, 9),
}

_FULL_ROWS: dict[ObservationType, tuple[int, int, int]] = {
    ObservationType.NEW_CONSISTENT: (850, 50, 0),
    ObservationType.EXPLICIT_CONFLICT: (700, 100, 100),
    ObservationType.IMPLICIT_CONFLICT: (650, 100, 150),
    ObservationType.STALE: (700, 150, 50),
    ObservationType.IRRELEVANT: (0, 400, 500),
    ObservationType.TOOL_ERROR_OR_LOW_TRUST: (800, 50, 50),
    ObservationType.INSUFFICIENT_OR_AMBIGUOUS: (500, 50, 50),
}


def _cells(
    rows: dict[ObservationType, tuple[int, int, int]],
    hold_verify: int,
) -> list[QuotaCell]:
    result: list[QuotaCell] = []
    remaining_verify = hold_verify
    for observation_type, (update, ignore, hold) in rows.items():
        result.append(QuotaCell(observation_type, Decision.UPDATE, update))
        result.append(QuotaCell(observation_type, Decision.IGNORE, ignore))
        verify = min(hold, remaining_verify)
        defer = hold - verify
        if verify:
            result.append(
                QuotaCell(
                    observation_type,
                    Decision.HOLD,
                    verify,
                    VerificationAction.VERIFY,
                )
            )
            remaining_verify -= verify
        if defer:
            result.append(
                QuotaCell(
                    observation_type,
                    Decision.HOLD,
                    defer,
                    VerificationAction.DEFER,
                )
            )
    if remaining_verify:
        raise RuntimeError("HOLD/VERIFY allocation exceeds HOLD quota")
    return result


def preview_quota_cells() -> list[QuotaCell]:
    return _cells(_PREVIEW_ROWS, hold_verify=21)


def full_quota_cells() -> list[QuotaCell]:
    return _cells(_FULL_ROWS, hold_verify=900)


def count_by(cells: Iterable[QuotaCell], attribute: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for cell in cells:
        value = getattr(cell, attribute)
        key = value.value if hasattr(value, "value") else str(value)
        counts[key] += cell.count
    return dict(counts)
