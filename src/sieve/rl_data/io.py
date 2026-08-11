from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .schema import RLScenario


def read_scenarios(path: str | Path) -> list[RLScenario]:
    source = Path(path)
    scenarios: list[RLScenario] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                scenarios.append(RLScenario.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid scenario at {source}:{line_number}: {error}") from error
    return scenarios


def write_scenarios(path: str | Path, scenarios: Iterable[RLScenario]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for scenario in scenarios:
                handle.write(
                    json.dumps(
                        scenario.to_dict(), ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
