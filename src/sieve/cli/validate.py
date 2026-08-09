from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import load_config, resolve_repo_path
from ..data.generate import generate_toy_trajectories, split_by_scenario
from ..data.io import read_jsonl
from ..data.schema import SFTRecord
from ..data.validation import validate_records, validate_split_isolation


def _load_or_generate_splits(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[SFTRecord]], str]:
    configured = config.get("paths", {})
    if not isinstance(configured, dict) or "data_dir" not in configured:
        raise ValueError("config.paths.data_dir is required")
    data_dir = resolve_repo_path(root, str(configured["data_dir"]))
    paths = {name: data_dir / f"{name}.jsonl" for name in ("train", "dev", "test")}
    if all(path.is_file() for path in paths.values()):
        return {name: read_jsonl(path) for name, path in paths.items()}, "jsonl"

    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("config.data must be a mapping")
    records = generate_toy_trajectories(
        int(data["scenarios"]),
        int(config["seed"]),
        int(data["max_slots"]),
    )
    return (
        split_by_scenario(
            records,
            float(data["train_fraction"]),
            float(data["dev_fraction"]),
            int(config["seed"]),
        ),
        "generated",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SIEVE data logic without training")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(resolve_repo_path(root, args.config))
    splits, source = _load_or_generate_splits(root, config)
    issues = validate_split_isolation(splits)
    for split_records in splits.values():
        issues.extend(validate_records(split_records))
    records = [record for split_records in splits.values() for record in split_records]
    report = {
        "source": source,
        "records": len(records),
        "scenarios": len({record.scenario_id for record in records}),
        "splits": {name: len(items) for name, items in splits.items()},
        "issues": [issue.__dict__ for issue in issues],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
