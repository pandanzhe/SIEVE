from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ensure_output_dirs, load_config, resolve_repo_path
from ..data.generate import generate_toy_trajectories, split_by_scenario, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired SIEVE training data")
    parser.add_argument("--config", default="configs/toy.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(resolve_repo_path(root, args.config))
    paths = ensure_output_dirs(root, config)
    data = config["data"]
    records = generate_toy_trajectories(int(data["scenarios"]), int(config["seed"]), int(data["max_slots"]))
    splits = split_by_scenario(
        records, float(data["train_fraction"]), float(data["dev_fraction"]), int(config["seed"])
    )
    write_splits(splits, paths["data_dir"])
    print({name: len(items) for name, items in splits.items()})


if __name__ == "__main__":
    main()
