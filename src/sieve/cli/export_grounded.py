from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data_factory.export import export_training_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export accepted audit data for SIEVE SFT")
    parser.add_argument("--root", default=".")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    source = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    counts = export_training_splits(source, output_dir, seed=args.seed)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
