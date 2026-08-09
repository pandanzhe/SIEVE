from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data_factory.publish import publish_canonical_sft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish canonical Stage-1 SFT data")
    parser.add_argument("--root", default=".")
    parser.add_argument("--input", required=True, help="Accepted audit JSONL")
    parser.add_argument("--split-dir", required=True, help="Exported split directory")
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--quality-report")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    manifest = publish_canonical_sft(
        root / args.input,
        root / args.split_dir,
        root / args.output_dir,
        prompt_version=args.prompt_version,
        quality_report_path=(root / args.quality_report if args.quality_report else None),
    )
    print(json.dumps({
        "output_dir": str((root / args.output_dir).resolve()),
        "source_records": manifest["source"]["records"],
        "train_records": manifest["train"]["records"],
        "dev_records": manifest["validation"]["records"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
