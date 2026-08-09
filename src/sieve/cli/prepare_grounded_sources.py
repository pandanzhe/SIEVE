from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data_factory.normalize import (
    normalize_agentbench,
    normalize_tau2,
    normalize_toolbench,
    normalize_webarena,
    write_normalized_jsonl,
)


VERSIONS = {
    "tau2-bench": "1d244f5dca42944b67a379b44bfeb9f5748f189d",
    "ToolBench": "d56fdd89faf8c91fa135090b212bb9057ee5cfc2",
    "AgentBench": "d1e4a10db08c87075c78972e48ecc182be03e2d5",
    "WebArena": "dce04686a56253aefba7b18a4fa0937cf1dc987b",
}


def prepare(root: Path, output_dir: Path, per_source_limit: int) -> dict[str, int]:
    repos = root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_records = []
    tau_root = repos / "tau2-bench" / "data" / "tau2" / "domains"
    for domain in ("retail", "airline", "telecom"):
        tau_records.extend(
            normalize_tau2(
                tau_root / domain / "tasks.json",
                VERSIONS["tau2-bench"],
                "MIT",
                limit=per_source_limit,
            )
        )
    tau_records = tau_records[:per_source_limit]

    tool_paths = sorted(
        (repos / "ToolBench" / "data_example" / "answer").glob("**/*.json")
    )
    tool_records = normalize_toolbench(
        tool_paths,
        VERSIONS["ToolBench"],
        "Apache-2.0",
        limit=per_source_limit,
    )

    agent_records = normalize_agentbench(
        repos / "AgentBench" / "data" / "dbbench" / "standard.jsonl",
        VERSIONS["AgentBench"],
        "Apache-2.0",
        limit=per_source_limit,
    )
    web_records = normalize_webarena(
        repos / "webarena" / "config_files" / "test.raw.json",
        VERSIONS["WebArena"],
        "Apache-2.0",
        limit=per_source_limit,
    )

    outputs = {
        "tau2-bench": (tau_records, "tau2-normalized.jsonl", "MIT"),
        "ToolBench": (tool_records, "toolbench-normalized.jsonl", "Apache-2.0"),
        "AgentBench": (agent_records, "agentbench-normalized.jsonl", "Apache-2.0"),
        "WebArena": (web_records, "webarena-normalized.jsonl", "Apache-2.0"),
    }
    counts: dict[str, int] = {}
    manifest_entries = []
    for dataset, (records, filename, license_name) in outputs.items():
        counts[dataset] = write_normalized_jsonl(records, output_dir / filename)
        manifest_entries.append(
            {
                "dataset": dataset,
                "version": VERSIONS[dataset],
                "license": license_name,
                "path": filename,
                "format": "jsonl",
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"sources": manifest_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize pinned open-source records")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="tmp/data_factory/normalized")
    parser.add_argument("--per-source-limit", type=int, default=120)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    counts = prepare(root, output_dir, args.per_source_limit)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

