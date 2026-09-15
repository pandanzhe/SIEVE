from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _string_value(node: ast.AST) -> str | None:
    value = _literal(node)
    if isinstance(value, str):
        return value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant):
                parts.append(str(item.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def _class_number(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _extract_tasks(path: Path, *, kind: str, version: str, suite: str) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prefix = "UserTask" if kind == "user" else "InjectionTask"
    prompt_field = "PROMPT" if kind == "user" else "GOAL"
    rows: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        number = _class_number(node.name, prefix)
        if number is None:
            continue
        fields: dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        fields[target.id] = _string_value(stmt.value) or _literal(stmt.value)
        rows.append(
            {
                "suite": suite,
                "version_dir": version,
                "kind": kind,
                "task_id": f"{kind}_task_{number}",
                "class_name": node.name,
                "difficulty": str(fields.get("DIFFICULTY", "")),
                "prompt": fields.get(prompt_field, ""),
                "ground_truth_output": fields.get("GROUND_TRUTH_OUTPUT", ""),
                "source_file": str(path),
            }
        )
    return rows


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _discover_default_root(root: Path) -> Path:
    candidates = [
        root / ".agentdojo-src" / "agentdojo",
        root / ".agentdojo-pydeps" / "agentdojo",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find local AgentDojo package. Expected .agentdojo-src/agentdojo "
        "or .agentdojo-pydeps/agentdojo under the SIEVE root."
    )


def build_manifest(agentdojo_root: Path) -> dict[str, Any]:
    data_root = agentdojo_root / "data" / "suites"
    default_suites_root = agentdojo_root / "default_suites"
    suites: dict[str, Any] = {}
    all_tasks: list[dict[str, Any]] = []
    for suite_dir in sorted(data_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        injection_vectors_path = suite_dir / "injection_vectors.yaml"
        environment_path = suite_dir / "environment.yaml"
        injection_vectors = _load_yaml(injection_vectors_path) if injection_vectors_path.exists() else {}
        suites[suite_dir.name] = {
            "environment_yaml": str(environment_path),
            "injection_vectors_yaml": str(injection_vectors_path),
            "num_injection_vectors": len(injection_vectors or {}),
            "injection_vectors": [
                {
                    "id": vector_id,
                    "description": payload.get("description", ""),
                    "default": payload.get("default", ""),
                }
                for vector_id, payload in sorted((injection_vectors or {}).items())
            ],
        }
    for version_dir in sorted(default_suites_root.iterdir()):
        if not version_dir.is_dir() or not version_dir.name.startswith("v"):
            continue
        for suite_dir in sorted(version_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            user_tasks_path = suite_dir / "user_tasks.py"
            injection_tasks_path = suite_dir / "injection_tasks.py"
            if user_tasks_path.exists():
                all_tasks.extend(
                    _extract_tasks(user_tasks_path, kind="user", version=version_dir.name, suite=suite_dir.name)
                )
            if injection_tasks_path.exists():
                all_tasks.extend(
                    _extract_tasks(
                        injection_tasks_path,
                        kind="injection",
                        version=version_dir.name,
                        suite=suite_dir.name,
                    )
                )
    summary: dict[str, Any] = {
        "agentdojo_root": str(agentdojo_root),
        "num_suites": len(suites),
        "num_task_definitions": len(all_tasks),
        "by_suite": {},
    }
    for suite_name in suites:
        suite_tasks = [row for row in all_tasks if row["suite"] == suite_name]
        summary["by_suite"][suite_name] = {
            "user_task_definitions": sum(row["kind"] == "user" for row in suite_tasks),
            "injection_task_definitions": sum(row["kind"] == "injection" for row in suite_tasks),
            "injection_vectors": suites[suite_name]["num_injection_vectors"],
        }
    return {"summary": summary, "suites": suites, "task_definitions": all_tasks}


def build_smoke_plan(
    manifest: dict[str, Any],
    *,
    suites: list[str],
    user_tasks_per_suite: int,
    injection_tasks_per_suite: int,
) -> dict[str, Any]:
    tasks = manifest["task_definitions"]
    selected: list[dict[str, Any]] = []
    for suite in suites:
        suite_tasks = [row for row in tasks if row["suite"] == suite]
        user_tasks = [row for row in suite_tasks if row["kind"] == "user"]
        injection_tasks = [row for row in suite_tasks if row["kind"] == "injection"]
        user_latest: dict[str, dict[str, Any]] = {}
        injection_latest: dict[str, dict[str, Any]] = {}
        for row in user_tasks:
            user_latest[row["task_id"]] = row
        for row in injection_tasks:
            injection_latest[row["task_id"]] = row
        chosen_users = list(user_latest.values())[:user_tasks_per_suite]
        chosen_injections = list(injection_latest.values())[:injection_tasks_per_suite]
        for user_task in chosen_users:
            for injection_task in chosen_injections:
                selected.append(
                    {
                        "suite": suite,
                        "user_task_id": user_task["task_id"],
                        "user_prompt": user_task.get("prompt", ""),
                        "injection_task_id": injection_task["task_id"],
                        "injection_goal": injection_task.get("prompt", ""),
                    }
                )
    return {
        "description": "AgentDojo external smoke subset. This is evaluation-only and must not be mixed into SIEVE training data.",
        "num_pairs": len(selected),
        "suites": suites,
        "user_tasks_per_suite": user_tasks_per_suite,
        "injection_tasks_per_suite": injection_tasks_per_suite,
        "pairs": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a locally installed AgentDojo package without importing its "
            "provider-heavy benchmark entrypoint."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="SIEVE repository root.")
    parser.add_argument("--agentdojo-root", type=Path, default=None, help="Path to the local agentdojo package.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/agentdojo/manifest.json"),
        help="Where to write the AgentDojo suite/task manifest.",
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=None,
        help="Optional path for a deterministic AgentDojo smoke-pair plan.",
    )
    parser.add_argument(
        "--smoke-suites",
        nargs="+",
        default=["workspace", "slack"],
        help="Suites to include in the optional smoke plan.",
    )
    parser.add_argument(
        "--user-tasks-per-suite",
        type=int,
        default=5,
        help="Number of user tasks per suite in the optional smoke plan.",
    )
    parser.add_argument(
        "--injection-tasks-per-suite",
        type=int,
        default=2,
        help="Number of injection tasks per suite in the optional smoke plan.",
    )
    parser.add_argument("--print-examples", type=int, default=3, help="Number of task examples to print.")
    args = parser.parse_args()

    root = args.root.resolve()
    agentdojo_root = args.agentdojo_root.resolve() if args.agentdojo_root else _discover_default_root(root)
    manifest = build_manifest(agentdojo_root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.smoke_output is not None:
        smoke_output = args.smoke_output if args.smoke_output.is_absolute() else root / args.smoke_output
        smoke_output.parent.mkdir(parents=True, exist_ok=True)
        smoke_plan = build_smoke_plan(
            manifest,
            suites=args.smoke_suites,
            user_tasks_per_suite=args.user_tasks_per_suite,
            injection_tasks_per_suite=args.injection_tasks_per_suite,
        )
        smoke_output.write_text(json.dumps(smoke_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = manifest["summary"]
    print(
        "[agentdojo-inspect] done "
        f"suites={summary['num_suites']} "
        f"task_definitions={summary['num_task_definitions']} "
        f"output={output}",
        flush=True,
    )
    if args.smoke_output is not None:
        print(
            "[agentdojo-inspect] smoke_plan "
            f"pairs={smoke_plan['num_pairs']} "
            f"output={smoke_output}",
            flush=True,
        )
    for suite_name, suite_summary in sorted(summary["by_suite"].items()):
        print(
            "[agentdojo-inspect] suite "
            f"{suite_name} "
            f"user_defs={suite_summary['user_task_definitions']} "
            f"injection_defs={suite_summary['injection_task_definitions']} "
            f"injection_vectors={suite_summary['injection_vectors']}",
            flush=True,
        )
    if args.print_examples > 0:
        for row in manifest["task_definitions"][: args.print_examples]:
            prompt = str(row.get("prompt") or "").replace("\n", " ")
            print(
                "[agentdojo-inspect] example "
                f"{row['version_dir']}/{row['suite']}/{row['task_id']} "
                f"kind={row['kind']} prompt={prompt[:180]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
