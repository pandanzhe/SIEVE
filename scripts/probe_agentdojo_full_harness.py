from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import yaml


def _ensure_agentdojo_paths(root: Path) -> None:
    for rel in (".agentdojo-src", ".agentdojo-extra"):
        path = root / rel
        if path.exists():
            sys.path.insert(0, str(path))
    agent_pipeline_path = root / ".agentdojo-src" / "agentdojo" / "agent_pipeline"
    if agent_pipeline_path.exists() and "agentdojo.agent_pipeline" not in sys.modules:
        # AgentDojo's package __init__ imports all provider SDK adapters. The
        # local environment has a broken anthropic wheel, but this probe only
        # needs the suite runtime/checkers. Pre-register the package path so
        # submodules can be imported without executing the provider-heavy init.
        package = types.ModuleType("agentdojo.agent_pipeline")
        package.__path__ = [str(agent_pipeline_path)]  # type: ignore[attr-defined]
        sys.modules["agentdojo.agent_pipeline"] = package
    google_llm_name = "agentdojo.agent_pipeline.llms.google_llm"
    if google_llm_name not in sys.modules:
        google_llm_stub = types.ModuleType(google_llm_name)
        google_llm_stub.EMPTY_FUNCTION_NAME = "empty_function_name"  # type: ignore[attr-defined]
        sys.modules[google_llm_name] = google_llm_stub


def _load_smoke_pairs(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("smoke plan must contain a list field named 'pairs'")
    return pairs[:limit] if limit is not None else pairs


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s"


def _text_block(content: str) -> dict[str, str]:
    return {"type": "text", "content": content}


def _tool_result_to_str(tool_result: Any) -> str:
    dump = getattr(tool_result, "model_dump", None)
    if callable(dump):
        return yaml.safe_dump(dump(), allow_unicode=True, sort_keys=False).strip()
    if isinstance(tool_result, list):
        values: list[Any] = []
        for item in tool_result:
            item_dump = getattr(item, "model_dump", None)
            values.append(item_dump() if callable(item_dump) else item)
        return yaml.safe_dump(values, allow_unicode=True, sort_keys=False).strip()
    return str(tool_result)


class _GroundTruthPipeline:
    name = "ground_truth_probe"

    def __init__(self, task: Any) -> None:
        self.task = task

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, Any, Any, list[dict[str, Any]], dict[str, Any]]:
        extra_args = extra_args or {}
        new_messages = list(messages)
        for tool_call in self.task.ground_truth(env):
            new_messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [tool_call],
                    "content": [_text_block("")],
                }
            )
            tool_result, error = runtime.run_function(env, tool_call.function, tool_call.args, raise_on_error=True)
            new_messages.append(
                {
                    "role": "tool",
                    "content": [_text_block(_tool_result_to_str(tool_result))],
                    "tool_call": tool_call,
                    "tool_call_id": getattr(tool_call, "id", None),
                    "error": error,
                }
            )
        new_messages.append(
            {
                "role": "assistant",
                "content": [_text_block(getattr(self.task, "GROUND_TRUTH_OUTPUT", ""))],
                "tool_calls": None,
            }
        )
        return query, runtime, env, new_messages, extra_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe AgentDojo's executable suite and official utility/security "
            "checkers without calling an external LLM provider. This validates "
            "the environment/checker layer, not SIEVE's full action-agent loop."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--smoke-plan",
        type=Path,
        default=Path("outputs/agentdojo/smoke_plan.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/agentdojo/full_harness_probe.json"),
    )
    parser.add_argument("--benchmark-version", default="v1")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="Raise on per-case probe failures.")
    args = parser.parse_args()

    root = args.root.resolve()
    _ensure_agentdojo_paths(root)

    from agentdojo.task_suite.load_suites import get_suite

    smoke_plan = args.smoke_plan if args.smoke_plan.is_absolute() else root / args.smoke_plan
    output = args.output if args.output.is_absolute() else root / args.output
    pairs = _load_smoke_pairs(smoke_plan, args.limit)

    records: list[dict[str, Any]] = []
    started = time.time()
    for index, pair in enumerate(pairs, start=1):
        suite_name = pair["suite"]
        user_task_id = pair["user_task_id"]
        injection_task_id = pair["injection_task_id"]
        record: dict[str, Any] = {
            "pair_index": index - 1,
            "suite": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
        }
        try:
            suite = get_suite(args.benchmark_version, suite_name)
            user_task = suite.get_user_task_by_id(user_task_id)
            injection_task = suite.get_injection_task_by_id(injection_task_id)

            user_pipeline = _GroundTruthPipeline(user_task)
            user_utility, user_security = suite.run_task_with_pipeline(
                user_pipeline,
                user_task,
                injection_task=None,
                injections={},
            )

            injection_pipeline = _GroundTruthPipeline(injection_task)
            injection_utility, injection_security = suite.run_task_with_pipeline(
                injection_pipeline,
                user_task,
                injection_task=injection_task,
                injections={},
            )
            record.update(
                {
                    "ok": True,
                    "user_ground_truth_utility": bool(user_utility),
                    "user_ground_truth_security": bool(user_security),
                    "injection_ground_truth_utility": bool(injection_utility),
                    "injection_ground_truth_security": bool(injection_security),
                    "user_prompt": getattr(user_task, "PROMPT", ""),
                    "injection_goal": getattr(injection_task, "GOAL", ""),
                }
            )
        except Exception as exc:
            record.update({"ok": False, "error": repr(exc)})
            if args.strict:
                raise
        records.append(record)

        elapsed = time.time() - started
        eta = elapsed / index * (len(pairs) - index) if index else 0
        print(
            "[agentdojo-full-probe] "
            f"progress {index}/{len(pairs)} "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )

    total = len(records)
    ok_count = sum(1 for row in records if row["ok"])
    user_utility = sum(1 for row in records if row.get("user_ground_truth_utility"))
    injection_security = sum(1 for row in records if row.get("injection_ground_truth_security"))
    summary = {
        "benchmark_version": args.benchmark_version,
        "smoke_plan": str(smoke_plan),
        "cases": total,
        "ok_rate": ok_count / max(total, 1),
        "user_ground_truth_utility_rate": user_utility / max(total, 1),
        "injection_ground_truth_security_rate": injection_security / max(total, 1),
        "note": (
            "This validates AgentDojo suite loading, environment execution, and official "
            "utility/security checkers with ground-truth pipelines. It does not run a "
            "learned action agent or the SIEVE wrapper inside AgentDojo."
        ),
    }
    payload = {"summary": summary, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[agentdojo-full-probe] done "
        f"cases={total} ok={summary['ok_rate']:.4f} "
        f"user_utility={summary['user_ground_truth_utility_rate']:.4f} "
        f"injection_security={summary['injection_ground_truth_security_rate']:.4f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
