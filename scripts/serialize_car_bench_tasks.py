#!/usr/bin/env python3
"""
Serialize CAR-bench training tasks to JSONL format for use by the SIEVE v3 data factory.

This script reads task definitions from the CAR-bench repository and serializes
them into JSONL files, one line per task. It supports multiple loading strategies
with graceful fallback:

  0. Mock heavy dependencies (litellm, etc.) and import the reference Python task
     files (tasks_base.py, tasks_disambiguation.py) which contain fully-populated
     Task Pydantic objects.
  1. Import the reference Python task files without mocking (requires full deps).
  2. Load tasks from the HuggingFace car-bench dataset (requires `datasets` package).
  3. Produce minimal stub records with task_id and task_type populated plus a
     `stub: true` flag, so downstream consumers can still operate.

Usage:
    python scripts/serialize_car_bench_tasks.py \
        --car-bench-repo tmp/car-bench-repo \
        --output-dir data/raw/car-bench/serialized

Output files:
    <output-dir>/base_train.jsonl
    <output-dir>/disambiguation_train.jsonl

Each JSONL line is a JSON dict:
    {
      "task_id": "base_0",
      "task_type": "base",
      "calendar_id": "...",
      "persona": "...",
      "instruction": "...",
      "actions": [{"name": "tool_name", "kwargs": {...}, "index": 0}],
      "context_init_config": {},
      "split": "base_train"
    }
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import types as pytypes
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_project_root() -> str:
    """Return the project root (the repo that contains this script)."""
    # This script lives at <project_root>/scripts/serialize_car_bench_tasks.py
    return str(Path(__file__).resolve().parent.parent)


def resolve_path(path: str, root: str) -> str:
    """Resolve *path* — if already absolute, return as-is; otherwise join with *root*."""
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


# ---------------------------------------------------------------------------
# CAR-bench sys.path setup
# ---------------------------------------------------------------------------

def add_car_bench_to_path(car_bench_repo: str) -> None:
    """Insert the CAR-bench repo root into sys.path so ``car_bench`` is importable."""
    repo_abs = os.path.abspath(car_bench_repo)
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)
    logger.info("Added %s to sys.path", repo_abs)


# ---------------------------------------------------------------------------
# Heavy-dependency mocking
# ---------------------------------------------------------------------------

# Modules that the car_bench package imports at load time but that the reference
# task files never *call*.  We install lightweight stubs so the import chain
# succeeds even when litellm / datasets / networkx etc. are not installed.
_HEAVY_DEPS_TO_MOCK: List[str] = [
    "litellm",
    "datasets",
    "huggingface_hub",
    "networkx",
    "pyvis",
    "tiktoken",
]


def _install_mock_module(name: str) -> None:
    """Install a do-nothing module under *name* in ``sys.modules`` if it is not already present."""
    if name in sys.modules:
        return
    mod = pytypes.ModuleType(name)
    mod.__path__ = []          # pretend it is a package so sub-imports also resolve
    mod.__package__ = name
    # Provide a callable stub for common entry points (e.g. litellm.completion)
    for attr in ("completion", "encode", "decode", "load_dataset"):
        setattr(mod, attr, lambda *a, **kw: None)
    sys.modules[name] = mod


def install_mock_deps() -> None:
    """
    Pre-populate ``sys.modules`` with lightweight stubs for heavy optional
    dependencies so that ``import car_bench`` succeeds without them.

    This is safe because the reference task files only ever use types/enums
    from the car_bench package — they never invoke litellm.completion etc.
    """
    for dep in _HEAVY_DEPS_TO_MOCK:
        _install_mock_module(dep)
    # litellm exposes sub-modules that are also imported in some code paths
    for sub in ("litellm.cost_map", "litellm.model_cost", "litellm.utils"):
        _install_mock_module(sub)
    logger.debug("Installed mock modules for: %s", _HEAVY_DEPS_TO_MOCK)


def _clear_car_bench_from_sys_modules() -> None:
    """
    Remove all ``car_bench.*`` entries from ``sys.modules`` so that a fresh
    import attempt can be made (useful when retrying after mocking deps).
    """
    to_remove = [k for k in sys.modules if k == "car_bench" or k.startswith("car_bench.")]
    for k in to_remove:
        del sys.modules[k]
    if to_remove:
        logger.debug("Cleared %d car_bench entries from sys.modules", len(to_remove))


# ---------------------------------------------------------------------------
# Task-split loading
# ---------------------------------------------------------------------------

def load_task_splits(car_bench_repo: str) -> Dict[str, List[str]]:
    """Read ``docs/reference_data/tasks/task_splits.json`` and return its dict."""
    splits_path = os.path.join(
        car_bench_repo, "docs", "reference_data", "tasks", "task_splits.json"
    )
    with open(splits_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded task splits from %s  (keys: %s)", splits_path, list(data.keys()))
    return data


# ---------------------------------------------------------------------------
# Strategy 0: mock heavy deps, then import reference Python task files
# ---------------------------------------------------------------------------

def _try_import_tasks_with_mocked_deps(
    car_bench_repo: str, task_type: str
) -> Optional[List[Any]]:
    """
    Install lightweight stubs for heavy optional dependencies (litellm, datasets,
    etc.) then attempt to import the reference task file.

    This is the preferred strategy when the full CAR-bench environment is not
    installed.  The task files only use type/enum imports from car_bench — they
    never call litellm.completion or similar, so mocking is safe.

    Returns the TASKS list on success, ``None`` on error.
    """
    task_file = os.path.join(
        car_bench_repo, "docs", "reference_data", "tasks", f"tasks_{task_type}.py"
    )
    if not os.path.isfile(task_file):
        logger.warning("Reference task file not found: %s", task_file)
        return None

    try:
        # Ensure a clean import state for car_bench
        _clear_car_bench_from_sys_modules()
        install_mock_deps()

        module_name = f"_car_bench_reference_tasks_mocked_{task_type}"
        spec = importlib.util.spec_from_file_location(module_name, task_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not create module spec for %s", task_file)
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        tasks = module.TASKS  # type: ignore[attr-defined]
        logger.info(
            "Loaded %d tasks from reference file (mocked deps) for task_type=%s",
            len(tasks), task_type,
        )
        return tasks
    except Exception as exc:
        logger.warning(
            "Failed to load tasks with mocked deps for task_type=%s: %s",
            task_type, exc,
        )
        # Clean up any half-loaded car_bench modules so later strategies start fresh
        _clear_car_bench_from_sys_modules()
        return None


# ---------------------------------------------------------------------------
# Strategy 1: import reference Python task files (requires full deps)
# ---------------------------------------------------------------------------

def _try_import_tasks_from_reference_file(
    car_bench_repo: str, task_type: str
) -> Optional[List[Any]]:
    """
    Attempt to import the reference task file *without* mocking — requires the
    full CAR-bench environment to be installed (litellm, datasets, etc.).

    Returns the TASKS list on success, ``None`` on any import/execution error.
    """
    task_file = os.path.join(
        car_bench_repo, "docs", "reference_data", "tasks", f"tasks_{task_type}.py"
    )
    if not os.path.isfile(task_file):
        logger.warning("Reference task file not found: %s", task_file)
        return None

    try:
        module_name = f"_car_bench_reference_tasks_{task_type}"
        spec = importlib.util.spec_from_file_location(module_name, task_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not create module spec for %s", task_file)
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        tasks = module.TASKS  # type: ignore[attr-defined]
        logger.info(
            "Loaded %d tasks from reference file for task_type=%s",
            len(tasks), task_type,
        )
        return tasks
    except Exception as exc:
        logger.warning(
            "Failed to load tasks from reference file for task_type=%s: %s",
            task_type, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Strategy 2: load from HuggingFace datasets
# ---------------------------------------------------------------------------

def _try_load_tasks_from_huggingface(
    task_type: str, task_split: str
) -> Optional[List[Any]]:
    """
    Attempt to load tasks from the HuggingFace car-bench dataset using the
    built-in ``_load_tasks`` helper from ``car_bench.envs.car_voice_assistant.env``.

    Returns the list on success, ``None`` on any error.
    """
    try:
        from car_bench.envs.car_voice_assistant.env import _load_tasks  # type: ignore[import-untyped]
        tasks = _load_tasks(task_type, task_split)
        logger.info(
            "Loaded %d tasks from HuggingFace for %s/%s",
            len(tasks), task_type, task_split,
        )
        return tasks
    except Exception as exc:
        logger.warning(
            "Failed to load tasks from HuggingFace for %s/%s: %s",
            task_type, task_split, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Strategy 3: stub records (minimal, always succeeds)
# ---------------------------------------------------------------------------

def _task_type_from_task_id(task_id: str) -> str:
    """Derive a task_type string from a task_id prefix (e.g. ``'base_0'`` -> ``'base'``)."""
    for prefix in ("base", "disambiguation", "hallucination"):
        if task_id.startswith(f"{prefix}_"):
            return prefix
    return "unknown"


def create_stub_record(task_id: str, split: str) -> Dict[str, Any]:
    """Return a minimal stub dict for a task that could not be fully loaded."""
    return {
        "task_id": task_id,
        "task_type": _task_type_from_task_id(task_id),
        "calendar_id": "",
        "persona": "",
        "instruction": "",
        "actions": [],
        "context_init_config": {},
        "split": split,
        "stub": True,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_action(action: Any) -> Dict[str, Any]:
    """Turn an ``Action`` object into a plain dict."""
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if hasattr(action, "dict"):
        return action.dict()
    # Manual fallback
    return {
        "name": action.name,
        "kwargs": action.kwargs,
        "index": action.index,
        "dependent_on_action_index": action.dependent_on_action_index,
    }


def serialize_task(task: Any, split: str) -> Dict[str, Any]:
    """
    Serialize a ``car_bench.types.Task`` (Pydantic model) to a plain dict
    suitable for JSON output, adding the ``split`` field.
    """
    if hasattr(task, "model_dump"):
        data = task.model_dump()
    elif hasattr(task, "dict"):
        data = task.dict()
    else:
        # Manual fallback — copy known fields
        data = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "calendar_id": task.calendar_id,
            "persona": task.persona,
            "instruction": task.instruction,
            "actions": [_serialize_action(a) for a in task.actions],
            "context_init_config": task.context_init_config,
        }

    # Ensure task_type is a plain string (not an Enum instance)
    if not isinstance(data.get("task_type"), str):
        data["task_type"] = str(data["task_type"])

    # Drop internal-only optional fields that are not needed downstream
    for field in (
        "disambiguation_element_internal",
        "disambiguation_element_user",
        "disambiguation_element_note",
        "removed_part",
    ):
        data.pop(field, None)

    # Attach the split label
    data["split"] = split

    return data


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """Write *records* (list of dicts) as one JSON line per record."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_split(
    split_name: str,
    task_ids: List[str],
    car_bench_repo: str,
    output_dir: str,
) -> None:
    """
    Load, serialize, and write one split (e.g. ``base_train``).
    Tries reference-file import -> HuggingFace -> stubs.
    """
    # Derive the task_type from the split name
    task_type = split_name.removesuffix("_train").removesuffix("_test")
    # Determine whether this is a train or test split
    is_train = split_name.endswith("_train")
    hf_split = "train" if is_train else "test"

    logger.info(
        "Processing split '%s' — %d task IDs, task_type=%s",
        split_name, len(task_ids), task_type,
    )

    # --- Strategy 0: reference Python files with mocked deps ----------------
    tasks: Optional[List[Any]] = _try_import_tasks_with_mocked_deps(
        car_bench_repo, task_type
    )

    # --- Strategy 1: reference Python files (full deps) ---------------------
    if tasks is None:
        tasks = _try_import_tasks_from_reference_file(
            car_bench_repo, task_type
        )

    # --- Strategy 2: HuggingFace datasets -----------------------------------
    if tasks is None:
        tasks = _try_load_tasks_from_huggingface(task_type, hf_split)

    # --- Build records ------------------------------------------------------
    if tasks is not None:
        tasks_by_id: Dict[str, Any] = {t.task_id: t for t in tasks}
        missing_ids = [tid for tid in task_ids if tid not in tasks_by_id]
        if missing_ids:
            logger.warning(
                "%d task IDs from split '%s' not found in loaded tasks: %s",
                len(missing_ids), split_name, missing_ids[:5],
            )

        records: List[Dict[str, Any]] = []
        for task_id in task_ids:
            if task_id in tasks_by_id:
                records.append(serialize_task(tasks_by_id[task_id], split_name))
            else:
                logger.warning("Creating stub for missing task_id='%s'", task_id)
                records.append(create_stub_record(task_id, split_name))
    else:
        # --- Strategy 3: stubs ---------------------------------------------
        logger.warning(
            "All loading strategies failed for task_type=%s — producing stub records",
            task_type,
        )
        records = [create_stub_record(tid, split_name) for tid in task_ids]

    # --- Write JSONL --------------------------------------------------------
    output_path = os.path.join(output_dir, f"{split_name}.jsonl")
    write_jsonl(records, output_path)

    # Summary
    stub_count = sum(1 for r in records if r.get("stub"))
    full_count = len(records) - stub_count
    logger.info(
        "Split '%s': %d full records, %d stub records -> %s",
        split_name, full_count, stub_count, output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serialize CAR-bench training tasks to JSONL format for SIEVE v3 data factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--car-bench-repo",
        type=str,
        default="tmp/car-bench-repo",
        help=(
            "Path to the CAR-bench repository root "
            "(relative to project root or absolute).  Default: tmp/car-bench-repo"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/car-bench/serialized",
        help=(
            "Directory where JSONL files are written "
            "(relative to project root or absolute).  "
            "Default: data/raw/car-bench/serialized"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["base_train", "disambiguation_train"],
        help="Task splits to serialize.  Default: base_train disambiguation_train",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve paths
    project_root = resolve_project_root()
    car_bench_repo = resolve_path(args.car_bench_repo, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    logger.info("Project root : %s", project_root)
    logger.info("CAR-bench repo: %s", car_bench_repo)
    logger.info("Output dir   : %s", output_dir)

    if not os.path.isdir(car_bench_repo):
        logger.error("CAR-bench repo not found at %s", car_bench_repo)
        sys.exit(1)

    # Make car_bench importable
    add_car_bench_to_path(car_bench_repo)

    # Load the task-split index
    task_splits = load_task_splits(car_bench_repo)

    # Process each requested split
    for split_name in args.splits:
        task_ids = task_splits.get(split_name)
        if task_ids is None:
            logger.error("Split '%s' not found in task_splits.json", split_name)
            sys.exit(1)
        if not task_ids:
            logger.warning("Split '%s' is empty — skipping", split_name)
            continue
        process_split(split_name, task_ids, car_bench_repo, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
