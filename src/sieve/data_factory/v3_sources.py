from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import GroundedSourceRecord, Provenance, ScenarioType


# ---------------------------------------------------------------------------
# SourceTask: unified intermediate representation for v3 source adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTask:
    """Intermediate task representation produced by v3 source adapters."""

    task_id: str
    split: str  # "train" or "test"
    domain: str
    goal: str
    entity: str
    field_id: str
    old_value: Any
    new_value: Any
    source: str
    source_text: str
    is_test: bool
    raw_task: dict[str, Any]
    parent_task_id: str
    reference_actions: tuple[dict[str, Any], ...] = ()
    initial_state: dict[str, Any] = field(default_factory=dict)

    @property
    def source_split(self) -> str:
        return self.split


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ENTITY_KEYS = (
    "order_id",
    "reservation_id",
    "booking_id",
    "user_id",
    "phone_number",
)


def _digest(raw: Any) -> str:
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provenance(
    dataset: str,
    version: str,
    license_name: str,
    path: Path,
    record_id: str,
    raw: Any,
) -> Provenance:
    return Provenance(
        dataset,
        version,
        record_id,
        f"{path.as_posix()}#{record_id}",
        _digest(raw),
        license_name,
    )


# ---------------------------------------------------------------------------
# tau3-bench (tau2-bench) adapter
# ---------------------------------------------------------------------------


def load_tau3_tasks(
    raw_dir: str | Path,
    *,
    domains: tuple[str, ...] = ("airline", "retail", "telecom"),
) -> list[SourceTask]:
    """Load train tasks from the tau3-bench (tau2-bench) raw data directory.

    Parameters
    ----------
    raw_dir : str | Path
        Root of the raw tau2-bench checkout (containing ``data/tau2/domains/``).
    domains : tuple[str, ...]
        Domain subdirectories to load.  Defaults to all three.

    Returns
    -------
    list[SourceTask]
        Train-split tasks only (``is_test=False``).
    """
    root = Path(raw_dir).resolve()
    base = root / "data" / "tau2" / "domains"
    tasks: list[SourceTask] = []

    for domain in domains:
        domain_dir = base / domain
        if not domain_dir.is_dir():
            continue

        # ---- split membership ----
        split_path = domain_dir / "split_tasks.json"
        if not split_path.is_file():
            continue
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = {str(tid) for tid in splits.get("train", [])}
        test_ids = {str(tid) for tid in splits.get("test", [])}

        # ---- task definitions ----
        tasks_path = domain_dir / "tasks.json"
        if not tasks_path.is_file():
            continue
        tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))

        for task in tasks_data:
            task_id = str(task.get("id"))

            # Determine split; skip tasks that appear in neither train nor test.
            if task_id in train_ids:
                split = "train"
                is_test = False
            elif task_id in test_ids:
                split = "test"
                is_test = True
            else:
                continue

            # v3 only keeps train tasks.
            if is_test:
                continue

            # ---- extract fields from evaluation actions ----
            instructions = task.get("user_scenario", {}).get("instructions", {})
            actions = task.get("evaluation_criteria", {}).get("actions", [])

            arguments: dict[str, Any] = {}
            tool_name = "tau2_environment"
            for action in actions:
                if isinstance(action, dict) and isinstance(action.get("arguments"), dict):
                    arguments = action["arguments"]
                    tool_name = str(action.get("name") or tool_name)
                    if arguments:
                        break

            # Entity identifier from known entity keys.
            entity = next(
                (str(arguments[key]) for key in _ENTITY_KEYS if key in arguments),
                None,
            )
            if entity is None:
                entity = f"{domain}-task-{task_id}"

            # field_id / new_value from the first non-entity argument key.
            value_keys = [key for key in arguments if key not in _ENTITY_KEYS]
            field_id = value_keys[0] if value_keys else next(iter(arguments), "unknown")
            new_value = arguments.get(field_id)

            goal = str(instructions.get("reason_for_call") or "complete the service task")
            known_info = instructions.get("known_info") or ""
            reason = instructions.get("reason_for_call") or ""
            source_text = f"{known_info} {reason}".strip()
            if not source_text:
                source_text = json.dumps(arguments, ensure_ascii=False)

            parent_task_id = f"tau3:{domain}:{task_id}"

            tasks.append(
                SourceTask(
                    task_id=task_id,
                    split=split,
                    domain=domain,
                    goal=goal,
                    entity=entity,
                    field_id=str(field_id),
                    old_value=arguments.get(f"old_{field_id}", "unknown"),
                    new_value=new_value,
                    source=tool_name,
                    source_text=source_text,
                    is_test=is_test,
                    raw_task=task,
                    parent_task_id=parent_task_id,
                    reference_actions=tuple(actions),
                )
            )

    return tasks


# ---------------------------------------------------------------------------
# CAR-bench adapter
# ---------------------------------------------------------------------------

_TASK_FIELDS = (
    "task_id",
    "calendar_id",
    "actions",
    "persona",
    "instruction",
    "context_init_config",
    "task_type",
)

_ACTION_FIELDS = (
    "name",
    "kwargs",
    "index",
    "dependent_on_action_index",
)


def _parse_tasks_python(path: Path) -> list[dict[str, Any]]:
    """Parse a CAR-bench ``TASKS=[...]`` assignment from Python source.

    Uses ``eval`` with a restricted namespace so that ``Task(...)`` and
    ``Action(...)`` constructor calls are converted to plain dicts without
    importing the benchmark package.
    """
    source = path.read_text(encoding="utf-8")
    match = re.search(r"TASKS\s*=\s*(.+)", source, re.DOTALL)
    if not match:
        return []
    expr = match.group(1).strip()

    # -- safe constructors that return plain dicts --
    def _Task(*args: Any, **kwargs: Any) -> dict[str, Any]:
        for i, arg in enumerate(args):
            if i < len(_TASK_FIELDS):
                kwargs[_TASK_FIELDS[i]] = arg
        return kwargs

    def _Action(*args: Any, **kwargs: Any) -> dict[str, Any]:
        for i, arg in enumerate(args):
            if i < len(_ACTION_FIELDS):
                kwargs[_ACTION_FIELDS[i]] = arg
        return kwargs

    class _TaskTypeProxy:
        """Proxy so that ``TaskType.BASE`` evaluates to the string ``"BASE"``."""

        def __getattr__(self, name: str) -> str:
            return name

    safe_ns: dict[str, Any] = {
        "__builtins__": {},
        "Task": _Task,
        "Action": _Action,
        "TaskType": _TaskTypeProxy(),
    }

    try:
        result = eval(expr, safe_ns)  # noqa: S307
    except Exception:
        return []

    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def _car_task_to_source(
    task_dict: dict[str, Any],
    *,
    split: str = "train",
) -> SourceTask:
    """Convert a single CAR-bench task dict to a SourceTask."""
    task_id = str(task_dict.get("task_id", ""))
    actions = task_dict.get("actions", [])
    instruction = str(task_dict.get("instruction") or "")
    persona = str(task_dict.get("persona") or "")
    context_init = task_dict.get("context_init_config") or {}

    first_action = actions[0] if actions else {}
    action_name = str(first_action.get("name", "unknown"))
    action_kwargs = first_action.get("kwargs", {})

    entity = str(task_dict.get("calendar_id") or task_id)
    field_id = action_name
    new_value = action_kwargs

    goal = instruction or "complete the calendar task"
    source_text = f"{persona} {instruction}".strip()

    parent_task_id = f"car:{task_id}"

    return SourceTask(
        task_id=task_id,
        split=split,
        domain="calendar",
        goal=goal,
        entity=entity,
        field_id=field_id,
        old_value="unknown",
        new_value=new_value,
        source=action_name,
        source_text=source_text,
        is_test=False,
        raw_task=task_dict,
        parent_task_id=parent_task_id,
        reference_actions=tuple(actions),
        initial_state=dict(context_init),
    )


def _load_car_python(root: Path) -> list[SourceTask]:
    """Load CAR-bench tasks by parsing the reference Python source files."""
    tasks_dir = root / "docs" / "reference_data" / "tasks"

    split_path = tasks_dir / "task_splits.json"
    if not split_path.is_file():
        return []
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids: set[str] = set()
    for key in ("base_train", "disambiguation_train"):
        train_ids.update(str(tid) for tid in splits.get(key, []))

    all_tasks: list[dict[str, Any]] = []
    for filename in ("tasks_base.py", "tasks_disambiguation.py"):
        py_path = tasks_dir / filename
        if py_path.is_file():
            all_tasks.extend(_parse_tasks_python(py_path))

    result: list[SourceTask] = []
    for task_dict in all_tasks:
        task_id = str(task_dict.get("task_id", ""))
        if task_id not in train_ids:
            continue
        result.append(_car_task_to_source(task_dict))

    return result


def _load_car_serialized(root: Path) -> list[SourceTask]:
    """Load CAR-bench tasks from pre-serialized JSONL files."""
    result: list[SourceTask] = []
    for split_name in ("base_train", "disambiguation_train"):
        jsonl_path = root / "serialized" / f"{split_name}.jsonl"
        if not jsonl_path.is_file():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                task_dict = json.loads(line)
                result.append(_car_task_to_source(task_dict))
    return result


def load_car_tasks(raw_dir: str | Path) -> list[SourceTask]:
    """Load train tasks from the CAR-bench raw data directory.

    If pre-serialized JSONL files exist under ``serialized/``, they are
    preferred.  Otherwise the adapter parses the reference Python source
    files in ``docs/reference_data/tasks/`` without importing the benchmark
    package.

    Parameters
    ----------
    raw_dir : str | Path
        Root of the raw car-bench checkout.

    Returns
    -------
    list[SourceTask]
        Train-split tasks only (``is_test=False``).
    """
    root = Path(raw_dir).resolve()
    serialized_dir = root / "serialized"

    if (serialized_dir / "base_train.jsonl").is_file():
        return _load_car_serialized(root)

    return _load_car_python(root)


# ---------------------------------------------------------------------------
# SourceTask -> GroundedSourceRecord conversion
# ---------------------------------------------------------------------------


def source_task_to_grounded_record(
    task: SourceTask,
    dataset: str,
    version: str,
    license_name: str,
    source_path: Path,
    index: int,
) -> GroundedSourceRecord:
    """Convert a SourceTask to a GroundedSourceRecord for the rules pipeline.

    Parameters
    ----------
    task : SourceTask
        Source task to convert.
    dataset : str
        Dataset identifier (e.g. ``"tau3-bench"``, ``"car-bench"``).
    version : str
        Dataset version string.
    license_name : str
        License identifier for provenance.
    source_path : Path
        Path to the source file for URI construction.
    index : int
        Sequential index used to compute ``observed_at`` and ``valid_from``
        timestamps.

    Returns
    -------
    GroundedSourceRecord
    """
    observed_at = 1_700_000_000 + index
    scenario_type = (
        ScenarioType.TRANSACTION
        if dataset == "tau3-bench"
        else ScenarioType.INFORMATION_TOOL
    )

    return GroundedSourceRecord(
        provenance=_provenance(
            dataset, version, license_name, source_path, task.parent_task_id, task.raw_task
        ),
        domain=task.domain,
        scenario_type=scenario_type,
        entity=task.entity,
        field_id=task.field_id,
        old_value=task.old_value,
        new_value=task.new_value,
        source=task.source,
        observed_at=observed_at,
        valid_from=observed_at,
        goal=task.goal,
        source_text=task.source_text,
        metadata={
            "raw_task_id": task.task_id,
            "split": task.split,
            "parent_task_id": task.parent_task_id,
        },
    )


# ---------------------------------------------------------------------------
# Audit / snapshot report
# ---------------------------------------------------------------------------


def audit_source_snapshot(
    tasks: list[SourceTask],
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Produce an audit report over a list of source tasks.

    Parameters
    ----------
    tasks : list[SourceTask]
        Tasks to audit.
    manifest_path : str | Path
        Path to the source manifest (recorded in the report for traceability).

    Returns
    -------
    dict[str, Any]
        Report with keys ``total``, ``by_domain``, ``by_split``,
        ``test_count``, ``task_ids_sample``, and optionally ``warning``.
    """
    by_domain: dict[str, int] = {}
    by_split: dict[str, int] = {}
    test_count = 0

    for task in tasks:
        by_domain[task.domain] = by_domain.get(task.domain, 0) + 1
        by_split[task.split] = by_split.get(task.split, 0) + 1
        if task.is_test:
            test_count += 1

    report: dict[str, Any] = {
        "total": len(tasks),
        "by_domain": by_domain,
        "by_split": by_split,
        "test_count": test_count,
        "task_ids_sample": [t.task_id for t in tasks[:10]],
        "manifest_path": str(manifest_path),
    }

    if test_count > 0:
        report["warning"] = f"found {test_count} test tasks in v3 source snapshot (expected 0)"

    return report
