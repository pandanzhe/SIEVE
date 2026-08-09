from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        current = yaml.safe_load(handle) or {}
    parent: dict[str, Any] = {}
    base = current.pop("base", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        parent = load_config(base_path)
    merged = deep_merge(parent, current)
    return deep_merge(merged, overrides or {})


def resolve_repo_path(root: str | Path, candidate: str | Path) -> Path:
    repo_root = Path(root).resolve()
    path = Path(candidate)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {candidate}") from exc
    return resolved


def ensure_output_dirs(root: str | Path, config: Mapping[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, value in config.get("paths", {}).items():
        resolved = resolve_repo_path(root, value)
        resolved.mkdir(parents=True, exist_ok=True)
        result[name] = resolved
    return result


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
