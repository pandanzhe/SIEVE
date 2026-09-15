#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "reward",
        (
            "mean_return",
            "dev_mean_return",
            "mean_penalized_score",
        ),
    ),
    (
        "success_parse",
        (
            "success_rate",
            "dev_success_rate",
            "parse_rate",
            "dev_parse_rate",
        ),
    ),
    (
        "returns",
        (
            "mean_return",
            "dev_mean_return",
            "mean_penalized_score",
        ),
    ),
    (
        "variance",
        (
            "group_reward_variance",
            "step_reward_variance",
            "mean_abs_step_advantage",
        ),
    ),
    (
        "loss_kl",
        (
            "policy_loss",
            "sampled_kl",
        ),
    ),
    (
        "costs",
        (
            "cost_false_update",
            "cost_stall",
            "cost_invalid_format",
            "cost_budget_violation",
        ),
    ),
)


def _coerce_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_curves_dir(path: Path) -> Path:
    if path.name == "curves":
        return path
    if (path / "curves").is_dir():
        return path / "curves"
    if path.is_file():
        return path.parent
    return path


def _load_rows(curves_dir: Path) -> list[dict[str, Any]]:
    jsonl = curves_dir / "metrics.jsonl"
    csv_path = curves_dir / "metrics.csv"
    if jsonl.is_file():
        return _read_jsonl(jsonl)
    if csv_path.is_file():
        return _read_csv(csv_path)
    raise FileNotFoundError(f"missing metrics.jsonl or metrics.csv in {curves_dir}")


def _smooth(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values
    radius = window // 2
    smoothed: list[float | None] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        local = [value for value in values[start:end] if value is not None]
        smoothed.append(sum(local) / len(local) if local else None)
    return smoothed


def _series(
    rows: list[dict[str, Any]], metric: str, window: int
) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float | None]] = []
    for row in rows:
        step = _coerce_number(row.get("iteration"))
        if step is None:
            continue
        points.append((step, _coerce_number(row.get(metric))))
    if not points:
        return [], []
    xs = [point[0] for point in points]
    ys = _smooth([point[1] for point in points], window)
    clean_x: list[float] = []
    clean_y: list[float] = []
    for x_value, y_value in zip(xs, ys, strict=True):
        if y_value is not None:
            clean_x.append(x_value)
            clean_y.append(y_value)
    return clean_x, clean_y


def _available_metrics(rows: list[dict[str, Any]], names: Iterable[str]) -> list[str]:
    available: list[str] = []
    for name in names:
        if any(_coerce_number(row.get(name)) is not None for row in rows):
            available.append(name)
    return available


def _plot_group(
    rows: list[dict[str, Any]],
    curves_dir: Path,
    title: str,
    metrics: tuple[str, ...],
    *,
    window: int,
    dpi: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    available = _available_metrics(rows, metrics)
    if not available:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    for metric in available:
        xs, ys = _series(rows, metric, window)
        if xs and ys:
            ax.plot(xs, ys, linewidth=2.0, label=metric)
            raw_xs, raw_ys = _series(rows, metric, 1)
            ax.plot(raw_xs, raw_ys, linewidth=0.8, alpha=0.18)

    ax.set_title(f"Stage-2 {title} (moving average window={window})")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(title.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output = curves_dir / f"{title}.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _write_summary(rows: list[dict[str, Any]], curves_dir: Path) -> Path:
    summary_path = curves_dir / "curve_summary.json"
    if rows:
        first = rows[0]
        last = rows[-1]
        payload = {
            "row_count": len(rows),
            "first_iteration": _coerce_number(first.get("iteration")),
            "last_iteration": _coerce_number(last.get("iteration")),
            "algorithm": last.get("algorithm") or first.get("algorithm"),
            "final": {
                key: value
                for key, value in last.items()
                if key == "algorithm" or _coerce_number(value) is not None
            },
        }
    else:
        payload = {"row_count": 0}
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def plot_curves(path: Path, *, window: int, dpi: int) -> list[Path]:
    curves_dir = _resolve_curves_dir(path)
    rows = _load_rows(curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for title, metrics in DEFAULT_GROUPS:
        output = _plot_group(rows, curves_dir, title, metrics, window=window, dpi=dpi)
        if output is not None:
            outputs.append(output)
    outputs.append(_write_summary(rows, curves_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot smoothed Stage-2 RL training curves from metrics.jsonl/csv."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, curves directories, or metrics files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Centered moving-average smoothing window. Use 1 for no smoothing.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    for raw_path in args.paths:
        outputs = plot_curves(Path(raw_path).resolve(), window=args.window, dpi=args.dpi)
        print(f"{raw_path}:")
        for output in outputs:
            print(f"  {output}")


if __name__ == "__main__":
    main()
