#!/usr/bin/env python3
"""Collect GO2 reproduction logs and test metrics into tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]
TRACKING_KEYS = [
    "root_pos_err",
    "root_rot_err",
    "body_pos_err",
    "body_rot_err",
    "dof_vel_err",
    "root_vel_err",
    "root_ang_vel_err",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    return parser.parse_args()


def select(values: list[str], all_values: list[str]) -> list[str]:
    if "all" in values:
        return all_values
    bad = [value for value in values if value not in all_values]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {all_values}")
    return values


def read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r") as f:
        return json.load(f)


def read_final_log_row(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}

    with path.open("r") as f:
        lines = [line.strip().replace(",", "\t") for line in f if line.strip()]
    if len(lines) < 2:
        return {}

    header = lines[0].split()
    values = lines[-1].split()
    row: dict[str, float] = {}
    for key, value in zip(header, values):
        try:
            row[key] = float(value)
        except ValueError:
            pass
    return row


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def fmt(value: float | None, std: float | None) -> str:
    if value is None:
        return ""
    if std is None:
        return f"{value:.6g}"
    return f"{value:.6g} +/- {std:.6g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |\n")


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    table_dir = root / "tables"
    algorithms = select(args.algorithms, ALGORITHMS)
    motions = select(args.motions, MOTIONS)

    return_rows: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []

    for motion in motions:
        for algorithm in algorithms:
            metrics_list = []
            final_logs = []
            for seed in args.seeds:
                metrics_path = root / "eval" / algorithm / motion / f"seed_{seed}" / "metrics.json"
                log_path = root / "runs" / algorithm / motion / f"seed_{seed}" / "log.txt"
                metrics = read_metrics(metrics_path)
                if metrics is not None:
                    metrics_list.append(metrics)
                final_log = read_final_log_row(log_path)
                if final_log:
                    final_logs.append(final_log)

            eval_returns = [float(m["mean_return"]) for m in metrics_list if "mean_return" in m]
            eval_lengths = [float(m["mean_ep_len"]) for m in metrics_list if "mean_ep_len" in m]
            train_final_returns = [float(r["Test_Return"]) for r in final_logs if "Test_Return" in r]
            samples = [float(r["Samples"]) for r in final_logs if "Samples" in r]

            eval_return_mean, eval_return_std = summarize(eval_returns)
            eval_len_mean, eval_len_std = summarize(eval_lengths)
            final_return_mean, final_return_std = summarize(train_final_returns)
            sample_mean, _ = summarize(samples)

            return_rows.append({
                "motion": motion,
                "algorithm": algorithm,
                "seeds_found": len(metrics_list),
                "train_logs_found": len(final_logs),
                "eval_return": fmt(eval_return_mean, eval_return_std),
                "eval_ep_len": fmt(eval_len_mean, eval_len_std),
                "final_train_log_test_return": fmt(final_return_mean, final_return_std),
                "final_samples_mean": "" if sample_mean is None else f"{sample_mean:.0f}",
            })

            track_row: dict[str, Any] = {
                "motion": motion,
                "algorithm": algorithm,
                "seeds_found": len(metrics_list),
                "train_logs_found": len(final_logs),
            }
            for key in TRACKING_KEYS:
                vals = [float(m[key]) for m in metrics_list if key in m]
                val_mean, val_std = summarize(vals)
                track_row[key] = fmt(val_mean, val_std)
            tracking_rows.append(track_row)

    write_csv(
        table_dir / "table_returns.csv",
        return_rows,
        ["motion", "algorithm", "seeds_found", "train_logs_found", "eval_return", "eval_ep_len", "final_train_log_test_return", "final_samples_mean"],
    )
    write_csv(
        table_dir / "table_tracking_errors.csv",
        tracking_rows,
        ["motion", "algorithm", "seeds_found", "train_logs_found", *TRACKING_KEYS],
    )
    write_markdown(
        table_dir / "table_tracking_errors.md",
        tracking_rows,
        ["motion", "algorithm", "seeds_found", "train_logs_found", *TRACKING_KEYS],
    )

    print(f"Wrote {table_dir / 'table_returns.csv'}")
    print(f"Wrote {table_dir / 'table_tracking_errors.csv'}")
    print(f"Wrote {table_dir / 'table_tracking_errors.md'}")


if __name__ == "__main__":
    main()
