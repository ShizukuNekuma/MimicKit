#!/usr/bin/env python3
"""Generate GO2 report figures and paper-style tracking tables."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ["amp", "deepmimic", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]
ALG_LABELS = {
    "amp": "AMP",
    "deepmimic": "DeepMimic",
    "add": "ADD",
}
MOTION_LABELS = {
    "go2_pace": "Pace",
    "go2_run": "Run",
    "go2_trot": "Trot",
    "go2_walk0": "Walk0",
    "go2_walk1": "Walk1",
    "go2_walk2": "Walk2",
    "go2_walk3": "Walk3",
}
DEFAULT_METRICS = {
    "deepmimic": "Test_Return",
    "amp": "Disc_Reward_Mean",
    "add": "Test_Return",
}
COLORS = {
    "deepmimic": "#1f77b4",
    "amp": "#ff7f0e",
    "add": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max_samples", type=float, default=None)
    parser.add_argument("--x_key", default="Samples")
    parser.add_argument("--metric", action="append", default=[], help="Override metric as algorithm:key, e.g. amp:Disc_Reward_Mean.")
    parser.add_argument("--position_key", default="body_pos_err")
    parser.add_argument("--velocity_key", default="dof_vel_err")
    return parser.parse_args()


def select(values: list[str], all_values: list[str]) -> list[str]:
    if "all" in values:
        return all_values
    bad = [value for value in values if value not in all_values]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {all_values}")
    return values


def metric_map(overrides: list[str]) -> dict[str, str]:
    metrics = dict(DEFAULT_METRICS)
    for item in overrides:
        if ":" not in item:
            raise ValueError(f"--metric must be algorithm:key, got {item!r}")
        algorithm, key = item.split(":", 1)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm in --metric: {algorithm}")
        metrics[algorithm] = key
    return metrics


def read_log(path: Path, x_key: str, y_key: str, max_x: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with path.open("r") as f:
        lines = [line.replace(",", "\t") for line in f if line.strip()]
    if len(lines) < 2:
        return None
    data = np.genfromtxt(lines, delimiter=None, dtype=None, names=True)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    if data.dtype.names is None or x_key not in data.dtype.names or y_key not in data.dtype.names:
        return None
    xs = np.asarray(data[x_key], dtype=float)
    ys = np.asarray(data[y_key], dtype=float)
    if max_x is not None:
        keep = xs <= max_x
        xs = xs[keep]
        ys = ys[keep]
    if len(xs) == 0:
        return None
    return xs, ys


def log_path(root: Path, algorithm: str, motion: str, seed: int) -> Path:
    override = root / "plot_logs" / algorithm / motion / f"seed_{seed}" / "log.txt"
    if override.is_file():
        return override
    return root / "runs" / algorithm / motion / f"seed_{seed}" / "log.txt"


def align_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not curves:
        return None
    min_len = min(len(xs) for xs, _ in curves)
    if min_len == 0:
        return None
    xs = curves[0][0][:min_len]
    ys = np.stack([ys[:min_len] for _, ys in curves], axis=0)
    return xs, ys


def plot_learning_curves(
    root: Path,
    algorithms: list[str],
    motions: list[str],
    seeds: list[int],
    metrics: dict[str, str],
    x_key: str,
    max_samples: float | None,
) -> int:
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for algorithm in algorithms:
        y_key = metrics[algorithm]
        for motion in motions:
            curves = []
            for seed in seeds:
                curve = read_log(log_path(root, algorithm, motion, seed), x_key, y_key, max_samples)
                if curve is not None:
                    curves.append(curve)
            aligned = align_curves(curves)
            if aligned is None:
                continue

            xs, ys = aligned
            mean_y = np.mean(ys, axis=0)
            std_y = np.std(ys, axis=0)
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            color = COLORS.get(algorithm, "#444444")
            ax.plot(xs, mean_y, color=color, linewidth=2.0, label=f"{ALG_LABELS[algorithm]} ({len(curves)} seed(s))")
            if len(curves) > 1:
                ax.fill_between(xs, mean_y - std_y, mean_y + std_y, color=color, alpha=0.18)
            ax.set_title(f"{ALG_LABELS[algorithm]} {MOTION_LABELS.get(motion, motion)}")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            ax.grid(linestyle="dotted", alpha=0.7)
            ax.legend()
            fig.tight_layout()
            stem = f"{algorithm}_{motion}_{y_key.lower()}"
            fig.savefig(out_dir / f"{stem}.png", dpi=220)
            fig.savefig(out_dir / f"{stem}.pdf")
            plt.close(fig)
            made += 1
    return made


def motion_length(motion: str) -> float | None:
    path = Path("data/motions/go2") / f"{motion}.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as f:
        data: dict[str, Any] = pickle.load(f)
    return len(data["frames"]) / float(data["fps"])


def read_metric(root: Path, algorithm: str, motion: str, seed: int, key: str) -> float | None:
    path = root / "eval" / algorithm / motion / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    with path.open("r") as f:
        data = json.load(f)
    value = data.get(key)
    return None if value is None else float(value)


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return mean(values), pstdev(values)


def fmt(value: float | None, std: float | None) -> str:
    if value is None:
        return "--"
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f}+/-{std:.3f}"


def make_rows(
    root: Path,
    algorithms: list[str],
    motions: list[str],
    seeds: list[int],
    position_key: str,
    velocity_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for motion in motions:
        row: dict[str, Any] = {
            "motion": motion,
            "motion_label": MOTION_LABELS.get(motion, motion),
            "length": motion_length(motion),
        }
        for algorithm in algorithms:
            pos_vals = [v for seed in seeds if (v := read_metric(root, algorithm, motion, seed, position_key)) is not None]
            vel_vals = [v for seed in seeds if (v := read_metric(root, algorithm, motion, seed, velocity_key)) is not None]
            row[f"{algorithm}_pos_mean"], row[f"{algorithm}_pos_std"] = summarize(pos_vals)
            row[f"{algorithm}_vel_mean"], row[f"{algorithm}_vel_std"] = summarize(vel_vals)
        rows.append(row)
    return rows


def write_tracking_csv(root: Path, rows: list[dict[str, Any]], algorithms: list[str]) -> None:
    out = root / "tables" / "table_tracking_errors_paper_style.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Motion", "Length"]
    fields += [f"{ALG_LABELS[alg]} Position Tracking Error [m]" for alg in algorithms]
    fields += [f"{ALG_LABELS[alg]} DoF Velocity Tracking Error [rad/s]" for alg in algorithms]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out_row = {
                "Motion": row["motion_label"],
                "Length": "--" if row["length"] is None else f"{row['length']:.2f}s",
            }
            for algorithm in algorithms:
                out_row[f"{ALG_LABELS[algorithm]} Position Tracking Error [m]"] = fmt(row[f"{algorithm}_pos_mean"], row[f"{algorithm}_pos_std"])
            for algorithm in algorithms:
                out_row[f"{ALG_LABELS[algorithm]} DoF Velocity Tracking Error [rad/s]"] = fmt(row[f"{algorithm}_vel_mean"], row[f"{algorithm}_vel_std"])
            writer.writerow(out_row)


def write_tracking_markdown(root: Path, rows: list[dict[str, Any]], algorithms: list[str]) -> None:
    out = root / "tables" / "table_tracking_errors_paper_style.md"
    headers = ["Motion", "Length"]
    headers += [f"Pos {ALG_LABELS[alg]}" for alg in algorithms]
    headers += [f"DoF Vel {ALG_LABELS[alg]}" for alg in algorithms]
    with out.open("w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            vals = [row["motion_label"], "--" if row["length"] is None else f"{row['length']:.2f}s"]
            vals += [fmt(row[f"{alg}_pos_mean"], row[f"{alg}_pos_std"]) for alg in algorithms]
            vals += [fmt(row[f"{alg}_vel_mean"], row[f"{alg}_vel_std"]) for alg in algorithms]
            f.write("| " + " | ".join(vals) + " |\n")


def best_algorithm(row: dict[str, Any], algorithms: list[str], suffix: str) -> str | None:
    vals = [(alg, row[f"{alg}_{suffix}_mean"]) for alg in algorithms if row[f"{alg}_{suffix}_mean"] is not None]
    if not vals:
        return None
    return min(vals, key=lambda item: item[1])[0]


def write_tracking_png(root: Path, rows: list[dict[str, Any]], algorithms: list[str]) -> None:
    out = root / "tables" / "table_tracking_errors_paper_style.png"
    header1 = ["Motion", "Length", "Position Tracking Error [m]"]
    header1 += [""] * (len(algorithms) - 1)
    header1 += ["DoF Velocity Tracking Error [rad/s]"]
    header1 += [""] * (len(algorithms) - 1)
    header2 = ["", "", *[ALG_LABELS[alg] for alg in algorithms], *[ALG_LABELS[alg] for alg in algorithms]]
    table_rows = [header1, header2]
    for row in rows:
        table_rows.append([
            row["motion_label"],
            "--" if row["length"] is None else f"{row['length']:.2f}s",
            *[fmt(row[f"{alg}_pos_mean"], row[f"{alg}_pos_std"]) for alg in algorithms],
            *[fmt(row[f"{alg}_vel_mean"], row[f"{alg}_vel_std"]) for alg in algorithms],
        ])

    fig_h = 1.0 + 0.62 * len(table_rows)
    fig_w = 4.2 + 1.75 * len(algorithms) * 2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=table_rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.55)

    shade = "#ddddff"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        if r in (0, 1):
            cell.set_text_props(weight="bold", fontsize=14)
            cell.set_facecolor("#f7f7f7")
        if r == 0 and (3 <= c < 2 + len(algorithms) or 3 + len(algorithms) <= c < 2 + 2 * len(algorithms)):
            cell.get_text().set_text("")

    pos_cols = {alg: 2 + idx for idx, alg in enumerate(algorithms)}
    vel_cols = {alg: 2 + len(algorithms) + idx for idx, alg in enumerate(algorithms)}
    for row_idx, row in enumerate(rows, start=2):
        pos_best = best_algorithm(row, algorithms, "pos")
        vel_best = best_algorithm(row, algorithms, "vel")
        if pos_best is not None:
            table[row_idx, pos_cols[pos_best]].set_facecolor(shade)
            table[row_idx, pos_cols[pos_best]].set_text_props(weight="bold")
        if vel_best is not None:
            table[row_idx, vel_cols[vel_best]].set_facecolor(shade)
            table[row_idx, vel_cols[vel_best]].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    algorithms = select(args.algorithms, ALGORITHMS)
    motions = select(args.motions, MOTIONS)
    metrics = metric_map(args.metric)
    made = plot_learning_curves(root, algorithms, motions, args.seeds, metrics, args.x_key, args.max_samples)
    rows = make_rows(root, algorithms, motions, args.seeds, args.position_key, args.velocity_key)
    write_tracking_csv(root, rows, algorithms)
    write_tracking_markdown(root, rows, algorithms)
    write_tracking_png(root, rows, algorithms)
    print(f"Wrote {made} figure set(s) and paper-style tracking tables under {root}")


if __name__ == "__main__":
    main()
