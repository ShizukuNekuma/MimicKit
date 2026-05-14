#!/usr/bin/env python3
"""Plot Fig. 5 style GO2 learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]
COLORS = {
    "deepmimic": "#1f77b4",
    "amp": "#ff7f0e",
    "add": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    parser.add_argument("--y_key", default="Test_Return")
    parser.add_argument("--x_key", default="Samples")
    parser.add_argument("--max_x", type=float, default=None)
    return parser.parse_args()


def select(values: list[str], all_values: list[str]) -> list[str]:
    if "all" in values:
        return all_values
    bad = [value for value in values if value not in all_values]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {all_values}")
    return values


def read_log(path: Path, x_key: str, y_key: str, max_x: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None

    with path.open("r") as f:
        clean_lines = [line.replace(",", "\t") for line in f if line.strip()]
    if len(clean_lines) < 2:
        return None

    data = np.genfromtxt(clean_lines, delimiter=None, dtype=None, names=True)
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


def align_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not curves:
        return None

    min_len = min(len(x) for x, _ in curves)
    if min_len == 0:
        return None

    xs = curves[0][0][:min_len]
    ys = np.stack([y[:min_len] for _, y in curves], axis=0)
    return xs, ys


def plot_motion(root: Path, motion: str, algorithms: list[str], seeds: list[int], x_key: str, y_key: str, max_x: float | None) -> bool:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = False

    for algorithm in algorithms:
        curves = []
        for seed in seeds:
            override_path = root / "plot_logs" / algorithm / motion / f"seed_{seed}" / "log.txt"
            log_path = override_path if override_path.is_file() else root / "runs" / algorithm / motion / f"seed_{seed}" / "log.txt"
            curve = read_log(log_path, x_key=x_key, y_key=y_key, max_x=max_x)
            if curve is not None:
                curves.append(curve)

        aligned = align_curves(curves)
        if aligned is None:
            continue

        xs, ys = aligned
        mean_y = np.mean(ys, axis=0)
        std_y = np.std(ys, axis=0)
        color = COLORS.get(algorithm)
        ax.plot(xs, mean_y, label=f"{algorithm} ({len(curves)} seeds)", color=color)
        if len(curves) > 1:
            ax.fill_between(xs, mean_y - std_y, mean_y + std_y, alpha=0.2, color=color)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(f"{motion} {y_key}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(linestyle="dotted")
    ax.legend()
    fig.tight_layout()

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{motion}_{y_key.lower()}.png", dpi=200)
    fig.savefig(out_dir / f"{motion}_{y_key.lower()}.pdf")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    algorithms = select(args.algorithms, ALGORITHMS)
    motions = select(args.motions, MOTIONS)

    made = 0
    for motion in motions:
        if plot_motion(root, motion, algorithms, args.seeds, args.x_key, args.y_key, args.max_x):
            made += 1

    print(f"Wrote {made} figure set(s) to {root / 'figures'}")


if __name__ == "__main__":
    main()
