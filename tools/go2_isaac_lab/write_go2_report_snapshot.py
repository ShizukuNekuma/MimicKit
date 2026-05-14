#!/usr/bin/env python3
"""Write a deadline-oriented GO2 reproduction report snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--report_file", default="docs/GO2_REPRODUCTION_REPORT.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--algorithms", nargs="+", default=["deepmimic", "amp"])
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    return parser.parse_args()


def select(values: list[str], all_values: list[str]) -> list[str]:
    if "all" in values:
        return all_values
    bad = [value for value in values if value not in all_values]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {all_values}")
    return values


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def read_csv_table(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        return list(csv.reader(f))


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_Not generated yet._"
    header = rows[0]
    body = rows[1:]
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def status_summary(status_rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    starts = {}
    finishes = {}
    failures = []
    for row in status_rows:
        key = (row.get("phase"), row.get("algorithm"), row.get("motion"), row.get("seed"))
        if row.get("event") == "start":
            starts[key] = row
        elif row.get("event") == "finish":
            finishes[key] = row
            if not row.get("ok", False):
                failures.append(row)

    lines = []
    for key, start in starts.items():
        finish = finishes.get(key)
        phase, algorithm, motion, seed = key
        if finish is None:
            lines.append(f"- `{phase} {algorithm}/{motion}/seed_{seed}`: running since {start.get('time')}")
        else:
            status = "ok" if finish.get("ok") else f"failed rc={finish.get('returncode')}"
            elapsed = float(finish.get("elapsed_seconds", 0.0)) / 3600.0
            lines.append(f"- `{phase} {algorithm}/{motion}/seed_{seed}`: {status}, {elapsed:.2f} h")

    failure_lines = []
    for row in failures:
        failure_lines.append(
            "- "
            f"`{row.get('phase')} {row.get('algorithm')}/{row.get('motion')}/seed_{row.get('seed')}` "
            f"failed with rc={row.get('returncode')}; see `{row.get('command_log')}`"
        )
    return lines, failure_lines


def training_snapshot(root: Path, algorithms: list[str], motions: list[str], seeds: list[int]) -> list[str]:
    lines = []
    for motion in motions:
        for algorithm in algorithms:
            for seed in seeds:
                log_path = root / "runs" / algorithm / motion / f"seed_{seed}" / "log.txt"
                row = read_final_log_row(log_path)
                if not row:
                    continue
                samples = row.get("Samples", 0.0)
                wall_time = row.get("Wall_Time", 0.0)
                test_return = row.get("Test_Return", 0.0)
                ep_len = row.get("Test_Episode_Length", 0.0)
                lines.append(
                    f"- `{algorithm}/{motion}/seed_{seed}`: "
                    f"{samples:.0f} samples, {wall_time:.2f} h, "
                    f"Test_Return={test_return:.2f}, Test_Episode_Length={ep_len:.2f}"
                )
    if not lines:
        lines.append("- No training logs generated yet.")
    return lines


def figure_links(root: Path, motions: list[str]) -> list[str]:
    lines = []
    for motion in motions:
        path = root / "figures" / f"{motion}_test_return.png"
        if path.is_file():
            lines.append(f"- `{path}`")
    if not lines:
        lines.append("- No learning-curve figures generated yet.")
    return lines


def video_links(root: Path, algorithms: list[str], motions: list[str]) -> list[str]:
    lines = []
    for motion in motions:
        for algorithm in algorithms:
            path = root / "videos" / algorithm / motion / "seed_0.mp4"
            if path.is_file():
                lines.append(f"- `{path}`")
    if not lines:
        lines.append("- No videos generated yet.")
    return lines


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    report_file = Path(args.report_file)
    algorithms = select(args.algorithms, ALGORITHMS)
    motions = select(args.motions, MOTIONS)
    status_lines, failure_lines = status_summary(read_jsonl(root / "job_status.jsonl"))
    returns = read_csv_table(root / "tables" / "table_returns.csv")
    tracking = read_csv_table(root / "tables" / "table_tracking_errors.csv")

    content = f"""# GO2 Reproduction Report

## Summary

This report analyzes GO2 motion imitation controllers trained with DeepMimic,
AMP, and ADD in Isaac Lab.

The deadline matrix is intentionally reduced to one seed and four representative
motions: `go2_pace`, `go2_run`, `go2_trot`, and `go2_walk0`. The completed
deadline comparison covers `{", ".join(algorithms)}`. Each full run uses a fixed
`300000000` sample budget. This is a pipeline validation and qualitative method
comparison, not a statistically complete reproduction.

## Experiment Setup

- Simulator backend: Isaac Lab / Isaac Sim
- Robot: Unitree Go2
- Algorithms: `{", ".join(algorithms)}`
- Motions: `{", ".join(motions)}`
- Seeds: `0`
- Max samples per run: `300000000`
- Training mode: headless, `--visualize false`
- Evaluation episodes: `4096`

## Job Status

{chr(10).join(status_lines) if status_lines else "- No jobs recorded yet."}

## Training Snapshot

{chr(10).join(training_snapshot(root, algorithms, motions, args.seeds))}

## Return Summary

{markdown_table(returns)}

## Tracking Error Summary

{markdown_table(tracking)}

## Learning Curves

{chr(10).join(figure_links(root, motions))}

With one seed, the curves should be read as single-run learning trends. They
are still useful for comparing convergence speed and failure modes.

## Video Review

{chr(10).join(video_links(root, algorithms, motions))}

Qualitative review should focus on gait rhythm, body height, torso pitch,
foot sliding, and early falls. Numerical return and visual quality may disagree,
especially for AMP-style objectives.

## Failure Cases

{chr(10).join(failure_lines) if failure_lines else "- No failed jobs recorded yet."}

## Preliminary Interpretation

- DeepMimic directly rewards reference tracking, so it is expected to converge
  quickly on motions where the reference clip and robot dynamics are compatible.
- AMP can produce plausible style but may be less precise on tracking metrics,
  because its objective is distribution matching through a discriminator.
- ADD is expected to combine tracking and adversarial style terms; its value in
  this experiment is whether it improves robustness without losing reference
  fidelity.
- `go2_walk0` is used as the walk representative for the deadline report. The
  other walk clips should be added in the full reproduction.

## Limitations

- Only seed `0` is included in the deadline run.
- Results may differ from the paper because this reproduction uses Isaac Lab,
  while the original GO2 defaults were oriented around Isaac Gym.
- Any over-budget diagnostic runs are excluded from the fair comparison unless
  explicitly labeled.
"""

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(content)
    print(f"Wrote {report_file}")


if __name__ == "__main__":
    main()
