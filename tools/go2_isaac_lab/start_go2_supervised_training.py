#!/usr/bin/env python3
"""Start a detached GO2 training job and install periodic agent supervision."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from install_go2_monitor_timer import enable_timer, write_units  # noqa: E402


DEFAULT_CONDA = shutil.which("conda") or "/home/zifeng/miniconda3/bin/conda"
ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="go2_train")
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--conda", default=DEFAULT_CONDA)
    parser.add_argument("--conda_env", default="isaaclab")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--order", choices=["motion_major", "algorithm_major"], default="motion_major")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--logger", choices=["txt", "tb", "wandb"], default="tb")
    parser.add_argument("--max_samples", type=int, default=300_000_000)
    parser.add_argument("--archive_existing", action="store_true")
    parser.add_argument("--job_timeout_seconds", type=int, default=0)
    parser.add_argument("--interval_seconds", type=int, default=300)
    parser.add_argument("--stale_minutes", type=float, default=20.0)
    parser.add_argument("--alert_codex", action="store_true", help="Timer will trigger `codex exec` on new monitor alerts.")
    parser.add_argument("--codex_model", default="")
    parser.add_argument("--codex_sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex_approval", default="never", choices=["untrusted", "on-request", "never"])
    parser.add_argument("--alert_command", default="", help="Optional custom alert hook command for the monitor.")
    parser.add_argument("--enable_timer", action="store_true", help="Enable the systemd user timer immediately.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def train_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.conda,
        "run",
        "-n",
        args.conda_env,
        "python",
        str(REPO_ROOT / "tools/go2_isaac_lab/run_go2_matrix.py"),
        "--phase",
        "train",
        "--algorithms",
        *args.algorithms,
        "--motions",
        *args.motions,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--order",
        args.order,
        "--num_envs",
        str(args.num_envs),
        "--visualize",
        "false",
        "--logger",
        args.logger,
        "--max_samples",
        str(args.max_samples),
        "--output_root",
        args.output_root,
    ]
    if args.archive_existing:
        cmd.append("--archive_existing")
    if args.job_timeout_seconds:
        cmd += ["--job_timeout_seconds", str(args.job_timeout_seconds)]
    return cmd


def write_pid_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def start_detached(args: argparse.Namespace, cmd: list[str]) -> dict[str, object]:
    supervisor_dir = Path(args.output_root) / "supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    log_path = supervisor_dir / f"{args.name}.train.log"
    pid_path = supervisor_dir / f"{args.name}.train.pid.json"

    with log_path.open("ab") as log_file:
        log_file.write((" ".join(cmd) + "\n\n").encode())
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    payload = {
        "name": args.name,
        "pid": proc.pid,
        "command": cmd,
        "log_file": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_pid_file(pid_path, payload)
    payload["pid_file"] = str(pid_path)
    return payload


def timer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        name=f"{args.name}-monitor",
        output_root=args.output_root,
        algorithms=args.algorithms,
        motions=args.motions,
        seeds=args.seeds,
        max_samples=args.max_samples,
        stale_minutes=args.stale_minutes,
        interval_seconds=args.interval_seconds,
        conda=args.conda,
        conda_env=args.conda_env,
        alert_codex=args.alert_codex,
        codex_model=args.codex_model,
        codex_sandbox=args.codex_sandbox,
        codex_approval=args.codex_approval,
        alert_command=args.alert_command,
        enable=args.enable_timer,
    )


def main() -> None:
    args = parse_args()
    if args.alert_codex and shutil.which("codex") is None:
        raise RuntimeError("`--alert_codex` was requested, but no `codex` executable is on PATH.")

    cmd = train_command(args)
    monitor_args = timer_args(args)

    if args.dry_run:
        print("Training command:")
        print(" ".join(cmd))
        print("Timer name:")
        print(f"{monitor_args.name}.timer")
        return

    train_payload = start_detached(args, cmd)
    service_path, timer_path = write_units(monitor_args)
    if args.enable_timer:
        enable_timer(monitor_args.name)

    print(f"Started training pid={train_payload['pid']}")
    print(f"Training log: {train_payload['log_file']}")
    print(f"PID file: {train_payload['pid_file']}")
    print(f"Monitor service: {service_path}")
    print(f"Monitor timer: {timer_path}")
    if args.enable_timer:
        print(f"Enabled {monitor_args.name}.timer")
    else:
        print("Timer written but not enabled. Enable it with:")
        print(f"  systemctl --user daemon-reload")
        print(f"  systemctl --user enable --now {monitor_args.name}.timer")


if __name__ == "__main__":
    sys.exit(main())
