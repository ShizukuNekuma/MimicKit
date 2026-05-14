#!/usr/bin/env python3
"""Install a systemd user timer for GO2 training supervision."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDA = shutil.which("conda") or "/home/zifeng/miniconda3/bin/conda"
ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mimickit-go2-monitor")
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max_samples", type=int, default=300_000_000)
    parser.add_argument("--stale_minutes", type=float, default=20.0)
    parser.add_argument("--interval_seconds", type=int, default=300)
    parser.add_argument("--conda", default=DEFAULT_CONDA)
    parser.add_argument("--conda_env", default="isaaclab")
    parser.add_argument("--alert_codex", action="store_true", help="Trigger `codex exec` on new monitor alerts.")
    parser.add_argument("--codex_model", default="")
    parser.add_argument("--codex_sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex_approval", default="never", choices=["untrusted", "on-request", "never"])
    parser.add_argument("--alert_command", default="", help="Optional custom alert hook command.")
    parser.add_argument("--enable", action="store_true", help="Run systemctl --user daemon-reload and enable the timer now.")
    return parser.parse_args()


def monitor_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.conda,
        "run",
        "-n",
        args.conda_env,
        "python",
        str(REPO_ROOT / "tools/go2_isaac_lab/monitor_go2_training.py"),
        "--output_root",
        args.output_root,
        "--algorithms",
        *args.algorithms,
        "--motions",
        *args.motions,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--max_samples",
        str(args.max_samples),
        "--stale_minutes",
        str(args.stale_minutes),
        "--alert_on_issues",
    ]
    if args.alert_codex:
        cmd.append("--alert_codex")
        cmd += ["--codex_sandbox", args.codex_sandbox, "--codex_approval", args.codex_approval]
        if args.codex_model:
            cmd += ["--codex_model", args.codex_model]
    if args.alert_command:
        cmd += ["--alert_command", args.alert_command]
    return cmd


def write_units(args: argparse.Namespace) -> tuple[Path, Path]:
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    service_path = unit_dir / f"{args.name}.service"
    timer_path = unit_dir / f"{args.name}.timer"

    command = shlex.join(monitor_command(args))
    service = f"""[Unit]
Description=MimicKit GO2 training monitor

[Service]
Type=oneshot
WorkingDirectory={REPO_ROOT}
ExecStart=/usr/bin/env bash -lc {shlex.quote(command)}
"""
    timer = f"""[Unit]
Description=Run MimicKit GO2 training monitor every {args.interval_seconds} seconds

[Timer]
OnBootSec=60
OnUnitActiveSec={args.interval_seconds}
AccuracySec=30
Persistent=true

[Install]
WantedBy=timers.target
"""
    service_path.write_text(service)
    timer_path.write_text(timer)
    return service_path, timer_path


def enable_timer(name: str) -> None:
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{name}.timer"], check=True)


def main() -> None:
    args = parse_args()
    service_path, timer_path = write_units(args)
    print(f"Wrote {service_path}")
    print(f"Wrote {timer_path}")
    if args.enable:
        enable_timer(args.name)
        print(f"Enabled {args.name}.timer")
    else:
        print("To enable:")
        print(f"  systemctl --user daemon-reload")
        print(f"  systemctl --user enable --now {args.name}.timer")
        print("To inspect:")
        print(f"  systemctl --user status {args.name}.timer")
        print(f"  journalctl --user -u {args.name}.service -f")


if __name__ == "__main__":
    sys.exit(main())
