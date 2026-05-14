#!/usr/bin/env python3
"""Launch a long GO2 command detached from the current terminal/session."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Short job name used for pid and log files.")
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a command after --")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    supervisor_dir = root / "supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    log_path = supervisor_dir / f"{args.name}.log"
    pid_path = supervisor_dir / f"{args.name}.pid.json"

    with log_path.open("ab") as log_file:
        log_file.write((" ".join(args.command) + "\n\n").encode())
        log_file.flush()
        proc = subprocess.Popen(
            args.command,
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
        "pgid": os.getpgid(proc.pid),
        "command": args.command,
        "cwd": str(REPO_ROOT),
        "log_file": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with pid_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Started detached job {args.name} pid={proc.pid}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_path}")


if __name__ == "__main__":
    sys.exit(main())
