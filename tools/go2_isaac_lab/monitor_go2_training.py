#!/usr/bin/env python3
"""Write compact GO2 training health snapshots for low-token supervision."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ["deepmimic", "amp", "add"]
MOTIONS = ["go2_pace", "go2_run", "go2_trot", "go2_walk0", "go2_walk1", "go2_walk2", "go2_walk3"]
DEFAULT_MAX_SAMPLES = 300_000_000
ISSUE_STATES = {"failed", "early_finish_under_budget", "stale_running", "orphaned_or_stopped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--motions", nargs="+", default=MOTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max_samples", type=float, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--min_completion_ratio", type=float, default=0.98)
    parser.add_argument("--stale_minutes", type=float, default=20.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval_seconds", type=float, default=300.0)
    parser.add_argument("--alert_on_issues", action="store_true", help="Write alert prompt and trigger the configured agent/hook when issues are detected.")
    parser.add_argument("--alert_cooldown_minutes", type=float, default=60.0, help="Minimum time before repeating the same alert signature.")
    parser.add_argument("--alert_command", default="", help="Optional shell command to run on a new alert. Supports {prompt_file}, {alert_file}, {snapshot_file}, and {output_root}.")
    parser.add_argument("--alert_codex", action="store_true", help="Run `codex exec` on a new alert using the generated prompt.")
    parser.add_argument("--codex_model", default="", help="Optional model passed to `codex exec -m`.")
    parser.add_argument("--codex_sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex_approval", default="never", choices=["untrusted", "on-request", "never"])
    return parser.parse_args()


def select(values: list[str], all_values: list[str]) -> list[str]:
    if "all" in values:
        return all_values
    bad = [value for value in values if value not in all_values]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {all_values}")
    return values


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
                rows.append({"event": "json_decode_error", "raw": line})
    return rows


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


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return f"unavailable: {exc}"


def process_snapshot(output_root: str) -> list[dict[str, Any]]:
    text = command_output(["ps", "-eo", "pid,ppid,stat,etime,pcpu,pmem,args"])
    rows = []
    for line in text.splitlines()[1:]:
        if "run_go2_matrix.py" not in line and "mimickit/run.py" not in line:
            continue
        if output_root not in line and "go2_reproduction" not in line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, stat, etime, pcpu, pmem, cmd = parts
        rows.append({
            "pid": int(pid),
            "ppid": int(ppid),
            "stat": stat,
            "etime": etime,
            "pcpu": pcpu,
            "pmem": pmem,
            "cmd": cmd,
        })
    return rows


def gpu_snapshot() -> str:
    return command_output([
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]).strip()


def status_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "finish":
            continue
        try:
            key = (str(row["phase"]), str(row["algorithm"]), str(row["motion"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        latest[key] = row
    return latest


def job_running(processes: list[dict[str, Any]], output_root: str, algorithm: str, motion: str, seed: int) -> bool:
    needles = [
        f"{algorithm}_{motion}_seed_{seed}",
        f"runs/{algorithm}/{motion}/seed_{seed}",
        f"{algorithm} --motion {motion}",
    ]
    return any(any(needle in proc["cmd"] for needle in needles) for proc in processes)


def inspect_train_job(
    root: Path,
    output_root_arg: str,
    algorithm: str,
    motion: str,
    seed: int,
    max_samples: float,
    min_completion_ratio: float,
    stale_minutes: float,
    finishes: dict[tuple[str, str, str, int], dict[str, Any]],
    processes: list[dict[str, Any]],
) -> dict[str, Any]:
    log_path = root / "runs" / algorithm / motion / f"seed_{seed}" / "log.txt"
    row = read_final_log_row(log_path)
    samples = row.get("Samples")
    test_return = row.get("Test_Return")
    disc_reward = row.get("Disc_Reward_Mean")
    age_minutes = None
    if log_path.is_file():
        age_minutes = (time.time() - log_path.stat().st_mtime) / 60.0

    finish = finishes.get(("train", algorithm, motion, seed))
    running = job_running(processes, output_root_arg, algorithm, motion, seed)
    target = max_samples * min_completion_ratio
    state = "not_started"
    issue = ""

    if finish is not None:
        if not finish.get("ok", False):
            state = "failed"
            issue = f"returncode={finish.get('returncode')}; see {finish.get('command_log')}"
        elif samples is not None and samples < target:
            state = "early_finish_under_budget"
            issue = f"samples={samples:.0f} < target~{target:.0f}"
        else:
            state = "done"
    elif running:
        if age_minutes is not None and age_minutes > stale_minutes:
            state = "stale_running"
            issue = f"log has not changed for {age_minutes:.1f} min"
        else:
            state = "running"
    elif log_path.is_file():
        state = "orphaned_or_stopped"
        issue = "log exists but no running process and no finish record"

    return {
        "algorithm": algorithm,
        "motion": motion,
        "seed": seed,
        "state": state,
        "issue": issue,
        "samples": samples,
        "test_return": test_return,
        "disc_reward_mean": disc_reward,
        "log_age_minutes": age_minutes,
        "log_path": str(log_path),
    }


def write_snapshot(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    algorithms = select(args.algorithms, ALGORITHMS)
    motions = select(args.motions, MOTIONS)
    monitor_dir = root / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(root / "job_status.jsonl")
    finishes = status_index(rows)
    processes = process_snapshot(args.output_root)
    jobs = [
        inspect_train_job(
            root,
            args.output_root,
            algorithm,
            motion,
            seed,
            args.max_samples,
            args.min_completion_ratio,
            args.stale_minutes,
            finishes,
            processes,
        )
        for algorithm in algorithms
        for motion in motions
        for seed in args.seeds
    ]
    issues = [job for job in jobs if job["state"] in ISSUE_STATES]
    running = [job for job in jobs if job["state"] == "running"]
    done = [job for job in jobs if job["state"] == "done"]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": args.output_root,
        "gpu": gpu_snapshot(),
        "processes": processes,
        "counts": {
            "done": len(done),
            "running": len(running),
            "issues": len(issues),
            "total": len(jobs),
        },
        "issues": issues,
        "running": running,
        "jobs": jobs,
    }
    with (monitor_dir / "latest.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    lines = [
        "# GO2 Training Monitor",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Counts: done `{len(done)}`, running `{len(running)}`, issues `{len(issues)}`, total `{len(jobs)}`",
        f"- GPU: `{payload['gpu']}`",
        f"- Matched processes: `{len(processes)}`",
        "",
        "## Issues",
    ]
    if not issues:
        lines.append("- None detected.")
    else:
        for job in issues[:20]:
            samples = "--" if job["samples"] is None else f"{job['samples']:.0f}"
            lines.append(
                f"- `{job['algorithm']}/{job['motion']}/seed_{job['seed']}`: "
                f"{job['state']}; samples={samples}; {job['issue']}"
            )
    lines += ["", "## Running"]
    if not running:
        lines.append("- None.")
    else:
        for job in running[:20]:
            samples = "--" if job["samples"] is None else f"{job['samples']:.0f}"
            age = "--" if job["log_age_minutes"] is None else f"{job['log_age_minutes']:.1f} min"
            lines.append(f"- `{job['algorithm']}/{job['motion']}/seed_{job['seed']}`: samples={samples}; log age={age}")
    lines += ["", "## Recent Processes"]
    if not processes:
        lines.append("- None matched.")
    else:
        for proc in processes[:10]:
            lines.append(f"- pid `{proc['pid']}` elapsed `{proc['etime']}` cpu `{proc['pcpu']}`: `{proc['cmd'][:180]}`")
    (monitor_dir / "latest.md").write_text("\n".join(lines) + "\n")
    with (monitor_dir / "events.jsonl").open("a") as f:
        f.write(json.dumps({"time": payload["generated_at"], "counts": payload["counts"], "issues": issues}, sort_keys=True) + "\n")
    if args.alert_on_issues:
        maybe_alert(args, payload, monitor_dir)
    print(f"Wrote {monitor_dir / 'latest.md'}")


def issue_signature(issues: list[dict[str, Any]]) -> str:
    compact = [
        {
            "algorithm": issue["algorithm"],
            "motion": issue["motion"],
            "seed": issue["seed"],
            "state": issue["state"],
            "issue": issue["issue"],
            "samples": issue["samples"],
        }
        for issue in issues
    ]
    blob = json.dumps(compact, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def read_alert_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r") as f:
        return json.load(f)


def write_alert_state(path: Path, state: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def should_alert(state_path: Path, signature: str, cooldown_minutes: float, now: float) -> bool:
    state = read_alert_state(state_path)
    last_signature = state.get("signature")
    last_alert_time = float(state.get("last_alert_time", 0.0))
    if signature != last_signature:
        return True
    return (now - last_alert_time) / 60.0 >= cooldown_minutes


def agent_prompt(payload: dict[str, Any], alert_file: Path, snapshot_file: Path) -> str:
    issue_lines = []
    for issue in payload["issues"]:
        samples = "--" if issue["samples"] is None else f"{issue['samples']:.0f}"
        issue_lines.append(
            f"- {issue['algorithm']}/{issue['motion']}/seed_{issue['seed']}: "
            f"{issue['state']}; samples={samples}; {issue['issue']}"
        )
    return f"""You are the GO2 training supervision agent for the MimicKit repository.

Workspace: {REPO_ROOT}

Read these files first:
- AGENTS.md
- progress.md
- {snapshot_file}
- {alert_file}

Current alert:
{chr(10).join(issue_lines)}

Your job:
1. Diagnose whether the issue is a real training failure, a harmless completed partial run, or a monitor false positive.
2. If a safe fix is possible, apply it. Examples: resume a missing matrix job, mark an expected partial run in progress.md, or rerun a failed job with the same output root after archiving as appropriate.
3. Do not kill unrelated processes. Only stop processes clearly launched for this GO2 job.
4. Preserve user edits and generated results unless the user explicitly asked to remove them.
5. Update progress.md with what happened and what you did.
6. Leave a concise summary in the final response/log.

Do not stream long logs. Start from the compact monitor snapshot and inspect full command logs only for the jobs listed in the alert.
"""


def run_hook(command_template: str, prompt_file: Path, alert_file: Path, snapshot_file: Path, output_root: str, log_file: Path) -> int:
    command = command_template.format(
        prompt_file=str(prompt_file),
        alert_file=str(alert_file),
        snapshot_file=str(snapshot_file),
        output_root=output_root,
    )
    with log_file.open("ab") as f:
        f.write((command + "\n\n").encode())
        f.flush()
        proc = subprocess.Popen(command, cwd=REPO_ROOT, shell=True, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc.pid


def run_codex(args: argparse.Namespace, prompt_file: Path, log_file: Path) -> int:
    prompt = prompt_file.read_text()
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        args.codex_sandbox,
        "--ask-for-approval",
        args.codex_approval,
    ]
    if args.codex_model:
        cmd += ["-m", args.codex_model]
    cmd.append(prompt)
    with log_file.open("ab") as f:
        f.write((" ".join(shlex.quote(part) for part in cmd[:-1]) + " <prompt>\n\n").encode())
        f.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc.pid


def maybe_alert(args: argparse.Namespace, payload: dict[str, Any], monitor_dir: Path) -> None:
    issues = payload["issues"]
    state_path = monitor_dir / "alert_state.json"
    if not issues:
        write_alert_state(state_path, {
            "signature": "",
            "last_alert_time": 0.0,
            "last_alert_at": payload["generated_at"],
            "status": "clear",
        })
        return

    signature = issue_signature(issues)
    now = time.time()
    if not should_alert(state_path, signature, args.alert_cooldown_minutes, now):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = monitor_dir / "latest.md"
    alert_file = monitor_dir / "alert.md"
    prompt_file = monitor_dir / "agent_prompt.md"
    agent_log_file = monitor_dir / f"agent_alert_{timestamp}.log"

    alert_lines = [
        "# GO2 Training Alert",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Signature: `{signature}`",
        f"- Issues: `{len(issues)}`",
        "",
        "## Issues",
    ]
    for issue in issues:
        samples = "--" if issue["samples"] is None else f"{issue['samples']:.0f}"
        alert_lines.append(
            f"- `{issue['algorithm']}/{issue['motion']}/seed_{issue['seed']}`: "
            f"{issue['state']}; samples={samples}; {issue['issue']}"
        )
    alert_file.write_text("\n".join(alert_lines) + "\n")
    prompt_file.write_text(agent_prompt(payload, alert_file, snapshot_file))

    triggered: list[dict[str, Any]] = []
    if args.alert_command:
        pid = run_hook(args.alert_command, prompt_file, alert_file, snapshot_file, args.output_root, agent_log_file)
        triggered.append({"kind": "alert_command", "pid": pid, "log_file": str(agent_log_file)})
    if args.alert_codex:
        pid = run_codex(args, prompt_file, agent_log_file)
        triggered.append({"kind": "codex", "pid": pid, "log_file": str(agent_log_file)})

    state = {
        "signature": signature,
        "last_alert_time": now,
        "last_alert_at": payload["generated_at"],
        "issues": issues,
        "alert_file": str(alert_file),
        "prompt_file": str(prompt_file),
        "triggered": triggered,
        "status": "alerted" if triggered else "written",
    }
    write_alert_state(state_path, state)
    with (monitor_dir / "alerts.jsonl").open("a") as f:
        f.write(json.dumps(state, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    while True:
        write_snapshot(args)
        if not args.watch:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
