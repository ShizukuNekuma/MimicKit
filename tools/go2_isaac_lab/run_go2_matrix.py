#!/usr/bin/env python3
"""Run the GO2 Isaac Lab reproduction matrix."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
ENGINE_CONFIG = "data/engines/isaac_lab_engine.yaml"

ALGORITHMS = {
    "deepmimic": {
        "env": "data/envs/deepmimic_go2_env.yaml",
        "agent": "data/agents/deepmimic_go2_ppo_agent.yaml",
    },
    "amp": {
        "env": "data/envs/amp_go2_env.yaml",
        "agent": "data/agents/amp_go2_agent.yaml",
    },
    "add": {
        "env": "data/envs/add_go2_env.yaml",
        "agent": "data/agents/add_go2_agent.yaml",
    },
}

MOTIONS = {
    "go2_pace": "data/motions/go2/go2_pace.pkl",
    "go2_run": "data/motions/go2/go2_run.pkl",
    "go2_trot": "data/motions/go2/go2_trot.pkl",
    "go2_walk0": "data/motions/go2/go2_walk0.pkl",
    "go2_walk1": "data/motions/go2/go2_walk1.pkl",
    "go2_walk2": "data/motions/go2/go2_walk2.pkl",
    "go2_walk3": "data/motions/go2/go2_walk3.pkl",
}

DEFAULT_MAX_SAMPLES = 300_000_000

TRACKING_ERROR_KEYS = [
    "root_pos_err",
    "root_rot_err",
    "body_pos_err",
    "body_rot_err",
    "dof_vel_err",
    "root_vel_err",
    "root_ang_vel_err",
]


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    return text.lower() in ("1", "true", "t", "yes", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "test", "video"], required=True)
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--motions", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--order", choices=["motion_major", "algorithm_major"], default="motion_major")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--test_episodes", type=int, default=4096)
    parser.add_argument("--visualize", type=parse_bool, default=False)
    parser.add_argument("--logger", choices=["txt", "tb", "wandb"], default="tb")
    parser.add_argument("--output_root", default="output/go2_reproduction")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--train_log_tracking_error",
        type=parse_bool,
        default=True,
        help="Enable tracking-error diagnostics in generated training env configs. Defaults to true for report-ready curves.",
    )
    parser.add_argument("--video", type=parse_bool, default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--job_timeout_seconds",
        type=int,
        default=0,
        help="Optional per-job timeout. The runner kills only the child process group it started.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--halt_on_error", action="store_true")
    parser.add_argument("--archive_existing", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), help=argparse.SUPPRESS)
    parser.add_argument("--motion", choices=sorted(MOTIONS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--env_config", help=argparse.SUPPRESS)
    parser.add_argument("--agent_config", help=argparse.SUPPRESS)
    parser.add_argument("--model_file", help=argparse.SUPPRESS)
    parser.add_argument("--metrics_file", help=argparse.SUPPRESS)
    parser.add_argument("--video_file", help=argparse.SUPPRESS)
    return parser.parse_args()


def select_names(values: list[str], choices: dict[str, Any]) -> list[str]:
    if "all" in values:
        return list(choices.keys())

    bad = [v for v in values if v not in choices]
    if bad:
        raise ValueError(f"Unknown value(s): {bad}. Choices: {list(choices)}")
    return values


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def make_env_config(
    output_root: Path,
    phase: str,
    algorithm: str,
    motion: str,
    log_tracking_error: bool,
) -> Path:
    base_env = REPO_ROOT / ALGORITHMS[algorithm]["env"]
    data = load_yaml(base_env)
    data["motion_file"] = MOTIONS[motion]
    data["log_tracking_error"] = bool(log_tracking_error)

    out_file = output_root / "configs" / phase / f"{algorithm}_{motion}.yaml"
    write_yaml(out_file, data)
    return out_file


def run_dir(output_root: Path, algorithm: str, motion: str, seed: int) -> Path:
    return output_root / "runs" / algorithm / motion / f"seed_{seed}"


def eval_dir(output_root: Path, algorithm: str, motion: str, seed: int) -> Path:
    return output_root / "eval" / algorithm / motion / f"seed_{seed}"


def video_dir(output_root: Path, algorithm: str, motion: str) -> Path:
    return output_root / "videos" / algorithm / motion


def archive_path(path: Path, output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        rel_path = path.relative_to(output_root)
    except ValueError:
        rel_path = Path(path.name)
    return output_root / "archived_runs" / stamp / rel_path


def archive_existing_run(path: Path, output_root: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    dst = archive_path(path, output_root)
    print(f"[archive] {rel(path)} -> {rel(dst)}", flush=True)
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))


def build_train_command(args: argparse.Namespace, algorithm: str, motion: str, seed: int) -> list[str]:
    output_root = Path(args.output_root)
    env_config = make_env_config(output_root, "train", algorithm, motion, log_tracking_error=args.train_log_tracking_error)
    out_dir = run_dir(output_root, algorithm, motion, seed)

    cmd = [
        sys.executable,
        "mimickit/run.py",
        "--mode",
        "train",
        "--num_envs",
        str(args.num_envs),
        "--engine_config",
        ENGINE_CONFIG,
        "--env_config",
        rel(env_config),
        "--agent_config",
        ALGORITHMS[algorithm]["agent"],
        "--visualize",
        str(args.visualize).lower(),
        "--logger",
        args.logger,
        "--out_dir",
        rel(out_dir),
        "--rand_seed",
        str(seed),
    ]

    cmd += ["--max_samples", str(args.max_samples)]

    return cmd


def build_worker_command(args: argparse.Namespace, algorithm: str, motion: str, seed: int) -> list[str]:
    output_root = Path(args.output_root)
    phase = args.phase
    env_config = make_env_config(output_root, phase, algorithm, motion, log_tracking_error=True)
    model_file = run_dir(output_root, algorithm, motion, seed) / "model.pt"

    if phase == "test":
        metrics_file = eval_dir(output_root, algorithm, motion, seed) / "metrics.json"
        video_file = ""
        record_video = False
    else:
        metrics_file = video_dir(output_root, algorithm, motion) / f"seed_{seed}.json"
        video_file = video_dir(output_root, algorithm, motion) / f"seed_{seed}.mp4"
        record_video = True

    cmd = [
        sys.executable,
        rel(Path(__file__).resolve()),
        "--worker",
        "--phase",
        phase,
        "--algorithm",
        algorithm,
        "--motion",
        motion,
        "--seed",
        str(seed),
        "--num_envs",
        str(args.num_envs),
        "--test_episodes",
        str(args.test_episodes),
        "--visualize",
        str(args.visualize).lower(),
        "--video",
        str(record_video or args.video).lower(),
        "--device",
        args.device,
        "--env_config",
        rel(env_config),
        "--agent_config",
        ALGORITHMS[algorithm]["agent"],
        "--model_file",
        rel(model_file),
        "--metrics_file",
        rel(metrics_file),
    ]

    if video_file:
        cmd += ["--video_file", rel(video_file)]

    return cmd


def command_log_path(args: argparse.Namespace, algorithm: str, motion: str, seed: int) -> Path:
    return Path(args.output_root) / "command_logs" / args.phase / f"{algorithm}_{motion}_seed_{seed}.log"


def append_status(args: argparse.Namespace, row: dict[str, Any]) -> None:
    path = Path(args.output_root) / "job_status.jsonl"
    if args.dry_run:
        return
    ensure_parent(path)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run_command(
    args: argparse.Namespace,
    cmd: list[str],
    algorithm: str,
    motion: str,
    seed: int,
) -> bool:
    printable = " ".join(cmd)
    print(printable, flush=True)
    if args.dry_run:
        return True

    log_path = command_log_path(args, algorithm, motion, seed)
    ensure_parent(log_path)
    start = time.time()
    append_status(args, {
        "event": "start",
        "phase": args.phase,
        "algorithm": algorithm,
        "motion": motion,
        "seed": seed,
        "command": printable,
        "command_log": rel(log_path),
        "time": datetime.now().isoformat(timespec="seconds"),
    })

    timed_out = False
    with log_path.open("w") as log_file:
        log_file.write(printable + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=args.job_timeout_seconds or None)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait()

    elapsed = time.time() - start
    ok = returncode == 0
    append_status(args, {
        "event": "finish",
        "phase": args.phase,
        "algorithm": algorithm,
        "motion": motion,
        "seed": seed,
        "returncode": returncode,
        "ok": ok,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "command_log": rel(log_path),
        "time": datetime.now().isoformat(timespec="seconds"),
    })
    if not ok:
        msg = f"[failed] {args.phase} {algorithm} {motion} seed={seed} rc={returncode}; see {rel(log_path)}"
        if args.halt_on_error:
            raise RuntimeError(msg)
        print(msg, flush=True)
    return ok


def tensor_to_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def run_worker(args: argparse.Namespace) -> None:
    if str(MIMICKIT_ROOT) not in sys.path:
        sys.path.insert(0, str(MIMICKIT_ROOT))

    import torch

    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import util.mp_util as mp_util
    import util.util as util

    if not Path(args.model_file).is_file():
        raise FileNotFoundError(f"Missing model file: {args.model_file}")

    torch.multiprocessing.set_start_method("spawn", force=True)
    mp_util.init(rank=0, num_procs=1, device=args.device, master_port=None)
    util.set_rand_seed(args.seed)

    env = env_builder.build_env(
        args.env_config,
        ENGINE_CONFIG,
        args.num_envs,
        args.device,
        visualize=args.visualize,
        record_video=args.video,
    )
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)

    result = agent.test_model(args.test_episodes)
    diagnostics = env.record_diagnostics()

    metrics: dict[str, Any] = {
        "algorithm": args.algorithm,
        "motion": args.motion,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "test_episodes": args.test_episodes,
        "model_file": args.model_file,
    }
    metrics.update({k: tensor_to_value(v) for k, v in result.items()})

    for key in TRACKING_ERROR_KEYS:
        if key in diagnostics:
            metrics[key] = tensor_to_value(diagnostics[key])

    if args.phase == "video":
        video = diagnostics.get("sim_recording")
        if video is None:
            raise RuntimeError("Video recording was requested, but no video was returned by the environment.")
        if video.get_num_frames() == 0:
            raise RuntimeError("Video recording produced zero frames.")
        video_path = Path(args.video_file)
        ensure_parent(video_path)
        video.save(str(video_path))
        metrics["video_file"] = str(video_path)
        metrics["video_frames"] = video.get_num_frames()

    metrics_path = Path(args.metrics_file)
    ensure_parent(metrics_path)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if hasattr(env, "_engine") and env._engine.enabled_record_video():
        env._engine.stop_video_recording()


def run_matrix(args: argparse.Namespace) -> None:
    algorithms = select_names(args.algorithms, ALGORITHMS)
    motions = select_names(args.motions, MOTIONS)

    if args.phase == "train" and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive for training.")

    if args.phase == "train":
        print(f"[train] max_samples={args.max_samples}", flush=True)

    if args.order == "motion_major":
        jobs = [(algorithm, motion, seed) for motion in motions for algorithm in algorithms for seed in args.seeds]
    else:
        jobs = [(algorithm, motion, seed) for algorithm in algorithms for motion in motions for seed in args.seeds]

    failures = 0
    for algorithm, motion, seed in jobs:
        if args.phase == "train":
            out_dir = run_dir(Path(args.output_root), algorithm, motion, seed)
            if args.archive_existing:
                archive_existing_run(out_dir, Path(args.output_root), args.dry_run)
            cmd = build_train_command(args, algorithm, motion, seed)
        else:
            cmd = build_worker_command(args, algorithm, motion, seed)
        if not run_command(args, cmd, algorithm, motion, seed):
            failures += 1

    if failures:
        print(f"[done-with-failures] {failures} job(s) failed.", flush=True)


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.worker:
        run_worker(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
