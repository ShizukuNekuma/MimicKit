#!/usr/bin/env python3
"""Export a trained GO2 pace policy as a self-contained deployment bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "output/go2_reproduction/runs/deepmimic/go2_pace/seed_0/model.pt"
DEFAULT_MOTION_FILE = REPO_ROOT / "data/motions/go2/go2_pace.pkl"
DEFAULT_CHAR_FILE = REPO_ROOT / "data/assets/go2/go2.xml"
DEFAULT_OUT_DIR = REPO_ROOT / "output/go2_deploy/deepmimic_go2_pace"

COMMON_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

UNITREE_MOTOR_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

KEY_BODY_ORDER = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


class ExportedActor(torch.nn.Module):
    def __init__(self, state_dict: OrderedDict[str, torch.Tensor]):
        super().__init__()
        obs_mean = state_dict["_obs_norm._mean"].float()
        obs_std = state_dict["_obs_norm._std"].float()
        action_mean = state_dict["_a_norm._mean"].float()
        action_std = state_dict["_a_norm._std"].float()

        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

        self.fc0 = torch.nn.Linear(obs_mean.numel(), 1024)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.action_head = torch.nn.Linear(512, action_mean.numel())
        self.load_state_dict(
            {
                "fc0.weight": state_dict["_model._actor_layers.0.weight"].float(),
                "fc0.bias": state_dict["_model._actor_layers.0.bias"].float(),
                "fc1.weight": state_dict["_model._actor_layers.2.weight"].float(),
                "fc1.bias": state_dict["_model._actor_layers.2.bias"].float(),
                "action_head.weight": state_dict["_model._action_dist._mean_net.weight"].float(),
                "action_head.bias": state_dict["_model._action_dist._mean_net.bias"].float(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "action_mean": action_mean,
                "action_std": action_std,
            },
            strict=True,
        )

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        norm_obs = torch.clamp((raw_obs - self.obs_mean) / self.obs_std, -10.0, 10.0)
        hidden = torch.relu(self.fc0(norm_obs))
        hidden = torch.relu(self.fc1(hidden))
        norm_action = self.action_head(hidden)
        return norm_action * self.action_std + self.action_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--motion_file", default=str(DEFAULT_MOTION_FILE))
    parser.add_argument("--char_file", default=str(DEFAULT_CHAR_FILE))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--control_freq", type=float, default=30.0)
    parser.add_argument("--low_level_freq", type=float, default=500.0)
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=3.0)
    parser.add_argument("--damping_kd", type=float, default=2.0)
    parser.add_argument("--phase_speed_min", type=float, default=0.5)
    parser.add_argument("--phase_speed_max", type=float, default=1.3)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--parity_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--handheld_config", default="config/joystick.yaml")
    return parser.parse_args()


def load_checkpoint(path: Path) -> OrderedDict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, OrderedDict):
        raise TypeError(f"Expected checkpoint state_dict OrderedDict, got {type(state_dict)!r}")
    required = [
        "_obs_norm._mean",
        "_obs_norm._std",
        "_a_norm._mean",
        "_a_norm._std",
        "_model._actor_layers.0.weight",
        "_model._actor_layers.0.bias",
        "_model._actor_layers.2.weight",
        "_model._actor_layers.2.bias",
        "_model._action_dist._mean_net.weight",
        "_model._action_dist._mean_net.bias",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise KeyError(f"Checkpoint is missing required tensors: {missing}")
    return state_dict


def parse_joint_limits(char_file: Path) -> dict[str, tuple[float, float]]:
    tree = ET.parse(char_file)
    limits: dict[str, tuple[float, float]] = {}
    for joint in tree.findall(".//joint"):
        name = joint.attrib.get("name")
        if not name or name == "root":
            continue
        range_text = joint.attrib.get("range")
        if range_text is None:
            continue
        low, high = [float(x) for x in range_text.split()]
        limits[name] = (low, high)
    missing = [name for name in COMMON_JOINT_ORDER if name not in limits]
    if missing:
        raise KeyError(f"Missing GO2 joint limits in {char_file}: {missing}")
    return limits


def exp_map_to_quat(exp_map: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(exp_map, axis=-1, keepdims=True)
    safe_angle = np.maximum(angle, 1e-8)
    axis = exp_map / safe_angle
    half = 0.5 * angle
    xyz = axis * np.sin(half)
    w = np.cos(half)
    quat = np.concatenate([xyz, w], axis=-1)
    small = (angle[..., 0] < 1e-8)
    if np.any(small):
        quat[small] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat.astype(np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_to_exp_map(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    neg = q[..., 3] < 0.0
    q[neg] *= -1.0
    length = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, q[..., 3:4])
    axis = q[..., :3] / np.maximum(length, 1e-8)
    return (axis * angle).astype(np.float32)


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(values, dtype=np.float32)
    vel[:-1] = fps * (values[1:] - values[:-1])
    vel[-1] = vel[-2] if len(values) > 1 else 0.0
    return vel


def load_motion_rows(motion_file: Path) -> tuple[float, int, np.ndarray]:
    with motion_file.open("rb") as f:
        data = pickle.load(f)
    fps = float(data["fps"])
    frames = np.asarray(data["frames"], dtype=np.float32)
    loop_mode = int(data["loop_mode"])
    return fps, loop_mode, frames


def write_reference_csv(out_file: Path, motion_file: Path) -> dict[str, object]:
    fps, loop_mode, frames = load_motion_rows(motion_file)
    root_pos = frames[:, 0:3]
    root_quat = exp_map_to_quat(frames[:, 3:6])
    dof_pos = frames[:, 6:18]
    root_vel = finite_difference(root_pos, fps)
    root_delta = quat_mul(root_quat[1:], quat_conj(root_quat[:-1]))
    root_ang_vel = np.zeros_like(root_pos, dtype=np.float32)
    root_ang_vel[:-1] = fps * quat_to_exp_map(root_delta)
    root_ang_vel[-1] = root_ang_vel[-2] if len(root_ang_vel) > 1 else 0.0
    dof_vel = finite_difference(dof_pos, fps)

    header = (
        ["time"]
        + [f"root_pos_{axis}" for axis in "xyz"]
        + [f"root_quat_{axis}" for axis in ["x", "y", "z", "w"]]
        + [f"root_vel_{axis}" for axis in "xyz"]
        + [f"root_ang_vel_{axis}" for axis in "xyz"]
        + [f"dof_pos_{name}" for name in COMMON_JOINT_ORDER]
        + [f"dof_vel_{name}" for name in COMMON_JOINT_ORDER]
    )
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(frames.shape[0]):
            t = i / fps
            row = (
                [t]
                + root_pos[i].tolist()
                + root_quat[i].tolist()
                + root_vel[i].tolist()
                + root_ang_vel[i].tolist()
                + dof_pos[i].tolist()
                + dof_vel[i].tolist()
            )
            writer.writerow([f"{x:.9g}" for x in row])

    return {
        "fps": fps,
        "loop_mode": loop_mode,
        "num_frames": int(frames.shape[0]),
        "length_seconds": float((frames.shape[0] - 1) / fps),
        "wrap_delta_xyz": (root_pos[-1] - root_pos[0]).astype(float).tolist(),
    }


def dump_yaml_scalar(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Non-finite YAML value: {value}")
        return repr(value)
    raise TypeError(type(value))


def write_yaml(out_file: Path, data: dict[str, object]) -> None:
    def emit(obj: object, indent: int = 0) -> list[str]:
        pad = " " * indent
        if isinstance(obj, dict):
            lines: list[str] = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {dump_yaml_scalar(value)}")
            return lines
        if isinstance(obj, list):
            lines = []
            for value in obj:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}- {dump_yaml_scalar(value)}")
            return lines
        return [f"{pad}{dump_yaml_scalar(obj)}"]

    out_file.write_text("\n".join(emit(data)) + "\n")


def write_joint_limits(out_file: Path, limits: dict[str, tuple[float, float]]) -> None:
    rows = {
        "common_order": COMMON_JOINT_ORDER,
        "unitree_motor_order": UNITREE_MOTOR_ORDER,
        "hardware_limits": {
            name: {"min": limits[name][0], "max": limits[name][1]} for name in COMMON_JOINT_ORDER
        },
    }
    write_yaml(out_file, rows)


def export_onnx(model: ExportedActor, out_file: Path, opset: int) -> None:
    model.eval()
    dummy = torch.zeros(1, model.obs_mean.numel(), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_file),
        input_names=["raw_obs"],
        output_names=["joint_position"],
        dynamic_axes={"raw_obs": {0: "batch"}, "joint_position": {0: "batch"}},
        opset_version=opset,
    )


def compute_parity(
    model: ExportedActor,
    onnx_file: Path,
    samples: int,
    seed: int,
    limits: dict[str, tuple[float, float]],
) -> dict[str, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    raw_obs = model.obs_mean.unsqueeze(0) + model.obs_std.unsqueeze(0) * torch.randn(
        samples, model.obs_mean.numel(), generator=gen
    )
    with torch.no_grad():
        torch_actions = model(raw_obs).numpy()

    report: dict[str, object] = {
        "samples": samples,
        "torch_action_min": torch_actions.min(axis=0).astype(float).tolist(),
        "torch_action_max": torch_actions.max(axis=0).astype(float).tolist(),
    }

    limit_low = np.asarray([limits[name][0] for name in COMMON_JOINT_ORDER], dtype=np.float32)
    limit_high = np.asarray([limits[name][1] for name in COMMON_JOINT_ORDER], dtype=np.float32)
    clipped = np.clip(torch_actions, limit_low, limit_high)
    report["clipped_action_min"] = clipped.min(axis=0).astype(float).tolist()
    report["clipped_action_max"] = clipped.max(axis=0).astype(float).tolist()
    report["clipped_within_hardware_limits"] = bool(
        np.all(clipped >= limit_low - 1e-6) and np.all(clipped <= limit_high + 1e-6)
    )

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host environment
        report["onnxruntime_available"] = False
        report["onnxruntime_error"] = str(exc)
        return report

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    onnx_actions = session.run(["joint_position"], {"raw_obs": raw_obs.numpy().astype(np.float32)})[0]
    report["onnxruntime_available"] = True
    report["max_abs_error"] = float(np.max(np.abs(onnx_actions - torch_actions)))
    report["mean_abs_error"] = float(np.mean(np.abs(onnx_actions - torch_actions)))
    report["passes_1e-4"] = bool(report["max_abs_error"] < 1e-4)
    return report


def metadata(args: argparse.Namespace, state_dict: OrderedDict[str, torch.Tensor], motion_info: dict[str, object]) -> dict[str, object]:
    action_mean = state_dict["_a_norm._mean"].float().tolist()
    action_std = state_dict["_a_norm._std"].float().tolist()
    checkpoint_path = Path(args.checkpoint).resolve()
    return {
        "name": "deepmimic_go2_pace",
        "checkpoint": os.path.relpath(checkpoint_path, REPO_ROOT),
        "policy_file": "policy.onnx",
        "pace_reference_file": "pace_reference.csv",
        "joint_limits_file": "joint_limits.yaml",
        "handheld_config": args.handheld_config,
        "obs_dim": int(state_dict["_obs_norm._mean"].numel()),
        "action_dim": int(state_dict["_a_norm._mean"].numel()),
        "control_freq_hz": float(args.control_freq),
        "low_level_freq_hz": float(args.low_level_freq),
        "pd_gains": {"kp": float(args.kp), "kd": float(args.kd), "damping_kd": float(args.damping_kd)},
        "phase_speed": {"min": float(args.phase_speed_min), "max": float(args.phase_speed_max), "default": 1.0},
        "common_joint_order": COMMON_JOINT_ORDER,
        "unitree_motor_order": UNITREE_MOTOR_ORDER,
        "key_body_order": KEY_BODY_ORDER,
        "action_normalizer": {"mean": action_mean, "std": action_std},
        "action_bounds_from_training": {
            "min": [float(m - s) for m, s in zip(action_mean, action_std)],
            "max": [float(m + s) for m, s in zip(action_mean, action_std)],
        },
        "reference_motion": motion_info,
        "safety": {
            "dry_run_default": True,
            "requires_arm_command": True,
            "clips_to_hardware_limits": True,
            "publishes_damping_on_disarm": True,
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    motion_file = Path(args.motion_file)
    char_file = Path(args.char_file)

    state_dict = load_checkpoint(checkpoint)
    limits = parse_joint_limits(char_file)
    model = ExportedActor(state_dict)

    onnx_file = out_dir / "policy.onnx"
    export_onnx(model, onnx_file, args.opset)

    motion_info = write_reference_csv(out_dir / "pace_reference.csv", motion_file)
    write_joint_limits(out_dir / "joint_limits.yaml", limits)
    write_yaml(out_dir / "deploy_metadata.yaml", metadata(args, state_dict, motion_info))

    report = compute_parity(model, onnx_file, args.parity_samples, args.seed, limits)
    (out_dir / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote GO2 deployment bundle to {out_dir}")
    if report.get("onnxruntime_available"):
        print(f"ONNX parity max_abs_error={report['max_abs_error']:.6g}")
    else:
        print("ONNX Runtime was not available; wrote PyTorch-only parity metadata.")


if __name__ == "__main__":
    main()
