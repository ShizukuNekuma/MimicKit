#!/usr/bin/env python3
"""Offline checks for the GO2 deployment constants and exported bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="output/go2_deploy/deepmimic_go2_pace")
    return parser.parse_args()


def assert_mapping_round_trip() -> None:
    common_to_unitree = [COMMON_JOINT_ORDER.index(name) for name in UNITREE_MOTOR_ORDER]
    unitree_to_common = [UNITREE_MOTOR_ORDER.index(name) for name in COMMON_JOINT_ORDER]
    sample = list(range(len(COMMON_JOINT_ORDER)))
    unitree = [sample[i] for i in common_to_unitree]
    restored = [unitree[i] for i in unitree_to_common]
    if restored != sample:
        raise AssertionError(f"Joint mapping round-trip failed: {restored} != {sample}")


def check_bundle(bundle: Path) -> None:
    required = ["policy.onnx", "deploy_metadata.yaml", "joint_limits.yaml", "pace_reference.csv", "parity_report.json"]
    missing = [name for name in required if not (bundle / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing deployment bundle files under {bundle}: {missing}")

    with (bundle / "pace_reference.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
        first = next(reader)
    expected_cols = 1 + 3 + 4 + 3 + 3 + 12 + 12
    if len(header) != expected_cols or len(first) != expected_cols:
        raise AssertionError(f"pace_reference.csv has {len(header)} columns, expected {expected_cols}")

    report = json.loads((bundle / "parity_report.json").read_text())
    if report.get("onnxruntime_available") and not report.get("passes_1e-4"):
        raise AssertionError(f"ONNX parity failed: {report}")
    if not report.get("clipped_within_hardware_limits", False):
        raise AssertionError("Clipped policy actions exceed hardware limits")


def main() -> None:
    args = parse_args()
    assert_mapping_round_trip()
    bundle = Path(args.bundle)
    if bundle.exists():
        check_bundle(bundle)
    print("GO2 deployment offline checks passed")


if __name__ == "__main__":
    main()
