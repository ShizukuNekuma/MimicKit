# GO2 Pace Deployment

This document describes the first GO2 pace deployment path:

1. Export the trained MimicKit policy to an ONNX deployment bundle.
2. Validate the ROS2 low-level controller in Unitree MuJoCo.
3. Move the same ROS2 node to the GO2 onboard computer for staged real-robot tests.

The first version is intentionally limited to safe enable/disable, emergency stop, and pace phase-speed scaling. It does not implement joystick velocity or yaw commands because the current policy was trained to imitate a fixed pace reference and does not receive command-conditioned observations.

## Export

From the MimicKit repository root:

```bash
conda run -n isaaclab python tools/go2_deploy/export_go2_policy.py
python3 tools/go2_deploy/check_go2_deploy.py --bundle output/go2_deploy/deepmimic_go2_pace
```

The default checkpoint is:

```text
output/go2_reproduction/runs/deepmimic/go2_pace/seed_0/model.pt
```

The export writes:

- `policy.onnx`
- `deploy_metadata.yaml`
- `pace_reference.csv`
- `joint_limits.yaml`
- `parity_report.json`

`policy.onnx` takes raw MimicKit observations with shape `[N, 484]` and returns 12 common-order joint position targets. The ROS2 node clips those targets to the GO2 hardware limits before publishing Unitree motor commands.

## Joint Order

The exported policy uses MimicKit common order:

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

Unitree low-level motor order is:

```text
FR_0, FR_1, FR_2,
FL_0, FL_1, FL_2,
RR_0, RR_1, RR_2,
RL_0, RL_1, RL_2
```

The ROS2 deploy node performs this mapping internally for both `/lowstate` reads and `/lowcmd` writes.

## Build The ROS2 Node

On a ROS2 Foxy machine with `unitree_ros2` already built and sourced:

```bash
source ~/unitree_ros2/setup.sh
mkdir -p ~/go2_mimickit_ws/src
ln -s /home/zifeng/MimicKit/ros2/go2_mimickit_deploy ~/go2_mimickit_ws/src/go2_mimickit_deploy
cd ~/go2_mimickit_ws
colcon build --packages-select go2_mimickit_deploy
source install/setup.bash
```

The package requires ONNX Runtime C++ headers and library. If they are not in a standard path, set:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

The default launch parameters keep `dry_run: true`, so the node subscribes, runs inference, and logs targets without publishing motor commands.

## Sim-To-Sim With Unitree MuJoCo

Start Unitree MuJoCo with GO2 on local DDS domain 1. Then run:

```bash
source ~/unitree_ros2/setup_local.sh
export ROS_DOMAIN_ID=1
ros2 launch go2_mimickit_deploy go2_policy.launch.py
```

After confirming `/lowstate`, `/wirelesscontroller`, and policy logs are healthy, copy the params file and explicitly set:

```yaml
dry_run: false
```

Use low gains first. The node publishes damping commands while disarmed, and only publishes policy targets after the configured arm command.

## Sim-To-Real Staging

On the Jetson Orin NX:

1. Connect the GO2 network interface and source `~/unitree_ros2/setup.sh`.
2. Set `ROS_DOMAIN_ID=0`.
3. Run the node with `dry_run: true`; verify state, joystick, and policy logs.
4. Calibrate `arm_button_mask`, `disarm_button_mask`, and `estop_button_mask` in `ros2/go2_mimickit_deploy/config/go2_policy_params.yaml`.
5. Test suspended with low gains and `dry_run: false`.
6. Test briefly on the ground with a spotter and immediate emergency stop access.

For real-robot runs, keep `enable_hold_button_mask` configured once the controller key mapping is known. Leaving it at `0` means the policy runs whenever armed, which is convenient in simulation but less conservative on hardware.

## Important Limits

- The deploy node reconstructs the MimicKit observation from `/lowstate` and the exported pace reference. Root x/y velocity is reference-based, while angular velocity comes from the IMU.
- The controller holds the most recent policy target between 30 Hz policy ticks and publishes low-level commands at 500 Hz.
- If observation construction or ONNX inference throws, the node disarms and publishes damping if configured to do so.
