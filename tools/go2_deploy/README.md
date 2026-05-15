# GO2 Deployment Tools

This directory contains the offline export path for the GO2 pace controller.

Export the default DeepMimic pace checkpoint:

```bash
conda run -n isaaclab python tools/go2_deploy/export_go2_policy.py
```

The command writes a bundle under `output/go2_deploy/deepmimic_go2_pace/`:

- `policy.onnx`: actor inference with observation normalization and action unnormalization.
- `deploy_metadata.yaml`: joint order, control rates, PD gains, safety defaults, and reference metadata.
- `pace_reference.csv`: reference root and joint trajectory used by the deploy node.
- `joint_limits.yaml`: GO2 hardware joint limits and joint order.
- `parity_report.json`: PyTorch vs ONNX Runtime parity when `onnxruntime` is installed.

Run offline checks:

```bash
python3 tools/go2_deploy/check_go2_deploy.py --bundle output/go2_deploy/deepmimic_go2_pace
```
