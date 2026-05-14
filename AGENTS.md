# Agent Guidelines for MimicKit

These instructions apply to the entire repository.

## Project Direction

MimicKit is now maintained as a GO2-focused motion imitation project. Future training work should assume:

- The primary robot is Unitree GO2.
- The training backend is Isaac Lab.
- GO2 training, evaluation, reporting, monitoring, and launch tooling under `tools/go2_isaac_lab/` is reusable project infrastructure, not one-off reproduction scaffolding.
- Avoid adding new Isaac Gym, Newton, or non-GO2 workflow surface unless the user explicitly asks for it.

Use `docs/README_GO2_ISAAC_LAB.md` as the durable GO2 workflow reference. Use `progress.md` only for branch-local status and handoff notes.

## Repository Layout

- `mimickit/`: core runtime code, including environments, learning agents, models, engines, animation utilities, and `mimickit/run.py`.
- `mimickit/engines/isaac_lab_engine.py`: active simulator backend for training work.
- `data/envs/`, `data/agents/`, `data/engines/`: YAML configs used by the training entrypoints.
- `args/*_go2_lab_args.txt`: Isaac Lab GO2 single-run argument files.
- `tools/go2_isaac_lab/`: reusable GO2 batch, evaluation, reporting, plotting, detached launch, and monitoring tools.
- `docs/README_GO2_ISAAC_LAB.md`: operational handbook for GO2 Isaac Lab training and evaluation.
- `output/`: generated experiment output; keep it out of source control unless the user explicitly asks to version selected results.

## Environment

Use the Isaac Lab environment for GO2 work:

```bash
conda activate isaaclab
```

For non-interactive checks, prefer:

```bash
conda run -n isaaclab python -c "import isaaclab; print('isaaclab ok')"
conda run -n isaaclab python -c "import isaacsim; print('isaacsim ok')"
```

Install MimicKit Python dependencies from the repository root when needed:

```bash
pip install -r requirements.txt
```

## Running GO2 Jobs

Run single GO2 Isaac Lab jobs with the lab argument files:

```bash
conda run -n isaaclab python mimickit/run.py --arg_file args/deepmimic_go2_ppo_lab_args.txt --visualize false
conda run -n isaaclab python mimickit/run.py --arg_file args/amp_go2_lab_args.txt --visualize false
conda run -n isaaclab python mimickit/run.py --arg_file args/add_go2_lab_args.txt --visualize false
```

Run matrix jobs with the reusable GO2 runner:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase train \
  --algorithms deepmimic amp add \
  --motions all \
  --seeds 0 1 2 \
  --num_envs 4096 \
  --visualize false \
  --logger tb \
  --max_samples 300000000 \
  --output_root output/go2_reproduction
```

Use `--motions all` for the full GO2 motion set, or pass explicit motion names such as `go2_pace go2_run go2_trot`.

Use `--dry_run` before long runs to inspect commands. Use `--max_samples`; without it, `mimickit/run.py` defaults to an effectively unbounded sample budget.

## Long-Running Training

Do not run multi-hour jobs attached to an interactive Codex tool session. Use the project helpers:

- `tools/go2_isaac_lab/launch_detached.py` for detached commands.
- `tools/go2_isaac_lab/start_go2_supervised_training.py` for detached training plus periodic monitoring.
- `tools/go2_isaac_lab/monitor_go2_training.py` for compact health snapshots under `output/go2_reproduction/monitor/`.

The monitor writes files and alerts; it should not kill or restart jobs by itself. If it reports an issue, inspect the specific command log before deciding whether to resume, archive, or rerun. Only stop processes that are clearly part of the current GO2 job.

## Evaluation And Reporting

Use the reusable GO2 tools for post-training work:

- `tools/go2_isaac_lab/run_go2_matrix.py --phase test` for quantitative metrics.
- `tools/go2_isaac_lab/run_go2_matrix.py --phase video` for validation videos.
- `tools/go2_isaac_lab/collect_go2_results.py` for return and tracking-error tables.
- `tools/go2_isaac_lab/make_go2_report_outputs.py` for learning curves and paper-style tables.

For cross-algorithm tracking tables, use `body_pos_err` as Position Tracking Error and `dof_vel_err` as DoF Velocity Tracking Error unless the user specifies a different report definition.

Video recording can hang in Isaac/Kit shutdown paths. Use per-job timeouts and kill only the child process group launched by the runner.

## Development Guidelines

- Preserve the existing import style inside `mimickit/`, which uses top-level imports such as `envs.*`, `learning.*`, and `util.*` when run from the repository root.
- When adding environments, agents, engines, or network definitions, update the corresponding builder so YAML configs can instantiate the new component.
- Prefer extending GO2 Isaac Lab configs and tools over adding parallel non-GO2 or non-Isaac-Lab workflows.
- Keep YAML config names and keys consistent with nearby examples in `data/envs/`, `data/agents/`, `data/engines/`, and `args/`.
- Keep generated configs, logs, checkpoints, videos, monitor snapshots, and report outputs under `output/go2_reproduction/` unless the user asks for a different output root.
- Preserve the Isaac Lab initial DOF position behavior for GO2. GO2 initialization depends on configured initial joint positions being respected.

## Validation

There is no configured repository-wide test suite. For Python changes, at minimum run syntax checks on changed files:

```bash
python -m py_compile <changed .py files>
```

When the Isaac Lab environment is available, prefer:

```bash
conda run -n isaaclab python -m py_compile <changed .py files>
```

For GO2 tooling changes, also run a dry-run check:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase train \
  --algorithms deepmimic \
  --motions go2_pace \
  --seeds 0 \
  --num_envs 1 \
  --visualize false \
  --max_samples 64 \
  --output_root /tmp/mimickit_go2_dry_run \
  --dry_run
```

For full simulator validation, use a small smoke run only when Isaac Lab, assets, motion data, and compute budget are available.

## Data And Artifacts

- Do not commit generated `output/`, `wandb/`, `__pycache__/`, `.pyc`, local editor files, logs, checkpoints, videos, or large derived assets unless explicitly requested.
- Documentation images may be committed when they are intentionally referenced by docs.
- Treat downloaded data under `data/` as external project assets. Keep path references stable so existing YAML configs and argument files continue to work.

## Safety

- Preserve user edits and unrelated local changes.
- Avoid destructive commands unless explicitly requested.
- Do not kill jobs that were not started by the current task.
- For repository history work, commit only after the staged diff has been reviewed and validation results are known.
