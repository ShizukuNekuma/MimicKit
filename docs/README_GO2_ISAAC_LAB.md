# GO2 Isaac Lab Reproduction

This note defines the reproduction workflow for training and evaluating Unitree
Go2 motion imitation controllers in MimicKit with Isaac Lab.

## Goal

The reproduction target is a systematic GO2 study over:

- Algorithms: `deepmimic`, `amp`, `add`
- Motions: `go2_pace`, `go2_run`, `go2_trot`, `go2_walk0`, `go2_walk1`, `go2_walk2`, `go2_walk3`
- Seeds: `0`, `1`, `2`

The final deliverables are:

- Table-style summaries similar to [table](../images/test_result_table.png).
- Learning-curve figures similar to [figure](../images/learning_curve_fig.png).
- Final validation videos for selected trained policies.
- A written analysis report in `docs/GO2_REPRODUCTION_REPORT.md`.

## Environment

Validated local status:

- `isaaclab` works: Python 3.11, PyTorch CUDA, Isaac Lab, and Isaac Sim import successfully.
  - `isaacsim` version: `5.1.0-rc.19+release.26219.9c81211b.gl`
  - `isaaclab` version: `2.3.2`
- MimicKit dependencies have been installed in the working Isaac Lab environment.

Use this environment:

```bash
conda activate isaaclab
```

For non-interactive checks:

```bash
conda run -n isaaclab python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n isaaclab python -c "import isaaclab; print('isaaclab ok')"
conda run -n isaaclab python -c "import isaacsim; print('isaacsim ok')"
```

## Outputs

The reproduction tools write all experiment artifacts under:

```text
output/go2_reproduction/
```

Important subdirectories:

- `runs/{algorithm}/{motion}/seed_{seed}/`: training logs, copied configs, and `model.pt`.
- `eval/{algorithm}/{motion}/seed_{seed}/metrics.json`: test return, episode length, and tracking errors.
- `tables/`: CSV and Markdown summary tables.
- `figures/`: learning-curve figures.
- `videos/{algorithm}/{motion}/seed_{seed}.mp4`: final validation videos.
- `configs/`: generated environment configs with the selected motion file.
- `command_logs/{phase}/`: stdout/stderr for each matrix job.
- `job_status.jsonl`: start/finish records for each matrix job.
- `monitor/latest.md`: compact health snapshot for low-token supervision.
- `supervisor/`: detached launch logs and pid files.
- `archived_runs/`: old runs moved aside when `--archive_existing` is used.

## Args

The original GO2 args use Isaac Gym. Use the Isaac Lab variants:

- `args/deepmimic_go2_ppo_lab_args.txt`
- `args/amp_go2_lab_args.txt`
- `args/add_go2_lab_args.txt`
- `args/view_motion_go2_lab_args.txt`

The batch scripts usually pass explicit configs instead of relying on these arg
files, but the files remain useful for single-run debugging.

## Sanity Check

Before launching the full matrix, run a tiny DeepMimic pace check:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase train \
  --algorithms deepmimic \
  --motions go2_pace \
  --seeds 0 \
  --num_envs 1 \
  --visualize false \
  --max_samples 64 \
  --output_root output/go2_reproduction_smoke
```

Then test it:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase test \
  --algorithms deepmimic \
  --motions go2_pace \
  --seeds 0 \
  --num_envs 1 \
  --test_episodes 1 \
  --output_root output/go2_reproduction_smoke
```

## Training

For matrix training, use the reusable runner:

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
Other CLI args include:
- `--order algorithm_major` prioritizes different motions for one algorithm before moving to the next algorithm. By default, it is motion_major.
- `--archive_existing` moves old runs aside before retraining so
overtrained or aborted experiments do not contaminate the fair comparison.
- `--dry_run` is used to inspect commands without starting training:

Important training rules:

- Do not run multi-hour jobs attached to a Codex tool session. Use detached
  launchers, `tmux`, `nohup`, or `systemd --user`.
- Do not kill jobs that were not started by the current task. Inspect process
  command lines first and only stop processes that are clearly yours.
- `--max_samples` is mandatory in practice; without it, MimicKit defaults to an effectively infinite sample budget.
- Keep tracking-error logging enabled for reproduction runs unless deliberately
  measuring its overhead. `run_go2_matrix.py` defaults
  `--train_log_tracking_error true`.

Training produces one `log.txt` and one final `model.pt` per run.

Useful live monitors:

```bash
tail -f output/go2_reproduction/command_logs/train/deepmimic_go2_pace_seed_0.log
tail -f output/go2_reproduction/runs/deepmimic/go2_pace/seed_0/log.txt
tail -f output/go2_reproduction/job_status.jsonl
conda run -n isaaclab tensorboard --logdir output/go2_reproduction/runs --port 6006
```

`log.txt` updates at MimicKit's output interval, not every environment step.
TensorBoard is the easiest way to watch learning curves while the batch runs.

## Detached Long Runs

For multi-hour training, do not attach the process to an interactive Codex tool
session. Launch it detached:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/launch_detached.py \
  --name go2_full_train \
  --output_root output/go2_reproduction \
  -- conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
    --phase train \
    --algorithms deepmimic amp add \
    --motions all \
    --seeds 0 \
    --order algorithm_major \
    --num_envs 4096 \
    --visualize false \
    --logger tb \
    --max_samples 300000000 \
    --output_root output/go2_reproduction
```

The launcher writes:

- `output/go2_reproduction/supervisor/go2_full_train.log`
- `output/go2_reproduction/supervisor/go2_full_train.pid.json`

## Supervised Training

For unattended runs, use the supervised launcher. It starts training detached,
writes a systemd user timer for periodic monitoring, and can trigger a
non-interactive Codex agent when new issues appear:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/start_go2_supervised_training.py \
  --name go2_full \
  --output_root output/go2_reproduction \
  --algorithms deepmimic amp add \
  --motions all \
  --seeds 0 1 2 \
  --order algorithm_major \
  --num_envs 4096 \
  --max_samples 300000000 \
  --interval_seconds 300 \
  --stale_minutes 20 \
  --alert_codex \
  --enable_timer
```

This creates:

- `output/go2_reproduction/supervisor/go2_full.train.log`
- `output/go2_reproduction/supervisor/go2_full.train.pid.json`
- `~/.config/systemd/user/go2_full-monitor.service`
- `~/.config/systemd/user/go2_full-monitor.timer`

On each timer tick, the monitor updates:

- `output/go2_reproduction/monitor/latest.md`
- `output/go2_reproduction/monitor/latest.json`

If a new issue is detected, it also writes:

- `output/go2_reproduction/monitor/alert.md`
- `output/go2_reproduction/monitor/agent_prompt.md`
- `output/go2_reproduction/monitor/agent_alert_*.log`

The default `--alert_codex` trigger runs:

```text
codex exec --cd <repo> --sandbox workspace-write --ask-for-approval never <generated prompt>
```

That spawned agent is separate from the original training launcher. It receives
the monitor snapshot, the alert, `AGENTS.md`, and `progress.md` as required
context. The prompt tells it not to kill unrelated processes and to inspect full
logs only for affected jobs.

Useful systemd commands:

```bash
systemctl --user status go2_full-monitor.timer
journalctl --user -u go2_full-monitor.service -f
systemctl --user disable --now go2_full-monitor.timer
```

## Low-Token Monitoring

Instead of streaming logs into chat, generate compact snapshots:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/monitor_go2_training.py \
  --output_root output/go2_reproduction \
  --motions all \
  --seeds 0 \
  --max_samples 300000000
```

Read this file first:

```text
output/go2_reproduction/monitor/latest.md
```

It summarizes running jobs, stale logs, early finishes under the sample budget,
GPU status, and matched training processes. For unattended monitoring, run:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/monitor_go2_training.py \
  --output_root output/go2_reproduction \
  --motions all \
  --seeds 0 \
  --max_samples 300000000 \
  --watch \
  --interval_seconds 300
```

This monitor writes files only; it does not kill or restart jobs. If it reports
an issue, inspect the command log before deciding whether to resume, archive, or
rerun.

To enable alert generation without launching Codex, add:

```bash
--alert_on_issues
```

To use a custom hook instead of Codex, pass a command template:

```bash
--alert_command "notify-send 'GO2 training alert' 'See {alert_file}'"
```

The placeholders `{prompt_file}`, `{alert_file}`, `{snapshot_file}`, and
`{output_root}` are available.

## Testing

For quantitative evaluation:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase test \
  --algorithms deepmimic amp add \
  --motions all \
  --seeds 0 1 2 \
  --num_envs 4096 \
  --test_episodes 4096 \
  --output_root output/go2_reproduction
```

GO2 evaluation metrics are written to:

```text
output/go2_reproduction/eval/{algorithm}/{motion}/seed_{seed}/metrics.json
```

These include:

- `mean_return`
- `mean_ep_len`
- `root_pos_err`
- `root_rot_err`
- `body_pos_err`
- `body_rot_err`
- `dof_vel_err`
- `root_vel_err`
- `root_ang_vel_err`

## Videos

To record videos:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/run_go2_matrix.py \
  --phase video \
  --algorithms deepmimic amp add \
  --motions all \
  --seeds 0 1 2 \
  --num_envs 1 \
  --test_episodes 1 \
  --video true \
  --job_timeout_seconds 600 \
  --output_root output/go2_reproduction
```

`--video true` enables MimicKit's Isaac Lab recorder. The batch script saves the
returned video object as `mp4` under `output/go2_reproduction/videos/`.
`--job_timeout_seconds` is recommended because Isaac/Kit video shutdown can
occasionally hang.

## Tables

Collect returns and tracking errors:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/collect_go2_results.py \
  --output_root output/go2_reproduction \
  --motions all \
  --seeds 0 1 2
```

Generated files:

- `output/go2_reproduction/tables/table_returns.csv`
- `output/go2_reproduction/tables/table_tracking_errors.csv`
- `output/go2_reproduction/tables/table_tracking_errors.md`

Use `table_tracking_errors.md` directly in the report draft, then discuss which
algorithm best tracks each motion and where distribution matching behaves
differently from explicit tracking.

## Figures And Paper-Style Tables

Generate report-ready learning curves and the paper-style tracking table:

```bash
conda run -n isaaclab python tools/go2_isaac_lab/make_go2_report_outputs.py \
  --output_root output/go2_reproduction \
  --motions all \
  --seeds 0 1 2 \
  --max_samples 300000000
```

Generated files:

- `output/go2_reproduction/figures/deepmimic_{motion}_test_return.png`
- `output/go2_reproduction/figures/amp_{motion}_disc_reward_mean.png`
- `output/go2_reproduction/tables/table_tracking_errors_paper_style.csv`
- `output/go2_reproduction/tables/table_tracking_errors_paper_style.md`
- `output/go2_reproduction/tables/table_tracking_errors_paper_style.png`

By default, DeepMimic curves use `Test_Return`, AMP curves use
`Disc_Reward_Mean`, and ADD curves use `Test_Return` if present. Use
`--metric algorithm:key` to override the metric mapping. With multiple seeds,
the shaded region is the standard deviation.

Metric gotchas:

- DeepMimic `Test_Return` is periodic training-time test-rollout return. It is
  useful for DeepMimic because the reward is explicit motion tracking, but it is
  distinct from final validation metrics under `eval/`.
- AMP environment `Test_Return` can be zero because AMP trains from
  discriminator reward. For AMP learning curves, prefer `Disc_Reward_Mean`.
- For GO2 paper-style tables, use `body_pos_err` as Position Tracking Error and
  `dof_vel_err` as DoF Velocity Tracking Error unless a report defines different
  columns.

## Report

Write the final analysis in:

```text
docs/GO2_REPRODUCTION_REPORT.md
```

The report should include:

- Experiment matrix and compute environment.
- Table summaries.
- Learning curves.
- Qualitative video observations.
- Failure cases and likely causes.
- Conclusions about DeepMimic, AMP, and ADD for GO2 motion imitation.
