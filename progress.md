# Progress

This file is branch/feature-local state for the GO2 Isaac Lab reproduction
workflow. Stable lessons belong in `AGENTS.md`; durable milestones should be
summarized in commit or PR text.

## Done

- Built GO2 Isaac Lab reproduction entrypoints for DeepMimic, AMP, and ADD.
- Added Isaac Lab args for GO2 train/test/view-motion workflows.
- Fixed GO2 Isaac Lab initialization so configured initial DOF positions are
  respected.
- Completed the May 8 deadline report with seed `0` results for DeepMimic and
  AMP on `go2_pace`, `go2_run`, `go2_trot`, and `go2_walk0`.
- Generated quantitative eval metrics, learning curves, a paper-style tracking
  table, and `docs/GO2_REPRODUCTION_REPORT.md`.
- Migrated long-lived agent instructions from non-standard `agent.md` into root
  `AGENTS.md`.
- Replaced deadline-only helpers with reusable tools:
  - `tools/go2_isaac_lab/launch_detached.py`
  - `tools/go2_isaac_lab/monitor_go2_training.py`
  - `tools/go2_isaac_lab/make_go2_report_outputs.py`
- Updated `run_go2_matrix.py` so training env configs default to
  `log_tracking_error: true` and jobs can use `--job_timeout_seconds`.
- Verified the cleaned tools with `python -m py_compile`, a train dry-run, a
  monitor snapshot, and regenerated report outputs.
- Refactored `AGENTS.md` after scope review into a project handbook structure:
  Project Overview, Environment, Build/Test, Training, Evaluation, Monitoring,
  Git Development, Code Style, and Safety.
- Implemented an unattended supervision path:
  `start_go2_supervised_training.py` starts detached training and installs a
  systemd user timer; `monitor_go2_training.py --alert_codex` generates an
  agent prompt and launches `codex exec` for new issues.
- Added persistent supervision-agent audit files for alert inputs and outputs:
  timestamped `alert_*.md`, `agent_prompt_*.md`, `snapshot_*.md`,
  `agent_alert_*.log`, plus `agent_audit.jsonl`.
- Archived the previous GO2 reproduction outputs under
  `output/go2_reproduction/archived_runs/20260514_211026/`.

## In Progress

- Fresh full GO2 training is running detached as
  `go2_full_motion_major_20260514`, using motion-major order over
  DeepMimic/AMP/ADD, motions `all`, seeds `0 1 2`, `num_envs=4096`, and
  `max_samples=300000000`.
- The systemd user timer
  `go2_full_motion_major_20260514-monitor.timer` checks training every
  1800 seconds and triggers `codex exec` on new monitor issues.
- Pace evaluation for DeepMimic/AMP/ADD seeds `0 1 2` completed with
  `num_envs=4096` and `test_episodes=4096`. Pace-only return/tracking tables
  and per-seed learning-curve figures were generated under
  `output/go2_reproduction/tables/` and `output/go2_reproduction/figures/`.

## Bugs And Risks

- The fresh full seven-motion, three-seed reproduction is currently running;
  final completion, evaluation, plots, tables, and videos are still pending.
- Video recording can hang in Isaac/Kit shutdown paths; use per-job timeouts
  and avoid running video export inside an attached Codex session.
- Existing deadline outputs include an intentionally mixed provenance for
  DeepMimic pace: the over-budget log was used for the clipped learning curve,
  while the shorter retrained model was used for final validation.
- Results are Isaac Lab results; do not assume exact numeric equivalence to
  Isaac Gym-based paper defaults.

## Decisions

- `AGENTS.md` is a project handbook for durable instructions, not a branch
  index. `progress.md` is branch/feature state.
- Important branch outcomes should be propagated through commit/PR messages,
  not by making `progress.md` a global project diary.
- For cross-algorithm tables, use `body_pos_err` as Position Tracking Error and
  `dof_vel_err` as DoF Velocity Tracking Error.
- For learning curves, use algorithm-appropriate metrics: DeepMimic
  `Test_Return`, AMP `Disc_Reward_Mean`, and explicit tracking errors when
  available.
- Long training runs should be detached from Codex and monitored via
  `output/go2_reproduction/monitor/latest.md`.
- Monitoring needs two layers: an external/scheduled process writes compact
  snapshots, and an automation/hook wakes an agent only when those snapshots
  need interpretation.
- In the current local environment, the available wakeup implementation is a
  systemd user timer plus `codex exec`; Codex app-native automations were not
  exposed as a callable tool in this session.

## Next Steps

1. Monitor `output/go2_reproduction/monitor/latest.md` and the systemd timer
   journal while the detached run proceeds.
2. When training completes, run GO2 test, video, collection, and report-output
   commands from `docs/README_GO2_ISAAC_LAB.md`.
3. Inspect diffs for `progress.md` and GO2 tools before committing source
   changes; keep generated `output/` artifacts local unless deliberately
   selected for version control.
