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

## In Progress

- Review before commit: inspect diffs and decide whether generated report
  outputs under `output/go2_reproduction/` should be committed or kept local.

## Bugs And Risks

- Full ADD training/evaluation has not been completed yet.
- Full seven-motion, three-seed reproduction has not been run.
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

1. Inspect diffs for `AGENTS.md`, `progress.md`, docs, and GO2 tools.
2. Decide what generated outputs, if any, belong in version control.
3. After this cleanup is committed, resume the full ADD and remaining
   motion/seed matrix when compute time is available.
