# Shared locomotion architecture refactor

## Changes

- Moved generic manager configs from `tasks/velocity/cfg/` to
  `tasks/locomotion/velocity/cfg/`.
- Moved generic assembly to `tasks/locomotion/velocity/velocity_env_cfg.py`.
- Added terrain-neutral `LocomotionVelocityEnvCfg` and sibling
  `LocomotionVelocityFlatEnvCfg` / `LocomotionVelocityRoughEnvCfg`.
- Body terrain scanning, height observations and terrain progression are
  enabled only by the generic rough variant. Shared foot-height sensing
  remains available on flat terrain. Disabled fields retain their positions
  to preserve observation and curriculum ordering in subclasses.
- Migrated every independent Microduck recipe to the generic flat base.
  Existing robot-specific inheritance for velstand, swizzle, roller_slope,
  roller_standup and Microduck rough variants is preserved.
- Updated imports throughout the task packages and tests, including stale
  pre-restructure test imports and private helper/constant locations.
- Updated the roller-standup reward-index regression test to use the existing
  canonical servo-only index contract; the reward implementation is unchanged.
- Updated standalone testbench to the API of the pinned mjlab 1.3.0:
  TerrainEntityCfg, RslRlModelCfg actor/critic configs, entity_name fields,
  and CommandTermCfg.build(). Preserved its policy observation-group name via
  an explicit runner mapping. Added the BAM startup field-expansion event
  required by its actuator. No new task registrations were added.

## Validation

- Full real-dependency CPU test suite: **230 passed, 1 skipped** (577 import
  subtests passed). The skipped check requires a linux-aarch64 GPU machine.
- Compared **70 serialized environment configurations** against the uploaded
  source: 33 registered task IDs x train/play, plus left/right BallKick x
  train/play. All values, callable references, manager key order and observation
  term order are identical after normalizing the checkout path.
- Checked generic flat/rough sensor and curriculum wiring, mutable-config
  isolation, play/training separation, and standalone testbench construction.
- Parsed/compiled all Python sources, tests and scripts; local imports resolve.
- Tested with mjlab 1.3.0, MuJoCo 3.10.0, warp-lang 1.12.0, PyTorch 2.9.1+cpu,
  and BAM revision 62bd8ce12154340be97e06f7f41a0ca8f116d967 from uv.lock.
  Project dependency declarations and uv.lock are unchanged.
- GPU training and a simulation training smoke run were not performed.

Run the suite in the project environment:

```bash
uv run --with pytest pytest tests/ -q
```

## Applying the archive

Extract into a fresh directory or remove the old `tasks/velocity/cfg/` and
`tasks/velocity/locomotion_velocity_env_cfg.py` when replacing an existing
checkout. The old internal import paths were removed rather than kept as
compatibility wrappers. Public Microduck package imports, factory signatures,
task IDs and training commands are unchanged.

See `src/mjlab_microduck/tasks/velocity/MIGRATION.md` for usage examples.
