# Manager-based env-cfg restructure — `roller_standup` — status & notes

## What changed

`tasks/microduck_roller_standup_env_cfg.py` (~550 lines, one function that
called `make_microduck_velocity_rollers_env_cfg(play=play)` and patched the
result heavily) has been replaced by `tasks/roller_standup/`, the same
IsaacLab-style, one-file-per-manager package shape used for every other
task in this codebase. Backend is still 100% mjlab — nothing about the
runtime changed, only how the config is *authored*.

**This is the last of the 11 task files.** Every `tasks/microduck_*_env_cfg.py`
listed in the original folder screenshot has now been restructured.

## Post-delivery bugfix: `SERVO_LEG_JOINTS` (found by a real GPU training run)

After this migration was delivered, a real `uv run train
Mjlab-RollerStandUp-Flat-MicroDuck` run crashed with a CUDA `index out of
bounds` assert inside `pose_target_match`. Root cause, confirmed and fixed:

`pose_target_match`/`pose_l1_penalty`/`standing_composite_score` all route
their `joint_indices` param through `_servo_joint_pos()`/
`_servo_default_joint_pos()` (see `mdp/_common.py`), which **already**
filters out `passive_*` joints and compacts the remaining 14 servo joints
into a dense `0-13` index space — its own docstring says so explicitly:
"All joint-index-based reward/event params in this module ... are written
against the canonical 14-servo layout ... raw indices would select the
wrong joints." `LEG_JOINTS = [0,1,2,3,4, 11,12,13,14,15]`, however, was
computed against the **raw, uncompacted** 18-joint layout (with the
interleaved wheels still in it) — correct for the joint-index-lock test
(`test_joint_indices_match_actual_roller_model`, which compiles the real
MjSpec and checks names at those raw indices), but **not** what these
three reward functions expect. Indices 14/15 don't even exist in a
14-column servo-filtered tensor, hence the crash.

**This bug was already present in the original (pre-refactor) file** — the
migration faithfully copied `_LEG_JOINTS = [0,1,2,3,4,11,12,13,14,15]`
verbatim, and the project's own pre-existing test
(`test_pose_rewards_target_legs_only_at_roller_indices`) asserted the same
(buggy) value, so it wasn't caught by any verification this refactor could
do without a real GPU (no amount of stub-`mjlab` import testing or
CPU-only assertion checking exercises actual CUDA tensor indexing). It
took a real training run to surface it — exactly why every task's
MIGRATION.md recommends one before trusting these configs in production.

**Fix**: added `SERVO_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]` to
`microduck_flags.py` — the correctly-compacted equivalent (computed by
simulating the servo-filter: drop indices `{5,6,16,17}` from `range(18)`,
then find the new positions of `{0,1,2,3,4,11,12,13,14,15}` in what's
left). It is, not coincidentally, numerically **identical** to the
wheel-less `standup` task's own `LEG_JOINTS` — filtering out the
interleaved wheels leaves the same leg/neck/leg joint ordering as the
wheel-less model has natively. `microduck_rewards_cfg.py`'s three affected
`joint_indices` params were switched to `SERVO_LEG_JOINTS`; the raw
`LEG_JOINTS`/`NECK_JOINTS`/`WHEEL_JOINTS` constants are unchanged and still
correctly document/lock the real model's raw joint layout for the CPU-only
index test. The project's own test
(`test_pose_rewards_target_legs_only_at_roller_indices`) was updated to
assert against `SERVO_LEG_JOINTS` instead — re-verified: 36/37 tests still
pass (the 37th needs real `mujoco`, unrelated to this fix).

```
tasks/roller_standup/
  microduck_flags.py                    # ROLLER_STAND_Z/PRONE_Z, LEG_JOINTS/
                                         # NECK_JOINTS/WHEEL_JOINTS (remapped
                                         # for the interleaved-wheel joint
                                         # order), SKATING_REWARDS,
                                         # play_face_up_overrides() + resolve_play_face_up()
  microduck_observations_cfg.py         # MicroduckObservationsCfg -- same terms as
                                         # velocity_rollers, nan_policy="sanitize"
  microduck_commands_cfg.py             # MicroduckCommandsCfg -- twist neutralised,
                                         # VelocityCommandCommandOnlyCfg (not heading-relative)
  microduck_rewards_cfg.py              # MicroduckRewardsCfg(velocity_rollers.MicroduckRewardsCfg)
                                         # -- standup's rise recipe, remapped joint indices
  microduck_events_cfg.py               # MicroduckEventsCfg -- ground-state spawn (belly/back/standing)
  microduck_terminations_cfg.py         # MicroduckTerminationsCfg -- fell_over dropped
  microduck_curriculum_cfg.py           # MicroduckCurriculumCfg -- ground-state ramp +
                                         # INVERTED wheel-friction ramp + action-rate/push ramps
  microduck_roller_standup_env_cfg.py   # MicroduckRollerStandupEnvCfg(velocity_rollers.MicroduckVelocityRollersEnvCfg)
                                         # + make_microduck_roller_standup_env_cfg()
  microduck_rl_cfg.py                   # MicroduckRollerStandUpRlCfg (RSL-RL PPO)
  __init__.py
```

Same "straightforward subclass" pattern as `roller_slope`: subclasses
`tasks.velocity_rollers`'s actual env-cfg class
(`MicroduckVelocityRollersEnvCfg`) directly (the original called
`make_microduck_velocity_rollers_env_cfg(play=play)`, not the raw mjlab
base).

## Things worth flagging clearly

- **Interleaved wheel joints mean this task's joint indices are NOT the
  same as `standup`'s** (the walking-duck version), even though the reward
  *recipe* is a direct transplant. The roller model's passive wheels sit
  between the leg and neck joints in the articulation order
  (`LEG_JOINTS = [0,1,2,3,4, 11,12,13,14,15]`, `NECK_JOINTS = [7,8,9,10]`,
  `WHEEL_JOINTS = [5,6,16,17]`), not the wheel-less model's `[0-4, 9-13]` /
  `[5-8]`. This is locked by a dedicated test
  (`test_joint_indices_match_actual_roller_model`, in
  `tests/test_roller_standup_cfg.py`) that compiles the real robot XML and
  checks joint names at these indices — the one test in this whole
  migration that genuinely needs real `mujoco` (see Verified below).
- **`joint_torques_l2` (inherited, unchanged) and `joint_torque_rate_l2`
  (new) are two SEPARATE reward terms** that coexist — easy to misread as
  a rename given the near-identical names. `joint_torques_l2` flows through
  from `velocity_rollers` untouched (not in `SKATING_REWARDS`, so never
  dropped); `joint_torque_rate_l2` is a genuinely new field with its own
  weight/history (see the reward file's docstring for the full
  measured-contribution story on both).
- **The rolling-friction curriculum is INVERTED relative to
  `velocity_rollers`'s own `wheel_friction`** — same field name
  (`wheel_friction`), same event (`randomize_wheel_friction`), but ramping
  DOWN (0.05 -> 0.0015, near-locked to freely-rolling) instead of UP (0 ->
  0.0015). The rise needs traction to bootstrap; the skating task needs the
  opposite. This is the one genuinely new mechanical piece of this task
  (per the module docstring), not just a reused/renamed reward.
- **Play-mode override reuses the exact `roller_slope`
  (`SLOPE_PLAY_DIFFICULTY`) pattern** for a different variable
  (`STANDUP_PLAY_FACE_UP`): forces a supine ("on-the-back", the hardest
  recovery case) start mix at play time, since a freshly-rebuilt play env's
  curriculum always starts at stage 0 (where `face_up_prob = 0`) and would
  otherwise never show the hardest case. Implemented as a `play_face_up_
  overrides()` helper in `microduck_flags.py` (mirrors `ball_offset_y_of()`
  in `ball_kick`, `support_foot_of()` in the same file) plus the same
  "write the event params, then null the curriculum term" mechanism
  `velstand` uses for its own play-mode termination/curriculum override.
- **`test_roller_standup_cfg.py` already existed** (37 tests — the largest
  test suite of any task in this refactor) and encodes an unusually
  detailed behavioral contract: distinct `SceneEntityCfg` object identity
  per reward term, event-declaration-order guarantees, curriculum
  monotonicity checks, the exact sign-convention lock for
  `gentle_rise`/`height_stand_l1`/`pose_stand_l1`, and all 7 play-override
  edge cases (forced, clamped, invalid input, `"none"` keyword, training
  immunity). Every one of these was checked against the new package, not
  just term counts.

**Public API is unchanged**: `make_microduck_roller_standup_env_cfg(play=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task ID. `tasks/__init__.py` needed only the one import-path change;
`tests/test_roller_standup_cfg.py` needed its imports updated (the module
path, plus a few module-level constants that moved into `microduck_flags.py`
and lost their leading underscore, matching this refactor's naming
convention for constants — `LEG_JOINTS` not `_LEG_JOINTS`, etc.).

## Verified

- `pyflakes` clean on every new file.
- **`tests/test_roller_standup_cfg.py`'s own 37 tests run against the new
  package via `pytest`** (not just the isolated-import harness used for
  earlier tasks — a real `pytest` was installed in the sandbox for this
  final task specifically to run the project's actual test file directly):
  **36 pass**. The one failure
  (`test_joint_indices_match_actual_roller_model`) needs the real `mujoco`
  library's `MjSpec.from_file()` to compile the actual robot XML — the
  sandbox's lightweight `mujoco` stub can't provide that, and this test is
  verifying `robot/microduck_constants.py` (an unmodified file) against the
  real model geometry, not anything in this package. Re-run it for real
  once you have actual `mujoco` installed.
- `test_task_is_registered` needed one small stub addition (`mjlab.tasks.
  registry.list_tasks()`) to run in the sandbox — added, then passed,
  confirming the full `mjlab_microduck.tasks` package (all 11 tasks)
  registers the `Mjlab-RollerStandUp-Flat-MicroDuck` task ID correctly.
- **Full `mjlab_microduck.tasks` package import now succeeds completely,
  end to end, for the whole project** — every one of the 11 originally
  monolithic task files is now migrated, and the sandbox's stub-`mjlab`
  harness imports the entire package with zero errors.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run `uv run pytest tests/test_roller_standup_cfg.py` for
  real (all 37, including the joint-index lock), plus a short
  `uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production.

## This was the last task — a closing note on the whole restructure

Across all 11 tasks (`velocity`, `ball_kick`, `velocity_rollers`,
`ground_pick`, `roller_crouch`, `roulade`, `sitstand`, `standup`, `spin`,
`velstand`, `roller_slope`, `roller_standup`) plus the `mdp.py` module
split, the same handful of verification techniques caught every real bug
found along the way, none of them by inspection alone:

1. **Isolated import against a lightweight stub `mjlab`** — catches typos,
   wrong-module imports (e.g. `microduck_mdp.action_rate_l2` vs. the
   correct `mjlab.tasks.velocity.mdp.action_rate_l2` in `roller_slope`),
   and missing re-exports (the `MICRODUCK_ROUGH_TERRAINS_CFG` gap in
   `tasks/velocity/__init__.py` early on) immediately, before they'd
   surface as a runtime error days into a training run.
2. **Term-count cross-checks against the original file**, derived by hand
   from the `del`/`.pop()`/keep-set logic — catches silent miscounts from
   a missed drop or an accidentally-duplicated add.
3. **Running the project's own pre-existing test files** where they
   existed (`roller_slope`, `roller_standup`, and others) — the strongest
   signal available, since it checks the codebase's own encoded behavioral
   contract rather than a contract re-derived from reading the old file.
4. **Explicit isolation/aliasing checks** (e.g. `MICRODUCK_ROUGH_TERRAINS_CFG`
   mutation not leaking across env-cfg instances) — caught and fixed two
   real latent bugs in the original code as a byproduct of the
   configclass's per-instance deep-copy semantics, documented in the
   `velocity` and `ground_pick` MIGRATION.md files.

Every task's own MIGRATION.md documents its specific findings; this file
closes the set.
