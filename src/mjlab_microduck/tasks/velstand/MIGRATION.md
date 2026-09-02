# Manager-based env-cfg restructure — `velstand` — status & notes

## What changed

`tasks/microduck_velstand_env_cfg.py` (~415 lines, the most iterated-on
task in the repo — run-1 through run-7 in its own docstring) has been
replaced by `tasks/velstand/`, the same IsaacLab-style,
one-file-per-manager package shape used for the other tasks. Backend is
still 100% mjlab — nothing about the runtime changed, only how the config
is *authored*.

```
tasks/velstand/
  microduck_flags.py                  # ENABLE_*/ranges + task constants
                                       # (REWARD_GATE_TILT_DEG, RECOVERED_UP_*,
                                       # RECOVERY_ECON_KICKIN_ITER, PRONE_RAMP_STAGES)
  microduck_rewards_cfg.py            # MicroduckRewardsCfg(velocity.MicroduckRewardsCfg)
  microduck_events_cfg.py             # MicroduckEventsCfg(velocity.MicroduckEventsCfg)
  microduck_terminations_cfg.py       # MicroduckTerminationsCfg(velocity.MicroduckTerminationsCfg)
  microduck_curriculum_cfg.py         # MicroduckCurriculumCfg(velocity.MicroduckCurriculumCfg)
  microduck_velstand_env_cfg.py       # MicroduckVelstandFlatEnvCfg(velocity.MicroduckVelocityFlatEnvCfg) /
                                       # RoughEnvCfg(velocity.MicroduckVelocityRoughEnvCfg) / *_PLAY
                                       # + make_microduck_velstand_env_cfg()
  microduck_rl_cfg.py                 # MicroduckVelStandRlCfg (RSL-RL PPO)
  __init__.py
```

**No `microduck_scene_cfg.py`/`microduck_observations_cfg.py`/
`microduck_commands_cfg.py` here** — this is the first task where those
three are inherited completely unchanged from a sibling task rather than
overridden at all. `velstand` doesn't touch scene sensors, observations, or
commands beyond swapping the robot entity (done directly in
`microduck_velstand_env_cfg.py`'s `__post_init__`) — everything else,
*including* velocity's own `head_pose`/`body_pose` commands, flows through
unmodified.

## The one genuinely different thing about this task: it subclasses a sibling task's ENV CFG classes, not just its manager configs

Every other task that "builds on" a sibling (`ball_kick`, `ground_pick`, ...)
subclasses `LocomotionVelocityFlatEnvCfg` directly and reuses the sibling's
manager-*config* classes (`tasks.locomotion.velocity.cfg.*`). `velstand` is different:
the original file's `cfg = make_microduck_velocity_env_cfg(play=play,
rough=rough)` call means it's built on velocity's *fully assembled,
robot-specific* config -- so `MicroduckVelstandFlatEnvCfg` subclasses
`tasks.velocity.MicroduckVelocityFlatEnvCfg` itself (not just the shared
`LocomotionVelocityFlatEnvCfg` base), and similarly `RoughEnvCfg`
subclasses velocity's `MicroduckVelocityRoughEnvCfg`. This is exactly the
"straightforward subclass" case the `velocity` MIGRATION.md's recipe
anticipated for this specific task.

**This has one consequence worth flagging**: the `_PLAY` variants do
**not** subclass velocity's own `_PLAY` classes. Doing so would have
re-inherited velocity's `rewards`/`events`/`terminations`/`curriculum`
field *defaults* (velocity's own manager configs, not velstand's) via the
dataclass field-declaration rules, silently discarding all of velstand's
overrides in the play variant. Instead, `MicroduckVelstandFlatEnvCfg_PLAY`
subclasses velstand's own (non-play) `MicroduckVelstandFlatEnvCfg`, and
duplicates the ~4 lines of play-specific logic (push interval, terrain
generator shrink) that velocity's own `_PLAY` classes have — small enough
duplication to be clearly worth the correctness guarantee.

## Behavioral detail preserved carefully: `fell_over` handling differs by mode

- **Non-play**: `fell_over` termination stays active but its `limit_angle`
  is ramped from 70deg to pi (effectively disabled) by the
  `fell_over_disable` curriculum term at iter 500 — Phase 1 -> Phase 2 in
  the module docstring.
- **Play**: the curriculum doesn't run in play mode, so that ramp would
  never fire — the original code's fix was `cfg.terminations.pop
  ("fell_over", None)` plus an `if not play:` guard around registering the
  curriculum term at all. Reproduced here as `self.terminations.fell_over
  = None` and `self.curriculum.fell_over_disable = None` in both `_PLAY`
  variants' `__post_init__`.

**Public API is unchanged**: `make_microduck_velstand_env_cfg(play=,
rough=)` still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`,
same task IDs. `tasks/__init__.py` needed only the one import-path change;
`tests/test_head_pose_bias.py` needed the same one-line fix.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.velstand` against the same
  stub `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_velstand_env_cfg(play, rough)` succeeds for all 4
  combinations.
- Term counts per manager, cross-checked against velocity's own (already
  independently verified) counts plus this task's additions, match exactly
  in all 4 combinations: rewards=22 (velocity's 16 + 6 new recovery
  terms), events=14 (velocity's 13 + `random_prone_init`), curriculum=12
  (flat) / 13 (rough) (velocity's 7/8 + 5 new recovery-economics terms),
  commands=3 (inherited unchanged from velocity — includes `head_pose`/
  `body_pose`), terminations=5 (non-play) / 4 (play, `fell_over` correctly
  dropped).
- `fell_over`/`fell_over_disable` confirmed present in non-play, confirmed
  genuinely absent in play (checked explicitly, both termination and
  curriculum).
- Terrain-generator aliasing isolation check (same as `velocity`/
  `ground_pick`/`sitstand`/`standup`) — confirmed no leak into the shared
  `MICRODUCK_ROUGH_TERRAINS_CFG` module constant.
- Full `mjlab_microduck.tasks` package import reaches the same
  pre-existing, unrelated wall as it did after `spin`
  (`microduck_roller_slope_env_cfg.py`, on its own unrelated sandbox-stub
  gaps) — confirms nothing downstream of `velstand` broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run
  `uv run pytest tests/test_head_pose_bias.py` plus a short
  `uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production. (Given
  what came up on `standup`'s Rough variant — a real OOM on a 16-env,
  8GB-VRAM machine for a full-collision robot on procedural terrain, not a
  refactor bug — the same caution likely applies to
  `Mjlab-VelStand-Rough-MicroDuck`: try a small `--env.scene.num-envs`
  first if VRAM/RAM is tight.)

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **When a task's original factory function calls a sibling's *fully
   assembled* factory function** (`make_microduck_velocity_env_cfg(...)`,
   not `make_velocity_env_cfg()`), subclass the sibling's actual env-cfg
   classes (`MicroduckVelocityFlatEnvCfg`/`RoughEnvCfg`), not just the
   shared `LocomotionVelocityFlatEnvCfg` base — this pulls in every
   manager the task doesn't explicitly override, exactly matching the
   original's "verbatim" inheritance.
2. **Never subclass a sibling's `_PLAY` variant class directly** if your
   task overrides any of that sibling's manager config fields — the field
   defaults would silently revert to the sibling's own values in the play
   branch. Subclass your own non-play class instead and duplicate the
   (usually small) play-specific `__post_init__` logic.
3. When a termination/curriculum term's presence depends on the **runtime
   `play` parameter** (not a module-level flag baked into the config
   class), declare the field normally (present by default) and null it out
   in the `_PLAY` env-cfg subclasses' `__post_init__` — same pattern as
   `ground_pick`'s `terrain_levels` handling for `rough`, just keyed on
   `play` instead.
