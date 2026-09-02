# Manager-based env-cfg restructure — `velocity_rollers` — status & notes

## What changed

`tasks/microduck_velocity_rollers_env_cfg.py` (~670 lines, one function that
built a base cfg via mjlab's own `make_velocity_env_cfg()` and then
imperatively poked `cfg.rewards["name"] = ...`) has been replaced by
`tasks/velocity_rollers/`, the same IsaacLab-style, one-file-per-manager
package shape used for `tasks/velocity/` and `tasks/ball_kick/`. Backend is
still 100% mjlab — nothing about the runtime changed, only how the config is
*authored*.

```
tasks/velocity_rollers/
  microduck_flags.py                     # ENABLE_*/ranges (symmetry OFF, DR toggles)
  microduck_scene_cfg.py                 # robot entity, feet/self-collision sensors
                                          # (subtree-based, matches the merged
                                          # ankle/blade geometry of the 2026-07 model)
  microduck_observations_cfg.py          # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py              # MicroduckCommandsCfg(CommandsCfg) --
                                          # RelativeHeadingVelocityCommandCfg
  microduck_rewards_cfg.py               # MicroduckRewardsCfg(RewardsCfg) --
                                          # skating stride/glide/anti-swizzle recipe
  microduck_events_cfg.py                # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py          # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py            # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_velocity_rollers_env_cfg.py  # MicroduckVelocityRollersEnvCfg(LocomotionVelocityFlatEnvCfg)
                                          # + make_microduck_velocity_rollers_env_cfg()
  microduck_rl_cfg.py                    # MicroduckRollersRlCfg (RSL-RL PPO)
  __init__.py
```

No new `cfg/` folder here either — like `ball_kick`, this task reuses the
generic base component files under `tasks/locomotion/velocity/cfg/` (it's built on
mjlab's `make_velocity_env_cfg()` too, just like the original file was), by
subclassing `tasks.locomotion.velocity.LocomotionVelocityFlatEnvCfg` directly.

**No structural play variant.** Unlike `velocity`, this task's old function
took a `play: bool = False` parameter that was **never referenced anywhere
in the function body** — a dead parameter kept only so the registration
call `make_microduck_velocity_rollers_env_cfg(play=True)` doesn't error.
`make_microduck_velocity_rollers_env_cfg()` still accepts (and ignores)
`play` for the same reason; there's no `_PLAY` subclass because there's
nothing play-specific to configure.

**Public API is unchanged**: `make_microduck_velocity_rollers_env_cfg(play=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task IDs (`Mjlab-Velocity-Flat-MicroDuck-Rollers`,
`Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers`). `tasks/__init__.py`
needed exactly one import-path change
(`.microduck_velocity_rollers_env_cfg` → `.velocity_rollers`).

## Downstream effect: this unblocked 3 dependent tasks

`microduck_roller_slope_env_cfg.py`, `microduck_roller_standup_env_cfg.py`,
and `microduck_velocity_swizzle_env_cfg.py` all build on this task by
calling `make_microduck_velocity_rollers_env_cfg(play=play)` and then
further patching the returned (dict-based) `ManagerBasedRlEnvCfg` — exactly
the "several tasks build on top of `make_microduck_velocity_env_cfg()`"
pattern the `velocity` MIGRATION.md flagged for the walk family, just for
the roller family instead.

Those three files are **not yet restructured** — only their one import line
changed (`tasks.microduck_velocity_rollers_env_cfg` →
`tasks.velocity_rollers`), same for the two tests that imported the old
path directly (`tests/test_roller_standup_cfg.py`,
`tests/test_swizzle_head_cfg.py`). They keep working exactly as before,
since `make_microduck_velocity_rollers_env_cfg()` still returns the same
real `ManagerBasedRlEnvCfg` object their imperative patching code expects.
Restructuring them properly (as `MicroduckRollerSlopeEnvCfg
(MicroduckVelocityRollersEnvCfg)` etc., reusing
`tasks/velocity_rollers/cfg`-equivalent overrides) is the natural next step
now that they're unblocked — see the recipe reminder below.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.velocity_rollers` against the
  same stub `mjlab` used for `velocity`/`ball_kick` — imports cleanly,
  `make_microduck_velocity_rollers_env_cfg(play)` succeeds for both `play`
  values.
- Term counts per manager, cross-checked by hand against the original
  670-line file, match exactly: rewards=21, obs(actor)=8, obs(critic)=13,
  events=13, curriculum=4, commands=1, terminations=4.
- **Downstream check specific to this task**: with the stub environment,
  `microduck_roller_standup_env_cfg.py` and `microduck_velocity_swizzle_env_cfg.py`
  both still import and build a real env cfg on top of the new package
  (rewards=19 and rewards=18 respectively — the expected shrink from their
  own drop/add patching on top of rollers' 21). `microduck_roller_slope_env_cfg.py`
  couldn't be independently verified in the sandbox (its own,
  unrelated `slope_terrain.py` dependency needed more stub surface than was
  worth building for this check) but received the identical one-line fix as
  the other two.
- Full `mjlab_microduck.tasks` package import (the same thing
  `tasks/__init__.py`'s registration does) now gets further than before
  this change and reaches the next not-yet-migrated file in registration
  order (`microduck_standup_env_cfg.py`, which still calls mjlab's own
  `make_velocity_env_cfg()` directly) — confirms nothing downstream of
  `velocity_rollers`'s registration broke, and that the "wall" moved
  forward exactly as expected.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step (needs the real heavy
  deps — torch, mujoco-warp, warp-lang — install them and run
  `uv run train Mjlab-Velocity-Flat-MicroDuck-Rollers --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production).

## Recipe reminder for the remaining environments

Same recipe as `velocity`'s and `ball_kick`'s MIGRATION.md, refined by one
more data point:

1. If task A's old function *calls* task B's old `make_microduck_..._cfg()`
   function and patches the result further (rather than building from
   scratch via mjlab's `make_velocity_env_cfg()`), restructure B first — A
   becomes a much smaller, cleaner subclass of B's new
   `Microduck...EnvCfg` once B exists.
2. A `play`/`rough`-shaped parameter that's **accepted but never used** in
   the function body is common in this codebase (kept only for calling-
   convention parity with `register_mjlab_task`'s `play_env_cfg=...(play=True)`
   pattern) — grep the parameter's name in the function body before
   deciding whether it needs a `_PLAY` subclass or just a harmlessly-ignored
   dataclass field.
3. `RelativeHeadingVelocityCommandCfg`/other custom command subclasses
   already have their `.Ranges`/`.VizCfg` nested classes via inheriting
   from `UniformVelocityCommandCfg` — no need to redeclare them, just
   reference `SomeCommandCfg.Ranges(...)` the same way the base
   `tasks/locomotion/velocity/cfg/commands_cfg.py` does.
