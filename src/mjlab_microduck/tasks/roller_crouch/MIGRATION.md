# Manager-based env-cfg restructure — `roller_crouch` — status & notes

## What changed

`tasks/microduck_roller_crouch_env_cfg.py` (~480 lines, one function that
built a base cfg via mjlab's own `make_velocity_env_cfg()` and then
imperatively poked `cfg.rewards["name"] = ...`) has been replaced by
`tasks/roller_crouch/`, the same IsaacLab-style, one-file-per-manager
package shape used for the other tasks. Backend is still 100% mjlab —
nothing about the runtime changed, only how the config is *authored*.

```
tasks/roller_crouch/
  microduck_flags.py                    # ENABLE_*/ranges + task constants
                                         # (CROUCH_PERIOD, DESCENT_END, HOLD_END, RISE_END,
                                         # STAND_POSE, CROUCH_POSE, ENTRY_VELOCITY_X, ...)
  microduck_scene_cfg.py                # thin re-export of velocity_rollers's
                                         # robot/sensors (identical set)
  microduck_commands_cfg.py             # MicroduckCommandsCfg(CommandsCfg) --
                                         # GroundPickPhaseCommandCfg, randomize_phase=False
  microduck_rewards_cfg.py              # MicroduckRewardsCfg(RewardsCfg) --
                                         # phase-interpolated crouch/stand pose recipe
  microduck_events_cfg.py               # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py         # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py           # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_roller_crouch_env_cfg.py    # MicroduckRollerCrouchEnvCfg(LocomotionVelocityFlatEnvCfg)
                                         # + make_microduck_roller_crouch_env_cfg()
  microduck_rl_cfg.py                   # MicroduckRollerCrouchRlCfg (RSL-RL PPO)
  __init__.py
```

## A genuinely hybrid task, and one thing worth flagging clearly

The docstring describes this task as built from two others' machinery:
robot/physics from `velocity_rollers`, one-shot phase command from
`ground_pick`. **But the original file built its base cfg from mjlab's raw
`make_velocity_env_cfg()` directly** — not by calling
`make_microduck_velocity_rollers_env_cfg()` or
`make_microduck_ground_pick_env_cfg()`. It's "hybrid" in *design* (borrows
the same recipes), not in *code* (doesn't call either sibling task's
factory function). This matters for the restructure:

- **Observations**: the original's obs-editing block is line-for-line
  identical in *content* to `velocity_rollers`'s (same drops/adds, same
  `delay_max_lag=1` IMU window — importantly, *not* `ground_pick`'s wider
  `delay_max_lag=3`). Since the values are identical, `microduck_
  roller_crouch_env_cfg.py` reuses `tasks.velocity_rollers.microduck_
  observations_cfg.MicroduckObservationsCfg` directly rather than
  duplicating the file — legitimate DRY reuse of *matching values*, not a
  claim that the original code called velocity_rollers.
- **Commands**: this is the one place the "raw mjlab base, not
  velocity_rollers" distinction is load-bearing. The command's inherited
  defaults (`resampling_time_range`, `rel_forward_envs`,
  `heading_command=True`, `ranges=(±1, ±1, ±0.5, ±pi)`) are the **standard
  velocity command's**, not velocity_rollers's modified
  `RelativeHeadingVelocityCommandCfg` ones (`heading_command=False`,
  `ranges=(-0.5,0.6; 0,0; 0,0; None)`). Only `rel_standing_envs`/
  `rel_heading_envs` are overridden, then the command TYPE is swapped
  entirely to `GroundPickPhaseCommandCfg`. Building this file from a copy of
  `roller_crouch`'s sibling `commands_cfg.py` would have silently pulled in
  the wrong base ranges — caught by re-deriving from `tasks/locomotion/velocity/cfg/
  commands_cfg.py` (the actual original base) instead, and called out in
  this file's own docstring so a future editor doesn't "simplify" it into
  inheriting from velocity_rollers's version.

**No structural play variant** — like `velocity_rollers`, `play` is
accepted by the old function but never referenced in its body; the new
`make_microduck_roller_crouch_env_cfg(play=)` keeps accepting (and
ignoring) it for calling-convention parity.

**Public API is unchanged**: same signature, same return type, same task
IDs (`Mjlab-RollerCrouch-Flat-MicroDuck`,
`Mjlab-RollerCrouch-Flat-Backlash-MicroDuck`). `tasks/__init__.py` needed no
changes at all (only the function-call sites, which already used the
correct new import in prior refactors' cleanup); `tests/test_roller_crouch_
cfg.py` and `tests/test_spin_cfg.py` (which imports this task's factory
function for a cross-task comparison test) needed the one-line import-path fix.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.roller_crouch` against the same
  stub `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_roller_crouch_env_cfg(play)` succeeds for both `play`
  values.
- Term counts per manager, cross-checked by hand against the original
  ~480-line file, match exactly: rewards=12, events=12, obs(actor)=8,
  obs(critic)=13 (both obs counts identical to `velocity_rollers`, as
  expected from the direct reuse), curriculum=3, commands=1,
  terminations=4.
- Full `mjlab_microduck.tasks` package import reaches the same pre-existing,
  unrelated wall as before this change (`microduck_standup_env_cfg.py`,
  the sandbox's deliberate "canary" for not-yet-migrated files) — confirms
  nothing downstream of `roller_crouch` broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/mujoco-warp
  and run `uv run pytest tests/test_roller_crouch_cfg.py tests/test_spin_cfg.py`
  plus a short
  `uv run train Mjlab-RollerCrouch-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production.

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
point:

1. **A task's docstring/name suggesting it "derives from" another task
   doesn't necessarily mean the code calls that task's factory function.**
   Always check the actual `cfg = make_...()` call at the top of the
   function body (`mjlab.tasks.velocity.velocity_env_cfg.make_velocity_env_cfg()`
   = raw mjlab base, vs. `mjlab_microduck.tasks.X.make_microduck_X_env_cfg()`
   = a sibling task's customized base) before assuming which set of
   defaults a re-declared manager section is actually overriding relative
   to.
2. When two tasks' edited sections are content-identical (not just
   similarly-shaped) — like this task's observations vs. `velocity_rollers`'s
   — importing the sibling's already-built config class directly is
   cleaner than duplicating the file, *as long as* the values are actually
   proven identical (diff them line-by-line, don't assume from the
   docstring).
