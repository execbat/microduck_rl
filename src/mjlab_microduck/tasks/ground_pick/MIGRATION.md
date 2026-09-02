# Manager-based env-cfg restructure — `ground_pick` — status & notes

## What changed

`tasks/microduck_ground_pick_env_cfg.py` (~700 lines, one function that
built a base cfg via mjlab's own `make_velocity_env_cfg()` and then
imperatively poked `cfg.rewards["name"] = ...`) has been replaced by
`tasks/ground_pick/`, the same IsaacLab-style, one-file-per-manager package
shape used for `tasks/velocity/` and friends. Backend is still 100% mjlab —
nothing about the runtime changed, only how the config is *authored*.

```
tasks/ground_pick/
  microduck_flags.py                 # ENABLE_*/ranges + task constants
                                      # (GP_PERIOD, DESCENT_END, HOLD_END, RISE_END,
                                      # LEG_JOINTS, NECK_JOINTS)
  microduck_scene_cfg.py             # robot, feet/self-collision/head-impact sensors
  microduck_observations_cfg.py      # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py          # MicroduckCommandsCfg(CommandsCfg) --
                                      # GroundPickPhaseCommandCfg (cyclic phase encoding)
  microduck_rewards_cfg.py           # MicroduckRewardsCfg(RewardsCfg) --
                                      # approach/return/regularisation recipe
  microduck_events_cfg.py            # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py      # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py        # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_ground_pick_env_cfg.py   # MicroduckGroundPickFlatEnvCfg / RoughEnvCfg /
                                      # *_PLAY, mirroring the velocity task's
                                      # Flat/Rough/_PLAY split exactly, +
                                      # make_microduck_ground_pick_env_cfg()
  microduck_rl_cfg.py                # MicroduckGroundPickRlCfg (RSL-RL PPO)
  __init__.py
```

Like `velocity`, this task has real Flat/Rough terrain variants *and* a
`play` flag that actually changes behavior (push interval) — so it gets the
full 4-class `Flat`/`Flat_PLAY`/`Rough`/`Rough_PLAY` treatment, unlike
`ball_kick`/`velocity_rollers` (flat-only) or `velocity_rollers`'s dead
`play` parameter.

**One quirk of the original file, preserved exactly rather than "fixed":**
the scene sensors are replaced wholesale
(`feet_ground_contact`/`self_collision`/`head_impact_contact`, dropping the
inherited terrain-scan sensors) **unconditionally** — even in the Rough
variant. Same for the `height_scan`/`foot_height` observation terms: they're
removed regardless of terrain type. So this task has no terrain-height
awareness even when walking on procedural rough terrain (reasonable for a
quasi-static reach-down motion, arguably not for anything more dynamic —
not this refactor's call to make). `MicroduckGroundPickRoughEnvCfg`'s
docstring calls this out explicitly so it doesn't look like an oversight
next time someone reads the code.

**Public API is unchanged**: `make_microduck_ground_pick_env_cfg(play=,
rough=)` still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`,
same task IDs (`Mjlab-GroundPick-Rough-MicroDuck`,
`Mjlab-GroundPick-Flat-Backlash-MicroDuck`,
`Mjlab-GroundPick-Rough-Backlash-MicroDuck`, ...). `tasks/__init__.py`
needed exactly one import-path change; `tests/test_ground_pick_cfg.py`
needed the same one-line fix.

## Behavioral fix that fell out of the restructure (not intentional scope)

Same class of fix as `velocity`'s and `standup`-family's rough-terrain
handling: the original code assigned
`cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG`
directly (the bare module-level constant, not a copy). Any subsequent
mutation of `.terrain_generator` (e.g. the play-mode num_rows/num_cols
shrink) would alias back into the shared constant and leak into every other
env cfg built from it. `MicroduckGroundPickRoughEnvCfg.__post_init__` now
wraps it in `dataclasses.replace(...)`, exactly like `velocity`'s and
mjlab's own `velocity_env_cfg.py`. Verified directly (see below) — mutating
one instance's `terrain_generator.num_rows` does not affect the shared
`MICRODUCK_ROUGH_TERRAINS_CFG` module constant.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.ground_pick` against the same
  stub `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_ground_pick_env_cfg(play, rough)` succeeds for all 4
  combinations.
- Term counts per manager, cross-checked by hand against the original
  ~700-line file, match exactly in all 4 combinations: rewards=19,
  obs(actor)=8, obs(critic)=12, events=14, curriculum=3 (flat)/4 (rough),
  commands=1, terminations=4.
- Terrain-generator aliasing isolation check (see above) — confirmed no
  leak into the shared `MICRODUCK_ROUGH_TERRAINS_CFG` module constant.
- Full `mjlab_microduck.tasks` package import (the same thing
  `tasks/__init__.py`'s registration does) reaches the same pre-existing,
  unrelated wall as before this change (`microduck_standup_env_cfg.py`,
  still calling mjlab's own `make_velocity_env_cfg()` directly — a
  deliberate "canary" left in the sandbox's stub `mjlab` that raises if a
  *migrated* file calls it, confirming nothing downstream of `ground_pick`
  broke, and that `standup` is next in line).
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step (needs the real heavy
  deps — torch, mujoco-warp, warp-lang — install them and run
  `uv run pytest tests/test_ground_pick_cfg.py` plus a short
  `uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production).

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. If a task has genuine Flat/Rough terrain variants *and* a `play` flag
   that changes real behavior (not just accepted-and-ignored like
   `velocity_rollers`'s), use the full 4-class
   `Flat`/`Flat_PLAY`/`Rough`/`Rough_PLAY` hierarchy exactly like `velocity`
   — don't try to collapse it into dataclass fields the way `ball_kick` did
   for its single-variant `kick_foot`/`play`.
2. Watch for **unconditional** overrides that look like they should be
   terrain-dependent but aren't in the original file (sensors, obs terms
   deleted regardless of `rough`) — preserve them as unconditional, and
   leave a comment explaining why, rather than "fixing" them to be
   conditional (that would be a behavior change, not a restructure).
3. When a curriculum term needs to be "kept on the Rough variant, dropped
   on Flat" (the opposite of most terms, which are usually Rough-only
   additions), it's cleanest to declare it in the *base* `CurriculumCfg`
   unchanged, delete it in `Flat.__post_init__`, and reconstruct it in
   `Rough.__post_init__` (which calls `super().__post_init__()` first) —
   mirrors exactly how `velocity`'s own `terrain_levels` curriculum term is
   handled.
