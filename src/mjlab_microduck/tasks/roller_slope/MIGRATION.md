# Manager-based env-cfg restructure — `roller_slope` — status & notes

## What changed

`tasks/microduck_roller_slope_env_cfg.py` (~250 lines, one function that
called `make_microduck_velocity_rollers_env_cfg(play=play)` and patched the
result heavily) has been replaced by `tasks/roller_slope/`, the same
IsaacLab-style, one-file-per-manager package shape used for the other
tasks. Backend is still 100% mjlab — nothing about the runtime changed,
only how the config is *authored*.

```
tasks/roller_slope/
  microduck_flags.py                  # ENTRY_VELOCITY_X, TILE_SIZE, SPAWN_YAW,
                                       # VOID_FLOOR, PLAY_DIFFICULTY + resolve_play_difficulty()
  microduck_scene_cfg.py              # build_terrain(play) -- flat+ramp+runout
                                       # generator factory (play-mode-dependent)
  microduck_observations_cfg.py       # MicroduckObservationsCfg -- same terms as
                                       # velocity_rollers, nan_policy="sanitize"
  microduck_commands_cfg.py           # MicroduckCommandsCfg -- twist neutralised
  microduck_rewards_cfg.py            # MicroduckRewardsCfg(velocity_rollers.MicroduckRewardsCfg)
                                       # -- free-balance recipe
  microduck_events_cfg.py             # MicroduckEventsCfg -- fixed spawn yaw + rolling entry
  microduck_terminations_cfg.py       # MicroduckTerminationsCfg -- fall + void, no edge cutoff
  microduck_curriculum_cfg.py         # MicroduckCurriculumCfg -- single stiffness-ramp term
  microduck_roller_slope_env_cfg.py   # MicroduckRollerSlopeEnvCfg(velocity_rollers.MicroduckVelocityRollersEnvCfg)
                                       # + make_microduck_roller_slope_env_cfg()
  microduck_rl_cfg.py                 # MicroduckRollerSlopeRlCfg (RSL-RL PPO)
  __init__.py
```

Subclasses `tasks.velocity_rollers`'s actual env-cfg class
(`MicroduckVelocityRollersEnvCfg`) directly — same "straightforward
subclass" situation as `velstand` (the original called
`make_microduck_velocity_rollers_env_cfg(play=play)`, not the raw mjlab
base), just for the roller family instead of the walk family.

## Things worth flagging clearly

- **`play` became a real dataclass field** (`MicroduckRollerSlopeEnvCfg.play:
  bool = False`), not just a factory-function parameter, because the
  terrain itself is play-mode-dependent (`build_terrain(play)`, called from
  `__post_init__` using `self.play`) — this is a genuinely different
  situation from `ball_kick`'s `kick_foot`/`play` fields (there, `play` only
  tweaked one event's interval; here it changes which terrain generator
  config gets built, including reading the `SLOPE_PLAY_DIFFICULTY`
  environment variable at construction time). Kept as a plain field for the
  same reason `ball_kick` did: no structural class-variant, just a
  parameter that flows into `__post_init__`.
- **Reward names are REUSED with brand-new values**, not renamed. The
  original code deletes velocity_rollers's whole reward set except
  `action_rate_l2`, then re-adds `upright`/`feet_flat`/
  `neck_action_rate_l2`/`neck_joint_pos_l2`/`joint_torques_l2`/
  `heading_hold` — SAME dict keys as velocity_rollers had, but wired to
  different functions/params (e.g. `upright` here is
  `body_upright_gaussian`, not velocity_rollers's `mdp.upright`). The first
  draft of this file accidentally suffixed these with `_free` to avoid a
  "redeclaring the same field twice" confusion — caught before finalizing:
  that would have changed the actual term *names* that show up in
  wandb/logs, a real behavioral difference from the original. Fixed to
  redeclare each field ONCE, with its final value, under the ORIGINAL name.
- **A real bug caught by the stub-import test, not by inspection**: this
  file's `action_rate_l2` override initially used
  `func=microduck_mdp.action_rate_l2` (copy-paste habit from writing so many
  `microduck_mdp.*`-heavy files) — that function doesn't exist in the
  microduck mdp package at all; `action_rate_l2` lives in
  `mjlab.tasks.velocity.mdp` (imported as plain `mdp` everywhere else in
  this codebase) and always has. The isolated-import test crashed
  immediately with `AttributeError: module 'mjlab_microduck.tasks.mdp' has
  no attribute 'action_rate_l2'` — exactly the kind of typo this
  verification step exists to catch before it reaches a real training run.
- **A companion test file already existed**: `tests/test_roller_slope_cfg.py`
  (12 tests) asserts on the exact same behaviors this migration needed to
  preserve (terrain shape, neutralised command, reused reward names,
  dropped skating rewards, void/fall terminations, `nan_policy`, curriculum
  presence, tile geometry). Every one of those 12 tests was run against the
  new package (via the same stub-`mjlab` harness used elsewhere) and all
  passed — this is the strongest verification any task in this refactor has
  had, since it's checking the project's own pre-existing behavioral
  contract, not just term counts derived by hand from the old file.

**Public API is unchanged**: `make_microduck_roller_slope_env_cfg(play=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task ID. `tasks/__init__.py` needed only the one import-path change;
`tests/test_roller_slope_cfg.py` needed the same one-line fix.

## Verified

- `pyflakes` clean on every new file.
- **All 12 tests in `tests/test_roller_slope_cfg.py` pass** against the new
  package (run via the stub-`mjlab` harness).
- Term counts per manager, cross-checked by hand against the original
  ~250-line file, match exactly for both `play` values: rewards=9
  (1 kept + 8 reused/new names), events=14 (velocity_rollers's 13 +
  `reset_rolling_entry`), curriculum=1 (`terrain_levels` only — everything
  else wholesale-dropped), commands=1, terminations=4
  (velocity_rollers's 4, `out_of_terrain_bounds` dropped, `fell_into_void`
  added, `fell_over` reused with a different function).
- Full `mjlab_microduck.tasks` package import now **succeeds completely**
  end to end (no wall at all) — this confirms `roller_slope` didn't break
  anything downstream, and as a side effect confirms the one remaining
  not-yet-migrated task (`roller_standup`) already imports fine on top of
  the now-fully-migrated `velocity_rollers`.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run `uv run pytest tests/test_roller_slope_cfg.py` for
  real, plus a short
  `uv run train Mjlab-RollerSlope-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production.

## Recipe reminder for the remaining environment (`roller_standup`, the last one)

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **Check for an existing test file before writing anything** — if one
   exists (`tests/test_X_cfg.py`), it's a ready-made, higher-confidence
   verification target than hand-deriving term counts, and it may reveal
   exact behavioral contracts (like the reused-reward-names case here)
   that aren't obvious from reading the original file alone.
2. **When "deleting everything and re-adding some terms" reuses the ORIGINAL
   dict keys** for some of the re-added terms, redeclare that dataclass
   field exactly once with its final value under the original name —
   resist the urge to rename for clarity; the term's *name* is part of its
   observable behavior (wandb/log keys), not just an implementation detail.
3. **Double-check which `mdp` a reused/kept builtin term's function actually
   comes from** before writing `func=microduck_mdp.X` — velocity-family
   builtins like `action_rate_l2`, `upright`, `self_collision_cost` live in
   `mjlab.tasks.velocity.mdp` (the plain `mdp` import), not in
   `mjlab_microduck.tasks.mdp`. The isolated-import test catches this
   immediately, but it's faster to just check before writing the line.
