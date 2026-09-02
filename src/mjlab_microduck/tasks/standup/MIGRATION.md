# Manager-based env-cfg restructure — `standup` — status & notes

## What changed

`tasks/microduck_standup_env_cfg.py` (~1160 lines, the largest of the 11
task files — one function that built a base cfg via mjlab's own
`make_velocity_env_cfg()` and then imperatively poked
`cfg.rewards["name"] = ...`) has been replaced by `tasks/standup/`, the same
IsaacLab-style, one-file-per-manager package shape used for the other
tasks. Backend is still 100% mjlab — nothing about the runtime changed,
only how the config is *authored*.

```
tasks/standup/
  microduck_flags.py                # ENABLE_*/ranges + task constants (SIT_Z, STAND_Z,
                                     # SITTING_JOINT_OVERRIDES, BODY_CMD_*, ENABLE_BODY_CONTROL)
  microduck_scene_cfg.py            # robot + feet/self-collision sensors
  microduck_observations_cfg.py     # MicroduckObservationsCfg(ObservationsCfg) --
                                     # NaN-safe critic wrappers, real head/body command slots
  microduck_commands_cfg.py         # MicroduckCommandsCfg(CommandsCfg) -- THREE commands:
                                     # twist (near-zero) + head_pose + body_pose (conditional)
  microduck_rewards_cfg.py          # MicroduckRewardsCfg(RewardsCfg) -- the /4-rescaled
                                     # minimum-viable standup stack + body control
  microduck_events_cfg.py           # MicroduckEventsCfg(EventsCfg) -- 4-way ground-state mix
  microduck_terminations_cfg.py     # MicroduckTerminationsCfg(TerminationsCfg) -- fell_over dropped
  microduck_curriculum_cfg.py       # MicroduckCurriculumCfg(CurriculumCfg) -- 14/15 terms,
                                     # 5 of them body-control-only
  microduck_standup_env_cfg.py      # MicroduckStandupFlatEnvCfg / RoughEnvCfg / *_PLAY
                                     # + make_microduck_standup_env_cfg()
  microduck_rl_cfg.py               # MicroduckStandUpRlCfg (RSL-RL PPO)
  __init__.py
```

Full 4-class `Flat`/`Flat_PLAY`/`Rough`/`Rough_PLAY` hierarchy (like
`ground_pick`, `sitstand`, `velocity`).

## The biggest task so far — things worth flagging clearly

- **Three active commands.** `sitstand` was the first task with two; this
  is the first with three: `twist` (near-zero noise, obs-parity only),
  `head_pose` (real, like every posture-family task), and `body_pose` (real
  6D trunk-delta command, tracked on z/roll/pitch only). `body_pose` is
  gated by the `ENABLE_BODY_CONTROL` master toggle — when off, the field is
  `None` (dropped entirely), the `body_command` obs slot falls back to
  zero-padding, and the whole "body-control curricula" block (5 curriculum
  terms) doesn't apply.
- **`ENABLE_BODY_CONTROL` as a master toggle spanning 4 managers at once.**
  The original file's own comment calls this out with an explicit early
  `return cfg` for the curriculum section when the flag is off. There's no
  early-return equivalent in a declarative `@configclass` tree, so each of
  the 6 affected fields (`body_pose_tracking` reward, `body_pose` command,
  `body_command` obs term, and 5 curriculum terms) is independently
  conditioned on the same flag instead — same net effect, expressed
  declaratively rather than imperatively.
- **NaN-safe critic observation wrappers.** `foot_air_time`/
  `foot_contact_forces` on the critic use `_safe` variants
  (`foot_air_time_safe`/`foot_contact_forces_safe` from the mdp package)
  instead of the base `CriticCfg`'s plain versions — standup lands and
  flips constantly, so degenerate (non-finite) contact forces are far more
  likely here, and a single NaN slipping past `robot_state_is_nan` (which
  only checks joint + root state) previously crashed a real training run
  via rsl_rl's `check_nan`.
- **`nan_state` termination carries an explicit `params={"sensor_names":
  ("feet_ground_contact",)}`** — every other migrated task's `nan_state`
  uses the function's default sensor set. Preserved as written rather than
  "simplified" to match the others.
- **`reset_base`/`reset_robot_joints` are inherited completely unchanged**
  — same situation as `roulade`: the actual starting pose comes entirely
  from `set_ground_state` (one of 4 poses, ramped by the
  `ground_state_mix` curriculum), which runs after them in "reset" mode and
  overwrites the pose anyway.
- **Every task-reward weight is documented as "/4 of the pre-2026-07
  values"** in the original file, with an explicit note that internal
  ratios between task terms are unchanged (uniform rescale to match
  velocity's task-mass-to-regularizer ratio). Preserved as the top of
  `microduck_rewards_cfg.py`'s module docstring, along with the
  `gentle_rise` sign-convention warning (same bug class as `sitstand`'s
  `gentle_motion`/`descent_speed`/`rise_speed`).

**Public API is unchanged**: `make_microduck_standup_env_cfg(play=, rough=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task IDs. `tasks/__init__.py` needed only the one import-path change;
`tests/test_obs_nan_guard.py` and `tests/test_head_pose_bias.py` (both of
which import this task's factory function for cross-task checks) needed the
same one-line fix.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.standup` against the same stub
  `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_standup_env_cfg(play, rough)` succeeds for all 4 combinations.
- Term counts per manager, cross-checked by hand against the original
  ~1160-line file, match exactly in all 4 combinations: rewards=20,
  obs(actor)=8, obs(critic)=12, events=14, curriculum=14 (flat)/15 (rough),
  commands=3, terminations=3 (`fell_over` confirmed genuinely absent).
- `commands.keys() == ["twist", "head_pose", "body_pose"]` confirmed.
- `nan_state`'s `params` confirmed to carry `sensor_names=
  ("feet_ground_contact",)` exactly as in the original.
- Terrain-generator aliasing isolation check (same as `velocity`/
  `ground_pick`/`sitstand`) — confirmed no leak into the shared
  `MICRODUCK_ROUGH_TERRAINS_CFG` module constant.
- **Full `mjlab_microduck.tasks` package import now gets PAST `standup`
  entirely** and reaches the next not-yet-migrated file in registration
  order (`microduck_roller_slope_env_cfg.py`, which fails only on
  unrelated sandbox-stub gaps — a missing `TerrainGeneratorCfg.
  difficulty_range` field and a missing `mjlab.envs.mdp.bad_orientation`
  stub, neither related to `standup`) — this is a stronger confirmation
  than the previous tasks got, since `standup` was the last thing standing
  between the registration chain and the (deliberately un-implemented)
  `make_velocity_env_cfg()` canary; that canary no longer fires at all now.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run a short
  `uv run pytest tests/test_obs_nan_guard.py tests/test_head_pose_bias.py`
  plus
  `uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production.

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **A flag that gates an early `return cfg` in the original imperative
   function** doesn't have a direct declarative equivalent — enumerate
   every field the early return would have skipped (reward terms, command
   fields, obs terms, curriculum terms) and condition each one
   independently on the same flag. Cross-check the count: `standup`'s
   `ENABLE_BODY_CONTROL` touches exactly 1 reward + 1 command + 1 obs term
   (x2 groups) + 5 curriculum terms — verify that count against the
   original before considering the task done.
2. **Multiple active commands compound multiplicatively with per-task
   quirks** — `sitstand` had 2 (one repurposed, one real), `standup` has 3
   (one repurposed, two real, one conditional). Always check every
   `cfg.commands["X"] = ...` line, not just `"twist"`, and check which of
   them are genuinely new fields (not on the base `CommandsCfg`) each time.
3. When the original file has a "canary" pattern like the sandbox's
   `make_velocity_env_cfg()` stub (something that only stops erroring once
   *every* task calling it directly has been migrated), use it as a
   coarse-grained progress signal — but don't rely on it alone; the
   per-manager term-count cross-check is what actually proves a given
   task's migration is correct.
