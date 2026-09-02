# Manager-based env-cfg restructure — `roulade` — status & notes

## What changed

`tasks/microduck_roulade_env_cfg.py` (~770 lines, one function that built a
base cfg via mjlab's own `make_velocity_env_cfg()` and then imperatively
poked `cfg.rewards["name"] = ...`) has been replaced by `tasks/roulade/`,
the same IsaacLab-style, one-file-per-manager package shape used for the
other tasks. Backend is still 100% mjlab — nothing about the runtime
changed, only how the config is *authored*.

```
tasks/roulade/
  microduck_flags.py                # ENABLE_*/ranges + task constants (STAND_Z,
                                     # MIDROLL_*, TUCK_OVERRIDES, LANDING_GATE_*, RISE_GATE_*)
  microduck_scene_cfg.py            # robot + 4 sensors (2 with load-bearing names)
  microduck_observations_cfg.py     # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py         # MicroduckCommandsCfg(CommandsCfg) -- near-zero twist
  microduck_rewards_cfg.py          # MicroduckRewardsCfg(RewardsCfg) -- the roulade
                                     # progress/landing/regularisation recipe
  microduck_events_cfg.py           # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py     # MicroduckTerminationsCfg(TerminationsCfg) -- fell_over dropped
  microduck_curriculum_cfg.py       # MicroduckCurriculumCfg(CurriculumCfg) -- reverse-curriculum spawn mix
  microduck_roulade_env_cfg.py      # MicroduckRouladeEnvCfg(LocomotionVelocityFlatEnvCfg)
                                     # + make_microduck_roulade_env_cfg()
  microduck_rl_cfg.py               # MicroduckRouladeRlCfg (RSL-RL PPO)
  __init__.py
```

## Task-specific things worth flagging clearly (not "fixed", just preserved and documented)

- **`ENABLE_SYMMETRY = True`** — the only microduck task with symmetry on.
  Every other v1.5+ task has it off (`SYMMETRY_CFG`'s obs permutation was
  hardcoded for the old 51D layout and breaks the new 61D one); roulade was
  the first task migrated to the rewritten `SYMMETRY_CFG` (2026-08-13,
  includes the "policy" -> "actor" output-key fix). Preserved as-is —
  `microduck_flags.py` calls this out explicitly so nobody "fixes" it to
  match the other tasks' `False`.
- **`fell_over` is dropped from terminations entirely** (`fell_over: DoneTerm
  | None = None` in `microduck_terminations_cfg.py`) — falling over IS the
  task here, unlike every other task, which keeps it inherited unchanged
  from the base `TerminationsCfg`.
- **`push_robot` is dropped entirely, unconditionally** (not just "disabled
  by a flag that happens to be False" like most other DR toggles) — "a push
  mid-roll is incoherent" per the original comment. Declared as
  `push_robot: EventTerm | None = None` directly, no `if
  ENABLE_VELOCITY_PUSHES` branch at all, matching the original's
  unconditional `del cfg.events["push_robot"]`.
- **`reset_base`/`reset_robot_joints` are inherited completely unchanged**
  from the base `EventsCfg` — the original file never touches their
  `pose_range`/`z` at all (unlike `velocity_rollers`/`roller_crouch`, which
  both override `z` for their stand height). The actual starting pose here
  comes entirely from `set_roulade_state` (standing OR mid-roll spawn,
  chosen per-env) instead.
- **Field-order dependency**: `set_roulade_state` must run after
  `reset_robot_joints` (a mid-roll spawn's tuck lerps FROM the HOME pose
  `reset_robot_joints` writes) — automatically satisfied here since
  `reset_robot_joints` is an *inherited* field (present in the base class,
  hence early in declaration order) and `set_roulade_state` is a *new*
  field declared in the subclass (hence appended after every inherited
  field). See `ball_kick`'s `microduck_events_cfg.py` docstring for the
  general mechanism.

**No structural play variant** — like `velocity_rollers`/`roller_crouch`,
`play` is accepted by the old function but never referenced in its body;
the new `make_microduck_roulade_env_cfg(play=)` keeps accepting (and
ignoring) it for calling-convention parity.

**Public API is unchanged**: same signature, same return type, same task
IDs. `tasks/__init__.py` needed only the one import-path change; no test
file imports this task's factory function directly.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.roulade` against the same stub
  `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_roulade_env_cfg(play)` succeeds for both `play` values.
- Term counts per manager, cross-checked by hand against the original
  ~770-line file, match exactly: rewards=20, obs(actor)=8, obs(critic)=12,
  events=13, curriculum=7, commands=1, terminations=3 (confirmed
  `fell_over` is genuinely absent, not just miscounted).
- `episode_length_s=5.0` override confirmed present (vs. the base 20.0).
- Full `mjlab_microduck.tasks` package import reaches the same
  pre-existing, unrelated wall as before this change
  (`microduck_standup_env_cfg.py`, the sandbox's deliberate canary for
  not-yet-migrated files) — confirms nothing downstream of `roulade` broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run a short
  `uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production. (No
  `tests/test_roulade_*.py` file exists in this repo to run alongside it —
  worth adding one, following the pattern of `test_ground_pick_cfg.py`
  etc., if this task gets iterated on further.)

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **Don't assume every task keeps `fell_over`/`push_robot`/`reset_base`
   overrides "because every other task has them"** — grep each specific
   `del cfg.X[...]`/`.pop(...)`/untouched-field pattern per task; roulade
   is a good example of a task that diverges from the "usual" pattern in
   several places at once (terminations, events, even which inherited
   fields get left completely untouched).
2. When a task's design deliberately breaks with the codebase's general
   convention (symmetry on, a termination dropped, a whole DR category
   turned off unconditionally rather than behind a flag) — it's worth a
   one-line callout in the relevant new file's docstring/comment, not just
   silent preservation, so a future reader doesn't mistake the divergence
   for a copy-paste bug.
