# Manager-based env-cfg restructure — `spin` — status & notes

## What changed

`tasks/microduck_spin_env_cfg.py` (~485 lines, one function that built a
base cfg via mjlab's own `make_velocity_env_cfg()` and then imperatively
poked `cfg.rewards["name"] = ...`) has been replaced by `tasks/spin/`, the
same IsaacLab-style, one-file-per-manager package shape used for the other
tasks. Backend is still 100% mjlab — nothing about the runtime changed,
only how the config is *authored*.

```
tasks/spin/
  microduck_flags.py                # ENABLE_*/ranges + task constants
                                     # (ENTRY_VELOCITY_X, NECK_PATTERN_NO_YAW)
  microduck_scene_cfg.py            # thin re-export of velocity_rollers's
                                     # robot/sensors (identical set)
  microduck_observations_cfg.py     # MicroduckObservationsCfg -- subclasses
                                     # velocity_rollers's, only the critic's
                                     # wheel_vel joint pattern differs
  microduck_commands_cfg.py         # MicroduckCommandsCfg(CommandsCfg) --
                                     # GroundPickPhaseCommandCfg driving a
                                     # target YAW RATE, not a pose
  microduck_rewards_cfg.py          # MicroduckRewardsCfg(RewardsCfg) --
                                     # spin-rate tracking + rolling/scissor bootstraps
  microduck_events_cfg.py           # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py     # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py       # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_spin_env_cfg.py         # MicroduckSpinEnvCfg(LocomotionVelocityRoughEnvCfg)
                                     # + make_microduck_spin_env_cfg()
  microduck_rl_cfg.py               # MicroduckSpinRlCfg (RSL-RL PPO)
  __init__.py
```

Same "hybrid in design, not in code" situation as `roller_crouch`: the
docstring calls this task a fusion of `velocity_rollers` (physics) and
`roller_crouch` (phase machinery), but the original file builds from
mjlab's raw `make_velocity_env_cfg()` directly, not by calling either
sibling task's factory function. Same command-defaults trap as
`roller_crouch` applies here too — see that task's MIGRATION.md for the
full explanation; `microduck_commands_cfg.py`'s docstring repeats the
short version.

## One genuine, deliberate difference from `roller_crouch`'s reused observations

`roller_crouch`'s observations turned out to be byte-identical to
`velocity_rollers`'s, so that task reuses the class directly with no
override. `spin`'s observations are *almost* identical — except the
critic's privileged `wheel_vel` term matches `r"^passive_.*"` (every
passive joint) instead of `r"^passive_.*wheel"` (wheel joints only, what
`velocity_rollers`/`roller_crouch` use). This was checked carefully (not
assumed) before writing `microduck_observations_cfg.py`: it's a real,
intentional divergence in the original file, not a typo, so `spin` gets a
small subclass overriding just that one field rather than reusing
`velocity_rollers`'s class wholesale.

## Reward set is genuinely different from every other roller-family task

- **`angular_momentum` is deliberately dropped** (kept in most other
  tasks): it penalises the 3D angular-momentum *norm*, which would
  directly fight the spin itself. `body_ang_vel` is kept instead — it only
  penalises x/y ("don't penalize z-angular velocity", per mjlab's own
  convention), so it tames roll/pitch wobble without opposing the rotation.
  This distinction is explicit in the original file's comment and preserved
  as a doc comment on `MicroduckRewardsCfg`.
- **`dof_pos_limits` is also dropped** here, unlike most other migrated
  tasks (which keep it inherited unchanged from the base `RewardsCfg`).
- **`fell_over` is kept inherited unchanged** (unlike `roulade`/`sitstand`/
  `standup`, which all drop it) — a spin gone wrong is still a fall, worth
  terminating on.

**No structural play variant** — like `velocity_rollers`/`roller_crouch`,
`play` is accepted by the old function but never referenced in its body;
the new `make_microduck_spin_env_cfg(play=)` keeps accepting (and ignoring)
it for calling-convention parity.

**Public API is unchanged**: same signature, same return type, same task
ID. `tasks/__init__.py` needed only the one import-path change;
`tests/test_spin_cfg.py` needed the same one-line fix.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.spin` against the same stub
  `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_spin_env_cfg(play)` succeeds for both `play` values.
- Term counts per manager, cross-checked by hand against the original
  ~485-line file, match exactly: rewards=14, obs(actor)=8, obs(critic)=13,
  events=12, curriculum=4, commands=1, terminations=4.
- `angular_momentum` confirmed genuinely absent from rewards;
  `fell_over` confirmed genuinely present in terminations (the opposite
  pattern from most other migrated tasks, both checked explicitly rather
  than assumed).
- Full `mjlab_microduck.tasks` package import reaches the same
  pre-existing, unrelated wall as it did after the `standup` migration
  (`microduck_roller_slope_env_cfg.py`, on its own unrelated sandbox-stub
  gaps) — confirms nothing downstream of `spin` broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run
  `uv run pytest tests/test_spin_cfg.py` plus a short
  `uv run train Mjlab-Spin-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production.

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **"Similar to a sibling task" doesn't mean "identical to a sibling
   task."** Diff the relevant section line-by-line before deciding to
   reuse a class wholesale (`roller_crouch`'s case) vs. writing a small
   override subclass (`spin`'s case) — a single differing regex pattern is
   easy to miss if you just skim for overall shape.
2. **A task can drop reward/termination terms that every other task in the
   same family keeps** (here: `angular_momentum` dropped while kept
   elsewhere, `fell_over` kept while dropped elsewhere) — don't pattern-
   match against sibling tasks' `RewardsCfg`/`TerminationsCfg` overrides;
   always re-derive from the specific task's own `del`/`.pop()`/keep-set
   logic.
