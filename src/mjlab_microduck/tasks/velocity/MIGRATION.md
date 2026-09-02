# Velocity configuration architecture

The generic locomotion package lives at `tasks/locomotion/velocity/`.
`tasks/velocity/` contains only the Microduck walking recipe.

## Shared bases

- `LocomotionVelocityEnvCfg`: common manager groups, simulation/viewer defaults,
  and `to_mjlab_cfg()`. Terrain is unset. The foot-height sensor is a shared
  placeholder, with frames supplied by the robot recipe.
- `LocomotionVelocityFlatEnvCfg`: a plane; no body terrain scanner, height-scan
  observation, or terrain-level curriculum.
- `LocomotionVelocityRoughEnvCfg`: generated rough terrain, body terrain scanner,
  `RoughObservationsCfg`, and `RoughCurriculumCfg`.

Both terrain variants inherit directly from the neutral base. All generic
manager components are exported from `tasks.locomotion.velocity.cfg`.
`PolicyCfg.height_scan` and `CurriculumCfg.terrain_levels` retain disabled
slots so subclass overrides preserve manager/observation order.

## Microduck recipes

`MicroduckVelocityFlatEnvCfg` inherits the shared flat base and supplies robot
assets, sensor wiring, and its own manager configs. `MicroduckVelocityRoughEnvCfg`
continues to extend that Microduck recipe with gentle terrain, solver settings,
and terrain curriculum. It intentionally does not enable a body terrain scanner:
the deployed Microduck policy retains its existing observation contract.

Independent tasks (standup, sitstand, ground_pick, ball_kick, velocity_rollers,
roller_crouch, spin, roulade) also inherit the generic flat base. Velstand,
velocity_swizzle, roller_slope and roller_standup retain their existing
robot-specific parent recipes. Testbench remains independent.

## Imports

```python
from mjlab_microduck.tasks.locomotion.velocity import (
    LocomotionVelocityEnvCfg,
    LocomotionVelocityFlatEnvCfg,
    LocomotionVelocityRoughEnvCfg,
)
from mjlab_microduck.tasks.locomotion.velocity.cfg import CommandsCfg, RewardsCfg
from mjlab_microduck.tasks.velocity import make_microduck_velocity_env_cfg
```

The old `tasks.velocity.cfg` and `tasks.velocity.locomotion_velocity_env_cfg`
modules were removed; update external consumers to these imports. The public
Microduck task packages, factory signatures, task IDs, and train/play/backlash
registrations are unchanged. No changes are needed to training commands.

For a new recipe, import the shared manager classes rather than copying their
files into a concrete task. Keep robot-specific overrides next to that task.
