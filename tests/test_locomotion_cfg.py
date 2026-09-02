"""CPU config regressions for the shared locomotion package (no simulation)."""

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from mjlab_microduck.tasks.locomotion.velocity import (
    LocomotionVelocityEnvCfg,
    LocomotionVelocityFlatEnvCfg,
    LocomotionVelocityRoughEnvCfg,
)
from mjlab_microduck.tasks.velocity import make_microduck_velocity_env_cfg


@pytest.mark.parametrize("cfg_type", [LocomotionVelocityEnvCfg, LocomotionVelocityFlatEnvCfg])
def test_common_configs_do_not_require_body_terrain_scans(cfg_type):
    cfg = cfg_type().to_mjlab_cfg()
    assert "terrain_scan" not in {sensor.name for sensor in cfg.scene.sensors}
    assert "height_scan" not in cfg.observations["actor"].terms
    assert "height_scan" not in cfg.observations["critic"].terms
    assert "terrain_levels" not in cfg.curriculum
    if cfg_type is LocomotionVelocityEnvCfg:
        assert cfg.scene.terrain is None
    else:
        assert cfg.scene.terrain.terrain_type == "plane"


def test_rough_config_wires_terrain_observations_and_curriculum_together():
    cfg = LocomotionVelocityRoughEnvCfg().to_mjlab_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    assert cfg.scene.terrain.max_init_terrain_level == 5
    assert "terrain_scan" in {sensor.name for sensor in cfg.scene.sensors}
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["height_scan"]
        assert term.params["sensor_name"] == "terrain_scan"
        assert term.scale == 0.2
    assert cfg.observations["actor"].terms["height_scan"].noise is not None
    assert cfg.observations["critic"].terms["height_scan"].noise is None
    assert cfg.observations["actor"].enable_corruption is True
    assert cfg.observations["critic"].enable_corruption is False
    assert tuple(cfg.curriculum) == ("terrain_levels", "command_vel")
    assert tuple(cfg.observations["critic"].terms).index("height_scan") < tuple(
        cfg.observations["critic"].terms
    ).index("foot_height")


def test_terrain_and_manager_mutations_are_instance_local():
    first = LocomotionVelocityRoughEnvCfg()
    second = LocomotionVelocityRoughEnvCfg()
    rows = second.scene.terrain.terrain_generator.num_rows
    template_rows = ROUGH_TERRAINS_CFG.num_rows
    first.scene.terrain.terrain_generator.num_rows = rows + 1
    first.scene.sensors[0].frame.name = "changed_body"
    first.observations.actor.height_scan.params["sensor_name"] = "changed_scan"
    first.curriculum.terrain_levels.params["command_name"] = "changed_command"
    assert second.scene.terrain.terrain_generator.num_rows == rows
    assert ROUGH_TERRAINS_CFG.num_rows == template_rows
    assert second.scene.sensors[0].frame.name == ""
    assert second.observations.actor.height_scan.params["sensor_name"] == "terrain_scan"
    assert second.curriculum.terrain_levels.params["command_name"] == "twist"


def test_microduck_rough_play_does_not_change_future_training_configs():
    before = make_microduck_velocity_env_cfg(rough=True)
    play = make_microduck_velocity_env_cfg(play=True, rough=True)
    after = make_microduck_velocity_env_cfg(rough=True)
    assert play.scene.terrain.terrain_generator.num_rows == 5
    assert after.scene.terrain.terrain_generator.num_rows == before.scene.terrain.terrain_generator.num_rows
    assert after.scene.terrain.terrain_generator.curriculum == before.scene.terrain.terrain_generator.curriculum
    assert tuple(after.observations["actor"].terms) == tuple(before.observations["actor"].terms)


# Importing the shared package also exercises the existing tasks/__init__.py
# registrations, including each backlash wrapper and each train/play factory.
_MICRODUCK_TASKS = [name for name in list_tasks() if "MicroDuck" in name]


@pytest.mark.parametrize("task_id", _MICRODUCK_TASKS)
@pytest.mark.parametrize("play", [False, True])
def test_registered_microduck_configs_keep_the_policy_contract(task_id, play):
    cfg = load_env_cfg(task_id, play=play)
    assert cfg.scene.entities["robot"] is not None
    assert cfg.decimation == 4
    assert cfg.sim.mujoco.timestep == 0.005
    assert "twist" in cfg.commands
    actor_terms = cfg.observations["actor"].terms
    assert {"command", "head_command", "body_command"} <= actor_terms.keys()
    # Some tasks keep head/body observation slots via zero-padding instead
    # of active command generators. Only active references need a manager term.
    for term in actor_terms.values():
        if "command_name" in term.params:
            assert term.params["command_name"] in cfg.commands
    assert "height_scan" not in actor_terms
    assert "terrain_scan" not in {sensor.name for sensor in cfg.scene.sensors}
    if cfg.scene.terrain.terrain_type == "plane":
        assert "terrain_levels" not in cfg.curriculum
    else:
        assert "terrain_levels" in cfg.curriculum


def test_registration_includes_all_existing_variants():
    assert len(_MICRODUCK_TASKS) == 33


@pytest.mark.parametrize("play", [False, True])
def test_standalone_testbench_uses_current_mjlab_config_api(play):
    from mjlab_microduck.tasks.testbench_env_cfg import (
        MicroduckTestbenchRlCfg,
        TargetAngleCommandCfg,
        make_testbench_env_cfg,
    )

    cfg = make_testbench_env_cfg(play=play)
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.actions["joint_pos"].entity_name == "robot"
    assert cfg.viewer.entity_name == "robot"
    assert isinstance(cfg.commands["target_angle"], TargetAngleCommandCfg)
    assert callable(cfg.commands["target_angle"].build)
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    assert MicroduckTestbenchRlCfg.actor.hidden_dims == (256, 128, 64)
    assert MicroduckTestbenchRlCfg.critic.hidden_dims == (256, 128, 64)
    for groups in MicroduckTestbenchRlCfg.obs_groups.values():
        assert set(groups) <= cfg.observations.keys()
