import json
from dataclasses import replace
from pathlib import Path

import pytest

from z_manip.configuration import load_stack_config


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"


def _load_config():
    return load_stack_config(
        ROOT / "configs/go2w_piper.json",
        environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )


def _write_modified_config(tmp_path, mutate, *, mutate_collision=None):
    values = json.loads((ROOT / "configs/go2w_piper.json").read_text())
    values["robot"]["urdf_path"] = str(URDF)
    if mutate_collision is None:
        values["collision_model"] = str(ROOT / "configs/piper_collision_capsules.json")
    else:
        collision = json.loads((ROOT / "configs/piper_collision_capsules.json").read_text())
        mutate_collision(collision)
        collision_path = tmp_path / "collision.json"
        collision_path.write_text(json.dumps(collision))
        values["collision_model"] = str(collision_path)
    mutate(values)
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(values))
    return path


def test_deployment_config_resolves_robot_path_and_builds_typed_settings():
    config = _load_config()

    assert config.schema_version == 2
    assert config.robot.urdf_path == URDF.resolve()
    assert config.robot.platform_base_frame == "base_link"
    assert config.robot.mount_parent_link == "base"
    assert config.robot.base_link == "piper_base_link"
    assert len(config.robot.acceleration_limits) == 6
    assert config.topics.arm_trajectory == "/piper/joint_trajectory"
    assert config.vlm_models == (
        "qwen/qwen3-vl-8b-instruct:nitro",
        "qwen/qwen3-vl-235b-a22b-instruct:nitro",
    )
    assert config.collision_model_path.name == "piper_collision_capsules.json"
    assert config.approach.visual_servo is config.visual_servo
    assert config.approach.navigation_quiet_speed_mps == pytest.approx(0.035)
    assert config.approach.navigation_quiet_yaw_rate_rps == pytest.approx(0.035)
    assert config.approach.handoff_timeout_s == pytest.approx(60.0)
    assert config.visual_servo.depth_tolerance_m == pytest.approx(0.06)
    assert config.visual_servo.depth_exit_hysteresis_m == pytest.approx(0.01)
    assert config.visual_servo.lateral_exit_hysteresis_m == pytest.approx(0.005)
    assert config.visual_servo.max_forward_mps == pytest.approx(0.05)
    assert config.visual_servo.max_yaw_rps == pytest.approx(0.12)
    assert config.visual_servo.rotate_only_bearing_rad == pytest.approx(
        0.0698131701,
    )
    assert config.grasp_plan.max_feasible_plans == 2
    assert config.grasp_plan.max_hypotheses == 32
    assert config.grasp_plan.search_timeout_s == pytest.approx(12.0)
    assert config.grasp_plan.min_width_m == pytest.approx(0.012)
    assert config.tool_geometry.contact_tcp_z_m == pytest.approx(0.116675)
    assert config.tool_geometry.collision_open_aperture_m == pytest.approx(0.07)
    assert config.tool_geometry.collision_grasp_margin_m == pytest.approx(0.004)
    assert config.standoff.max_hypotheses >= max(
        config.standoff.max_candidates,
        config.standoff.depth_samples,
    )
    assert config.work_pose.radial_distances_m == pytest.approx((0.56, 0.50, 0.62))
    assert config.work_pose.preferred_target_x_m == pytest.approx(0.56)
    assert config.work_pose.max_base_translation_m == pytest.approx(1.5)
    # Objects smaller than 30 mm are outside the current picking scope.  Keep
    # a sub-centimetre position gate while avoiding false IK rejection from
    # camera/hand-eye/model residuals on the real arm.
    assert config.ik.position_tolerance_m == pytest.approx(0.004)
    assert config.ik.orientation_tolerance_rad == pytest.approx(0.3490658504)
    assert config.ik.continuation_timeout_s == pytest.approx(0.18)
    assert config.ik.continuation_seed_timeout_s == pytest.approx(0.08)
    assert config.ik.continuation_fallback_seeds == 2
    assert config.grasp_plan.pregrasp_distance_m == pytest.approx(0.06)
    assert config.grasp_plan.approach_steps == 4
    assert config.grasp_plan.lift_distance_m == pytest.approx(0.07)
    assert config.grasp_plan.lift_steps == 4
    assert config.grasp_plan.symmetry_samples == 4
    assert config.grasp_plan.max_cartesian_joint_step_rad == pytest.approx(0.55)


def test_loaded_frozen_configuration_contains_no_mutable_json_arrays():
    config = _load_config()

    assert isinstance(config.robot.acceleration_limits, tuple)
    assert isinstance(config.tool_geometry.tip_closing_axis, tuple)
    assert isinstance(config.tool_geometry.finger_contact_z_interval_m, tuple)
    assert isinstance(config.work_pose.radial_distances_m, tuple)
    assert isinstance(config.work_pose.target_lateral_offsets_m, tuple)
    assert isinstance(config.work_pose.yaw_offsets_rad, tuple)
    assert isinstance(config.grasp_plan.lift_direction_base, tuple)
    assert isinstance(config.grasp_plan.tool_from_tip, tuple)
    assert all(isinstance(row, tuple) for row in config.grasp_plan.tool_from_tip)


def test_schema_v1_requires_explicit_safety_geometry_migration(tmp_path):
    values = json.loads((ROOT / "configs/go2w_piper.json").read_text())
    values["schema_version"] = 1
    values.pop("tool_geometry")
    values.pop("work_pose")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match=r"schema_version 1 requires migration to 2"):
        load_stack_config(path, environ={})


def test_deployment_config_fails_on_missing_environment_or_unknown_schema(tmp_path):
    with pytest.raises(ValueError, match="Z_MANIP_ROBOT_URDF"):
        load_stack_config(ROOT / "configs/go2w_piper.json", environ={})

    invalid = tmp_path / "bad.json"
    invalid.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema_version"):
        load_stack_config(invalid)


def test_tool_geometry_rejects_invalid_axes_interval_and_tcp():
    geometry = _load_config().tool_geometry

    with pytest.raises(ValueError, match="unit vectors"):
        replace(geometry, tip_closing_axis=(0.0, 2.0, 0.0))
    with pytest.raises(ValueError, match="orthogonal"):
        replace(geometry, tip_approach_axis=(0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="interval"):
        replace(geometry, finger_contact_z_interval_m=(0.14, 0.06))
    with pytest.raises(ValueError, match="TCP"):
        replace(geometry, contact_tcp_z_m=0.17)
    with pytest.raises(ValueError, match="collision grasp margin"):
        replace(geometry, collision_grasp_margin_m=0.08)


def test_loader_rejects_tool_axis_drift_from_tool_transform(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda values: values["tool_geometry"].update(
            tip_closing_axis=[0.0, -1.0, 0.0],
        ),
    )

    with pytest.raises(ValueError, match="tip_closing_axis.*tool_from_tip"):
        load_stack_config(path)


def test_loader_rejects_tcp_drift_from_tool_transform(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda values: values["tool_geometry"].update(contact_tcp_z_m=0.11),
    )

    with pytest.raises(ValueError, match="tool_from_tip translation.*contact TCP"):
        load_stack_config(path)


def test_loader_rejects_grasp_width_beyond_collision_aperture(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda values: values["grasp_plan"].update(max_width_m=0.071),
    )

    with pytest.raises(ValueError, match="max_width_m.*collision_open_aperture_m"):
        load_stack_config(path)


def test_loader_rejects_collision_frames_absent_from_robot_urdf(tmp_path):
    def mutate_collision(values):
        values["capsules"][0]["start_frame"] = "missing_robot_link"

    path = _write_modified_config(
        tmp_path,
        lambda values: None,
        mutate_collision=mutate_collision,
    )

    with pytest.raises(ValueError, match="unknown URDF links.*missing_robot_link"):
        load_stack_config(path)


def test_loader_rejects_collision_model_without_target_contact_geometry(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda values: None,
        mutate_collision=lambda values: values.update(target_contact_capsules=[]),
    )

    with pytest.raises(ValueError, match="identify target_contact_capsules"):
        load_stack_config(path)


def test_loader_rejects_contact_capsule_that_misses_measured_finger_region(tmp_path):
    def mutate_collision(values):
        for capsule in values["capsules"]:
            if capsule["name"] in values["target_contact_capsules"]:
                capsule["radius"] = 0.005

    path = _write_modified_config(
        tmp_path,
        lambda values: None,
        mutate_collision=mutate_collision,
    )

    with pytest.raises(ValueError, match="(leave a gap|do not cover).*finger contact interval"):
        load_stack_config(path)


@pytest.mark.parametrize("limits", ([1.0, 0.0], [1.0, float("nan")]))
def test_loader_rejects_non_positive_or_non_finite_acceleration_limits(
    tmp_path,
    limits,
):
    path = _write_modified_config(
        tmp_path,
        lambda values: values["robot"].update(acceleration_limits=limits),
    )

    with pytest.raises(ValueError, match="acceleration limits.*finite and positive"):
        load_stack_config(path)


@pytest.mark.parametrize("models", ("qwen/model", [""], [42]))
def test_loader_requires_vlm_models_to_be_a_string_array(tmp_path, models):
    path = _write_modified_config(
        tmp_path,
        lambda values: values.update(vlm_models=models),
    )

    with pytest.raises(ValueError, match="vlm_models.*array"):
        load_stack_config(path)
