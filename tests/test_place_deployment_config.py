"""The deployed placement section, and the loader gate that keeps it required."""

import json
from pathlib import Path

import numpy as np
import pytest

from z_manip.configuration import load_stack_config
from z_manip.planning.placement import (
    ObservedPlacementConfig,
    PlacementDeploymentConfig,
)


ROOT = Path(__file__).resolve().parents[1]
STACK_CONFIG = ROOT / "configs/go2w_piper.json"


def _raw():
    return json.loads(STACK_CONFIG.read_text())


def _deployment():
    return PlacementDeploymentConfig.from_mapping(_raw()["placement"])


def test_deployed_placement_section_parses_and_matches_the_measured_gripper():
    values = _raw()
    deployment = _deployment()
    tool = values["tool_geometry"]

    assert deployment.release_clearance_m == pytest.approx(0.015)
    assert deployment.max_geometric_candidates == 24
    assert deployment.tool_probe_clearance_m == pytest.approx(
        tool["collision_grasp_margin_m"],
    )
    probes = np.asarray(deployment.gripper_probe_points_tool_m)
    assert probes.shape == (6, 3)

    # The probe points model the finger pair that straddles the object, so they
    # must lie inside the measured open aperture and reach the distal pads.
    half_aperture = 0.5 * tool["collision_open_aperture_m"]
    assert float(np.abs(probes[:, 1]).min()) >= half_aperture
    lower, upper = tool["finger_contact_z_interval_m"]
    assert float(probes[:, 2].max()) <= upper
    assert float(probes[:, 2].max()) >= lower


def test_release_clearance_clears_the_band_where_obstacles_are_ignored():
    deployment = _deployment()
    search = deployment.planner_config()

    # Releasing at or below plane_exclusion_m would drop the object inside the
    # band the obstacle test deliberately ignores, and below max_plane_rms_m it
    # would be inside the fit's own noise.
    assert deployment.release_clearance_m > search.plane_exclusion_m
    assert deployment.release_clearance_m > search.max_plane_rms_m


def test_planner_config_carries_the_gripper_model_and_accepts_search_overrides():
    deployment = _deployment()
    search = deployment.planner_config(sample_spacing_m=0.05, yaw_samples=4)

    assert search.tool_probe_points_m == deployment.gripper_probe_points_tool_m
    assert search.release_clearance_m == deployment.release_clearance_m
    assert search.max_geometric_candidates == 24
    assert search.sample_spacing_m == pytest.approx(0.05)
    assert search.yaw_samples == 4
    # Everything the deployment does not speak to stays robot-agnostic.
    defaults = ObservedPlacementConfig()
    assert search.plane_exclusion_m == defaults.plane_exclusion_m
    assert search.tool_clearance_radius_m == defaults.tool_clearance_radius_m


def test_placement_section_is_parsed_strictly_and_never_defaulted():
    values = _raw()["placement"]

    with pytest.raises(ValueError, match=r"missing=\['release_clearance_m'\]"):
        PlacementDeploymentConfig.from_mapping(
            {k: v for k, v in values.items() if k != "release_clearance_m"},
        )
    with pytest.raises(ValueError, match=r"unknown=\['relase_clearance_m'\]"):
        PlacementDeploymentConfig.from_mapping(
            dict(values, relase_clearance_m=0.015),
        )
    with pytest.raises(ValueError, match="release clearance"):
        PlacementDeploymentConfig.from_mapping(
            dict(values, release_clearance_m=-0.001),
        )
    with pytest.raises(ValueError, match="geometric candidate"):
        PlacementDeploymentConfig.from_mapping(
            dict(values, max_geometric_candidates=0),
        )
    with pytest.raises(ValueError, match="probe points"):
        PlacementDeploymentConfig.from_mapping(
            dict(values, gripper_probe_points_tool_m=[]),
        )


def test_stack_loader_accepts_a_config_that_predates_the_placement_section(tmp_path):
    # The stack config is bind-mounted into the planning container as a single
    # FILE, so a checkout that adds this section leaves the running container on
    # the old inode while the package hot-reloads. Requiring the section would
    # take live planning down until the container is recreated.
    values = _raw()
    values.pop("placement")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match="robot URDF does not exist"):
        load_stack_config(path, environ={"Z_MANIP_ROBOT_URDF": "/unused.urdf"})


def test_stack_loader_still_rejects_a_misspelled_placement_section(tmp_path):
    values = _raw()
    values["placemnt"] = values.pop("placement")
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match=r"unknown=\['placemnt'\]"):
        load_stack_config(path, environ={"Z_MANIP_ROBOT_URDF": "/unused.urdf"})


def test_stack_loader_parses_a_present_placement_section_strictly(tmp_path):
    urdf = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
    if not urdf.exists():
        pytest.skip("robot URDF asset is not checked out")
    values = _raw()
    values["collision_model"] = str(ROOT / "configs/piper_collision_capsules.json")
    values["placement"]["release_clearance_m"] = -1.0
    path = tmp_path / "bad_placement.json"
    path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match="release clearance"):
        load_stack_config(path, environ={"Z_MANIP_ROBOT_URDF": str(urdf)})


def test_loaded_stack_config_exposes_the_deployed_placement_section():
    urdf = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
    if not urdf.exists():
        pytest.skip("robot URDF asset is not checked out")

    config = load_stack_config(
        STACK_CONFIG, environ={"Z_MANIP_ROBOT_URDF": str(urdf)},
    )

    assert config.placement == _deployment()
