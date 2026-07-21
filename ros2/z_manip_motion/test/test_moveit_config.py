from pathlib import Path
from xml.etree import ElementTree

import pytest

from z_manip_motion.moveit_config import (
    MoveItConfigError,
    moveit_sensor_parameters,
    normalized_robot_description,
)


def _write_urdf(path: Path, mesh_references: list[str]) -> None:
    meshes = "".join(
        f'<visual><geometry><mesh filename="{reference}"/></geometry></visual>'
        for reference in mesh_references
    )
    path.write_text(f'<robot name="test"><link name="base">{meshes}</link></robot>')


def test_relative_meshes_are_resolved_from_urdf_directory(tmp_path: Path):
    urdf_directory = tmp_path / "robot" / "urdf"
    mesh = tmp_path / "robot" / "meshes" / "arm link.stl"
    urdf_directory.mkdir(parents=True)
    mesh.parent.mkdir()
    mesh.write_bytes(b"solid mesh")
    urdf = urdf_directory / "robot.urdf"
    _write_urdf(urdf, ["../meshes/arm link.stl"])

    root = ElementTree.fromstring(normalized_robot_description(urdf))

    assert root.find(".//mesh").attrib["filename"] == mesh.resolve().as_uri()


def test_file_and_package_meshes_are_checked_and_canonicalized(tmp_path: Path):
    urdf = tmp_path / "robot.urdf"
    direct_mesh = tmp_path / "direct.stl"
    package_share = tmp_path / "share" / "test_description"
    package_mesh = package_share / "meshes" / "package.stl"
    package_mesh.parent.mkdir(parents=True)
    direct_mesh.write_bytes(b"solid direct")
    package_mesh.write_bytes(b"solid package")
    _write_urdf(
        urdf,
        [direct_mesh.as_uri(), "package://test_description/meshes/package.stl"],
    )

    root = ElementTree.fromstring(
        normalized_robot_description(
            urdf,
            package_share_lookup=lambda package: str(package_share),
        )
    )

    assert [mesh.attrib["filename"] for mesh in root.iter("mesh")] == [
        direct_mesh.resolve().as_uri(),
        package_mesh.resolve().as_uri(),
    ]


@pytest.mark.parametrize(
    "reference",
    ["../meshes/missing.stl", "https://example.invalid/mesh.stl", ""],
)
def test_missing_unsupported_or_empty_mesh_reference_fails_closed(
    tmp_path: Path,
    reference: str,
):
    urdf = tmp_path / "robot.urdf"
    _write_urdf(urdf, [reference])

    with pytest.raises(MoveItConfigError):
        normalized_robot_description(urdf)


def test_named_point_cloud_profile_is_flattened_with_launch_overrides():
    profile = {
        "sensors": {
            "wrist_point_cloud": {
                "sensor_plugin": "occupancy_map_monitor/PointCloudOctomapUpdater",
                "point_cloud_topic": "/profile/cloud",
                "max_range": 1.0,
                "point_subsample": 2,
                "filtered_cloud_topic": "/profile/filtered",
            },
            "base_point_cloud": {
                "sensor_plugin": "occupancy_map_monitor/PointCloudOctomapUpdater",
            },
        }
    }

    parameters = moveit_sensor_parameters(
        profile,
        point_cloud_topic="/runtime/cloud",
        depth_image_topic="/runtime/depth",
        filtered_cloud_topic="/runtime/filtered",
        max_range=2.5,
    )

    assert parameters["sensors"] == ["wrist_point_cloud", "base_point_cloud"]
    for name in parameters["sensors"]:
        assert parameters[f"{name}.point_cloud_topic"] == "/runtime/cloud"
        assert parameters[f"{name}.max_range"] == 2.5
        assert parameters[f"{name}.filtered_cloud_topic"] == "/runtime/filtered"
    assert parameters["wrist_point_cloud.point_subsample"] == 2
    assert all(not isinstance(value, dict) for value in parameters.values())


def test_depth_profile_uses_depth_topic_and_range_overrides():
    parameters = moveit_sensor_parameters(
        {
            "sensors": {
                "wrist_depth": {
                    "sensor_plugin": "occupancy_map_monitor/DepthImageOctomapUpdater",
                    "image_topic": "/profile/depth",
                    "far_clipping_plane_distance": 1.0,
                    "filtered_cloud_topic": "/profile/filtered",
                }
            }
        },
        point_cloud_topic="/runtime/cloud",
        depth_image_topic="/runtime/depth",
        filtered_cloud_topic="/runtime/filtered",
        max_range=2.5,
    )

    assert parameters["sensors"] == ["wrist_depth"]
    assert parameters["wrist_depth.image_topic"] == "/runtime/depth"
    assert parameters["wrist_depth.far_clipping_plane_distance"] == 2.5
    assert parameters["wrist_depth.filtered_cloud_topic"] == "/runtime/filtered"


@pytest.mark.parametrize(
    "profile",
    [
        {"sensors": []},
        {"sensors": [{"sensor_plugin": "plugin"}]},
        {"sensors": {"bad.name": {"sensor_plugin": "plugin"}}},
        {"sensors": {"camera": {"point_cloud_topic": "/cloud"}}},
        {"sensors": {"camera": {"sensor_plugin": "plugin", "nested": {"x": 1}}}},
    ],
)
def test_invalid_sensor_profiles_fail_closed(profile):
    with pytest.raises(MoveItConfigError):
        moveit_sensor_parameters(
            profile,
            point_cloud_topic="/cloud",
            depth_image_topic="/depth",
            filtered_cloud_topic="/filtered",
            max_range=2.5,
        )
