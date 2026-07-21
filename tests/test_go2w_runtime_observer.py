from __future__ import annotations

import ast
from array import array
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_runtime_observer.py"
LAUNCHER = ROOT / "scripts" / "runtime" / "go2w_runtime_observer.sh"
SERVICE = ROOT / "configs" / "z-manip-runtime-observer.service"
SPEC = importlib.util.spec_from_file_location("go2w_runtime_observer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OBSERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OBSERVER
SPEC.loader.exec_module(OBSERVER)


def _header(stamp_ns: int = 1_700_000_000_123_456_789, frame: str = "base"):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        ),
        frame_id=frame,
    )


def _depth_filter_report() -> dict[str, object]:
    return {
        "method": "motion_adaptive_temporal_median",
        "frame_count": 5,
        "window_size": 5,
        "minimum_observations": 3,
        "mode": "static_temporal",
        "reset_reason": None,
        "motion_threshold_mm": 12.0,
        "global_changed_fraction": 0.002,
        "dynamic_pixels": 0,
        "stable_pixels": 300_000,
        "rejected_low_support_pixels": 10,
        "rejected_unstable_pixels": 12,
        "mad_p95_mm": 1.4,
        "applied_to": ["target_pointcloud", "scene_pointcloud"],
    }


def test_joint_summary_and_fresh_schema_without_ros():
    message = SimpleNamespace(
        header=_header(),
        name=[f"piper_joint{index}" for index in range(1, 7)],
        position=[0.1 * index for index in range(6)],
        velocity=[],
        effort=[],
    )
    summary = OBSERVER.summarize_joint_state(message)
    state = OBSERVER.RuntimeObserverState(
        OBSERVER.DEFAULT_TOPICS,
        ros_domain_id=20,
        started_unix_ns=100,
    )
    state.observe(
        "joint_state",
        summary,
        OBSERVER.stamp_ns(message),
        received_unix_ns=200,
        received_monotonic_ns=1_000_000_000,
    )

    document = state.snapshot(
        publisher_counts={"joint_state": 1},
        generated_unix_ns=300,
        now_monotonic_ns=1_100_000_000,
    )

    assert document["schema"] == "z_manip.runtime_observer.v1"
    assert document["sequence"] == 1
    assert document["observer"]["read_only"] is True
    assert document["observer"]["motion_commands_published"] == 0
    assert document["joint_state"]["available"] is True
    assert document["joint_state"]["names"] == message.name
    assert document["joint_state"]["positions_rad"] == message.position
    assert document["topics"]["joint_state"]["age_s"] == 0.1


def test_missing_joint_publisher_is_explicit_and_does_not_block_camera():
    state = OBSERVER.RuntimeObserverState(
        OBSERVER.DEFAULT_TOPICS,
        ros_domain_id=20,
        started_unix_ns=100,
    )
    valid = {"valid": True, "width": 640, "height": 480}
    for key in ("color", "depth", "camera_info"):
        state.observe(
            key,
            valid,
            10,
            received_unix_ns=20,
            received_monotonic_ns=1_000_000_000,
        )

    document = state.snapshot(
        publisher_counts={
            "joint_state": 0,
            "color": 1,
            "depth": 1,
            "camera_info": 1,
        },
        generated_unix_ns=30,
        now_monotonic_ns=1_100_000_000,
    )

    assert document["joint_state"]["available"] is False
    assert document["joint_state"]["reason"] == "no_publishers"
    assert document["summary"]["camera_rgbd_fresh"] is True


def test_camera_cloud_and_tf_summaries_are_metadata_only():
    image = SimpleNamespace(
        header=_header(frame="camera"),
        width=4,
        height=3,
        step=12,
        encoding="rgb8",
        data=bytes(36),
    )
    cloud = SimpleNamespace(
        header=_header(frame="camera"),
        width=5,
        height=1,
        point_step=16,
        row_step=80,
        data=bytes(80),
        fields=[SimpleNamespace(name="x"), SimpleNamespace(name="y")],
        is_dense=True,
    )
    transform = SimpleNamespace(
        header=_header(frame="base"),
        child_frame_id="tool",
    )

    image_summary = OBSERVER.summarize_image(image)
    cloud_summary = OBSERVER.summarize_point_cloud(cloud)
    tf_summary = OBSERVER.summarize_tf(SimpleNamespace(transforms=[transform]))

    assert image_summary["valid"] is True
    assert image_summary["data_bytes"] == 36
    assert cloud_summary["point_count"] == 5
    assert cloud_summary["fields"] == ["x", "y"]
    assert tf_summary["frame_pairs"] == [{"parent": "base", "child": "tool"}]


def test_raw_rgb_camera_frame_is_bounded_encoded_and_atomically_manifested(tmp_path):
    width, height = 800, 600
    row = bytes([255, 0, 0]) * width + b"padding"
    message = SimpleNamespace(
        header=_header(frame="camera_color_optical_frame"),
        width=width,
        height=height,
        step=len(row),
        encoding="rgb8",
        data=row * height,
    )
    image_path = tmp_path / "camera-latest.jpg"

    OBSERVER.CameraFrameWriter(image_path).write(message)

    jpeg = image_path.read_bytes()
    metadata = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype="uint8"), cv2.IMREAD_COLOR)
    assert decoded.shape == (480, 640, 3)
    assert int(decoded[240, 320, 2]) > 245
    assert int(decoded[240, 320, 0]) < 10
    assert len(jpeg) <= OBSERVER.MAX_CAMERA_JPEG_BYTES
    assert metadata["schema"] == "z_manip.camera_frame.v1"
    assert metadata["width"] == 640
    assert metadata["height"] == 480
    assert metadata["source_encoding"] == "rgb8"
    assert metadata["jpeg_bytes"] == len(jpeg)
    assert not list(tmp_path.glob(".*.tmp"))


def test_camera_encoder_rejects_ambiguous_or_short_raw_images():
    message = SimpleNamespace(
        header=_header(frame="camera"),
        width=4,
        height=3,
        step=12,
        encoding="yuv422",
        data=bytes(36),
    )
    with pytest.raises(ValueError, match="unsupported"):
        OBSERVER.encode_color_image_jpeg(message)
    message.encoding = "rgb8"
    message.data = bytes(35)
    with pytest.raises(ValueError, match="payload"):
        OBSERVER.encode_color_image_jpeg(message)


def test_ros_array_fields_are_accepted_without_ros_imports():
    joint = SimpleNamespace(
        header=_header(),
        name=["joint1", "joint2"],
        position=array("d", [0.1, -0.2]),
        velocity=array("d"),
        effort=array("d"),
    )
    info = SimpleNamespace(
        header=_header(frame="camera"),
        width=640,
        height=480,
        k=array("d", [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]),
        d=array("d"),
        distortion_model="plumb_bob",
    )

    assert OBSERVER.summarize_joint_state(joint)["valid"] is True
    assert OBSERVER.summarize_camera_info(info)["valid"] is True


def test_depth_filter_manifest_is_bounded_and_propagated_to_runtime_telemetry():
    report = _depth_filter_report()
    message = SimpleNamespace(data=json.dumps({
        "schema": "z_manip.tracker_frame.v1",
        "result_stamp_ns": 1_700_000_000_123_456_789,
        "depth_filter": report,
    }))
    summary = OBSERVER.summarize_depth_filter(message)
    assert summary == {"valid": True, "error": None, "depth_filter": report}

    state = OBSERVER.RuntimeObserverState(
        OBSERVER.DEFAULT_TOPICS,
        ros_domain_id=20,
        started_unix_ns=100,
    )
    state.observe(
        "depth_filter",
        summary,
        None,
        received_unix_ns=200,
        received_monotonic_ns=1_000_000_000,
    )
    diagnostic = state.snapshot(
        publisher_counts={"depth_filter": 1},
        generated_unix_ns=300,
        now_monotonic_ns=1_100_000_000,
    )
    runtime = OBSERVER.build_runtime_state(diagnostic)

    assert diagnostic["summary"]["depth_filter_fresh"] is True
    assert runtime["telemetry"]["depth_filter"] == {
        "available": True,
        "fresh": True,
        "report": report,
    }


def test_depth_filter_manifest_rejects_unknown_fields_and_impossible_bounds():
    report = _depth_filter_report()
    report["window_size"] = 2
    malformed = SimpleNamespace(data=json.dumps({
        "schema": "z_manip.tracker_frame.v1",
        "depth_filter": report,
    }))
    assert OBSERVER.summarize_depth_filter(malformed)["valid"] is False

    report = _depth_filter_report()
    report["untrusted"] = "field"
    unknown = SimpleNamespace(data=json.dumps({
        "schema": "z_manip.tracker_frame.v1",
        "depth_filter": report,
    }))
    assert OBSERVER.summarize_depth_filter(unknown)["valid"] is False


def test_atomic_snapshot_replaces_valid_json(tmp_path):
    output = tmp_path / "runtime.json"
    OBSERVER.atomic_write_json(output, {"sequence": 1})
    OBSERVER.atomic_write_json(output, {"sequence": 2, "complete": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "complete": True,
        "sequence": 2,
    }
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_runtime_state_exports_verified_platform_and_arm_camera_transforms():
    class Chain:
        base_link = "piper_base_link"
        tip_link = "piper_gripper_base"

        @staticmethod
        def forward(_values):
            transform = np.eye(4)
            transform[:3, 3] = (0.30, -0.10, 0.20)
            return transform

        @staticmethod
        def link_transforms(_values):
            return {"piper_base_link": np.eye(4)}

    diagnostic = {
        "generated_unix_ns": 1_700_000_000_000_000_000,
        "sequence": 7,
        "joint_state": {
            "available": True,
            "positions_rad": [0.0] * 6,
            "source_stamp_ns": 1_699_999_999_990_000_000,
            "topic": "/piper/state",
            "publisher_count": 1,
        },
        "summary": {},
        "observer": {"read_only": True, "motion_commands_published": 0},
        "topics": {},
    }
    calibration = {
        "calibrated": True,
        "synthetic": False,
        "mount_type": "eye_in_hand",
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "calibration_id": "measured-test",
        "tip_from_camera": np.eye(4).tolist(),
    }
    platform_from_arm = np.eye(4)
    platform_from_arm[:3, 3] = (0.06, 0.0, 0.067)

    runtime = OBSERVER.build_runtime_state(
        diagnostic,
        chain=Chain(),
        calibration=calibration,
        platform_from_arm_base=platform_from_arm,
        platform_frame="base_link",
    )

    transforms = runtime["kinematic_transforms"]
    assert transforms["verified"] is True
    assert transforms["camera_frame"] == "camera_color_optical_frame"
    assert transforms["arm_base_frame"] == "piper_base_link"
    assert transforms["platform_base_frame"] == "base_link"
    assert np.asarray(transforms["arm_base_from_camera"])[:3, 3] == pytest.approx(
        (0.30, -0.10, 0.20),
    )
    assert np.asarray(transforms["platform_base_from_camera"])[:3, 3] == pytest.approx(
        (0.36, -0.10, 0.267),
    )


def test_observer_restart_recovers_sequence_and_advances(tmp_path, monkeypatch):
    output = tmp_path / "runtime.json"
    output.write_text(
        json.dumps({"schema": OBSERVER.SCHEMA, "sequence": 46_135}),
        encoding="utf-8",
    )
    initial = OBSERVER.load_initial_sequence(output)
    state = OBSERVER.RuntimeObserverState(
        OBSERVER.DEFAULT_TOPICS,
        ros_domain_id=20,
        initial_sequence=initial,
    )

    assert initial == 46_135
    assert state.snapshot(generated_unix_ns=1)["sequence"] == 46_136

    monkeypatch.setattr(OBSERVER, "MAX_EXISTING_STATE_BYTES", 4)
    assert OBSERVER.load_initial_sequence(output) == 0


def test_observer_source_has_no_publish_send_subprocess_or_actuator_transport():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {
        "can",
        "piper_sdk",
        "pyAgxArm",
        "socket",
        "subprocess",
    }
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden_calls = {
        "call_async",
        "create_client",
        "create_publisher",
        "publish",
        "send",
        "send_goal",
        "send_goal_async",
        "sendall",
        "sendmsg",
        "sendto",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
    }

    assert imports.isdisjoint(forbidden_imports)
    assert calls == set()
    assert "create_subscription" in source
    assert "rclpy" not in OBSERVER.__dict__


def test_launcher_and_service_are_isolated_and_non_blocking():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")

    assert "--device" not in launcher
    assert "/dev/can" not in launcher
    assert "/dev/tty" not in launcher
    assert "--cap-drop ALL" in launcher
    assert "--read-only" in launcher
    assert "ROS_LOG_DIR=/tmp/ros-log" in launcher
    assert ":/ssh/" not in launcher
    assert "cmd_vel" not in launcher
    assert "piper_sdk" not in launcher.lower()
    assert "pyagxarm" not in launcher.lower()
    assert "can0" not in launcher.lower()
    assert "WantedBy=default.target" in service
    assert "Restart=on-failure" in service
    assert "Requires=" not in service
    assert "go2w_runtime_observer.sh run" in service
    assert "--camera-output" in launcher
