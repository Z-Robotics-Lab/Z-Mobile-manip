from __future__ import annotations

import ast
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("go2w_planning_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


HTML = ROOT / "web" / "debug_dashboard" / "index.html"
SERVICE = ROOT / "configs" / "z-manip-planning-workbench.service"


def _executor_start_evidence() -> dict[str, object]:
    return {
        "schema": "z_manip.piper_executor_start_receipt.v1",
        "event": "transport_opened",
        "artifact_id": "a" * 64,
        "planning_report_sha256": "b" * 64,
        "planned_grasp_sha256": "c" * 64,
        "planning_session_id": "20260722-120000",
        "executor_started_unix_ns": 1_800_000_001_000_000_000,
        "executor_started_monotonic_ns": 123_456_789,
        "monotonic_clock_domain": "nuc_piper_executor_process",
        "transport": "piper_can",
        "transport_opened": True,
        "commands_sent": 0,
        "motion_started": False,
    }


def test_executor_start_receipt_requires_real_zero_command_transport_evidence(tmp_path):
    receipt = tmp_path / "executor-start-receipt.json"
    receipt.write_text(json.dumps(_executor_start_evidence()), encoding="utf-8")

    validated = CONTROL.PiperGraspRunner._validate_executor_start_receipt(receipt)

    assert validated["transport_opened"] is True
    assert validated["commands_sent"] == 0
    assert validated["motion_started"] is False


def test_executor_start_receipt_fails_closed_for_worker_only_claim(tmp_path):
    document = _executor_start_evidence()
    document["transport_opened"] = False
    receipt = tmp_path / "executor-start-receipt.json"
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity/zero-command"):
        CONTROL.PiperGraspRunner._validate_executor_start_receipt(receipt)


def test_executor_start_receipt_is_bound_to_exact_plan_files_and_session(tmp_path):
    document = _executor_start_evidence()
    receipt = tmp_path / "executor-start-receipt.json"
    receipt.write_text(json.dumps(document), encoding="utf-8")

    validated = CONTROL.PiperGraspRunner._validate_executor_start_receipt(
        receipt,
        expected_artifact_id="a" * 64,
        expected_planning_report_sha256="b" * 64,
        expected_planned_grasp_sha256="c" * 64,
        expected_planning_session_id="20260722-120000",
    )
    assert validated == document

    for field in (
        "artifact_id",
        "planning_report_sha256",
        "planned_grasp_sha256",
        "planning_session_id",
    ):
        with pytest.raises(RuntimeError, match=field):
            CONTROL.PiperGraspRunner._validate_executor_start_receipt(
                receipt,
                expected_artifact_id=("d" * 64 if field == "artifact_id" else "a" * 64),
                expected_planning_report_sha256=("d" * 64 if field == "planning_report_sha256" else "b" * 64),
                expected_planned_grasp_sha256=("d" * 64 if field == "planned_grasp_sha256" else "c" * 64),
                expected_planning_session_id=("foreign-session" if field == "planning_session_id" else "20260722-120000"),
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


def _runtime_state(timestamp_ns: int, *, sequence: int = 1) -> dict[str, object]:
    return {
        "schema": "z_manip.runtime_state.v1",
        "sequence": sequence,
        "source_timestamp_ns": timestamp_ns,
        "joint_positions_rad": [0.0, 0.1, -0.2, 0.3, -0.4, 0.5],
        "robot_links": [
            {"name": "piper_link1", "transform": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]},
        ],
        "point_clouds": {
            "target": {
                "frame": "piper_base_link",
                "points_xyz_m": [[0.1, 0.0, 0.3], [0.11, 0.01, 0.3]],
                "colors_rgb": [[255, 0, 0], [255, 0, 0]],
            },
        },
        "candidates": [{
            "candidate_id": 2,
            "frame": "piper_base_link",
            "pose": [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "score": 0.8,
            "width_m": 0.03,
            "status": "selected",
        }],
        "plan_overlay": {
            "frame": "piper_base_link",
            "joint_names": [f"joint{i}" for i in range(1, 7)],
            "segments": {
                "transit": {
                    "positions_rad": [[0.0] * 6, [0.1] * 6],
                    "times_s": [0.0, 1.0],
                },
            },
            "tcp_path_xyz_m": [[0.0, 0.0, 0.2], [0.1, 0.0, 0.3]],
            "selected_candidate_id": 2,
        },
    }


def _bundle(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema": "z_manip.debug_bundle.v1",
        "mode": {"read_only": True},
        "safety": {"motion_commands_published": 0},
        "stages": [],
        "artifacts": {},
        "visualization": {},
    }), encoding="utf-8")
    return path


class _FakeInteractiveService:
    def __init__(self) -> None:
        self.perception_targets: list[str] = []
        self.planning_calls = 0
        self.clear_calls = 0
        self.state: dict[str, object] = {
            "schema": "z_manip.interactive_session_state.v1",
            "read_only": True,
            "busy": False,
            "selected_perception_session_id": None,
            "actions": {
                "perception": {"latest_attempt": None, "last_good": None},
                "planning": {"latest_attempt": None, "last_good": None},
            },
            "safety": {
                "motion_commands_available": False,
                "client_paths_accepted": False,
                "client_commands_accepted": False,
                "client_environment_accepted": False,
            },
        }

    def status(self):
        return self.state

    def start_perception(self, target):
        self.perception_targets.append(target)
        return {
            "schema": "z_manip.interactive_session_attempt.v1",
            "action": "perception",
            "session_id": "20260717-170000",
            "status": "succeeded",
            "target": target,
        }

    def start_planning(self):
        self.planning_calls += 1
        return {
            "schema": "z_manip.interactive_session_attempt.v1",
            "action": "planning",
            "session_id": "20260717-170001",
            "status": "succeeded",
        }

    def clear_current_context(self):
        self.clear_calls += 1
        return {"cleared": True}


class _FakeGraspRunner:
    def __init__(self) -> None:
        self.starts = 0
        self.running = False
        self.speed_percent = None
        self.selected_starts = 0
        self.target = None
        self.home_resets = 0
        self.home_starts = 0
        self.mobile_handoff_starts = 0
        self.superseded_perception_session_id = None
        self.base_stopped_unix_ns = None
        self.base_stopped_monotonic_ns = None

    def status(self):
        return {
            "schema": "z_manip.grasp_action.v1",
            "available": True,
            "running": self.running,
            "state": "running" if self.running else "idle",
            "motion_commands_permitted": True,
        }

    def start(self, target, speed_percent=5):
        self.starts += 1
        self.target = target
        self.speed_percent = speed_percent
        if self.running:
            return {"started": False, "grasp": self.status()}
        self.running = True
        return {"started": True, "grasp": self.status()}

    def start_at_mobile_handoff(
        self,
        target,
        speed_percent=5,
        *,
        superseded_perception_session_id=None,
        base_stopped_unix_ns=None,
        base_stopped_monotonic_ns=None,
    ):
        self.mobile_handoff_starts += 1
        self.starts += 1
        self.target = target
        self.speed_percent = speed_percent
        self.superseded_perception_session_id = superseded_perception_session_id
        self.base_stopped_unix_ns = base_stopped_unix_ns
        self.base_stopped_monotonic_ns = base_stopped_monotonic_ns
        if self.running:
            return {"started": False, "grasp": self.status()}
        self.running = True
        return {"started": True, "grasp": self.status()}

    def start_selected(self, speed_percent=5):
        self.selected_starts += 1
        self.starts += 1
        self.speed_percent = speed_percent
        if self.running:
            return {"started": False, "grasp": self.status()}
        self.running = True
        return {"started": True, "grasp": self.status()}

    def reset_after_home(self):
        self.home_resets += 1

    def reset_for_home(self):
        self.home_starts += 1


class _FakeApproachRunner:
    def __init__(self) -> None:
        self.mode = None
        self.running = False
        self.stops = 0
        self.options = {}

    def status(self):
        return {
            "schema": "z_manip.depth_servo_action.v1",
            "available": True,
            "running": self.running,
            "mode": self.mode,
            "phase": "approach" if self.running else "idle",
        }

    def start(self, mode, **options):
        self.mode = mode
        self.options = options
        self.running = True
        return {"started": True, "approach": self.status()}

    def stop(self):
        self.stops += 1
        self.running = False
        return {"stopped": True, "approach": self.status()}


class _FakeHomeRunner:
    def __init__(self) -> None:
        self.starts = 0
        self.running = False
        self.speed_percent = None

    def status(self):
        return {
            "schema": "z_manip.piper_home_action.v1",
            "available": True,
            "running": self.running,
            "state": "running" if self.running else "idle",
        }

    def start(self, speed_percent=2):
        self.starts += 1
        self.speed_percent = speed_percent
        if self.running:
            return {"started": False, "home": self.status()}
        self.running = True
        return {"started": True, "home": self.status()}


def _interactive_headers(port: int, action: str) -> dict[str, str]:
    return {
        "Origin": f"http://127.0.0.1:{port}",
        "Sec-Fetch-Site": "same-origin",
        "X-Z-Manip-Action": action,
        "Content-Type": "application/json; charset=UTF-8",
    }


def _manifest(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema": "z_manip.immutable_artifact_manifest.v1",
        "file_count": len(files),
        "files": [
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(files.items())
        ],
    }


def test_runner_serializes_one_fixed_argument_free_planning_script(tmp_path, monkeypatch):
    session = tmp_path / "go2w_planning_session.sh"
    session.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    session.chmod(0o755)
    run_root = tmp_path / "runs"
    latest = run_root / "latest"
    latest.mkdir(parents=True)
    (latest / "debug_bundle.json").write_text("{}", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["shell"] = kwargs["shell"]
        observed["stdin"] = kwargs["stdin"]
        entered.set()
        assert release.wait(timeout=2)
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(CONTROL.subprocess, "run", fake_run)
    runner = CONTROL.PlanningOnlyRunner(session, run_root)

    assert runner.start()["started"] is True
    assert entered.wait(timeout=2)
    assert runner.start()["started"] is False
    release.set()
    for _ in range(100):
        if not runner.status()["running"]:
            break
        time.sleep(0.01)

    status = runner.status()
    assert status["running"] is False
    assert status["outcome"] == "blocked"
    assert status["motion_commands_permitted"] is False
    assert observed["args"] == [str(session.resolve())]
    assert observed["shell"] is False
    assert observed["stdin"] is subprocess.DEVNULL


def test_control_source_has_no_direct_actuator_protocol_surface():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "cansend",
        "ros2 topic pub",
        "joint_trajectory",
        "cmd_vel",
        "pyagxarm",
        "shell=true",
    ):
        assert forbidden not in source
    assert "motion_commands_permitted" in source
    assert "[str(self.session_script)]" in source


def test_runtime_reader_reports_live_stale_and_sequence_updates(tmp_path):
    now = [1_800_000_000_000_000_000]
    state_path = tmp_path / "runtime.json"
    state_path.write_text(json.dumps(_runtime_state(now[0])), encoding="utf-8")
    reader = CONTROL.RuntimeStateReader(
        state_path,
        stale_after_s=1.0,
        clock_ns=lambda: now[0],
    )

    live, live_etag = reader.snapshot()
    assert live["status"] == "live"
    assert live["sequence"] == 1
    assert live["received_timestamp_ns"] == now[0]
    assert live["joint_positions_rad"][2] == -0.2
    assert live["point_clouds"]["target"]["points_xyz_m"][1] == [0.11, 0.01, 0.3]

    now[0] += 1_500_000_000
    stale, stale_etag = reader.snapshot()
    assert stale["status"] == "stale"
    assert stale_etag != live_etag
    assert stale["received_timestamp_ns"] < now[0]

    updated = _runtime_state(now[0], sequence=2)
    updated["joint_positions_rad"][0] = 0.25
    state_path.write_text(json.dumps(updated), encoding="utf-8")
    refreshed, _etag = reader.snapshot()
    assert refreshed["status"] == "live"
    assert refreshed["sequence"] == 2
    assert refreshed["joint_positions_rad"][0] == 0.25


def test_runtime_validator_preserves_strict_depth_filter_telemetry():
    state = _runtime_state(1_800_000_000_000_000_000)
    state["telemetry"] = {
        "depth_filter": {
            "available": True,
            "fresh": True,
            "report": _depth_filter_report(),
        },
    }

    normalized = CONTROL.validate_runtime_state(state)

    assert normalized["telemetry"]["depth_filter"]["report"] == _depth_filter_report()


def test_runtime_validator_accepts_verified_measured_kinematic_transforms():
    state = _runtime_state(1_800_000_000_000_000_000)
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    state["kinematic_transforms"] = {
        "schema": "z_manip.kinematic_transforms.v1",
        "verified": True,
        "source": "passive_joints+deployed_urdf+measured_hand_eye",
        "source_timestamp_ns": state["source_timestamp_ns"],
        "joint_source_timestamp_ns": state["source_timestamp_ns"] - 10_000_000,
        "camera_frame": "camera_color_optical_frame",
        "arm_base_frame": "piper_base_link",
        "platform_base_frame": "base_link",
        "arm_base_from_camera": identity,
        "platform_base_from_camera": identity,
        "calibration_id": "measured-hand-eye-test",
        "calibration_synthetic": False,
    }

    normalized = CONTROL.validate_runtime_state(state)

    assert normalized["kinematic_transforms"]["verified"] is True
    assert normalized["kinematic_transforms"]["platform_base_frame"] == "base_link"
    assert normalized["kinematic_transforms"]["platform_base_from_camera"] == identity


def test_runtime_validator_rejects_inconsistent_or_malformed_filter_telemetry():
    state = _runtime_state(1_800_000_000_000_000_000)
    state["telemetry"] = {
        "depth_filter": {"available": True, "fresh": True, "report": None},
    }
    try:
        CONTROL.validate_runtime_state(state)
    except CONTROL.RuntimeStateError as error:
        assert "requires a report" in str(error)
    else:
        raise AssertionError("missing available filter report was accepted")

    report = _depth_filter_report()
    report["global_changed_fraction"] = 1.5
    state["telemetry"]["depth_filter"]["report"] = report
    try:
        CONTROL.validate_runtime_state(state)
    except CONTROL.RuntimeStateError as error:
        assert "thresholds violate bounds" in str(error)
    else:
        raise AssertionError("out-of-bounds filter telemetry was accepted")


def test_runtime_validator_accepts_observer_tracker_with_live_joint_feedback():
    state = _runtime_state(1_800_000_000_000_000_000)
    state["joint_state_available"] = True
    state["telemetry"] = {
        "read_only": True,
        "motion_commands_published": 0,
        "joint_state_available": True,
        "tracker": {
            "phase": "tracking",
            "tracking": True,
            "target_fresh": True,
            "target_source_stamp_ns": 1_800_000_000_000_000_000,
            "failure": None,
        },
    }

    normalized = CONTROL.validate_runtime_state(state)

    assert normalized["joint_state_available"] is True
    assert normalized["telemetry"]["tracker"] == state["telemetry"]["tracker"]


def test_runtime_validator_rejects_malformed_observer_tracker():
    state = _runtime_state(1_800_000_000_000_000_000)
    state["telemetry"] = {
        "tracker": {
            "phase": "tracking",
            "tracking": "yes",
            "target_fresh": True,
            "target_source_stamp_ns": None,
            "failure": None,
        },
    }

    try:
        CONTROL.validate_runtime_state(state)
    except CONTROL.RuntimeStateError as error:
        assert "tracking must be boolean or null" in str(error)
    else:
        raise AssertionError("malformed tracker telemetry was accepted")


def test_runtime_reader_missing_invalid_and_oversized_files_are_offline(tmp_path, monkeypatch):
    missing = CONTROL.RuntimeStateReader(tmp_path / "missing.json").snapshot()[0]
    assert missing["status"] == "offline"
    assert missing["error"]["code"] == "RUNTIME_STATE_MISSING"
    assert missing["joint_positions_rad"] is None

    path = tmp_path / "runtime.json"
    invalid = _runtime_state(time.time_ns())
    invalid["client_path"] = "/etc/passwd"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    document, _etag = CONTROL.RuntimeStateReader(path).snapshot()
    assert document["status"] == "offline"
    assert document["error"]["code"] == "RUNTIME_STATE_INVALID"
    assert "unsupported fields" in document["error"]["message"]

    monkeypatch.setattr(CONTROL, "MAX_RUNTIME_STATE_BYTES", 4)
    document, _etag = CONTROL.RuntimeStateReader(path).snapshot()
    assert document["error"]["code"] == "RUNTIME_STATE_TOO_LARGE"


def test_runtime_reader_rejects_future_timestamp_and_sequence_reuse(tmp_path):
    now = [1_800_000_000_000_000_000]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_runtime_state(now[0] + 300_000_000)), encoding="utf-8")
    reader = CONTROL.RuntimeStateReader(path, clock_ns=lambda: now[0])
    future, _etag = reader.snapshot()
    assert future["status"] == "offline"
    assert future["error"]["code"] == "RUNTIME_TIMESTAMP_IN_FUTURE"

    path.write_text(json.dumps(_runtime_state(now[0], sequence=1)), encoding="utf-8")
    assert reader.snapshot()[0]["status"] == "live"
    changed = _runtime_state(now[0], sequence=1)
    changed["joint_positions_rad"][0] = 0.2
    path.write_text(json.dumps(changed), encoding="utf-8")
    reused, _etag = reader.snapshot()
    assert reused["status"] == "offline"
    assert reused["error"]["code"] == "RUNTIME_SEQUENCE_NOT_ADVANCED"


def test_runtime_endpoint_is_fixed_loopback_conditional_get_and_keeps_post(tmp_path):
    class FakeControl:
        def __init__(self):
            self.starts = 0

        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            self.starts += 1
            return {"started": True, "control": self.status()}

    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(_runtime_state(time.time_ns())), encoding="utf-8")
    other_path = tmp_path / "other.json"
    other_path.write_text(
        json.dumps(_runtime_state(time.time_ns(), sequence=99)),
        encoding="utf-8",
    )
    control = FakeControl()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=control,
        runtime_state=runtime_path,
    )
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_address[1],
        timeout=3,
    )
    try:
        connection.request("GET", "/api/runtime")
        response = connection.getresponse()
        assert response.status == 200
        etag = response.getheader("ETag")
        assert response.getheader("X-Z-Manip-Poll-Interval-Ms") == "200"
        document = json.loads(response.read())
        assert document["status"] == "live"
        assert document["sequence"] == 1
        assert control.starts == 0

        connection.request("GET", "/api/runtime", headers={"If-None-Match": etag})
        response = connection.getresponse()
        assert response.status == 304
        assert response.getheader("ETag") == etag
        assert response.read() == b""

        connection.request("GET", f"/api/runtime?path={other_path}")
        response = connection.getresponse()
        assert response.status == 400
        assert b"no query" in response.read()
        assert control.starts == 0

        connection.request(
            "POST",
            "/api/runs",
            headers={"X-Z-Manip-Action": "planning-only"},
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["started"] is True
        assert control.starts == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_interactive_session_api_exposes_separate_strict_actions_and_status(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("interactive actions must not invoke the legacy runner")

    interactive = _FakeInteractiveService()
    run_root = tmp_path / "interactive"
    (run_root / "perception").mkdir(parents=True)
    (run_root / "planning").mkdir()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        interactive_run_root=run_root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/api/sessions/status")
        response = connection.getresponse()
        assert response.status == 200
        state = json.loads(response.read())
        assert state["busy"] is False
        assert state["read_only"] is True

        perception_body = json.dumps(
            {"target": "白色 USB 适配器"},
            ensure_ascii=False,
        ).encode("utf-8")
        connection.request(
            "POST",
            "/api/sessions/perception",
            body=perception_body,
            headers=_interactive_headers(port, "perception"),
        )
        response = connection.getresponse()
        assert response.status == 200
        result = json.loads(response.read())
        assert result["action"] == "perception"
        assert result["status"] == "succeeded"
        assert result["busy"] is False
        assert result["session"]["read_only"] is True
        assert interactive.perception_targets == ["白色 USB 适配器"]

        connection.request(
            "POST",
            "/api/sessions/planning",
            body=b"{}",
            headers=_interactive_headers(port, "planning"),
        )
        response = connection.getresponse()
        assert response.status == 200
        result = json.loads(response.read())
        assert result["action"] == "planning"
        assert result["attempt"]["status"] == "succeeded"
        assert interactive.planning_calls == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_depth_servo_api_has_separate_shadow_live_and_stop_actions(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

    approach = _FakeApproachRunner()
    interactive = _FakeInteractiveService()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        approach_runner=approach,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/api/approach/status")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["running"] is False

        connection.request(
            "POST",
            "/api/approach/start",
            body=b'{"mode":"shadow"}',
            headers=_interactive_headers(port, "approach-start"),
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["approach"]["mode"] == "shadow"
        assert approach.mode == "shadow"
        assert approach.options == {
            "target": None,
            "acquire_target": False,
            "auto_handoff": False,
            "operator_present": False,
            "speed_percent": 5,
        }

        connection.request(
            "POST",
            "/api/approach/stop",
            body=b"{}",
            headers=_interactive_headers(port, "approach-stop"),
        )
        response = connection.getresponse()
        assert response.status == 200
        stopped = json.loads(response.read())
        assert stopped["approach"]["running"] is False
        assert stopped["task_context_clear_requested"] is True
        assert approach.stops == 1
        deadline = time.monotonic() + 1.0
        while interactive.clear_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert interactive.clear_calls == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_depth_servo_api_accepts_server_owned_automatic_workflow(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

    approach = _FakeApproachRunner()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=_FakeInteractiveService(),
        grasp_runner=_FakeGraspRunner(),
        approach_runner=approach,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        payload = json.dumps({
            "mode": "live",
            "target": "白色充电器",
            "acquire_target": True,
            "auto_handoff": True,
            "operator_present": True,
            "speed_percent": 20,
        }).encode()
        connection.request(
            "POST",
            "/api/approach/start",
            body=payload,
            headers=_interactive_headers(port, "approach-start"),
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert approach.options == {
            "target": "白色充电器",
            "acquire_target": True,
            "auto_handoff": True,
            "operator_present": True,
            "speed_percent": 20,
        }
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _servo_status_script(path: Path, phase: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        f"phase = {phase!r}\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema':'z_manip.depth_servo_status.v1','running':True,'phase':phase,"
        "'updated_unix_ns':time.time_ns()}))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _servo_phase_sequence_script(path: Path, phases: list[str]) -> Path:
    counter = path.with_suffix(".count")
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        f"phases = {phases!r}\n"
        f"counter = pathlib.Path({str(counter)!r})\n"
        "count = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(count + 1))\n"
        "phase = phases[min(count, len(phases) - 1)]\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema':'z_manip.depth_servo_status.v1','running':True,'phase':phase,"
        "'updated_unix_ns':time.time_ns()}))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _servo_posture_stall_script(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema':'z_manip.depth_servo_status.v1',"
        "'running':True,'phase':'posture_adjust',"
        "'reactive':{'arm_view':{'mode':'track'}},"
        "'posture_status':{'age_s':9.0,'document':None},"
        "'output':{'published_linear_x':0.0,'published_angular_z':0.0}}))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _servo_with_child_script(path: Path) -> Path:
    child_pid_path = path.with_suffix(".child-pid")
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "sleep 30 &\n"
        "child=$!\n"
        f"printf '%s' \"$child\" > {str(child_pid_path)!r}\n"
        "wait \"$child\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _servo_frozen_heartbeat_script(path: Path, phase: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        f"phase = {phase!r}\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema':'z_manip.depth_servo_status.v1','running':True,'phase':phase,"
        "'updated_unix_ns':time.time_ns()}))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_depth_servo_server_hands_reached_target_to_grasp(tmp_path):
    grasp = _FakeGraspRunner()
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "reached"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=_FakeInteractiveService(),
        grasp_runner=grasp,
    )

    result = runner.start(
        "live",
        target="charger",
        auto_handoff=True,
        speed_percent=17,
    )
    deadline = time.monotonic() + 3.0
    while grasp.starts == 0 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["started"] is True
    assert grasp.starts == 1
    assert grasp.mobile_handoff_starts == 1
    assert grasp.target == "charger"
    assert grasp.speed_percent == 17
    assert isinstance(grasp.base_stopped_unix_ns, int)
    assert isinstance(grasp.base_stopped_monotonic_ns, int)
    assert runner.status()["workflow"]["phase"] == "grasp_preparing"


def test_depth_servo_stop_terminates_launcher_process_group(tmp_path):
    script = _servo_with_child_script(tmp_path / "servo.sh")
    runner = CONTROL.DepthServoRunner(
        script,
        tmp_path / "status.json",
        tmp_path / "servo.log",
    )

    assert runner.start("shadow")["started"] is True
    child_pid_path = script.with_suffix(".child-pid")
    deadline = time.monotonic() + 2.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text())

    assert runner.stop()["stopped"] is True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("depth-servo child survived Full Stop")


def test_depth_servo_handoff_phases_stop_base_before_fresh_grasp(tmp_path):
    for phase in ("handoff_probe", "handoff_ready"):
        phase_dir = tmp_path / phase
        phase_dir.mkdir()
        grasp = _FakeGraspRunner()
        runner = CONTROL.DepthServoRunner(
            _servo_status_script(phase_dir / "servo.py", phase),
            phase_dir / "status.json",
            phase_dir / "servo.log",
            session_service=_FakeInteractiveService(),
            grasp_runner=grasp,
        )

        result = runner.start(
            "live",
            target="charger",
            auto_handoff=True,
            speed_percent=12,
        )
        deadline = time.monotonic() + 3.0
        while grasp.starts == 0 and time.monotonic() < deadline:
            time.sleep(0.02)

        assert result["started"] is True
        assert grasp.starts == 1
        assert grasp.mobile_handoff_starts == 1
        assert isinstance(grasp.base_stopped_unix_ns, int)
        assert runner._process is not None
        assert runner._process.poll() is not None
        assert isinstance(grasp.base_stopped_monotonic_ns, int)
        assert runner.status()["workflow"]["phase"] == "grasp_preparing"


def test_depth_servo_supervisor_accepts_structured_handoff_evidence(tmp_path):
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
    )

    assert runner._runtime_requests_handoff({"phase": "approach"}) is False
    assert runner._runtime_requests_handoff({
        "phase": "approach",
        "output": {"needs_ik_probe": True},
    }) is True
    assert runner._runtime_requests_handoff({
        "phase": "whole_body_approach",
        "reactive": {"phase": "handoff_ready"},
    }) is True


def test_depth_servo_server_reacquires_after_bounded_tracking_loss(tmp_path):
    interactive = _FakeInteractiveService()
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "tracking_lost"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=interactive,
    )

    result = runner.start("shadow", target="charger")
    deadline = time.monotonic() + 3.0
    while not interactive.perception_targets and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["started"] is True
    assert interactive.perception_targets == ["charger"]
    assert runner.status()["workflow"]["reacquisition_attempts"] == 1
    runner.stop()


def _acquire_runner(tmp_path, sessions, search):
    return CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=sessions,
        wrist_search=search,
    )


class _RecordingSearch:
    def __init__(self, found, events, status):
        self.found = found
        self.events = events
        self._status = status
        self.calls = []

    def run(self, target, *, mode, speed_percent, cancel, operator_present=False):
        self.events.append("search")
        self.calls.append((
            target, mode, speed_percent, cancel.is_set(), operator_present,
        ))
        return self.found

    def status(self):
        return dict(self._status)

    def stop(self):
        return None


def test_depth_servo_confirms_current_view_perception_without_wrist_motion(tmp_path):
    events = []

    class Sessions:
        def __init__(self):
            self.calls = 0

        def start_perception(self, target):
            self.calls += 1
            events.append("perception")
            return {"status": "succeeded", "target": target}

    sessions = Sessions()
    search = _RecordingSearch(True, events, {"phase": "idle", "failure": None})
    runner = _acquire_runner(tmp_path, sessions, search)

    result = runner.start(
        "shadow",
        target="charger",
        acquire_target=True,
        speed_percent=8,
    )
    deadline = time.monotonic() + 3.0
    while sessions.calls < 1 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["started"] is True
    assert sessions.calls == 1
    assert events == ["perception"]
    assert search.calls == []
    runner.stop()


def test_depth_servo_current_view_miss_falls_back_to_bounded_wrist_search(tmp_path):
    events = []

    class Sessions:
        def __init__(self):
            self.calls = 0

        def start_perception(self, target):
            self.calls += 1
            events.append("perception")
            return {"status": "failed", "target": target}

    sessions = Sessions()
    search = _RecordingSearch(
        False,
        events,
        {"phase": "exhausted", "failure": "bounded local search exhausted"},
    )
    runner = _acquire_runner(tmp_path, sessions, search)

    result = runner.start(
        "shadow",
        target="charger",
        acquire_target=True,
        speed_percent=8,
    )
    deadline = time.monotonic() + 3.0
    while runner.status()["workflow"]["active"] and time.monotonic() < deadline:
        time.sleep(0.01)

    status = runner.status()
    assert result["started"] is True
    assert events == ["perception", "search"]
    assert sessions.calls == 1
    assert status["workflow"]["phase"] == "blocked"
    assert status["workflow"]["failure"] == "bounded local search exhausted"
    assert runner._process is None


def test_depth_servo_wrist_search_find_seeds_second_perception(tmp_path):
    events = []

    class Sessions:
        def __init__(self):
            self.calls = 0

        def start_perception(self, target):
            self.calls += 1
            events.append("perception")
            status = "failed" if self.calls == 1 else "succeeded"
            return {"status": status, "target": target}

    sessions = Sessions()
    search = _RecordingSearch(True, events, {"phase": "found", "failure": None})
    runner = _acquire_runner(tmp_path, sessions, search)

    result = runner.start(
        "shadow",
        target="charger",
        acquire_target=True,
        speed_percent=8,
    )
    deadline = time.monotonic() + 3.0
    while sessions.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["started"] is True
    assert events == ["perception", "search", "perception"]
    assert search.calls == [("charger", "shadow", 8, False, False)]
    assert runner.status()["wrist_search"]["phase"] == "found"
    runner.stop()


def test_depth_servo_view_recovery_stops_base_before_wrist_search(tmp_path):
    for recovery_phase in ("view_recovery", "search_required"):
        phase_dir = tmp_path / recovery_phase
        phase_dir.mkdir()

        class Search:
            def __init__(self):
                self.calls = 0
                self.base_was_stopped = False

            def run(
                self,
                target,
                *,
                mode,
                speed_percent,
                cancel,
                operator_present=False,
            ):
                self.calls += 1
                process = runner._process
                self.base_was_stopped = (
                    process is not None and process.poll() is not None
                )
                return True

            def status(self):
                return {"phase": "found", "failure": None}

            def stop(self):
                return None

        interactive = _FakeInteractiveService()
        search = Search()
        runner = CONTROL.DepthServoRunner(
            _servo_phase_sequence_script(
                phase_dir / "servo.py",
                [recovery_phase, "base_approach"],
            ),
            phase_dir / "status.json",
            phase_dir / "servo.log",
            session_service=interactive,
            wrist_search=search,
        )

        result = runner.start(
            "shadow",
            target="charger",
            operator_present=True,
            speed_percent=9,
        )
        deadline = time.monotonic() + 3.0
        while search.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        while not interactive.perception_targets and time.monotonic() < deadline:
            time.sleep(0.02)

        assert result["started"] is True
        assert search.calls == 1
        assert search.base_was_stopped is True
        assert interactive.perception_targets == ["charger"]
        assert runner.status()["workflow"]["reacquisition_attempts"] == 1
        runner.stop()


def test_depth_servo_full_stop_is_idempotent_when_already_stopped(tmp_path):
    script = tmp_path / "servo.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({
        "schema": "z_manip.depth_servo_status.v1",
        "running": True,
        "phase": "approach",
    }), encoding="utf-8")
    runner = CONTROL.DepthServoRunner(
        script,
        status_path,
        tmp_path / "servo.log",
    )

    result = runner.stop()

    assert result["stopped"] is True
    assert result["approach"]["running"] is False
    assert result["approach"]["phase"] == "idle"
    assert not status_path.exists()


def test_depth_servo_posture_wait_degrades_instead_of_waiting_forever(tmp_path):
    grasp = _FakeGraspRunner()
    runner = CONTROL.DepthServoRunner(
        _servo_posture_stall_script(tmp_path / "servo.py"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=_FakeInteractiveService(),
        grasp_runner=grasp,
        posture_wait_timeout_s=0.15,
    )

    result = runner.start("shadow", target="charger")
    deadline = time.monotonic() + 3.0
    while runner.status()["phase"] != "degraded" and time.monotonic() < deadline:
        time.sleep(0.02)
    status = runner.status()

    assert result["started"] is True
    assert status["running"] is False
    assert status["phase"] == "degraded"
    assert status["supervision"]["code"] == "POSTURE_FEEDBACK_TIMEOUT"
    assert status["supervision"]["owners"]["arm_view"] == "intent_only"
    assert runner._process is not None and runner._process.poll() is not None
    assert grasp.starts == 0


def test_live_depth_servo_process_with_frozen_status_is_terminated(tmp_path):
    grasp = _FakeGraspRunner()
    runner = CONTROL.DepthServoRunner(
        _servo_frozen_heartbeat_script(tmp_path / "servo.py", "base_approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=_FakeInteractiveService(),
        grasp_runner=grasp,
        state_heartbeat_timeout_s=0.15,
    )

    result = runner.start("live", target="charger", auto_handoff=True)
    deadline = time.monotonic() + 3.0
    while runner.status()["phase"] != "degraded" and time.monotonic() < deadline:
        time.sleep(0.02)
    status = runner.status()

    assert result["started"] is True
    assert status["running"] is False
    assert status["phase"] == "degraded"
    assert status["supervision"]["code"] == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"
    assert (
        status["supervision"]["heartbeat_elapsed_s"] >= 0.15
        or status["supervision"]["heartbeat_age_s"] >= 0.15
    )
    assert runner._process is not None and runner._process.poll() is not None
    assert grasp.starts == 0


def test_stale_handoff_status_can_never_start_grasp(tmp_path):
    grasp = _FakeGraspRunner()
    script = tmp_path / "servo.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema':'z_manip.depth_servo_status.v1','running':True,"
        "'phase':'handoff_ready','updated_unix_ns':1}))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    runner = CONTROL.DepthServoRunner(
        script,
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=_FakeInteractiveService(),
        grasp_runner=grasp,
        state_heartbeat_timeout_s=0.15,
    )

    result = runner.start("live", target="charger", auto_handoff=True)
    deadline = time.monotonic() + 3.0
    while runner.status()["phase"] != "degraded" and time.monotonic() < deadline:
        time.sleep(0.02)
    status = runner.status()

    assert result["started"] is True
    assert status["phase"] == "degraded"
    assert status["supervision"]["code"] == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"
    assert grasp.starts == 0


def test_interactive_session_post_security_and_fields_fail_closed(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("legacy runner must remain isolated")

    interactive = _FakeInteractiveService()
    run_root = tmp_path / "interactive"
    (run_root / "perception").mkdir(parents=True)
    (run_root / "planning").mkdir()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        interactive_run_root=run_root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def request(body: bytes, headers: dict[str, str], path: str = "/api/sessions/perception"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        headers = _interactive_headers(port, "perception")
        without_origin = dict(headers)
        without_origin.pop("Origin")
        status, document = request(b'{"target":"object"}', without_origin)
        assert status == 403
        assert document["error"]["code"] == "CROSS_ORIGIN_FORBIDDEN"

        wrong_action = dict(headers)
        wrong_action["X-Z-Manip-Action"] = "planning"
        status, document = request(b'{"target":"object"}', wrong_action)
        assert status == 403
        assert document["error"]["code"] == "ACTION_HEADER_REQUIRED"

        wrong_type = dict(headers)
        wrong_type["Content-Type"] = "text/plain"
        status, document = request(b'{"target":"object"}', wrong_type)
        assert status == 415
        assert document["error"]["code"] == "INVALID_CONTENT_TYPE"

        status, document = request(
            b'{"target":"object","path":"/tmp/input"}',
            headers,
        )
        assert status == 400
        assert document["error"]["code"] == "INVALID_ACTION_FIELDS"

        status, document = request(
            b'{"target":"object","target":"other"}',
            headers,
        )
        assert status == 400
        assert document["error"]["code"] == "INVALID_JSON"

        status, document = request(
            b'{"target":"../object"}',
            headers,
        )
        assert status == 400
        assert document["error"]["code"] == "INVALID_TARGET_PATH"

        status, document = request(
            b'{"command":"run"}',
            _interactive_headers(port, "planning"),
            path="/api/sessions/planning",
        )
        assert status == 400
        assert document["error"]["code"] == "INVALID_ACTION_FIELDS"

        status, document = request(
            b"{}",
            _interactive_headers(port, "planning"),
            path="/api/sessions/planning?path=/tmp/run",
        )
        assert status == 400
        assert document["error"]["code"] == "QUERY_FORBIDDEN"

        status, document = request(b"x" * 513, headers)
        assert status == 413
        assert document["error"]["code"] == "INVALID_BODY_SIZE"

        assert interactive.perception_targets == []
        assert interactive.planning_calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_grasp_api_accepts_target_driven_full_action_without_selected_perception(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("fixed grasp must not invoke the planning runner")

    interactive = _FakeInteractiveService()
    grasp = _FakeGraspRunner()
    run_root = tmp_path / "interactive"
    (run_root / "perception").mkdir(parents=True)
    (run_root / "planning").mkdir()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        interactive_run_root=run_root,
        grasp_runner=grasp,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/api/grasp/status")
        response = connection.getresponse()
        assert response.status == 200
        status = json.loads(response.read())
        assert status["schema"] == "z_manip.grasp_action.v1"
        assert status["running"] is False
        assert grasp.starts == 0

        connection.request(
            "POST",
            "/api/grasp",
            body=b'{"target":"white adapter"}',
            headers=_interactive_headers(port, "grasp"),
        )
        response = connection.getresponse()
        assert response.status == 202
        result = json.loads(response.read())
        assert result["started"] is True
        assert result["grasp"]["running"] is True
        assert grasp.starts == 1
        assert grasp.target == "white adapter"

        grasp.running = False
        connection.request(
            "POST",
            "/api/grasp",
            body=b'{"target":"white adapter","speed_percent":10}',
            headers=_interactive_headers(port, "grasp"),
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert grasp.speed_percent == 10

        grasp.running = False
        connection.request(
            "POST",
            "/api/grasp",
            body=b'{"target":"white adapter","speed_percent":50}',
            headers=_interactive_headers(port, "grasp"),
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert grasp.speed_percent == 50

        grasp.running = False
        connection.request(
            "POST",
            "/api/grasp/selected",
            body=b'{"speed_percent":7}',
            headers=_interactive_headers(port, "grasp-selected"),
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert grasp.selected_starts == 1
        assert grasp.speed_percent == 7

        connection.request("GET", "/api/grasp/status")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["running"] is True
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_full_grasp_runner_perceives_target_exactly_once_before_planning(tmp_path):
    class FakeSessions:
        def __init__(self):
            self.perception_targets = []
            self.planning_calls = 0
            self.clear_calls = 0

        def start_perception(self, target):
            self.perception_targets.append(target)
            return {"status": "succeeded", "session_id": "20260720-120000"}

        def start_planning(self):
            self.planning_calls += 1
            return {"status": "succeeded", "session_id": "20260720-120001"}

        def clear_current_context(self):
            self.clear_calls += 1

    sessions = FakeSessions()
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = sessions
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "home",
        "outcome": None,
    }
    runner._wait_home = lambda speed_percent: "fresh_read_only_joint_feedback"
    runner._planning_artifacts = lambda attempt: (
        tmp_path / "planning_report.json",
        tmp_path / "planned_grasp.npz",
    )
    runner._run_full = lambda **kwargs: None

    runner._run("white adapter", 7)

    assert sessions.perception_targets == ["white adapter"]
    assert sessions.planning_calls == 1
    assert sessions.clear_calls == 1
    assert runner.status()["outcome"] == "passed"
    assert runner.status()["phase"] == "returned_home"


def test_mobile_handoff_invalidates_old_capture_and_plans_from_fresh_session(tmp_path):
    events: list[object] = []

    class FakeSessions:
        def clear_current_context(self):
            events.append("clear")

        def start_perception(self, target):
            events.append(("perception", target))
            return {"status": "succeeded", "session_id": "20260722-120010"}

        def start_planning(self):
            events.append("planning")
            return {"status": "succeeded", "session_id": "20260722-120011"}

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = FakeSessions()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner._planning_artifacts = lambda attempt: (
        tmp_path / "planning_report.json",
        tmp_path / "planned_grasp.npz",
    )

    class FreshJoints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            events.append("joint_ready")
            return True, "fresh", {
                "sequence": 42,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner.home_verifier = FreshJoints()
    runner._validate_mobile_handoff_capture_evidence = lambda **_kwargs: {
        "validated": True,
    }

    def execute(**kwargs):
        events.append(("execute", kwargs["speed_percent"]))
        return _executor_start_evidence()

    runner._run_full = execute
    # No home_runner/_wait_home is installed: a call to either would fail the
    # test and prove that the handoff discarded the current arm/view pose.
    runner._run_mobile_handoff(
        "floor bottle",
        9,
        "20260722-115900",
        1_800_000_000_000_000_000,
    )

    assert events[0] == "clear"
    assert set(events[1:3]) == {"joint_ready", ("perception", "floor bottle")}
    assert events[3:] == ["planning", ("execute", 9), "clear"]
    status = runner.status()
    assert status["outcome"] == "passed"
    assert status["phase"] == "returned_home"
    assert status["fresh_perception_session_id"] == "20260722-120010"
    assert status["planning_session_id"] == "20260722-120011"
    assert status["handoff_joint_evidence"]["sequence"] == 42


def test_mobile_handoff_surfaces_need_base_approach_without_execution(tmp_path):
    events: list[object] = []

    class FakeSessions:
        def clear_current_context(self):
            events.append("clear")

        def start_perception(self, target):
            events.append(("perception", target))
            return {"status": "succeeded", "session_id": "20260722-120020"}

        def start_planning(self):
            events.append("planning")
            return {
                "status": "blocked",
                "session_id": "20260722-120021",
                "error": {
                    "code": "NEED_BASE_APPROACH",
                    "message": "target remains outside handoff workspace",
                },
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = FakeSessions()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner.home_verifier = type("FreshJoints", (), {
        "current_joint_snapshot": lambda self, *, not_before_unix_ns: (
            True,
            "fresh",
            {
                "sequence": 44,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            },
        ),
    })()
    runner._validate_mobile_handoff_capture_evidence = lambda **_kwargs: {
        "validated": True,
    }
    runner._planning_artifacts = lambda _attempt: (_ for _ in ()).throw(
        AssertionError("typed base-approach disposition must bypass artifacts"),
    )
    runner._run_full = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("typed base-approach disposition must not execute"),
    )

    runner._run_mobile_handoff(
        "floor bottle",
        5,
        "20260722-120010",
        1_800_000_000_000_000_000,
    )

    assert events == ["clear", ("perception", "floor bottle"), "planning"]
    status = runner.status()
    assert status["running"] is False
    assert status["outcome"] == "recoverable"
    assert status["phase"] == "needs_base_approach"
    assert status["planning_disposition"] == "NEED_BASE_APPROACH"
    assert status["recovery_action"] == "approach_only"
    assert status["retryable"] is True
    assert "no execution was started" in status["message"]


def test_mobile_handoff_overlaps_joint_wait_with_fresh_perception(tmp_path):
    joint_started = threading.Event()
    perception_started = threading.Event()

    class Sessions:
        def clear_current_context(self):
            return None

        def start_perception(self, _target):
            perception_started.set()
            assert joint_started.wait(timeout=0.25)
            time.sleep(0.04)
            return {"status": "succeeded", "session_id": "20260722-120030"}

        def start_planning(self):
            return {"status": "succeeded", "session_id": "20260722-120031"}

    class Joints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            joint_started.set()
            assert perception_started.wait(timeout=0.25)
            time.sleep(0.04)
            return True, "fresh", {
                "sequence": 45,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = Sessions()
    runner.home_verifier = Joints()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner._validate_mobile_handoff_capture_evidence = lambda **_kwargs: {
        "validated": True,
    }
    runner._planning_artifacts = lambda _attempt: (
        tmp_path / "planning_report.json",
        tmp_path / "planned_grasp.npz",
    )
    runner._run_full = lambda **_kwargs: _executor_start_evidence()

    runner._run_mobile_handoff(
        "floor bottle",
        5,
        "20260722-120029",
        1_800_000_000_000_000_000,
    )

    status = runner.status()
    assert status["outcome"] == "passed"
    assert status["timings_s"]["handoff_capture_parallel"] < 0.2
    assert status["timings_s"]["handoff_joint_ready"] >= 0.04
    assert status["timings_s"]["handoff_perception"] >= 0.04


def test_mobile_handoff_capture_gate_accepts_one_coherent_post_stop_epoch(tmp_path):
    boundary = 1_800_000_000_000_000_000
    session_id = "20260722-120040"
    session_root = tmp_path / "sessions"
    perception_root = session_root / "perception" / session_id / "perception"
    perception_root.mkdir(parents=True)
    observation_start = boundary + 100_000_000
    first_feedback = boundary + 110_000_000
    last_feedback = boundary + 190_000_000
    observation_end = boundary + 200_000_000
    report_stamp = boundary + 210_000_000
    (perception_root / "selected_passive_joint_report.json").write_text(
        json.dumps({
            "schema": "z_manip.piper_passive_joint_report.v1",
            "read_only": True,
            "complete_joint_feedback": True,
            "zero_transmit_verified": True,
            "interface_tx_packet_delta": 0,
            "observation_start_unix_ns": observation_start,
            "first_feedback_unix_ns": first_feedback,
            "last_feedback_unix_ns": last_feedback,
            "observation_end_unix_ns": observation_end,
        }),
        encoding="utf-8",
    )
    (perception_root / "report.json").write_text(
        json.dumps({
            "instruction": "floor bottle",
            "read_only": True,
            "grasp_generation_valid": True,
            "stamp_ns": report_stamp,
            "passive_capture": {
                "synchronized": True,
                "observation_start_unix_ns": observation_start,
                "observation_end_unix_ns": observation_end,
                "selected_stamp_ns": report_stamp,
            },
        }),
        encoding="utf-8",
    )
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.session_run_root = session_root

    evidence = runner._validate_mobile_handoff_capture_evidence(
        perception_session_id=session_id,
        target="floor bottle",
        base_stopped_unix_ns=boundary,
        joint_evidence={
            "source_timestamp_ns": boundary + 150_000_000,
            "joint_positions_rad": [0.0] * 6,
            "read_only": True,
            "motion_commands_published": 0,
        },
    )

    assert evidence["validated"] is True
    assert evidence["observer_passive_skew_ms"] == 0.0
    assert evidence["zero_transmit_verified"] is True


def test_mobile_handoff_capture_gate_rejects_pre_stop_passive_evidence(tmp_path):
    boundary = 1_800_000_000_000_000_000
    session_id = "20260722-120050"
    session_root = tmp_path / "sessions"
    perception_root = session_root / "perception" / session_id / "perception"
    perception_root.mkdir(parents=True)
    (perception_root / "selected_passive_joint_report.json").write_text(
        json.dumps({
            "schema": "z_manip.piper_passive_joint_report.v1",
            "read_only": True,
            "complete_joint_feedback": True,
            "zero_transmit_verified": True,
            "interface_tx_packet_delta": 0,
            "observation_start_unix_ns": boundary,
            "first_feedback_unix_ns": boundary + 1,
            "last_feedback_unix_ns": boundary + 2,
            "observation_end_unix_ns": boundary + 3,
        }),
        encoding="utf-8",
    )
    (perception_root / "report.json").write_text(
        json.dumps({
            "instruction": "floor bottle",
            "read_only": True,
            "grasp_generation_valid": True,
            "stamp_ns": boundary + 4,
            "passive_capture": {
                "synchronized": True,
                "observation_start_unix_ns": boundary,
                "observation_end_unix_ns": boundary + 3,
                "selected_stamp_ns": boundary + 4,
            },
        }),
        encoding="utf-8",
    )
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.session_run_root = session_root

    with pytest.raises(RuntimeError, match="strictly post-stop"):
        runner._validate_mobile_handoff_capture_evidence(
            perception_session_id=session_id,
            target="floor bottle",
            base_stopped_unix_ns=boundary,
            joint_evidence={
                "source_timestamp_ns": boundary + 2,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
                "motion_commands_published": 0,
            },
        )


def test_mobile_handoff_never_plans_when_parallel_evidence_gate_fails(tmp_path):
    class Sessions:
        def clear_current_context(self):
            return None

        def start_perception(self, _target):
            return {"status": "succeeded", "session_id": "20260722-120059"}

        def start_planning(self):
            raise AssertionError("invalid parallel evidence must block planning")

    boundary = 1_800_000_000_000_000_000
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = Sessions()
    runner.home_verifier = type("FreshJoints", (), {
        "current_joint_snapshot": lambda self, *, not_before_unix_ns: (
            True,
            "fresh",
            {
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            },
        ),
    })()
    runner._validate_mobile_handoff_capture_evidence = lambda **_kwargs: (
        (_ for _ in ()).throw(RuntimeError("evidence epochs disagree"))
    )
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }

    runner._run_mobile_handoff(
        "floor bottle",
        5,
        "20260722-120058",
        boundary,
    )

    status = runner.status()
    assert status["outcome"] == "blocked"
    assert "evidence epochs disagree" in status["message"]


def test_depth_servo_mirrors_recoverable_handoff_disposition(tmp_path):
    class RecoverableGrasp:
        def status(self):
            return {
                "running": False,
                "phase": "needs_base_approach",
                "outcome": "recoverable",
                "planning_disposition": "NEED_BASE_APPROACH",
            }

    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "idle"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        grasp_runner=RecoverableGrasp(),
    )
    boundary = 1_800_000_000_000_000_000
    runner._set_workflow(
        active=False,
        phase="grasp_started",
        handoff_boundary_unix_ns=boundary,
    )

    runner._watch_mobile_handoff(
        handoff_boundary_unix_ns=boundary,
        cancel=threading.Event(),
        timeout_s=0.1,
    )

    status = runner.status()
    assert status["phase"] == "needs_base_approach"
    assert status["workflow"]["planning_disposition"] == "NEED_BASE_APPROACH"
    assert status["workflow"]["recovery_action"] == "approach_only"
    assert status["workflow"]["failure"] is None
    assert "resume Approach Only" in status["message"]


@pytest.mark.parametrize(
    ("grasp_status", "expected_phase", "expected_failure"),
    [
        (
            {
                "running": False,
                "phase": "returned_home",
                "outcome": "passed",
                "message": "fresh handoff completed",
            },
            "grasp_completed",
            None,
        ),
        (
            {
                "running": False,
                "phase": "handoff_planning",
                "outcome": "blocked",
                "message": "fresh planning did not produce a valid plan",
            },
            "blocked",
            "fresh planning did not produce a valid plan",
        ),
    ],
)
def test_depth_servo_mirrors_terminal_handoff_result(
    tmp_path,
    grasp_status,
    expected_phase,
    expected_failure,
):
    class TerminalGrasp:
        def status(self):
            return dict(grasp_status)

    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "idle"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        grasp_runner=TerminalGrasp(),
    )
    boundary = 1_800_000_000_000_000_001
    runner._set_workflow(
        active=False,
        phase="grasp_started",
        handoff_boundary_unix_ns=boundary,
    )

    runner._watch_mobile_handoff(
        handoff_boundary_unix_ns=boundary,
        cancel=threading.Event(),
        timeout_s=0.1,
    )

    status = runner.status()
    assert status["running"] is False
    assert status["phase"] == expected_phase
    assert status["workflow"]["failure"] == expected_failure
    assert status["message"] == grasp_status["message"]


@pytest.mark.parametrize("grasp_running", [False, True])
def test_depth_servo_rejects_second_approach_during_mobile_handoff(
    tmp_path,
    grasp_running,
):
    class HandoffGrasp:
        def status(self):
            return {"running": grasp_running, "phase": "handoff_planning"}

    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "idle"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        grasp_runner=HandoffGrasp(),
    )
    runner._set_workflow(
        active=False,
        phase="grasp_started",
        handoff_boundary_unix_ns=1_800_000_000_000_000_002,
    )

    result = runner.start(
        "live",
        target="floor bottle",
        acquire_target=True,
        auto_handoff=True,
        operator_present=True,
    )

    assert result["started"] is False
    assert result["error"]["code"] == "APPROACH_ACTION_BUSY"
    assert result["approach"]["running"] is True
    assert result["approach"]["phase"] == "grasp_started"


def test_depth_servo_recovery_start_clears_typed_handoff_state(tmp_path):
    class IdleGrasp:
        def status(self):
            return {"running": False, "phase": "needs_base_approach"}

    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "idle"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        grasp_runner=IdleGrasp(),
    )
    runner._set_workflow(
        active=False,
        phase="needs_base_approach",
        planning_disposition="NEED_BASE_APPROACH",
        recovery_action="approach_only",
        handoff_boundary_unix_ns=1_800_000_000_000_000_003,
    )

    result = runner.start(
        "shadow",
        target="floor bottle",
        acquire_target=False,
        auto_handoff=False,
    )

    assert result["started"] is True
    workflow = result["approach"]["workflow"]
    assert workflow["planning_disposition"] is None
    assert workflow["recovery_action"] is None
    assert "handoff_boundary_unix_ns" not in workflow
    runner.stop()


def test_mobile_handoff_rejects_reused_pre_servo_perception_session(tmp_path):
    class FakeSessions:
        def clear_current_context(self):
            return None

        def start_perception(self, _target):
            return {"status": "succeeded", "session_id": "20260722-120010"}

        def start_planning(self):
            raise AssertionError("reused perception must never reach planning")

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = FakeSessions()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner.home_verifier = type("FreshJoints", (), {
        "current_joint_snapshot": lambda self, *, not_before_unix_ns: (
            True,
            "fresh",
            {
                "sequence": 43,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            },
        ),
    })()
    runner._run_mobile_handoff(
        "floor bottle",
        5,
        "20260722-120010",
        1_800_000_000_000_000_000,
    )

    status = runner.status()
    assert status["outcome"] == "blocked"
    assert "reused the pre-servo perception session" in status["message"]


def test_mobile_handoff_blocks_before_capture_without_post_stop_joints(tmp_path):
    class Sessions:
        def clear_current_context(self):
            raise AssertionError("stale joints must block before clearing/capture")

    class StaleJoints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            return False, "predates stop", {
                "source_timestamp_ns": not_before_unix_ns - 1,
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = Sessions()
    runner.home_verifier = StaleJoints()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }

    try:
        runner._wait_mobile_handoff_joints(
            not_before_unix_ns=1_800_000_000_000_000_000,
            timeout_s=0.01,
        )
    except RuntimeError as error:
        assert "not observed after base stop" in str(error)
    else:
        raise AssertionError("stale passive feedback unexpectedly passed")


def test_mobile_handoff_joint_wait_polls_quickly_without_relaxing_epoch(tmp_path, monkeypatch):
    boundary = 1_800_000_000_000_000_000
    calls = []

    class DelayedFreshJoints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            calls.append(not_before_unix_ns)
            if len(calls) < 3:
                return False, "predates stop", {
                    "source_timestamp_ns": not_before_unix_ns - 1,
                    "joint_positions_rad": [0.0] * 6,
                    "read_only": True,
                }
            return True, "fresh", {
                "sequence": 44,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    sleeps = []
    monkeypatch.setattr(CONTROL.time, "sleep", sleeps.append)
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.home_verifier = DelayedFreshJoints()

    evidence = runner._wait_mobile_handoff_joints(
        not_before_unix_ns=boundary,
        timeout_s=1.0,
    )

    assert calls == [boundary, boundary, boundary]
    assert sleeps == [
        CONTROL.MOBILE_HANDOFF_JOINT_READY_POLL_S,
        CONTROL.MOBILE_HANDOFF_JOINT_READY_POLL_S,
    ]
    assert CONTROL.MOBILE_HANDOFF_JOINT_READY_POLL_S == 0.01
    assert evidence["source_timestamp_ns"] > boundary
    assert len(evidence["joint_positions_rad"]) == 6
    assert evidence["read_only"] is True


def test_grasp_post_requires_exact_loopback_origin_header_and_target_body(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("invalid grasp requests cannot start planning")

    grasp = _FakeGraspRunner()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        grasp_runner=grasp,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def request(
        body: bytes,
        headers: dict[str, str],
        path: str = "/api/grasp",
    ) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        valid = _interactive_headers(port, "grasp")
        missing_origin = dict(valid)
        missing_origin.pop("Origin")
        status, document = request(b"{}", missing_origin)
        assert status == 403
        assert document["error"]["code"] == "CROSS_ORIGIN_FORBIDDEN"

        alias_origin = dict(valid)
        alias_origin["Origin"] = f"http://localhost:{port}"
        status, document = request(b"{}", alias_origin)
        assert status == 403
        assert document["error"]["code"] == "CROSS_ORIGIN_FORBIDDEN"

        wrong_action = dict(valid)
        wrong_action["X-Z-Manip-Action"] = "planning"
        status, document = request(b"{}", wrong_action)
        assert status == 403
        assert document["error"]["code"] == "ACTION_HEADER_REQUIRED"

        for body in (
            b'{}',
            b'{"speed_percent":5}',
            b'{"path":"/tmp/trajectory.json"}',
            b'{"command":"execute"}',
            b'{"target_joints":[0,0,0,0,0,0]}',
        ):
            status, document = request(body, valid)
            assert status == 400
            assert document["error"]["code"] == "INVALID_ACTION_FIELDS"

        status, document = request(b'{"target":""}', valid)
        assert status == 400
        assert document["error"]["code"] == "INVALID_TARGET_LENGTH"

        status, document = request(b"{}", valid, "/api/grasp?path=/tmp/plan")
        assert status == 400
        assert document["error"]["code"] == "QUERY_FORBIDDEN"

        status, document = request(b"[]", valid)
        assert status == 400
        assert document["error"]["code"] == "INVALID_JSON_OBJECT"

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("GET", "/api/grasp/status?command=run")
            response = connection.getresponse()
            assert response.status == 400
            assert json.loads(response.read())["error"]["code"] == "QUERY_FORBIDDEN"
        finally:
            connection.close()
        assert grasp.starts == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_grasp_home_and_session_actions_are_mutually_exclusive(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("mutual-exclusion checks cannot start planning")

    interactive = _FakeInteractiveService()
    grasp = _FakeGraspRunner()
    home = _FakeHomeRunner()
    run_root = tmp_path / "interactive"
    (run_root / "perception").mkdir(parents=True)
    (run_root / "planning").mkdir()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        interactive_run_root=run_root,
        home_runner=home,
        grasp_runner=grasp,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def post(path: str, action: str, body: bytes = b"{}"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers=_interactive_headers(port, action),
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        interactive.state["busy"] = True
        status, document = post(
            "/api/grasp",
            "grasp",
            b'{"target":"white adapter"}',
        )
        assert status == 409
        assert document["error"]["code"] == "ACTION_BUSY"
        assert grasp.starts == 0

        interactive.state["busy"] = False
        home.running = True
        status, document = post(
            "/api/grasp",
            "grasp",
            b'{"target":"white adapter"}',
        )
        assert status == 409
        assert document["error"]["code"] == "ACTION_BUSY"
        assert grasp.starts == 0

        home.running = False
        grasp.running = True
        status, document = post(
            "/api/sessions/perception",
            "perception",
            b'{"target":"white adapter"}',
        )
        assert status == 409
        assert document["error"]["code"] == "ACTION_BUSY"
        status, document = post("/api/sessions/planning", "planning")
        assert status == 409
        assert document["error"]["code"] == "ACTION_BUSY"
        status, document = post("/api/home", "home")
        assert status == 409
        assert document["error"]["code"] == "ACTION_BUSY"
        assert interactive.perception_targets == []
        assert interactive.planning_calls == 0
        assert home.starts == 0

        # A stale held-object workflow must never lock out the operator's
        # measured Home recovery. Once no physical action is running, Home is
        # accepted and its completion callback can clear the stale workflow.
        grasp.running = False
        grasp.status = lambda: {
            "schema": "z_manip.grasp_action.v1",
            "available": True,
            "running": False,
            "state": "idle",
            "workflow": {"phase": "holding_at_lift", "holding_object": True},
        }
        status, document = post("/api/home", "home")
        assert status == 202
        assert document["started"] is True
        assert home.starts == 1
        assert grasp.home_starts == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_measured_home_clears_stale_grasp_workflow_and_action_lock(tmp_path):
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner._lock = threading.Lock()
    runner._workflow_path = tmp_path / "workflow.json"
    runner._workflow = {
        "phase": "holding_at_lift",
        "artifact_id": "a" * 64,
        "planning_session_id": "20260720-070533",
        "holding_object": True,
        "at_home": False,
        "receipt_dir": "/tmp/old-receipt",
        "planning_report": "/tmp/old-report.json",
        "planned_grasp": "/tmp/old-grasp.npz",
    }
    runner._status = {
        "running": False,
        "state": "finished",
        "phase": "return_home_holding",
        "outcome": "blocked",
        "revision": 7,
        "started_unix_ns": 1,
        "finished_unix_ns": 2,
        "message": "old blocked action",
    }

    runner.reset_after_home()

    status = runner.status()
    assert status["state"] == "idle"
    assert status["phase"] == "idle"
    assert status["outcome"] is None
    assert status["workflow"]["phase"] == "ready_at_home"
    assert status["workflow"]["holding_object"] is False
    assert status["workflow"]["artifact_id"] is None
    persisted = json.loads(runner._workflow_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "ready_at_home"
    assert persisted["holding_object"] is False


def test_home_start_immediately_clears_stale_grasp_workflow(tmp_path):
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner._lock = threading.Lock()
    runner._workflow_path = tmp_path / "workflow.json"
    runner._workflow = {
        "phase": "holding_at_lift",
        "artifact_id": "b" * 64,
        "planning_session_id": "20260720-070534",
        "holding_object": True,
        "at_home": False,
        "receipt_dir": "/tmp/old-receipt",
        "planning_report": "/tmp/old-report.json",
        "planned_grasp": "/tmp/old-grasp.npz",
    }
    runner._status = {
        "running": False,
        "state": "finished",
        "phase": "pick_hold",
        "outcome": "blocked",
        "revision": 2,
        "started_unix_ns": 1,
        "finished_unix_ns": 2,
        "message": "stale blocked state",
    }

    runner.reset_for_home()

    status = runner.status()
    assert status["workflow"]["phase"] == "ready_at_home"
    assert status["workflow"]["holding_object"] is False
    assert status["running"] is False
    assert status["outcome"] is None
    assert status["message"].startswith("Home recovery accepted")


def test_measured_home_verifier_accepts_only_fresh_read_only_joint_feedback(
    tmp_path,
):
    now_ns = 1_800_000_000_000_000_000
    runtime = tmp_path / "runtime.json"
    home = tmp_path / "piper_home.json"
    joints = [0.01, 0.02, -0.03, 0.04, 0.05, 0.0]
    home.write_text(json.dumps({
        "schema": "z_manip.piper_software_home.v1",
        "joint_radians": joints,
    }), encoding="utf-8")
    runtime.write_text(json.dumps({
        "schema": "z_manip.runtime_state.v1",
        "sequence": 1,
        "source_timestamp_ns": now_ns - 100_000_000,
        "joint_state_available": True,
        "joint_positions_rad": [value + 0.001 for value in joints],
        "telemetry": {
            "read_only": True,
            "motion_commands_published": 0,
        },
    }), encoding="utf-8")

    verifier = CONTROL.MeasuredHomeVerifier(runtime, home)
    verifier.reader._clock_ns = lambda: now_ns
    verified, detail = verifier.verify()

    assert verified is True
    assert "verify Home" in detail
    ready, detail, evidence = verifier.current_joint_snapshot(
        not_before_unix_ns=now_ns - 200_000_000,
    )
    assert ready is True
    assert evidence["source_timestamp_ns"] == now_ns - 100_000_000
    assert evidence["joint_positions_rad"] == [value + 0.001 for value in joints]

    ready, detail, evidence = verifier.current_joint_snapshot(
        not_before_unix_ns=now_ns,
    )
    assert ready is False
    assert "predates" in detail

    verifier.reader._clock_ns = lambda: now_ns + 2_000_000_000
    verified, detail = verifier.verify()
    assert verified is False
    assert "not live" in detail


def test_grasp_home_fast_path_never_starts_actuator_home(tmp_path):
    class Verifier:
        def verify(self):
            return True, "fresh measured Home"

    class Home:
        def start(self, _speed):
            raise AssertionError("fresh verified Home must not start actuator recovery")

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.home_verifier = Verifier()
    runner.home_runner = Home()
    runner.log_path = tmp_path / "grasp.log"

    assert runner._wait_home(10) == "fresh_read_only_joint_feedback"
    assert "Home fast verification" in runner.log_path.read_text(encoding="utf-8")


def test_interactive_artifacts_prefer_latest_blocked_bundle_and_are_manifest_bound(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("artifact reads cannot start any action")

    interactive = _FakeInteractiveService()
    run_root = tmp_path / "interactive"
    perception_id = "20260717-170010"
    perception_session = run_root / "perception" / perception_id
    perception_dir = perception_session / "perception"
    perception_dir.mkdir(parents=True)
    pngs = {
        "edgetam_mask.png": b"\x89PNG\r\n\x1a\nmask",
        "edgetam_overlay.png": b"\x89PNG\r\n\x1a\noverlay",
        "grasp_candidates_overlay.png": b"\x89PNG\r\n\x1a\ncandidates",
    }
    for name, payload in pngs.items():
        (perception_dir / name).write_bytes(payload)
    (perception_session / "attempt.json").write_text(json.dumps({
        "schema": "z_manip.interactive_session_attempt.v1",
        "action": "perception",
        "session_id": perception_id,
        "status": "succeeded",
    }), encoding="utf-8")
    (perception_session / "perception_manifest.json").write_text(
        json.dumps(_manifest(pngs)),
        encoding="utf-8",
    )

    planning_id = "20260717-170011"
    planning_session = run_root / "planning" / planning_id
    planning_artifacts = planning_session / "artifacts"
    planning_artifacts.mkdir(parents=True)
    bundle = json.dumps({
        "schema": "z_manip.debug_bundle.v1",
        "run_id": "successful-old",
        "mode": {"read_only": True},
        "safety": {"motion_commands_published": 0},
        "stages": [],
        "artifacts": {},
        "visualization": {},
    }).encode("utf-8")
    (planning_artifacts / "debug_bundle.json").write_bytes(bundle)
    (planning_session / "attempt.json").write_text(json.dumps({
        "schema": "z_manip.interactive_session_attempt.v1",
        "action": "planning",
        "session_id": planning_id,
        "status": "succeeded",
    }), encoding="utf-8")
    (planning_session / "planning_manifest.json").write_text(
        json.dumps(_manifest({"debug_bundle.json": bundle})),
        encoding="utf-8",
    )

    blocked_id = "20260717-170012"
    blocked_session = run_root / "planning" / blocked_id
    blocked_artifacts = blocked_session / "artifacts"
    blocked_artifacts.mkdir(parents=True)
    blocked_bundle = json.dumps({
        "schema": "z_manip.debug_bundle.v1",
        "run_id": "blocked-new",
        "mode": {"read_only": True},
        "safety": {"motion_commands_published": 0},
        "stages": [],
        "artifacts": {},
        "visualization": {},
    }).encode("utf-8")
    (blocked_artifacts / "debug_bundle.json").write_bytes(blocked_bundle)
    (blocked_session / "attempt.json").write_text(json.dumps({
        "schema": "z_manip.interactive_session_attempt.v1",
        "action": "planning",
        "session_id": blocked_id,
        "status": "blocked",
    }), encoding="utf-8")
    (blocked_session / "planning_manifest.json").write_text(
        json.dumps(_manifest({"debug_bundle.json": blocked_bundle})),
        encoding="utf-8",
    )

    interactive.state["selected_perception_session_id"] = perception_id
    interactive.state["actions"] = {
        "perception": {
            "latest_attempt": {"session_id": perception_id, "status": "succeeded"},
            "last_good": {"session_id": perception_id, "status": "succeeded"},
        },
        "planning": {
            "latest_attempt": {"session_id": blocked_id, "status": "blocked"},
            "last_good": {"session_id": planning_id, "status": "succeeded"},
        },
    }
    server = CONTROL.create_server(
        _bundle(tmp_path / "recorded_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        interactive_service=interactive,
        interactive_run_root=run_root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_address[1],
        timeout=3,
    )
    try:
        connection.request("GET", "/api/sessions/perception/artifacts/mask.png")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/png"
        assert response.read() == pngs["edgetam_mask.png"]

        connection.request("GET", "/api/sessions/planning/bundle")
        response = connection.getresponse()
        assert response.status == 200
        served_bundle = json.loads(response.read())
        assert served_bundle["schema"] == "z_manip.debug_bundle.v1"
        assert served_bundle["run_id"] == "blocked-new"

        (blocked_session / "planning_manifest.json").write_text(
            json.dumps(_manifest({})),
            encoding="utf-8",
        )
        connection.request("GET", "/api/sessions/planning/bundle")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["run_id"] == "successful-old"

        (blocked_session / "planning_manifest.json").write_text(
            json.dumps(_manifest({"debug_bundle.json": blocked_bundle})),
            encoding="utf-8",
        )
        (blocked_artifacts / "debug_bundle.json").write_bytes(blocked_bundle + b"tampered")
        connection.request("GET", "/api/sessions/planning/bundle")
        response = connection.getresponse()
        assert response.status == 409
        document = json.loads(response.read())
        assert document["error"]["code"] == "INTERACTIVE_ARTIFACT_UNAVAILABLE"

        (perception_dir / "edgetam_mask.png").write_bytes(
            b"\x89PNG\r\n\x1a\ntampered",
        )
        connection.request("GET", "/api/sessions/perception/artifacts/mask.png")
        response = connection.getresponse()
        assert response.status == 409
        document = json.loads(response.read())
        assert document["error"]["code"] == "INTERACTIVE_ARTIFACT_UNAVAILABLE"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_camera_endpoint_is_fixed_bounded_fresh_conditional_get_and_head(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("camera reads must never start planning")

    camera_path = tmp_path / "camera-latest.jpg"
    jpeg = b"\xff\xd8bounded-camera-frame\xff\xd9"
    camera_path.write_bytes(jpeg)
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        camera_image=camera_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/camera/latest.jpg")
        response = connection.getresponse()
        assert response.status == 200
        etag = response.getheader("ETag")
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Z-Manip-Camera-State") == "live"
        # Real-time RGB tile: the live response advertises the fast poll interval
        # so the browser can refresh the tile at >=10 Hz instead of a flat 5 Hz.
        assert response.getheader("X-Z-Manip-Poll-Interval-Ms") == "80"
        assert response.read() == jpeg

        connection.request("HEAD", "/api/camera/latest.jpg", headers={"If-None-Match": etag})
        response = connection.getresponse()
        assert response.status == 304
        assert response.getheader("ETag") == etag
        assert response.read() == b""

        connection.request("GET", "/api/camera/latest.jpg?path=/etc/passwd")
        response = connection.getresponse()
        assert response.status == 400
        assert b"no query" in response.read()

        connection.request("GET", "/api/camera/other.jpg")
        response = connection.getresponse()
        assert response.status == 404
        response.read()

        stale_ns = time.time_ns() - 3_000_000_000
        os.utime(camera_path, ns=(stale_ns, stale_ns))
        connection.request("GET", "/api/camera/latest.jpg")
        response = connection.getresponse()
        assert response.status == 503
        assert response.getheader("X-Z-Manip-Camera-State") == "stale"
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_camera_reader_rejects_oversized_or_non_jpeg_files(tmp_path):
    path = tmp_path / "camera.jpg"
    path.write_bytes(b"x" * (CONTROL.MAX_CAMERA_JPEG_BYTES + 1))
    status, payload, _etag, _age, message = CONTROL.CameraSnapshotReader(path).snapshot()
    assert status == "invalid"
    assert payload is None
    assert "size limit" in message

    path.write_bytes(b"not-a-jpeg")
    status, payload, _etag, _age, message = CONTROL.CameraSnapshotReader(path).snapshot()
    assert status == "invalid"
    assert payload is None
    assert "JPEG" in message


def test_depth_endpoint_is_fixed_bounded_fresh_conditional_get_and_head(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("depth reads must never start planning")

    depth_path = tmp_path / "depth-latest.jpg"
    jpeg = b"\xff\xd8bounded-depth-frame\xff\xd9"
    depth_path.write_bytes(jpeg)
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        depth_image=depth_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/depth/latest.jpg")
        response = connection.getresponse()
        assert response.status == 200
        etag = response.getheader("ETag")
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Z-Manip-Camera-State") == "live"
        assert response.getheader("X-Z-Manip-Poll-Interval-Ms") == "75"
        assert response.read() == jpeg

        connection.request("HEAD", "/api/depth/latest.jpg", headers={"If-None-Match": etag})
        response = connection.getresponse()
        assert response.status == 304
        assert response.getheader("ETag") == etag
        assert response.read() == b""

        connection.request("GET", "/api/depth/latest.jpg?path=/etc/passwd")
        response = connection.getresponse()
        assert response.status == 400
        assert b"no query" in response.read()

        connection.request("GET", "/api/depth/other.jpg")
        response = connection.getresponse()
        assert response.status == 404
        response.read()

        stale_ns = time.time_ns() - 3_000_000_000
        os.utime(depth_path, ns=(stale_ns, stale_ns))
        connection.request("GET", "/api/depth/latest.jpg")
        response = connection.getresponse()
        assert response.status == 503
        assert response.getheader("X-Z-Manip-Camera-State") == "stale"
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_depth_endpoint_is_offline_without_configuration(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("depth reads must never start planning")

    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/depth/latest.jpg")
        response = connection.getresponse()
        assert response.status == 404
        assert response.getheader("X-Z-Manip-Camera-State") == "offline"
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_workbench_service_passes_only_the_fixed_camera_artifact_path():
    source = SERVICE.read_text(encoding="utf-8")
    assert "--camera-image %h/Z-Robotics-Lab/artifacts/go2w_real/latest/camera-latest.jpg" in source
    assert "--depth-image %h/Z-Robotics-Lab/artifacts/go2w_real/latest/depth-latest.jpg" in source
    assert "--cloud-bin %h/Z-Robotics-Lab/artifacts/go2w_real/latest/cloud-latest.bin" in source
    assert "/api/camera" not in source
    assert "/api/depth" not in source
    assert "/api/cloud" not in source


def _cloud_binary(count: int, *, source_flag: int = 1, stamp_ns: int = 1700, magic: bytes = b"ZMPC") -> bytes:
    import numpy as np

    header = CONTROL.CLOUD_HEADER_STRUCT.pack(magic, 1, source_flag, count, stamp_ns)
    xyz = np.tile(np.array([0.1, -0.2, 1.0], dtype="<f4"), (count, 1))
    rgb = np.tile(np.array([12, 34, 56], dtype=np.uint8), (count, 1))
    return header + xyz.tobytes() + rgb.tobytes()


def test_cloud_reader_rejects_torn_foreign_or_oversized_binaries(tmp_path):
    path = tmp_path / "cloud-latest.bin"

    # A valid ZMPC binary reads live and exposes its source and point count.
    path.write_bytes(_cloud_binary(4, source_flag=1))
    status, payload, etag, _age, source, count, message = CONTROL.CloudSnapshotReader(path).snapshot()
    assert status == "live" and source == "ffs" and count == 4
    assert payload is not None and etag is not None and message == ""

    # Wrong magic is not a cloud.
    path.write_bytes(_cloud_binary(4, magic=b"XXXX"))
    status, payload, *_rest = CONTROL.CloudSnapshotReader(path).snapshot()
    assert status == "invalid" and payload is None

    # A truncated body (declared count does not match length) is rejected.
    path.write_bytes(_cloud_binary(4)[:-6])
    status, payload, *_rest = CONTROL.CloudSnapshotReader(path).snapshot()
    assert status == "invalid" and payload is None

    # Oversized files never load.
    path.write_bytes(b"\x00" * (CONTROL.MAX_CLOUD_BIN_BYTES + 1))
    status, payload, *_rest = CONTROL.CloudSnapshotReader(path).snapshot()
    assert status == "invalid" and payload is None


def test_cloud_endpoint_is_fixed_bounded_fresh_conditional_get_and_head(tmp_path):
    class FakeControl:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            raise AssertionError("cloud reads must never start planning")

    cloud_path = tmp_path / "cloud-latest.bin"
    binary = _cloud_binary(6, source_flag=0)  # 0 == raw D435 aligned depth
    cloud_path.write_bytes(binary)
    cloud_path.with_suffix(".json").write_text(
        json.dumps({
            "schema": "z_manip.point_cloud_frame.v1",
            "count": 6,
            "source": "d435_raw",
            "valid_fraction": 0.71,
        }),
        encoding="utf-8",
    )
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=FakeControl(),
        runtime_state=None,
        cloud_bin=cloud_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/cloud/latest.bin")
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        etag = response.getheader("ETag")
        assert response.getheader("Content-Type") == "application/octet-stream"
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Z-Manip-Cloud-State") == "live"
        assert response.getheader("X-Z-Manip-Cloud-Source") == "d435_raw"
        assert response.getheader("X-Z-Manip-Cloud-Count") == "6"
        assert response.getheader("X-Z-Manip-Poll-Interval-Ms") == str(CONTROL.CLOUD_POLL_INTERVAL_MS)
        assert body == binary

        connection.request("HEAD", "/api/cloud/latest.bin", headers={"If-None-Match": etag})
        response = connection.getresponse()
        assert response.status == 304
        assert response.getheader("ETag") == etag
        assert response.getheader("X-Z-Manip-Cloud-Source") == "d435_raw"
        assert response.read() == b""

        connection.request("GET", "/api/cloud/latest.bin?path=/etc/passwd")
        response = connection.getresponse()
        assert response.status == 400
        assert b"no query" in response.read()

        connection.request("GET", "/api/cloud/latest.json")
        response = connection.getresponse()
        manifest = json.loads(response.read())
        assert response.status == 200
        assert manifest["source"] == "d435_raw"
        assert manifest["valid_fraction"] == 0.71

        stale_ns = time.time_ns() - 3_000_000_000
        os.utime(cloud_path, ns=(stale_ns, stale_ns))
        connection.request("GET", "/api/cloud/latest.bin")
        response = connection.getresponse()
        assert response.status == 503
        assert response.getheader("X-Z-Manip-Cloud-State") == "stale"
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_runtime_reader_has_no_process_ros_can_or_transport_surface():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    reader = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeStateReader"
    )
    calls = {
        node.func.attr
        for node in ast.walk(reader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {
        "Popen",
        "run",
        "system",
        "create_publisher",
        "publish",
        "send",
        "sendall",
        "sendmsg",
        "sendto",
    }
    assert calls.isdisjoint(forbidden)


def test_depth_servo_blocks_when_joint_feedback_is_stale(tmp_path):
    class Sessions:
        def start_perception(self, target):
            return {"status": "succeeded", "target": target}

    stale = tmp_path / "runtime.json"
    stale.write_text(json.dumps({
        "joint_state_available": True,
        "source_timestamp_ns": 1,
    }), encoding="utf-8")
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=Sessions(),
        runtime_state=stale,
        joint_feedback_timeout_s=0.2,
    )

    result = runner.start(
        "shadow",
        target="charger",
        acquire_target=True,
        speed_percent=8,
    )
    deadline = time.monotonic() + 3.0
    while runner.status()["workflow"]["active"] and time.monotonic() < deadline:
        time.sleep(0.02)

    status = runner.status()
    assert result["started"] is True
    assert status["workflow"]["phase"] == "blocked"
    assert "passive joint feedback" in status["workflow"]["failure"]
    assert runner._process is None


def test_depth_servo_starts_when_joint_feedback_is_fresh(tmp_path):
    class Sessions:
        def start_perception(self, target):
            return {"status": "succeeded", "target": target}

    fresh = tmp_path / "runtime.json"
    fresh.write_text(json.dumps({
        "joint_state_available": True,
        "source_timestamp_ns": time.time_ns(),
    }), encoding="utf-8")
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=Sessions(),
        runtime_state=fresh,
        joint_feedback_timeout_s=0.2,
    )

    result = runner.start(
        "shadow",
        target="charger",
        acquire_target=True,
        speed_percent=8,
    )
    deadline = time.monotonic() + 3.0
    while runner.status()["workflow"]["phase"] not in {"approaching", "blocked"} and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["started"] is True
    assert runner.status()["workflow"]["phase"] == "approaching"
    runner.stop()


def _handoff_retry_runner(tmp_path, events, validate):
    class FakeSessions:
        def __init__(self):
            self.perception_calls = 0

        def clear_current_context(self):
            events.append("clear")

        def start_perception(self, target):
            self.perception_calls += 1
            events.append(("perception", self.perception_calls))
            return {
                "status": "succeeded",
                "session_id": f"20260723-12000{self.perception_calls}",
            }

        def start_planning(self):
            events.append("planning")
            return {"status": "succeeded", "session_id": "20260723-120009"}

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = FakeSessions()
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner._planning_artifacts = lambda attempt: (
        tmp_path / "planning_report.json",
        tmp_path / "planned_grasp.npz",
    )

    class FreshJoints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            events.append("joint_ready")
            return True, "fresh", {
                "sequence": 42,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner.home_verifier = FreshJoints()
    runner._validate_mobile_handoff_capture_evidence = validate

    def execute(**kwargs):
        events.append("execute")
        return _executor_start_evidence()

    runner._run_full = execute
    return runner


def test_mobile_handoff_recaptures_once_when_passive_window_straddles_stop(tmp_path):
    events = []
    attempts = {"n": 0}

    def flaky_validate(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "passive handoff feedback is not strictly post-stop and ordered"
            )
        return {"validated": True}

    runner = _handoff_retry_runner(tmp_path, events, flaky_validate)
    runner._run_mobile_handoff(
        "white charger",
        9,
        "20260723-115900",
        1_800_000_000_000_000_000,
    )

    assert events.count(("perception", 1)) == 1
    assert events.count(("perception", 2)) == 1
    assert attempts["n"] == 2
    assert runner.status()["outcome"] == "passed"


def test_mobile_handoff_does_not_recapture_for_other_evidence_failures(tmp_path):
    events = []
    attempts = {"n": 0}

    def hard_validate(**_kwargs):
        attempts["n"] += 1
        raise RuntimeError("observer joint evidence is not strictly post-stop/read-only")

    runner = _handoff_retry_runner(tmp_path, events, hard_validate)
    runner._run_mobile_handoff(
        "white charger",
        9,
        "20260723-115900",
        1_800_000_000_000_000_000,
    )

    assert attempts["n"] == 1
    assert events.count(("perception", 1)) == 1
    assert ("perception", 2) not in events
    assert runner.status()["outcome"] != "passed"
