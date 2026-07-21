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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("go2w_planning_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


HTML = ROOT / "web" / "debug_dashboard" / "index.html"
SERVICE = ROOT / "configs" / "z-manip-planning-workbench.service"


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
    assert grasp.target == "charger"
    assert grasp.speed_percent == 17
    assert runner.status()["workflow"]["phase"] == "grasp_started"


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
        assert runner._process is not None
        assert runner._process.poll() is not None
        assert runner.status()["workflow"]["phase"] == "grasp_started"


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


def test_depth_servo_uses_bounded_wrist_search_after_initial_detection_miss(tmp_path):
    class Sessions:
        def __init__(self):
            self.calls = 0

        def start_perception(self, target):
            self.calls += 1
            return {
                "status": "failed" if self.calls == 1 else "succeeded",
                "target": target,
            }

    class Search:
        def __init__(self):
            self.calls = []

        def run(self, target, *, mode, speed_percent, cancel, operator_present=False):
            self.calls.append((
                target, mode, speed_percent, cancel.is_set(), operator_present,
            ))
            return True

        def status(self):
            return {"phase": "found", "failure": None}

        def stop(self):
            return None

    sessions = Sessions()
    search = Search()
    runner = CONTROL.DepthServoRunner(
        _servo_status_script(tmp_path / "servo.py", "approach"),
        tmp_path / "status.json",
        tmp_path / "servo.log",
        session_service=sessions,
        wrist_search=search,
    )

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
    assert sessions.calls == 2
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


def test_workbench_service_passes_only_the_fixed_camera_artifact_path():
    source = SERVICE.read_text(encoding="utf-8")
    assert "--camera-image %h/Z-Robotics-Lab/artifacts/go2w_real/latest/camera-latest.jpg" in source
    assert "/api/camera" not in source


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
