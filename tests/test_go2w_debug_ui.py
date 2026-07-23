from __future__ import annotations

import ast
import http.client
import importlib.util
import json
from pathlib import Path
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_debug_ui.py"
SHELL = ROOT / "scripts" / "runtime" / "go2w_debug_ui.sh"
HTML = ROOT / "web" / "debug_dashboard" / "index.html"
SPEC = importlib.util.spec_from_file_location("go2w_debug_ui", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DEBUG_UI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEBUG_UI)


def _bundle(image_path: str = "mask.png") -> dict[str, object]:
    blocked = {
        "name": "calibration_gate",
        "status": "failed",
        "metrics": {"calibrated": False},
        "error": {
            "code": "UNCALIBRATED_OR_SYNTHETIC_CAMERA",
            "message": "measured calibration is unavailable",
        },
    }
    return {
        "schema": "z_manip.debug_bundle.v1",
        "run_id": "debug-test",
        "created_unix_ns": 1,
        "mode": {
            "read_only": True,
            "planning_only": True,
            "offline_artifact_reader": True,
        },
        "status": {
            "ok": False,
            "state": "blocked",
            "first_failed_stage": "calibration_gate",
        },
        "safety": {
            "motion_commands_published": 0,
            "transport_opened": False,
            "ros_imported": False,
            "can_opened": False,
        },
        "frames": {
            "perception": "camera_color_optical_frame",
            "planning": None,
        },
        "inputs": {"source_stamp_ns": 123},
        "stages": [
            {"name": "perception_bundle", "status": "ok", "metrics": {}, "error": None},
            blocked,
            {
                "name": "frame_transform",
                "status": "blocked",
                "metrics": {},
                "error": {"code": "MISSING_TRANSFORM", "message": "transform blocked"},
            },
            {
                "name": "motion_plan",
                "status": "blocked",
                "metrics": {},
                "error": {"code": "MISSING_PLAN", "message": "planning report unavailable"},
            },
        ],
        "artifacts": {"segmentation_mask": {"path": image_path}},
        "candidates": [
            {
                "candidate_id": 0,
                "rank": 1,
                "pose_source": [
                    [1, 0, 0, 0.1],
                    [0, 1, 0, -0.1],
                    [0, 0, 1, 0.5],
                    [0, 0, 0, 1],
                ],
                "source_frame": "camera_color_optical_frame",
            },
        ],
        "planning": {"available": False, "rejections": []},
        "selected_plan": None,
        "visualization": {
            "images": {"segmentation_mask": image_path},
            "target_cloud": {"points_xyz_m": [[0.1, -0.1, 0.5]]},
            "scene_cloud": {"points_xyz_m": [[0.0, 0.0, 0.6]]},
            "candidate_axes": [],
            "joint_trajectory": None,
        },
    }


def _write_bundle(tmp_path: Path) -> Path:
    (tmp_path / "mask.png").write_bytes(b"\x89PNG\r\n\x1a\nrecorded")
    path = tmp_path / "debug_bundle.json"
    path.write_text(json.dumps(_bundle()), encoding="utf-8")
    return path


def test_loads_blocked_real_bundle_without_requiring_a_plan(tmp_path):
    path = _write_bundle(tmp_path)

    document = DEBUG_UI.load_bundle(path)

    assert document["status"]["state"] == "blocked"
    assert document["planning"]["available"] is False
    assert document["stages"][1]["status"] == "failed"


def test_rejects_unknown_schema_and_oversized_or_missing_shape(tmp_path, monkeypatch):
    path = tmp_path / "bundle.json"
    path.write_text('{"schema":"unknown"}', encoding="utf-8")
    with pytest.raises(DEBUG_UI.BundleError, match="unsupported"):
        DEBUG_UI.load_bundle(path)

    document = _bundle()
    del document["visualization"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DEBUG_UI.BundleError, match="visualization"):
        DEBUG_UI.load_bundle(path)

    monkeypatch.setattr(DEBUG_UI, "MAX_BUNDLE_BYTES", 4)
    with pytest.raises(DEBUG_UI.BundleError, match="64 MiB"):
        DEBUG_UI.load_bundle(path)


def test_server_is_loopback_only_read_only_and_serves_declared_images(tmp_path):
    bundle_path = _write_bundle(tmp_path)
    server = DEBUG_UI.create_server(bundle_path, port=0, index_path=HTML)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/bundle")
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        assert body["schema"] == "z_manip.debug_bundle.v1"
        assert response.getheader("Access-Control-Allow-Origin") is None
        assert "connect-src 'self'" in response.getheader("Content-Security-Policy")

        connection.request("GET", "/artifact/segmentation_mask")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read().startswith(b"\x89PNG")
        assert response.getheader("Content-Type") == "image/png"

        connection.request("GET", "/artifact/../secret")
        response = connection.getresponse()
        assert response.status == 404
        response.read()

        connection.request("POST", "/api/bundle", body=b"{}")
        response = connection.getresponse()
        assert response.status == 405
        assert b"read-only" in response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_opt_in_control_endpoint_requires_explicit_same_origin_planning_header(tmp_path):
    class FakeControl:
        def __init__(self):
            self.starts = 0

        def status(self):
            return {"available": True, "running": False, "state": "idle"}

        def start(self):
            self.starts += 1
            return {"started": True, "control": self.status()}

    control = FakeControl()
    server = DEBUG_UI.create_server(
        _write_bundle(tmp_path),
        port=0,
        index_path=HTML,
        control_backend=control,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        connection.request("GET", "/api/control")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["available"] is True

        connection.request("POST", "/api/runs")
        response = connection.getresponse()
        assert response.status == 403
        response.read()

        connection.request(
            "POST",
            "/api/runs",
            headers={
                "Origin": "http://malicious.example",
                "X-Z-Manip-Action": "planning-only",
            },
        )
        response = connection.getresponse()
        assert response.status == 403
        response.read()

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


def test_server_source_has_no_robot_transport_or_subprocess_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {
        "can",
        "rclpy",
        "socket",
        "subprocess",
        "piper_sdk",
        "pyAgxArm",
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
        "create_publisher",
        "publish",
        "send",
        "sendall",
        "sendmsg",
        "sendto",
        "Popen",
        "run",
        "system",
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
    assert 'LOOPBACK = "127.0.0.1"' in source
    assert "0.0.0.0" not in source


def test_dashboard_is_offline_and_calibration_gates_base_overlays():
    source = HTML.read_text(encoding="utf-8")
    lowered = source.lower()

    assert "<canvas" in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "cdn" not in lowered
    assert "calibration invalid" in lowered
    assert "if (state.calibrationvalid)" in lowered
    assert "drawrobotoverlay(context, project)" in lowered
    assert "drawcartesiantrajectory(context, project)" in lowered
    assert "? sourcecandidateaxes()" in lowered
    assert "state.bundle?.candidates" in lowered
    assert ".fatal[hidden]" in lowered
    assert "innerhtml" not in lowered
    assert 'data-testid="rejection-disclosure"' in lowered
    assert 'data-testid="rejection-stage-filter"' in lowered
    assert 'data-testid="rejection-candidate-filter"' in lowered
    assert "max-height: min(42vh, 420px)" in lowered
    assert 'id="session-run-perception"' in lowered
    assert 'id="session-run-planning"' in lowered
    assert 'id="session-clear-demo"' in lowered
    assert 'id="session-restart-service"' in lowered
    assert '"/api/sessions/clear"' in lowered
    assert '"/api/service/restart"' in lowered
    assert 'runmaintenanceaction("clear-demo")' in lowered
    assert 'runmaintenanceaction("restart-workbench")' in lowered
    assert "freshbutton.disabled = actionbusy || !targetready" in lowered
    assert "directbutton.disabled = actionbusy || !athome || !selectedplanready" in lowered
    assert "perception + perform" in lowered
    assert "json.stringify({ target, speed_percent: selectedspeedpercent() })" in lowered
    assert 'id="perception-target"' in lowered
    assert '`/api/sessions/${action}`' in lowered
    assert '"/api/sessions/status"' in lowered
    assert 'data-testid="candidate-selector"' in lowered
    assert 'id="show-all-candidates"' in lowered
    assert "item.frame === displayframe" in lowered
    assert 'id="collision-evidence"' in lowered
    assert 'id="trajectory-refinement-evidence"' in lowered
    assert "function rendertrajectoryrefinementevidence()" in lowered
    assert "selected.trajectory_refinement" in lowered
    assert "selected.lift_pose_base" in lowered
    assert "fixed clearance" in lowered
    assert "safe rrt seed retained" in lowered
    assert "refined path accepted" in lowered
    assert "rendertrajectoryrefinementevidence();" in lowered
    assert "drawcollisionwitness(context, project)" in lowered
    assert '"no feasible candidate"' in lowered
    assert "state.selectedcandidate = candidateid" not in lowered
    assert '"x-z-manip-action": "planning-only"' in lowered
    assert "depth stabilization" in lowered
    assert "static / stable" in lowered
    assert "mad95" in lowered
    assert 'data-testid="live-camera-frame"' in lowered
    assert 'data-testid="live-depth-frame"' in lowered
    assert '"/api/camera/latest.jpg"' in lowered
    assert '"/api/depth/latest.jpg"' in lowered
    assert "fetch(tile.endpoint" in lowered
    assert 'id="recorded-image-stack"' in lowered
    assert lowered.index('data-testid="live-camera-frame"') < lowered.index('id="recorded-image-stack"')
    assert lowered.index('data-testid="live-depth-frame"') < lowered.index('id="recorded-image-stack"')
    assert "position: sticky" in lowered
    assert ".image-frame img[hidden], .empty-image[hidden] { display: none; }" in lowered
    assert 'data-feed-key="live_camera"' in lowered
    assert 'data-feed-key="live_depth"' in lowered
    assert 'data-reorder="up"' in lowered
    assert 'data-reorder="down"' in lowered
    assert 'aria-live="polite"' in lowered
    assert "z-manip.perception-feed-order.v2" in lowered
    assert "new set(parsed).size === default_feed_order.length" in lowered
    assert "parsed.every(key => default_feed_order.includes(key))" in lowered
    assert "window.localstorage.setitem" in lowered
    assert "applyfeedorder();" in lowered
    assert "const latestplanning = sessions?.actions?.planning?.latest_attempt;" in lowered
    assert "latestplanning?.selected_perception_session_id === perceptionid" in lowered
    assert 'await loadbundle("/api/sessions/planning/bundle")' in lowered
    assert "graspresultiscurrent" in lowered
    assert "graspfinishedms >= sessionfinishedms" in lowered
    assert "state.interactiveplanningid = null" in lowered
    assert "state.runtimescene.setbundle(null)" in lowered
    assert "scheduleperceptionfeedpoll(25)" in lowered
    assert 'id="session-pick-hold"' not in lowered
    assert 'id="session-return-home-holding"' not in lowered
    assert 'id="session-place-back"' not in lowered
    assert '"/api/grasp/pick-hold"' not in lowered
    assert '"/api/grasp/return-home-holding"' not in lowered
    assert '"/api/grasp/place-back"' not in lowered
    assert "staged_grasp_actions" not in lowered
    assert 'id="session-go-home"' in lowered
    assert "reset + home" in lowered
    assert "state.sessions?.busy === true || speed === null" not in lowered
    assert 'id="approach-camera-range"' in lowered
    assert 'id="approach-base-range"' in lowered
    assert 'id="approach-arm-range"' in lowered
    assert 'id="approach-posture"' in lowered
    assert 'id="approach-feedback-age"' in lowered
    assert 'id="approach-owner"' in lowered
    assert 'id="approach-optimizer"' in lowered
    assert 'id="approach-handoff"' in lowered
    assert 'id="approach-reactive"' in lowered
    assert 'id="approach-sport-proof"' in lowered
    assert 'id="approach-posture-proof"' in lowered
    assert 'id="approach-arm-proof"' in lowered
    assert 'id="approach-fixed-guard"' in lowered
    assert "calculated" in lowered
    assert "gated" in lowered
    assert "acked" in lowered
    assert "measured" in lowered
    assert "transport.mode_epoch" in lowered
    assert "posturecommand.euler_ack_generation" in lowered
    assert "posturecommand.euler_ack_code" in lowered
    assert "armstatus.accepted_seq" in lowered
    assert "armstatus.actual_joints_rad" in lowered
    assert "armstatus.fixed_collision_guard" in lowered
    assert "armstatus.collision_rejections" in lowered
    assert "armstatus.unsafe_target_forwarded" in lowered
    assert "forwarding evidence —" in lowered
    assert "geometry.base_planar_distance_m" in lowered
    assert "geometry.camera_range_m" in lowered
    assert "geometry.target_height_m" in lowered
    assert "geometry.arm_range_m" in lowered
    assert "reactive.handoff_ready" in lowered
    assert "posturestatus.body_height" in lowered
    assert "posturestatus.attitude" in lowered
    assert "postureenvelope.document" in lowered
    assert "supervision.feedback?.age_s" in lowered
    assert "optimizer.primal_residual" in lowered
    assert 'data-approach-phase="posture"' in lowered
    assert 'data-approach-phase="reacquire"' in lowered
    assert 'data-approach-phase="handoff"' in lowered
    assert "ground-plane distance for base translation" in lowered
    assert 'id="approach-only"' in lowered
    assert 'function startapproachonly()' in lowered
    assert 'auto_handoff: false' in lowered
    assert 'byid("approach-only").addeventlistener("click", startapproachonly)' in lowered


def test_dashboard_explains_blocked_attempt_while_showing_last_good_bundle():
    source = HTML.read_text(encoding="utf-8")

    assert 'control.latest_bundle_available === true' in source
    assert 'control.latest_attempt_bundle_available === true' in source
    assert 'control.latest_bundle !== control.latest_attempt_bundle' in source
    assert 'latest attempt blocked · showing last good' in source
    assert 'status.title = control.message || control.log_tail || ""' in source


def test_launcher_contains_no_remote_or_actuator_commands():
    source = SHELL.read_text(encoding="utf-8").lower()
    forbidden = (
        "sudo ",
        "ssh ",
        "ros2 topic pub",
        "create_publisher",
        "candump",
        "cansend",
        "/cmd_vel",
        "/joint_trajectory",
        "docker exec",
    )
    assert all(value not in source for value in forbidden)
    assert "127.0.0.1" in source
    assert "--no-open" in source
    assert "go2w_debug_safety_gate.py" in source
    assert "--artifact-root" in source
    assert "--output" in source
