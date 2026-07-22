from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts" / "runtime" / "go2w_component_manager.sh"
LAB = ROOT / "scripts" / "runtime" / "go2w_perception_lab.sh"
CONTROL_SCRIPT = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
DEBUG_SCRIPT = ROOT / "scripts" / "runtime" / "go2w_debug_ui.py"
HTML = ROOT / "web" / "debug_dashboard" / "index.html"
sys.path.insert(0, str(CONTROL_SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("component_control_test", CONTROL_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


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


class _Control:
    def status(self):
        return {"available": True, "running": False, "state": "idle"}

    def start(self):
        raise AssertionError("component routes must not start planning")


class _Sessions:
    def __init__(self) -> None:
        self.clears = 0

    def status(self):
        return {
            "schema": "z_manip.interactive_session_state.v1",
            "read_only": True,
            "busy": False,
            "selected_perception_session_id": None,
            "actions": {},
        }

    def clear_current_context(self):
        self.clears += 1
        return {"cleared": True}


class _Components:
    script = MANAGER

    def __init__(self) -> None:
        self.restarts: list[str] = []
        self.bringups = 0

    def status(self):
        return {
            "schema": "z_manip.visual_components.v1",
            "available": True,
            "busy": False,
            "active_component": None,
            "components": {
                "edgetam": {"name": "edgetam", "state": "healthy", "summary": "HTTP health OK"},
            },
            "last_result": None,
            "error": None,
        }

    def logs(self, component):
        return {
            "schema": "z_manip.visual_component_log.v1",
            "component": component,
            "ok": True,
            "text": "bounded log tail",
        }

    def restart(self, component):
        self.restarts.append(component)
        return {"started": True, "component": component}

    def bringup(self):
        self.bringups += 1
        return {"started": True, "component": "bringup"}


def _headers(port: int, action: str) -> dict[str, str]:
    return {
        "Origin": f"http://127.0.0.1:{port}",
        "Sec-Fetch-Site": "same-origin",
        "X-Z-Manip-Action": action,
        "Content-Type": "application/json",
    }


def test_component_manager_is_syntax_checked_singleton_and_motion_free():
    subprocess.run(["bash", "-n", str(MANAGER)], check=True)
    subprocess.run(["bash", "-n", str(LAB)], check=True)
    source = MANAGER.read_text(encoding="utf-8").lower()
    lab = LAB.read_text(encoding="utf-8").lower()

    assert "flock -n" in source
    assert "install_pc_units" in source
    assert "daemon-reload" in source
    assert "default.target.wants" in source
    assert "bringup)" in source
    assert "restart-edgetam" in source
    assert "z-mobile-manip-yoloe:latest" in source
    assert "grounding_ready" in source
    assert "restart_one grounding" in source
    assert "yoloe_source_hash" in source
    assert "org.zlab.yoloe.source-sha256" in source
    assert "restart-rgbd" in source
    assert "restart-perception" in source
    assert "resident_runners_current" in source
    assert "org.zlab.z-manip.runtime-sha256" in source
    assert "runner_socket_private" in source
    assert '$systemctl --user restart "$ui_unit"' in source
    assert "wait_until" in source
    assert "state error-active" in source
    assert "systemctl --user restart d435i.service" in source
    assert 'd435 usb device is absent' in source
    assert 'camera_artifact_fresh' in source
    assert '/dev/v4l/by-id/*realsense*' in source
    assert 'reconnect its usb cable before restarting' in source
    assert 'container_running z-manip-rgbd && camera_artifact_fresh' in source
    assert "sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8" in source
    assert "restart-edgetam)" in lab
    assert "restart-rgbd)" in lab
    assert "restart-perception)" in lab
    assert "z_manip_runtime_fingerprint.py" in lab
    assert "z_manip_runtime_fingerprint=$fingerprint" in lab
    assert 'rm -f -- "$perception_runner_socket"' in lab
    assert 'rm -f -- "$planning_runner_socket"' in lab
    for forbidden in ("cansend", "ip link set", "--execute", "motionenable"):
        assert forbidden not in source


def test_component_api_has_bounded_status_logs_restart_and_bringup(tmp_path):
    sessions = _Sessions()
    components = _Components()
    run_root = tmp_path / "interactive"
    (run_root / "perception").mkdir(parents=True)
    (run_root / "planning").mkdir()
    server = CONTROL.create_server(
        _bundle(tmp_path / "debug_bundle.json"),
        port=0,
        index_path=HTML,
        control_backend=_Control(),
        runtime_state=None,
        interactive_service=sessions,
        interactive_run_root=run_root,
        component_manager=components,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/api/components/status")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["components"]["edgetam"]["state"] == "healthy"

        connection.request("GET", "/api/components/logs/edgetam")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["text"] == "bounded log tail"

        connection.request(
            "POST",
            "/api/components/restart",
            body=json.dumps({"component": "edgetam"}),
            headers=_headers(port, "restart-component"),
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["restart"]["component"] == "edgetam"
        assert components.restarts == ["edgetam"]
        assert sessions.clears == 1

        connection.request(
            "POST",
            "/api/components/bringup",
            body=b"{}",
            headers=_headers(port, "bringup-components"),
        )
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        assert components.bringups == 1
        assert sessions.clears == 2
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_dashboard_component_controls_keep_live_feeds_bounded_and_updating():
    html = HTML.read_text(encoding="utf-8").lower()
    server = DEBUG_SCRIPT.read_text(encoding="utf-8").lower()

    assert 'id="component-bringup"' in html
    assert 'data-component-restart="edgetam"' in html
    assert 'data-component-restart="perception-all"' in html
    assert '"/api/components/status"' in html
    assert '"/api/components/restart"' in html
    assert '"/api/components/bringup"' in html
    assert "scheduleperceptionfeedpoll(tracking || perceptionstarting ? 500 : 1500)" in html
    assert "math.max(200, suggested)" in html
    assert "route.startswith(\"/api/perception/live/\")" in server
    assert 'status in {"200", "304", "409", "503"}' in server
    assert "grid-template-rows: auto auto auto auto minmax(620px, 1fr) auto" in html
