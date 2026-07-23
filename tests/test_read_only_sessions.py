from __future__ import annotations

import ast
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import socket
import sys
import threading
import time
import tempfile
from types import SimpleNamespace

import pytest

import z_manip.read_only_sessions as session_contracts
from z_manip.read_only_sessions import (
    BackendResult,
    ReadOnlySessionService,
    SessionContractError,
    validate_session_id,
    validate_target_description,
)


ROOT = Path(__file__).resolve().parents[1]
PURE_MODULE = ROOT / "z_manip" / "read_only_sessions.py"
INTEGRATION = ROOT / "scripts" / "runtime" / "go2w_interactive_sessions.py"


def _integration_module():
    name = "go2w_interactive_sessions_test"
    spec = importlib.util.spec_from_file_location(name, INTEGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _passive_report():
    return {
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
    }


def _write_perception_success(output: Path, target: str) -> None:
    (output / "report.json").write_text(json.dumps({
        "read_only": True,
        "instruction": target,
        "elapsed_s": 0.25,
    }), encoding="utf-8")
    for name in (
        "edgetam_mask.png",
        "edgetam_overlay.png",
        "grasp_candidates_overlay.png",
        "grasp_candidates.npz",
        "scene_collision_points.npy",
        "target_points.npy",
    ):
        (output / name).write_bytes(b"fixed")


def test_fixed_worker_request_uses_private_bounded_unix_socket(tmp_path):
    module = _integration_module()
    socket_path = tmp_path / "worker.sock"
    log_path = tmp_path / "worker.log"
    ready = threading.Event()
    received = []

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                payload = bytearray()
                while True:
                    block = connection.recv(4096)
                    if not block:
                        break
                    payload.extend(block)
                received.append(json.loads(bytes(payload)))
                connection.sendall(json.dumps({
                    "return_code": 0,
                    "elapsed_s": 0.125,
                    "output": "fixed worker output\n",
                }).encode("utf-8"))

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    assert ready.wait(timeout=1.0)

    result = module._run_fixed_worker_request(
        socket_path,
        {"argv": ["--output", "/fixed"]},
        log_path,
    )
    server_thread.join(timeout=1.0)

    assert result.returncode == 0
    assert result.worker_elapsed_s == pytest.approx(0.125)
    assert received == [{"argv": ["--output", "/fixed"]}]
    assert log_path.read_text(encoding="utf-8") == "fixed worker output\n"


def test_fixed_worker_request_rejects_public_socket(tmp_path):
    module = _integration_module()
    socket_path = tmp_path / "worker.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o666)
        assert module._fixed_worker_socket_available(socket_path) is False
        with pytest.raises(OSError, match="unavailable or unsafe"):
            module._run_fixed_worker_request(
                socket_path,
                {"argv": []},
                tmp_path / "worker.log",
            )


def test_fixed_worker_request_rejects_stale_resident_module(tmp_path):
    module = _integration_module()
    socket_path = tmp_path / "worker.sock"
    ready = threading.Event()

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                while connection.recv(4096):
                    pass
                connection.sendall(json.dumps({
                    "return_code": 0,
                    "elapsed_s": 0.001,
                    "output": "stale worker must not be accepted\n",
                    "worker_fingerprint": "old-source",
                }).encode("utf-8"))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        module._run_fixed_worker_request(
            socket_path,
            {"argv": []},
            tmp_path / "worker.log",
            expected_fingerprint="current-source",
        )
    thread.join(timeout=1.0)


class FakeBackend:
    def __init__(self) -> None:
        self.perception_result = BackendResult(0)
        self.planning_result = BackendResult(0)
        self.perception_calls: list[tuple[str, Path]] = []
        self.planning_calls: list[Path] = []

    def run_perception(self, *, target, output_dir, log_path):
        self.perception_calls.append((target, output_dir))
        log_path.write_text("fixed perception backend\n", encoding="utf-8")
        (output_dir / "report.json").write_text(json.dumps({
            "read_only": True,
            "grasp_generation_valid": self.perception_result.exit_code == 0,
        }), encoding="utf-8")
        (output_dir / "grasp_candidates.npz").write_bytes(b"immutable")
        return self.perception_result

    def run_planning(self, *, perception_dir, output_dir, log_path):
        self.planning_calls.append(perception_dir)
        log_path.write_text("fixed planning backend\n", encoding="utf-8")
        (output_dir / "session_gate.json").write_text(json.dumps({
            "planning_ready": self.planning_result.exit_code == 0,
            "read_only": True,
            "planning_only": True,
            "motion_commands_published": 0,
            "transport_opened": False,
        }), encoding="utf-8")
        planning = output_dir / "planning"
        planning.mkdir()
        (planning / "planning_report.json").write_text(json.dumps({
            "read_only": True,
            "planning_only": True,
            "motion_commands_published": 0,
            "plan_valid": self.planning_result.exit_code == 0,
        }), encoding="utf-8")
        return self.planning_result


def _service(tmp_path: Path, backend: FakeBackend) -> ReadOnlySessionService:
    return ReadOnlySessionService(
        tmp_path / "sessions",
        backend,
        now=lambda: datetime(2026, 7, 17, 8, 9, 10, tzinfo=timezone.utc),
        random_token=lambda: "1" * 32,
    )


@pytest.mark.parametrize("target", [
    "白色 USB 电源适配器",
    "small white adapter with red port",
    "a" * 160,
])
def test_target_description_accepts_strict_utf8_text(target):
    assert validate_target_description(target) == target


@pytest.mark.parametrize("target, code", [
    ("", "INVALID_TARGET_LENGTH"),
    ("a" * 161, "INVALID_TARGET_LENGTH"),
    (" leading", "INVALID_TARGET_WHITESPACE"),
    ("trailing ", "INVALID_TARGET_WHITESPACE"),
    ("two\nlines", "INVALID_TARGET_CONTROL_CHARACTER"),
    ("nul\0byte", "INVALID_TARGET_CONTROL_CHARACTER"),
    ("hidden\u200bformat", "INVALID_TARGET_CONTROL_CHARACTER"),
    ("/tmp/object", "INVALID_TARGET_PATH"),
    (r"C:\object", "INVALID_TARGET_PATH"),
    ("../object", "INVALID_TARGET_PATH"),
    ("file:object", "INVALID_TARGET_PATH"),
])
def test_target_description_rejects_control_path_and_length(target, code):
    with pytest.raises(SessionContractError) as caught:
        validate_target_description(target)
    assert caught.value.code == code


def test_target_description_rejects_non_utf8_surrogate():
    with pytest.raises(SessionContractError) as caught:
        validate_target_description("object\udcff")
    assert caught.value.code == "INVALID_TARGET_UTF8"


@pytest.mark.parametrize("session_id", [
    "20260717-160000",
    "s-0123456789abcdef0123456789abcdef",
])
def test_session_id_accepts_only_server_formats(session_id):
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize("session_id", [
    "../../etc/passwd",
    "20260230-120000",
    "20260717-160000/other",
    "s-not-random",
    "/absolute/path",
])
def test_session_id_rejects_paths_and_invalid_ids(session_id):
    with pytest.raises(SessionContractError) as caught:
        validate_session_id(session_id)
    assert caught.value.code == "INVALID_SESSION_ID"


def test_perception_creates_immutable_session_and_distinct_good_pointer(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)

    good = service.start_perception("白色适配器")
    backend.perception_result = BackendResult(7, "PERCEPTION_FAILED", "no target")
    failed = service.start_perception("绿色瓶子")
    state = service.status()

    assert good["status"] == "succeeded"
    assert failed["status"] == "failed"
    assert good["session_id"] == "20260717-080910"
    assert failed["session_id"] == "s-" + "1" * 32
    perception_state = state["actions"]["perception"]
    assert perception_state["latest_attempt"]["session_id"] == failed["session_id"]
    assert perception_state["last_good"]["session_id"] == good["session_id"]
    session = tmp_path / "sessions" / "perception" / good["session_id"]
    assert (session / "attempt.json").stat().st_mode & 0o222 == 0
    assert (session / "perception" / "report.json").stat().st_mode & 0o222 == 0
    assert (session / "perception_manifest.json").is_file()
    assert backend.perception_calls[0][0] == "白色适配器"


def test_planning_consumes_only_selected_verified_success(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    perception = service.start_perception("white adapter")

    selected = service.select_perception(perception["session_id"])
    planning = service.start_planning()
    state = service.status()

    assert selected["selected_perception_session_id"] == perception["session_id"]
    assert planning["status"] == "succeeded"
    assert planning["selected_perception_session_id"] == perception["session_id"]
    assert backend.planning_calls == [
        tmp_path
        / "sessions"
        / "perception"
        / perception["session_id"]
        / "perception"
    ]
    planning_state = state["actions"]["planning"]
    assert planning_state["latest_attempt"] == planning_state["last_good"]
    assert state["safety"] == {
        "motion_commands_available": False,
        "actuator_transport_available": False,
        "can_tx_available": False,
        "client_paths_accepted": False,
        "client_commands_accepted": False,
        "client_environment_accepted": False,
    }
    json.dumps(state)


def test_planning_attempt_preserves_recoverable_backend_disposition(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    service.start_perception("floor bottle")
    backend.planning_result = BackendResult(
        8,
        "NEED_BASE_APPROACH",
        "continue base approach before close-range planning",
    )

    planning = service.start_planning()

    assert planning["status"] == "blocked"
    assert planning["error"] == {
        "code": "NEED_BASE_APPROACH",
        "message": "continue base approach before close-range planning",
    }


def test_home_context_clear_invalidates_current_tasks_but_retains_history(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    perception = service.start_perception("white adapter")
    planning = service.start_planning()

    cleared = service.clear_current_context()
    state = service.status()

    assert cleared["cleared"] is True
    assert cleared["history_retained"] is True
    assert state["selected_perception_session_id"] is None
    assert state["actions"]["perception"]["latest_attempt"] is None
    assert state["actions"]["perception"]["last_good"] is None
    assert state["actions"]["planning"]["latest_attempt"] is None
    assert state["actions"]["planning"]["last_good"] is None
    assert (tmp_path / "sessions" / "perception" / perception["session_id"]).is_dir()
    assert (tmp_path / "sessions" / "planning" / planning["session_id"]).is_dir()


def test_new_successful_perception_invalidates_previous_plan(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    first = service.start_perception("white adapter")
    plan = service.start_planning()
    second = service.start_perception("black earphones")
    state = service.status()

    assert first["session_id"] != second["session_id"]
    assert plan["status"] == "succeeded"
    assert state["selected_perception_session_id"] == second["session_id"]
    assert state["actions"]["planning"]["latest_attempt"] is None
    assert state["actions"]["planning"]["last_good"] is None


def test_planning_without_selection_records_blocked_attempt(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)

    planning = service.start_planning()
    state = service.status()

    assert planning["status"] == "blocked"
    assert planning["error"]["code"] == "NO_SELECTED_PERCEPTION"
    assert backend.planning_calls == []
    assert state["actions"]["planning"]["latest_attempt"]["status"] == "blocked"
    assert state["actions"]["planning"]["last_good"] is None


def test_failed_perception_cannot_be_selected(tmp_path):
    backend = FakeBackend()
    backend.perception_result = BackendResult(1, "NO_TARGET", "no target")
    service = _service(tmp_path, backend)
    failed = service.start_perception("missing object")

    with pytest.raises(SessionContractError) as caught:
        service.select_perception(failed["session_id"])
    assert caught.value.code == "PERCEPTION_NOT_SUCCESSFUL"


def test_changed_perception_is_rejected_before_planning(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    attempt = service.start_perception("white adapter")
    report = (
        tmp_path
        / "sessions"
        / "perception"
        / attempt["session_id"]
        / "perception"
        / "report.json"
    )
    report.chmod(0o600)
    report.write_text("{}", encoding="utf-8")

    planning = service.start_planning()

    assert planning["status"] == "blocked"
    assert planning["error"]["code"] == "PERCEPTION_ARTIFACT_CHANGED"
    assert backend.planning_calls == []


def test_resolved_session_must_remain_under_server_run_root(tmp_path):
    backend = FakeBackend()
    service = _service(tmp_path, backend)
    outside = tmp_path / "outside"
    outside.mkdir()
    session_id = "20260717-080910"
    link = tmp_path / "sessions" / "perception" / session_id
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionContractError) as caught:
        service.select_perception(session_id)
    assert caught.value.code == "INVALID_SESSION_PATH"


def test_pure_control_has_no_transport_or_command_execution_imports():
    tree = ast.parse(PURE_MODULE.read_text(encoding="utf-8"))
    forbidden = {"subprocess", "socket", "rclpy", "can", "piper_sdk", "pyAgxArm"}
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
    assert imports.isdisjoint(forbidden)


def test_fixed_integration_has_no_actuator_or_can_tx_surface():
    source = INTEGRATION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
    ]

    assert {node.args[0].value for node in parser_calls} == {
        "perception",
        "select",
        "planning",
        "status",
    }
    assert "shell=True" not in source
    assert "cansend" not in source
    assert "joint_trajectory" not in source
    assert "piper/cmd" not in source
    assert "local_movement_cmd_vel" not in source
    assert "ros2 action" not in source
    assert "piper_passive_probe.py" in source
    assert 'PASSIVE_CAPTURE_SECONDS = "0.25"' in source
    assert 'REMOTE_PASSIVE_REPORT = "/tmp/z-manip-passive-live.json"' in source
    assert '"sudo"' not in source
    assert '"--network",\n                "none"' in source
    assert re.search(r'"--duration",\s*PASSIVE_CAPTURE_SECONDS', source)
    assert 'SEARCH_TIMEOUT_S = "6"' in source
    assert 'SYMMETRY_SAMPLES = "4"' in source
    assert 'MAX_HYPOTHESES = "64"' in source
    assert "environment: Mapping[str, str] | None" in source
    assert '"Z_MANIP_RUNTIME_IMAGE"' in source
    assert '"Z_MANIP_IK_BACKEND"' in source
    assert "z-manip-runtime:jazzy" not in source
    assert 'add_argument("environment"' not in source
    assert 'add_argument("command"' not in source
    assert 'add_argument("path"' not in source


def test_server_runtime_defaults_to_deployed_pinocchio_image():
    module = _integration_module()

    default = module.ServerRuntimeConfig.from_server_environment({})
    selected = module.ServerRuntimeConfig.from_server_environment({
        "Z_MANIP_RUNTIME_IMAGE": "z-manip-runtime:pinocchio-verified",
        "Z_MANIP_IK_BACKEND": "pinocchio",
        "IGNORED_CLIENT_LIKE_VALUE": "cansend can0 arbitrary",
    })

    assert default.runtime_image == "z-manip-runtime:pinocchio"
    assert default.ik_backend == "pinocchio"
    assert selected.runtime_image == "z-manip-runtime:pinocchio-verified"
    assert selected.ik_backend == "pinocchio"
    with pytest.raises(ValueError):
        module.ServerRuntimeConfig.from_server_environment({
            "Z_MANIP_RUNTIME_IMAGE": "docker.io/untrusted/image:latest",
        })
    with pytest.raises(ValueError):
        module.ServerRuntimeConfig.from_server_environment({
            "Z_MANIP_IK_BACKEND": "client-selected-backend",
        })


def test_passive_ssh_reuses_only_the_fixed_nuc_transport(tmp_path, monkeypatch):
    module = _integration_module()
    key = tmp_path / "server-key"
    monkeypatch.setattr(module, "NUC_KEY", key)

    prefix = module.FixedReadOnlyBackend._ssh_prefix()

    assert prefix[0] == "/usr/bin/ssh"
    assert "ControlMaster=auto" in prefix
    assert "ControlPersist=60" in prefix
    assert f"ControlPath={tmp_path / 'z-manip-%C'}" in prefix
    assert prefix[-1] == module.NUC_HOST


def test_passive_capture_uses_probe_stdout_without_second_ssh_fetch(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    output = tmp_path / "capture"
    output.mkdir()
    log = tmp_path / "capture.log"
    calls = []

    class Completed:
        returncode = 0

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        kwargs["stdout"].write(
            (json.dumps(_passive_report()) + "\n").encode("utf-8"),
        )
        return Completed()

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(
        backend,
        "_ssh_prefix",
        lambda: ("/usr/bin/ssh", "fixed-nuc"),
    )
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = backend._capture_passive_window(output, log, {})

    assert result.exit_code == 0
    assert len(calls) == 1
    assert module.REMOTE_PASSIVE_PROBE in calls[0]
    assert "cat" not in calls[0]
    assert backend._passive_report_valid(
        output / "live_passive_joint_report.json",
    )


def test_perception_passes_immutable_output_and_captures_synchronized_joints(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(module, "NUC_KEY", key)
    output = tmp_path / "immutable-output"
    output.mkdir()
    log = tmp_path / "perception.log"
    _write_perception_success(output, "white adapter")

    captured = {}

    class FakeProcess:
        def __init__(self):
            # Keep the perception worker alive for several supervisor polls.
            # Once the exact selected report exists, the backend must wait for
            # completion without repeatedly opening SSH capture windows.
            self.polls = iter((None, None, None, None, 0))

        def poll(self):
            return next(self.polls, 0)

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = tuple(argv)
        captured["environment"] = dict(kwargs["env"])
        return FakeProcess()

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        captured["passive_calls"] = captured.get("passive_calls", 0) + 1
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    argv = captured["argv"]
    assert argv[:5] == (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
    )
    assert argv[argv.index("--network") + 1] == "host"
    assert f"{module.DDS_CONFIG}:/config/cyclonedds.xml:ro" in argv
    assert (
        f"{module.PERCEPTION}:"
        "/usr/local/bin/z-manip-go2w-perception-dry-run:ro"
    ) in argv
    assert "PYTHONPATH=/opt/z_manip/python" in argv
    assert f"{module.STACK_ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro" in argv
    assert f"{output}:/artifacts" in argv
    assert "z-manip-go2w-perception-dry-run" in argv
    assert argv[argv.index("--instruction") + 1] == "white adapter"
    assert argv[argv.index("--output") + 1] == "/artifacts"
    assert "--passive-window" in argv
    assert "--selected-passive-window" in argv
    assert argv[argv.index("--timeout") + 1] == "15"
    assert argv[argv.index("--min-bundle-target-points") + 1] == "400"
    assert "--reuse-valid-tracking" in argv
    assert captured["environment"]["Z_MANIP_ARTIFACT_DIR"] == str(output)
    assert captured["environment"]["Z_MANIP_REQUIRE_PASSIVE_WINDOW"] == "1"
    assert captured["environment"]["Z_MANIP_RUNTIME_IMAGE"] == (
        "z-manip-runtime:pinocchio"
    )
    assert (output / "selected_passive_joint_report.json").is_file()
    timing_events = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    attempt_timing = next(
        event for event in timing_events
        if event.get("stage") == "perception_attempt"
    )
    total_timing = next(
        event for event in timing_events
        if event.get("stage") == "perception_total"
    )
    assert attempt_timing["passive_capture_count"] == 1
    assert captured["passive_calls"] == 1
    assert attempt_timing["process_launch_s"] >= 0.0
    assert total_timing["target_identity_valid"] is True
    assert total_timing["internal_elapsed_s"] == 0.25


def test_perception_output_validation_rejects_different_target_identity(tmp_path):
    module = _integration_module()
    output = tmp_path / "mismatched-target"
    output.mkdir()
    _write_perception_success(output, "black box")
    passive = json.dumps(_passive_report())
    (output / "selected_passive_joint_report.json").write_text(
        passive,
        encoding="utf-8",
    )

    assert module.FixedReadOnlyBackend._perception_outputs_valid(
        output,
        "white adapter",
    ) is False


def test_perception_uses_warm_runner_for_workspace_artifacts(tmp_path, monkeypatch):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "go2w_real" / "interactive_sessions" / "sample"
    output.mkdir(parents=True)
    log = tmp_path / "perception.log"
    _write_perception_success(output, "white adapter")
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.polls = iter((None, 0))

        def poll(self):
            return next(self.polls, 0)

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = tuple(argv)
        return FakeProcess()

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module, "NUC_KEY", key)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(backend, "_perception_runner_running", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    argv = captured["argv"]
    assert argv[:3] == (
        "/usr/bin/docker",
        "exec",
        module.PERCEPTION_RUNNER_CONTAINER,
    )
    assert argv[3:6] == (
        "z-manip-go2w-perception-worker",
        "client",
        "--",
    )
    assert "run" not in argv[:4]
    assert argv[argv.index("--tracking-reuse-max-age") + 1] == "0.5"
    assert argv[argv.index("--output") + 1] == (
        "/workspace-artifacts/go2w_real/interactive_sessions/sample"
    )


def test_perception_calls_resident_worker_socket_without_client_process(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    # Keep the AF_UNIX path below Linux's 108-byte sockaddr_un limit even
    # when pytest's temporary directory name is long.
    artifact_root = Path(tempfile.mkdtemp(prefix="zmi-", dir="/tmp"))
    output = artifact_root / "go2w_real" / "interactive_sessions" / "sample"
    output.mkdir(parents=True)
    log = tmp_path / "perception.log"
    socket_path = artifact_root / "go2w_real" / ".perception_runner.sock"
    ready = threading.Event()
    received = []

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                payload = bytearray()
                while True:
                    block = connection.recv(4096)
                    if not block:
                        break
                    payload.extend(block)
                received.append(json.loads(bytes(payload)))
                deadline = time.monotonic() + 1.0
                selected = output / "selected_passive_joint_report.json"
                while not selected.is_file() and time.monotonic() < deadline:
                    time.sleep(0.001)
                _write_perception_success(output, "white adapter")
                connection.sendall(json.dumps({
                    "return_code": 0,
                    "elapsed_s": 0.01,
                    "output": "fixed resident request\n",
                    "worker_fingerprint": module.runtime_fingerprint(),
                }).encode("utf-8"))

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    assert ready.wait(timeout=1.0)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module, "NUC_KEY", key)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)
    monkeypatch.setattr(
        backend,
        "_perception_runner_running",
        lambda: pytest.fail("docker inspect must not run when socket is ready"),
    )

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )
    server_thread.join(timeout=1.0)

    assert result.exit_code == 0
    assert received[0]["argv"][0:2] == ["--instruction", "white adapter"]
    assert received[0]["argv"][3] == (
        "/workspace-artifacts/go2w_real/interactive_sessions/sample"
    )
    timing = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    attempt = next(item for item in timing if item["stage"] == "perception_attempt")
    assert attempt["runner_transport"] == "unix_socket"
    assert attempt["passive_capture_count"] == 1
    shutil.rmtree(artifact_root)


def _serve_fingerprint_worker(socket_path, output, module, state, ready):
    """Serve resident-worker requests, healing the fingerprint after a restart.

    Each connection replies with a stale ``worker_fingerprint`` until
    ``state['healed']`` flips (set by the faked component restart), after which
    it returns the live fingerprint and writes a valid perception bundle.
    """

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        server.settimeout(5.0)
        ready.set()
        while state["connections"] < state["max_connections"]:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            state["connections"] += 1
            with connection:
                payload = bytearray()
                while True:
                    block = connection.recv(4096)
                    if not block:
                        break
                    payload.extend(block)
                if state["healed"]:
                    deadline = time.monotonic() + 1.0
                    selected = output / "selected_passive_joint_report.json"
                    while not selected.is_file() and time.monotonic() < deadline:
                        time.sleep(0.001)
                    _write_perception_success(output, "white adapter")
                    fingerprint = module.runtime_fingerprint()
                else:
                    fingerprint = "stale-resident-fingerprint"
                connection.sendall(json.dumps({
                    "return_code": 0,
                    "elapsed_s": 0.01,
                    "output": "resident request\n",
                    "worker_fingerprint": fingerprint,
                }).encode("utf-8"))


def test_perception_selfheals_once_on_resident_fingerprint_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    artifact_root = Path(tempfile.mkdtemp(prefix="zmi-", dir="/tmp"))
    output = artifact_root / "go2w_real" / "interactive_sessions" / "sample"
    output.mkdir(parents=True)
    log = tmp_path / "perception.log"
    socket_path = artifact_root / "go2w_real" / ".perception_runner.sock"
    state = {"healed": False, "connections": 0, "max_connections": 2}
    ready = threading.Event()

    server_thread = threading.Thread(
        target=_serve_fingerprint_worker,
        args=(socket_path, output, module, state, ready),
        daemon=True,
    )
    server_thread.start()
    assert ready.wait(timeout=1.0)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    restart_calls = []

    def fake_run_logged(argv, _log_path, *, environment):
        restart_calls.append(tuple(argv))
        # The restarted read-only component recreates the worker with the live
        # fingerprint; model that by healing the fake server.
        state["healed"] = True
        return module.subprocess.CompletedProcess(tuple(argv), 0)

    monkeypatch.setattr(module, "NUC_KEY", key)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "_run_logged", fake_run_logged)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)
    monkeypatch.setattr(
        backend,
        "_perception_runner_running",
        lambda: pytest.fail("docker inspect must not run when socket is ready"),
    )

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )
    server_thread.join(timeout=2.0)

    assert result.exit_code == 0
    # Exactly one restart+retry: the stale worker is healed, not hammered.
    assert len(restart_calls) == 1
    assert restart_calls[0][-2:] == ("restart", "perception")
    assert restart_calls[0][0] == str(module.COMPONENT_MANAGER)
    assert state["connections"] == 2
    log_text = log.read_text(encoding="utf-8")
    heal = next(
        json.loads(line)
        for line in log_text.splitlines()
        if line.startswith("{")
        and json.loads(line).get("stage") == "perception_fingerprint_selfheal"
    )
    assert heal["trigger"] == "resident_worker_fingerprint_mismatch"
    assert heal["action"] == "restart_perception"
    shutil.rmtree(artifact_root)


def test_perception_fingerprint_selfheal_is_capped_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    artifact_root = Path(tempfile.mkdtemp(prefix="zmi-", dir="/tmp"))
    output = artifact_root / "go2w_real" / "interactive_sessions" / "sample"
    output.mkdir(parents=True)
    log = tmp_path / "perception.log"
    socket_path = artifact_root / "go2w_real" / ".perception_runner.sock"
    # Never heals: a genuine mid-edit checkout keeps mismatching.
    state = {"healed": False, "connections": 0, "max_connections": 2}
    ready = threading.Event()

    server_thread = threading.Thread(
        target=_serve_fingerprint_worker,
        args=(socket_path, output, module, state, ready),
        daemon=True,
    )
    server_thread.start()
    assert ready.wait(timeout=1.0)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    restart_calls = []

    def fake_run_logged(argv, _log_path, *, environment):
        # Restart runs but the checkout is still mid-edit, so no heal happens.
        restart_calls.append(tuple(argv))
        return module.subprocess.CompletedProcess(tuple(argv), 0)

    monkeypatch.setattr(module, "NUC_KEY", key)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "_run_logged", fake_run_logged)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )
    server_thread.join(timeout=2.0)

    # One restart only (no loop) and the legible rc=70 error still surfaces.
    assert len(restart_calls) == 1
    assert result.exit_code == 70
    assert state["connections"] == 2
    shutil.rmtree(artifact_root)


def test_perception_retries_one_geometric_mask_failure(tmp_path, monkeypatch):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(module, "NUC_KEY", key)
    output = tmp_path / "retry-output"
    output.mkdir()
    log = tmp_path / "perception.log"
    attempts = []

    class FakeProcess:
        def __init__(self, return_code):
            self.return_code = return_code
            self.polls = iter((None, return_code))

        def poll(self):
            return next(self.polls, self.return_code)

        def wait(self, timeout=None):
            return self.return_code

    attempt_argvs = []

    def fake_popen(argv, **_kwargs):
        attempt = len(attempts) + 1
        attempts.append(attempt)
        attempt_argvs.append(tuple(argv))
        if attempt == 2:
            _write_perception_success(output, "white adapter")
        return FakeProcess(4 if attempt == 1 else 0)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    assert attempts == [1, 2]
    assert json.loads((output / "report.json").read_text())["instruction"] == (
        "white adapter"
    )
    assert "Retrying perception" in log.read_text(encoding="utf-8")
    # The first attempt may reuse a live same-instruction track; the retry
    # exists to recover with a fresh segmentation seed, so it must not offer
    # reuse of the exact mask that just failed.
    assert "--reuse-valid-tracking" in attempt_argvs[0]
    assert "--reuse-valid-tracking" not in attempt_argvs[1]
    assert "--tracking-reuse-max-age" not in attempt_argvs[1]


def test_perception_retries_one_explicit_tracker_failure(tmp_path, monkeypatch):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(module, "NUC_KEY", key)
    output = tmp_path / "retry-output"
    output.mkdir()
    log = tmp_path / "perception.log"
    attempts = []

    class FakeProcess:
        def __init__(self, return_code):
            self.return_code = return_code
            self.polls = iter((None, return_code))

        def poll(self):
            return next(self.polls, self.return_code)

        def wait(self, timeout=None):
            return self.return_code

    attempt_argvs = []

    def fake_popen(argv, **_kwargs):
        attempts.append(len(attempts) + 1)
        attempt_argvs.append(tuple(argv))
        if len(attempts) == 1:
            (output / "report.json").write_text(json.dumps({
                "perception_failure": "tracker_reported_loss: transient seed loss",
            }))
        if len(attempts) == 2:
            _write_perception_success(output, "white adapter")
        return FakeProcess(5 if len(attempts) == 1 else 0)

    backend = module.FixedReadOnlyBackend()

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    assert attempts == [1, 2]
    assert "--reuse-valid-tracking" in attempt_argvs[0]
    assert "--reuse-valid-tracking" not in attempt_argvs[1]


def test_perception_camera_timeout_fails_without_replaying_full_timeout(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(module, "NUC_KEY", key)
    output = tmp_path / "camera-timeout"
    output.mkdir()
    log = tmp_path / "perception.log"
    attempts = []

    class FakeProcess:
        def poll(self):
            return 5

        def wait(self, timeout=None):
            return 5

    def fake_popen(_argv, **_kwargs):
        attempts.append(1)
        (output / "report.json").write_text(json.dumps({
            "perception_failure": "camera_frame_timeout",
        }))
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    result = module.FixedReadOnlyBackend().run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 5
    assert attempts == [1]
    assert "Retrying perception" not in log.read_text(encoding="utf-8")
    assert result.error_code == "PERCEPTION_CAMERA_FRAME_TIMEOUT"


def test_perception_camera_timeout_retries_when_rgbd_metadata_is_live(tmp_path):
    module = _integration_module()
    output = tmp_path / "dds-startup-race"
    output.mkdir()
    (output / "report.json").write_text(json.dumps({
        "perception_failure": "camera_frame_timeout",
        "message_counts": {"info": 120, "overlay": 0, "mask": 0, "cloud": 0},
    }))

    assert module.FixedReadOnlyBackend._perception_retryable(output, 5) is True


def test_perception_does_not_retry_object_larger_than_gripper(tmp_path):
    module = _integration_module()
    output = tmp_path / "wrong-large-object"
    output.mkdir()
    detail = (
        "object has no OBB dimension within gripper aperture; "
        "extent=[0.218, 0.186, 0.074]"
    )
    (output / "report.json").write_text(json.dumps({
        "grasp_generation_error": detail,
    }))

    assert module.FixedReadOnlyBackend._perception_retryable(output, 4) is False
    result = module.FixedReadOnlyBackend._perception_failure_result(output, 4)
    assert result.error_code == "GRASP_GEOMETRY_FAILED"
    assert result.message == detail


def test_perception_bringup_defaults_to_verified_pinocchio_image():
    source = (
        ROOT / "scripts" / "runtime" / "go2w_perception_lab.sh"
    ).read_text(encoding="utf-8")

    assert 'IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"' in source


def test_failed_container_attempt_still_updates_latest_attempt(tmp_path):
    backend = FakeBackend()
    backend.perception_result = BackendResult(
        2,
        "PERCEPTION_PROCESS_FAILED",
        "target not found",
    )
    service = _service(tmp_path, backend)

    attempt = service.start_perception("changed object")
    state = service.status()["actions"]["perception"]

    assert attempt["status"] == "failed"
    assert state["latest_attempt"]["session_id"] == attempt["session_id"]
    assert state["latest_attempt"]["error"]["code"] == (
        "PERCEPTION_PROCESS_FAILED"
    )
    assert state["last_good"] is None


def test_freeze_accepts_only_non_writable_foreign_regular_file(
    tmp_path,
    monkeypatch,
):
    session = tmp_path / "session"
    session.mkdir()
    artifact = session / "root-owned-output.json"
    artifact.write_text("{}", encoding="utf-8")
    artifact.chmod(0o444)
    real_chmod = Path.chmod

    def deny_artifact_chmod(path, mode, *args, **kwargs):
        if path == artifact:
            raise PermissionError("simulated foreign ownership")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", deny_artifact_chmod)

    session_contracts._freeze_tree(session)

    assert session.stat().st_mode & 0o777 == 0o500


def test_freeze_rejects_foreign_group_writable_file(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    artifact = session / "unsafe-output.json"
    artifact.write_text("{}", encoding="utf-8")
    artifact.chmod(0o464)
    real_chmod = Path.chmod

    def deny_artifact_chmod(path, mode, *args, **kwargs):
        if path == artifact:
            raise PermissionError("simulated foreign ownership")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", deny_artifact_chmod)

    with pytest.raises(PermissionError):
        session_contracts._freeze_tree(session)


def test_perception_lab_maps_server_artifact_dir_to_fixed_outputs():
    source = (
        ROOT / "scripts" / "runtime" / "go2w_perception_lab.sh"
    ).read_text(encoding="utf-8")
    dry_run = (
        ROOT / "scripts" / "runtime" / "go2w_perception_dry_run.py"
    ).read_text(encoding="utf-8")

    assert 'ARTIFACT_DIR="${Z_MANIP_ARTIFACT_DIR:-' in source
    assert '-v "$ARTIFACT_DIR:/artifacts"' in source
    assert '--output /artifacts' in source
    assert 'Z_MANIP_REQUIRE_PASSIVE_WINDOW' in source
    assert '--passive-window /artifacts/live_passive_joint_report.json' in source
    assert (
        '--selected-passive-window '
        '/artifacts/selected_passive_joint_report.json'
    ) in source
    assert 'args.output / "edgetam_mask.png"' in dry_run
    assert 'args.output / "edgetam_overlay.png"' in dry_run
    assert 'args.output / "grasp_candidates_overlay.png"' in dry_run
    assert 'max_candidates=64' in dry_run


def test_full_session_uses_fast_unprivileged_passive_snapshots():
    source = (
        ROOT / "scripts" / "runtime" / "go2w_planning_session.sh"
    ).read_text(encoding="utf-8")

    assert 'z-manip-runtime:pinocchio' in source
    assert 'PLANNING_ONLY_SEARCH_TIMEOUT_S:-6' in source
    assert 'PLANNING_ONLY_SYMMETRY_SAMPLES:-4' in source
    assert '$STACK_ROOT/z_manip:/opt/z_manip/python/z_manip:ro' in source
    assert (
        '$STACK_ROOT/configs/piper_collision_capsules.json:'
        '/opt/z_manip/configs/piper_collision_capsules.json:ro'
    ) in source
    assert (
        '$STACK_ROOT/ros2/z_manip_task/z_manip_task:'
        '$TASK_PACKAGE_CONTAINER:ro'
    ) in source
    assert 'PLANNING_ONLY_MAX_HYPOTHESES:-64' in source
    assert 'Z_MANIP_PASSIVE_CAPTURE_SECONDS:-0.25' in source
    assert '/usr/local/libexec/z-manip/piper_passive_probe.py' in source
    assert '/tmp/z-manip-passive-live.json' in source
    assert 'sudo -n' not in source


def test_planning_reuses_capture_time_report_and_pinocchio_runtime(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    perception = tmp_path / "perception"
    perception.mkdir()
    joint_report = perception / "selected_passive_joint_report.json"
    joint_report.write_text(json.dumps(_passive_report()), encoding="utf-8")
    output = tmp_path / "planning-attempt"
    output.mkdir()
    log = tmp_path / "planning.log"
    commands = []

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        commands.append(command)
        if str(module.SESSION_GATE) in command:
            destination = Path(command[command.index("--output") + 1])
            destination.write_text(json.dumps({
                "planning_ready": True,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    gate_command, planner_command = commands
    assert gate_command[gate_command.index("--joint-report") + 1] == str(
        joint_report
    )
    assert "/usr/bin/ssh" not in planner_command
    assert "z-manip-runtime:pinocchio" in planner_command
    assert "Z_MANIP_IK_BACKEND=pinocchio" in planner_command
    assert planner_command[planner_command.index("--network") + 1] == "none"
    assert (
        f"{module.STACK_CONFIG}:/opt/z_manip/configs/go2w_piper.json:ro"
    ) in planner_command
    assert (
        f"{module.STACK_ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro"
    ) in planner_command
    assert planner_command[planner_command.index("--scene-clearance-m") + 1] == (
        module.SUPERVISED_SCENE_CLEARANCE_M
    )
    assert planner_command[planner_command.index("--search-timeout-s") + 1] == "6"
    assert planner_command[planner_command.index("--symmetry-samples") + 1] == "4"
    assert planner_command[planner_command.index("--max-hypotheses") + 1] == "64"
    assert planner_command[planner_command.index("--max-feasible-plans") + 1] == "1"
    assert (
        planner_command[planner_command.index("--support-approach-prior-weight") + 1]
        == "0.05"
    )
    assert planner_command[planner_command.index("--user") + 1] == (
        f"{os.geteuid()}:{os.getegid()}"
    )


def test_planning_warm_runner_uses_read_only_inputs_and_atomic_scratch(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    artifact_root = tmp_path / "artifacts"
    perception = artifact_root / "sessions" / "capture" / "perception"
    perception.mkdir(parents=True)
    joint_report = perception / "selected_passive_joint_report.json"
    joint_report.write_text(json.dumps(_passive_report()), encoding="utf-8")
    output = artifact_root / "sessions" / "planning-attempt"
    output.mkdir(parents=True)
    calibration = artifact_root / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    scratch_root = artifact_root / ".planning-runner-scratch"
    log = tmp_path / "planning.log"
    commands = []

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "PLANNING_RUNNER_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(backend, "_planning_runner_running", lambda: True)
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        commands.append(command)
        if str(module.SESSION_GATE) in command:
            destination = Path(command[command.index("--output") + 1])
            destination.write_text(json.dumps({
                "planning_ready": True,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }))
        else:
            container_output = Path(command[command.index("--output") + 1])
            host_output = scratch_root / container_output.name
            (host_output / "planning_report.json").write_text(
                '{"success": true}',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 0
    _, planner_command = commands
    assert planner_command[:5] == (
        "/usr/bin/docker",
        "exec",
        "-e",
        "Z_MANIP_IK_BACKEND=pinocchio",
        module.PLANNING_RUNNER_CONTAINER,
    )
    assert planner_command[planner_command.index("--artifacts") + 1] == (
        "/workspace-artifacts/sessions/capture/perception"
    )
    assert planner_command[planner_command.index("--camera-calibration") + 1] == (
        "/workspace-artifacts/calibration.json"
    )
    assert (output / "planning" / "planning_report.json").is_file()
    assert list(scratch_root.iterdir()) == []


def test_planning_calls_private_runner_socket_and_promotes_atomic_output(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    artifact_root = Path(tempfile.mkdtemp(prefix="zmp-", dir="/tmp"))
    perception = artifact_root / "sessions" / "capture" / "perception"
    perception.mkdir(parents=True)
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = artifact_root / "sessions" / "planning-attempt"
    output.mkdir(parents=True)
    calibration = artifact_root / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    scratch_root = artifact_root / ".planning-runner-scratch"
    scratch_root.mkdir()
    socket_path = scratch_root / ".planner.sock"
    log = tmp_path / "planning.log"
    ready = threading.Event()
    received = []

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                payload = bytearray()
                while True:
                    block = connection.recv(4096)
                    if not block:
                        break
                    payload.extend(block)
                request = json.loads(bytes(payload))
                received.append(request)
                argv = request["argv"]
                container_output = Path(argv[argv.index("--output") + 1])
                host_output = scratch_root / container_output.name
                (host_output / "planning_report.json").write_text(
                    '{"success": true}',
                    encoding="utf-8",
                )
                connection.sendall(json.dumps({
                    "return_code": 0,
                    "elapsed_s": 0.02,
                    "output": "fixed planning request\n",
                    "worker_fingerprint": module.runtime_fingerprint(),
                }).encode("utf-8"))

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    assert ready.wait(timeout=1.0)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "PLANNING_RUNNER_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(
        backend,
        "_planning_runner_running",
        lambda: pytest.fail("docker inspect must not run when socket is ready"),
    )
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        destination = Path(argv[argv.index("--output") + 1])
        destination.write_text(json.dumps({
            "planning_ready": True,
            "measured_joints_rad": [0.0] * 6,
            "planning_start_joints_rad": [0.0] * 6,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_logged", fake_run)
    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )
    server_thread.join(timeout=1.0)

    assert result.exit_code == 0
    assert received[0]["ik_backend"] == "pinocchio"
    assert (output / "planning" / "planning_report.json").is_file()
    timing = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    search = next(item for item in timing if item["stage"] == "planning_search")
    assert search["runner_transport"] == "unix_socket"
    assert search["worker_elapsed_s"] == pytest.approx(0.02)
    shutil.rmtree(artifact_root)


def test_planning_warm_runner_exit_without_report_fails_closed(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    artifact_root = tmp_path / "artifacts"
    perception = artifact_root / "sessions" / "capture" / "perception"
    perception.mkdir(parents=True)
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = artifact_root / "sessions" / "planning-attempt"
    output.mkdir(parents=True)
    calibration = artifact_root / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    scratch_root = artifact_root / ".planning-runner-scratch"
    stale_sibling = scratch_root / "planning-unrelated"
    stale_sibling.mkdir(parents=True)
    (stale_sibling / "planning_report.json").write_text(
        '{"success": true, "old": true}',
        encoding="utf-8",
    )
    log = tmp_path / "planning.log"
    visualization_calls = []
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "PLANNING_RUNNER_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(backend, "_planning_runner_running", lambda: True)
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: visualization_calls.append(True),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        if str(module.SESSION_GATE) in command:
            destination = Path(command[command.index("--output") + 1])
            destination.write_text(json.dumps({
                "planning_ready": True,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }))
            return SimpleNamespace(returncode=0)
        # Simulate inspect succeeding immediately before the runner exits.
        return SimpleNamespace(returncode=125)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 125
    assert result.error_code == "PLANNING_RUNNER_OUTPUT_MISSING"
    assert list((output / "planning").iterdir()) == []
    assert stale_sibling.is_dir()
    assert visualization_calls == []
    assert not any(
        path.name != stale_sibling.name for path in scratch_root.iterdir()
    )


def test_planning_warm_runner_atomic_promotion_failure_cleans_scratch(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    artifact_root = tmp_path / "artifacts"
    perception = artifact_root / "sessions" / "capture" / "perception"
    perception.mkdir(parents=True)
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = artifact_root / "sessions" / "planning-attempt"
    output.mkdir(parents=True)
    calibration = artifact_root / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    scratch_root = artifact_root / ".planning-runner-scratch"
    log = tmp_path / "planning.log"
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "PLANNING_RUNNER_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(backend, "_planning_runner_running", lambda: True)

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        destination = Path(command[command.index("--output") + 1])
        if str(module.SESSION_GATE) in command:
            destination.write_text(json.dumps({
                "planning_ready": True,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }))
        else:
            host_output = scratch_root / destination.name
            (host_output / "planning_report.json").write_text(
                '{"success": true}',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_logged", fake_run)
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(
            OSError("synthetic cross-device failure"),
        ),
    )

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.error_code == "PLANNING_RUNNER_OUTPUT_INVALID"
    assert (output / "planning").is_dir()
    assert list((output / "planning").iterdir()) == []
    assert list(scratch_root.iterdir()) == []


def test_planning_runner_scratch_cleanup_is_old_prefix_only(tmp_path):
    module = _integration_module()
    scratch_root = tmp_path / "scratch"
    old = scratch_root / "planning-old"
    fresh = scratch_root / "planning-active"
    unrelated = scratch_root / "operator-note"
    old.mkdir(parents=True)
    fresh.mkdir()
    unrelated.mkdir()
    symlink = scratch_root / "planning-link"
    symlink.symlink_to(old, target_is_directory=True)
    os.utime(old, (100.0, 100.0))
    os.utime(fresh, (9_950.0, 9_950.0))
    os.utime(unrelated, (100.0, 100.0))

    module._cleanup_stale_planning_runner_scratch(
        scratch_root,
        now_s=10_000.0,
        max_age_s=1_000.0,
    )

    assert not old.exists()
    assert fresh.is_dir()
    assert unrelated.is_dir()
    assert symlink.is_symlink()


def test_planning_runner_bringup_keeps_artifacts_read_only():
    source = (
        ROOT / "scripts" / "runtime" / "go2w_perception_lab.sh"
    ).read_text(encoding="utf-8")

    function = source.split("start_planning_runner() {", 1)[1].split("\n}", 1)[0]
    assert "--network none" in function
    assert "--cap-drop ALL" in function
    assert "--security-opt no-new-privileges" in function
    assert '$PERCEPTION_RUNNER_ARTIFACT_ROOT:/workspace-artifacts:ro' in function
    assert '$PLANNING_RUNNER_SCRATCH_ROOT:/workspace-planning-output' in function
    assert "--device" not in function


def test_planning_propagates_verified_need_base_approach_disposition(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    perception = tmp_path / "perception"
    perception.mkdir()
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = tmp_path / "planning-attempt"
    output.mkdir()
    log = tmp_path / "planning.log"
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        assert str(module.SESSION_GATE) in command
        destination = Path(command[command.index("--output") + 1])
        destination.write_text(json.dumps({
            "schema": "z_manip.piper_planning_session_gate.v1",
            "planning_ready": False,
            "read_only": True,
            "planning_only": True,
            "motion_commands_published": 0,
            "transport_opened": False,
            "planning_disposition": "NEED_BASE_APPROACH",
            "handoff_workspace": {
                "state": "NEED_BASE_APPROACH",
                "planning_allowed": False,
                "frame": "piper_base_link",
                "target_range_m": 0.83,
                "maximum_handoff_range_m": 0.70,
            },
            "errors": [{"code": "NEED_BASE_APPROACH"}],
        }), encoding="utf-8")
        return SimpleNamespace(returncode=8)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 8
    assert result.error_code == "NEED_BASE_APPROACH"
    assert "0.830 m > 0.700 m" in result.message


def test_planning_does_not_trust_incomplete_need_base_approach_report(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    perception = tmp_path / "perception"
    perception.mkdir()
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = tmp_path / "planning-attempt"
    output.mkdir()
    log = tmp_path / "planning.log"
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        destination = Path(command[command.index("--output") + 1])
        # The typed string alone is insufficient: missing zero-motion safety
        # evidence and workspace/error agreement must remain fail-closed.
        destination.write_text(json.dumps({
            "schema": "z_manip.piper_planning_session_gate.v1",
            "planning_ready": False,
            "planning_disposition": "NEED_BASE_APPROACH",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=8)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 8
    assert result.error_code == "SESSION_GATE_BLOCKED"


def test_planning_failure_surfaces_bounded_report_rejection_summary(
    tmp_path,
    monkeypatch,
):
    module = _integration_module()
    perception = tmp_path / "perception"
    perception.mkdir()
    (perception / "selected_passive_joint_report.json").write_text(
        json.dumps(_passive_report()),
        encoding="utf-8",
    )
    output = tmp_path / "planning-attempt"
    output.mkdir()
    log = tmp_path / "planning.log"
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )
    monkeypatch.setattr(backend, "_required_planning_files", lambda: ())
    monkeypatch.setattr(
        backend,
        "_build_visualization_bundle",
        lambda **_kwargs: module.BackendResult(0),
    )

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        if str(module.SESSION_GATE) in command:
            destination = Path(command[command.index("--output") + 1])
            destination.write_text(json.dumps({
                "planning_ready": True,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }))
            return SimpleNamespace(returncode=0)
        (output / "planning" / "planning_report.json").write_text(json.dumps({
            "plan_valid": False,
            "error": "GraspPlanningError: no candidate survived",
            "rejection_count": 4,
            "rejections": [
                {"stage": "ik"},
                {"stage": "ik"},
                {"stage": "ik"},
                {"stage": "approach_collision"},
            ],
        }))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module, "_run_logged", fake_run)

    result = backend.run_planning(
        perception_dir=perception,
        output_dir=output,
        log_path=log,
    )

    assert result.exit_code == 7
    assert result.error_code == "OFFLINE_PLANNER_BLOCKED"
    assert result.message == (
        "GraspPlanningError: no candidate survived; "
        "rejection summary: 4 total (ik=3, approach_collision=1)"
    )
    assert "network-disabled" not in result.message


def test_planning_failure_report_reader_rejects_symlink_and_oversize(tmp_path):
    module = _integration_module()
    output = tmp_path / "attempt"
    planning = output / "planning"
    planning.mkdir(parents=True)
    report = planning / "planning_report.json"
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"error": "must not be read"}))
    report.symlink_to(foreign)

    symlink_message = module._planning_failure_message(output)

    assert "must not be read" not in symlink_message
    assert "no valid grasp plan" in symlink_message
    report.unlink()
    report.write_bytes(b"x" * (module.MAX_PLANNING_REPORT_BYTES + 1))

    oversize_message = module._planning_failure_message(output)

    assert "no valid grasp plan" in oversize_message


def test_planning_bundle_is_built_and_safety_audited(tmp_path, monkeypatch):
    module = _integration_module()
    perception = tmp_path / "perception"
    perception.mkdir()
    joint_report = perception / "selected_passive_joint_report.json"
    joint_report.write_text(json.dumps(_passive_report()))
    output = tmp_path / "planning"
    (output / "planning").mkdir(parents=True)
    (output / "planning" / "planning_report.json").write_text("{}")
    log = tmp_path / "planning.log"
    commands = []

    def fake_run(argv, _log_path, *, environment):
        command = tuple(argv)
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        if str(module.DEBUG_BUNDLE) in command:
            destination.write_text(json.dumps({
                "schema": "z_manip.debug_bundle.v1",
                "visualization": {"images": {
                    "segmentation_mask": "mask",
                    "segmentation_overlay": "overlay",
                    "candidate_overlay": "candidates",
                }},
                "safety": {
                    "motion_commands_published": 0,
                    "transport_opened": False,
                    "can_opened": False,
                },
            }))
        else:
            destination.write_text(json.dumps({
                "schema": "z_manip.debug_safety_audit.v1",
                "passed": True,
                "motion_commands_published": 0,
            }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_logged", fake_run)
    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    result = backend._build_visualization_bundle(
        perception_dir=perception,
        output_dir=output,
        joint_report=joint_report,
        log_path=log,
        environment={},
    )

    assert result.exit_code == 0
    assert (output / "debug_bundle.json").is_file()
    assert (output / "debug_bundle.safety-audit.json").is_file()
    assert str(module.DEBUG_BUNDLE) in commands[0]
    assert "--planning-dir" in commands[0]
    assert str(module.SAFETY_GATE) in commands[1]
    assert "--joint-report" in commands[1]


def test_integration_module_is_importable_without_running_hardware():
    module = _integration_module()

    assert module.FixedReadOnlyBackend is not None
    assert module.RUN_ROOT.name == "interactive_sessions"


def test_perception_does_not_retry_grasp_geometry_failure_with_valid_seed(
    tmp_path,
    monkeypatch,
):
    """A seeded rc4 must return immediately: the mobile flow advances on it,
    and a fresh-grounding retry only adds seconds and churns the live track."""

    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(module, "NUC_KEY", key)
    output = tmp_path / "seeded-rc4"
    output.mkdir()
    log = tmp_path / "perception.log"
    attempts = []

    class FakeProcess:
        def __init__(self, return_code):
            self.return_code = return_code
            self.polls = iter((None, return_code))

        def poll(self):
            return next(self.polls, self.return_code)

        def wait(self, timeout=None):
            return self.return_code

    def fake_popen(_argv, **_kwargs):
        attempts.append(len(attempts) + 1)
        (output / "report.json").write_text(json.dumps({
            "instruction": "white adapter",
            "filtered_target_points": 1777,
            "grasp_generation_valid": False,
            "grasp_generation_error": (
                "no antipodal or aperture-bounded OBB grasp; observed "
                "extent=[0.118, 0.116, 0.069]"
            ),
        }))
        return FakeProcess(4)

    backend = module.FixedReadOnlyBackend(
        module.ServerRuntimeConfig.from_server_environment({}),
    )

    def fake_capture(output_dir, _log_path, _environment):
        payload = json.dumps(_passive_report())
        (output_dir / "live_passive_joint_report.json").write_text(payload)
        (output_dir / "selected_passive_joint_report.json").write_text(payload)
        return module.BackendResult(0)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backend, "_capture_passive_window", fake_capture)

    result = backend.run_perception(
        target="white adapter",
        output_dir=output,
        log_path=log,
    )

    assert attempts == [1]
    assert result.exit_code == 4
    assert "Retrying perception" not in log.read_text(encoding="utf-8")
