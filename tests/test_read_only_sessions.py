from __future__ import annotations

import ast
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
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
    for name in (
        "report.json",
        "edgetam_mask.png",
        "edgetam_overlay.png",
        "grasp_candidates_overlay.png",
    ):
        (output / name).write_bytes(b"fixed")

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
        captured["environment"] = dict(kwargs["env"])
        return FakeProcess()

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


def test_perception_uses_warm_runner_for_workspace_artifacts(tmp_path, monkeypatch):
    module = _integration_module()
    key = tmp_path / "server-key"
    key.write_text("test", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "go2w_real" / "interactive_sessions" / "sample"
    output.mkdir(parents=True)
    log = tmp_path / "perception.log"
    for name in (
        "report.json",
        "edgetam_mask.png",
        "edgetam_overlay.png",
        "grasp_candidates_overlay.png",
    ):
        (output / name).write_bytes(b"fixed")
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
    assert "run" not in argv[:4]
    assert argv[argv.index("--output") + 1] == (
        "/workspace-artifacts/go2w_real/interactive_sessions/sample"
    )


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

    def fake_popen(_argv, **_kwargs):
        attempt = len(attempts) + 1
        attempts.append(attempt)
        if attempt == 2:
            for name in (
                "report.json",
                "edgetam_mask.png",
                "edgetam_overlay.png",
                "grasp_candidates_overlay.png",
            ):
                (output / name).write_bytes(f"attempt-{attempt}".encode())
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
    assert (output / "report.json").read_bytes() == b"attempt-2"
    assert "Retrying perception" in log.read_text(encoding="utf-8")


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

    def fake_popen(_argv, **_kwargs):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            (output / "report.json").write_text(json.dumps({
                "perception_failure": "tracker_reported_loss: transient seed loss",
            }))
        if len(attempts) == 2:
            for name in (
                "report.json",
                "edgetam_mask.png",
                "edgetam_overlay.png",
                "grasp_candidates_overlay.png",
            ):
                (output / name).write_bytes(b"recovered")
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
