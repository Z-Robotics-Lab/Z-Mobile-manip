from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "runtime"
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location(
    "piper_full_grasp_executor_contract",
    RUNTIME / "piper_full_grasp_executor.py",
)
assert SPEC is not None and SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTOR)
CONTROL_SPEC = importlib.util.spec_from_file_location(
    "go2w_planning_control_staged_contract",
    RUNTIME / "go2w_planning_control.py",
)
assert CONTROL_SPEC is not None and CONTROL_SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(CONTROL)


ARTIFACT_ID = "a" * 64
SESSION_ID = "20260720-120000"
Q_HOME = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
Q_PRE = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
Q_GRASP = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
Q_LIFT = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def artifact(*, artifact_id: str = ARTIFACT_ID):
    return SimpleNamespace(
        artifact_id=artifact_id,
        arrays={"approach_raw": np.asarray([Q_PRE, Q_GRASP])},
    )


def write_workflow(
    directory: Path,
    *,
    phase: str,
    artifact_id: str = ARTIFACT_ID,
    planning_session_id: str = SESSION_ID,
    holding_object: bool = True,
) -> Path:
    directory.mkdir()
    path = directory / "workflow-state.json"
    path.write_text(
        json.dumps(
            {
                "schema": "z_manip.piper_grasp_workflow_state.v1",
                "artifact_id": artifact_id,
                "planning_session_id": planning_session_id,
                "prior_workflow_sha256": None,
                "phase": phase,
                "holding_object": holding_object,
                "at_home": phase == "holding_at_home",
                "final_joints_rad": Q_HOME.tolist(),
                "finished_unix_ns": time.time_ns(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_executor_start_receipt_is_bound_and_pre_motion(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()

    document = EXECUTOR._write_executor_start_receipt(
        receipt_dir,
        artifact=artifact(),
        workflow_phase="full",
        planning_session_id=SESSION_ID,
        started_unix_ns=1_800_000_000_000_000_000,
        started_monotonic_ns=987_654_321,
    )

    persisted = json.loads(
        (receipt_dir / "executor-start-receipt.json").read_text(encoding="utf-8"),
    )
    assert persisted == document
    assert document["artifact_id"] == ARTIFACT_ID
    assert document["transport_opened"] is True
    assert document["commands_sent"] == 0
    assert document["motion_started"] is False


def patch_paths(monkeypatch, events: list[tuple[str, np.ndarray]]):
    # Holding feedback has its own focused executor tests. Keep this contract
    # test concerned with which already-checked paths a continuation executes.
    monkeypatch.setattr(EXECUTOR, "_verify_holding_object", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        EXECUTOR.stage_executor,
        "validate_stage_context",
        lambda *_args, **_kwargs: np.asarray([Q_HOME, Q_PRE]),
    )
    monkeypatch.setattr(
        EXECUTOR.stage_executor,
        "timed_stage_path",
        lambda *_args, **_kwargs: (
            np.asarray([Q_GRASP, Q_LIFT]),
            np.asarray([0.0, 1.0]),
        ),
    )

    def timed(_robot, path, _times, _guard, **_kwargs):
        copied = np.asarray(path, dtype=float).copy()
        events.append(("timed", copied))
        return copied[-1]

    def joint(_robot, path, _guard, **_kwargs):
        copied = np.asarray(path, dtype=float).copy()
        events.append(("joint", copied))
        return copied[-1]

    monkeypatch.setattr(EXECUTOR.stage_executor, "execute_timed_joint_path", timed)
    monkeypatch.setattr(EXECUTOR.stage_executor, "execute_joint_path", joint)


def test_workflow_receipt_binds_artifact_session_and_prior_digest(tmp_path):
    receipt_dir = tmp_path / "next"
    receipt_dir.mkdir()
    document = EXECUTOR._workflow_state(
        receipt_dir,
        artifact=artifact(),
        phase="holding_at_home",
        final_joints=Q_HOME,
        holding_object=True,
        at_home=True,
        planning_session_id=SESSION_ID,
        prior_workflow_sha256="b" * 64,
    )

    assert document["artifact_id"] == ARTIFACT_ID
    assert document["planning_session_id"] == SESSION_ID
    assert document["prior_workflow_sha256"] == "b" * 64
    assert document["phase"] == "holding_at_home"
    assert document["holding_object"] is True
    assert document["at_home"] is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"artifact_id": "c" * 64}, "different planning artifact"),
        ({"planning_session_id": "foreign-session"}, "different planning session"),
        ({"phase": "placed_back_at_home"}, "workflow phase"),
        ({"holding_object": False}, "held object"),
    ],
)
def test_continuation_rejects_cross_task_or_out_of_order_receipt(
    tmp_path,
    override,
    message,
):
    prior = tmp_path / "prior"
    path = write_workflow(prior, phase="holding_at_lift")
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(override)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(EXECUTOR.stage_executor.SafetyError, match=message):
        EXECUTOR._load_workflow_state(
            prior,
            artifact=artifact(),
            expected_phase="holding_at_lift",
            planning_session_id=SESSION_ID,
        )


def test_return_home_holding_is_exact_reverse_and_never_opens_gripper(
    tmp_path,
    monkeypatch,
):
    prior = tmp_path / "pick"
    prior_path = write_workflow(prior, phase="holding_at_lift")
    events: list[tuple[str, np.ndarray]] = []
    patch_paths(monkeypatch, events)
    monkeypatch.setattr(
        EXECUTOR,
        "_open_gripper",
        lambda *_args, **_kwargs: pytest.fail("return-home-holding opened the gripper"),
    )

    output = tmp_path / "return"
    result = EXECUTOR.execute_workflow_phase(
        object(),
        object(),
        artifact(),
        workflow_phase="return-home-holding",
        planning_session_id=SESSION_ID,
        receipt_dir=output,
        prior_receipt_dir=prior,
        speed_percent=5,
        segment_timeout_s=12.0,
        gripper_force_n=1.0,
    )

    assert [name for name, _path in events] == ["timed", "joint", "joint"]
    np.testing.assert_allclose(events[0][1], [Q_LIFT, Q_GRASP])
    np.testing.assert_allclose(events[1][1], [Q_GRASP, Q_PRE])
    np.testing.assert_allclose(events[2][1], [Q_PRE, Q_HOME])
    workflow = result["workflow"]
    assert workflow["phase"] == "holding_at_home"
    assert workflow["holding_object"] is True
    assert workflow["at_home"] is True
    assert workflow["prior_workflow_sha256"] == hashlib.sha256(
        prior_path.read_bytes(),
    ).hexdigest()


def test_place_back_opens_only_at_original_grasp_then_reverses_home(
    tmp_path,
    monkeypatch,
):
    prior = tmp_path / "home-holding"
    prior_path = write_workflow(prior, phase="holding_at_home")
    events: list[tuple[str, np.ndarray]] = []
    patch_paths(monkeypatch, events)

    def opened(*_args, **_kwargs):
        events.append(("open", np.empty((0, 6))))

    monkeypatch.setattr(EXECUTOR, "_open_gripper", opened)
    output = tmp_path / "placed"
    result = EXECUTOR.execute_workflow_phase(
        object(),
        object(),
        artifact(),
        workflow_phase="place-back",
        planning_session_id=SESSION_ID,
        receipt_dir=output,
        prior_receipt_dir=prior,
        speed_percent=5,
        segment_timeout_s=12.0,
        gripper_force_n=1.0,
    )

    assert [name for name, _path in events] == ["joint", "joint", "open", "joint", "joint"]
    np.testing.assert_allclose(events[0][1], [Q_HOME, Q_PRE])
    np.testing.assert_allclose(events[1][1], [Q_PRE, Q_GRASP])
    np.testing.assert_allclose(events[3][1], [Q_GRASP, Q_PRE])
    np.testing.assert_allclose(events[4][1], [Q_PRE, Q_HOME])
    workflow = result["workflow"]
    assert workflow["phase"] == "placed_back_at_home"
    assert workflow["holding_object"] is False
    assert workflow["at_home"] is True
    assert workflow["prior_workflow_sha256"] == hashlib.sha256(
        prior_path.read_bytes(),
    ).hexdigest()


def test_place_back_from_lift_lowers_releases_and_reverses_home(
    tmp_path,
    monkeypatch,
):
    prior = tmp_path / "lift-holding"
    prior_path = write_workflow(prior, phase="holding_at_lift")
    events: list[tuple[str, np.ndarray]] = []
    patch_paths(monkeypatch, events)

    def opened(*_args, **_kwargs):
        events.append(("open", np.empty((0, 6))))

    monkeypatch.setattr(EXECUTOR, "_open_gripper", opened)
    result = EXECUTOR.execute_workflow_phase(
        object(),
        object(),
        artifact(),
        workflow_phase="place-back",
        planning_session_id=SESSION_ID,
        receipt_dir=tmp_path / "placed-from-lift",
        prior_receipt_dir=prior,
        speed_percent=5,
        segment_timeout_s=12.0,
        gripper_force_n=1.0,
    )

    assert [name for name, _path in events] == ["timed", "open", "joint", "joint"]
    np.testing.assert_allclose(events[0][1], [Q_LIFT, Q_GRASP])
    np.testing.assert_allclose(events[2][1], [Q_GRASP, Q_PRE])
    np.testing.assert_allclose(events[3][1], [Q_PRE, Q_HOME])
    workflow = result["workflow"]
    assert workflow["phase"] == "placed_back_at_home"
    assert workflow["holding_object"] is False
    assert workflow["at_home"] is True
    assert workflow["prior_workflow_sha256"] == hashlib.sha256(
        prior_path.read_bytes(),
    ).hexdigest()


def test_pick_hold_stops_at_lift_without_implicit_return_or_release(
    tmp_path,
    monkeypatch,
):
    stages: list[str] = []
    monkeypatch.setattr(
        EXECUTOR.stage_executor,
        "validate_stage_context",
        lambda _artifact, stage, _prior: np.asarray([Q_HOME, Q_PRE]),
    )

    def execute_stage(_robot, _effector, _artifact, stage, path, **_kwargs):
        stages.append(stage)
        return np.asarray(path)[-1], None

    monkeypatch.setattr(EXECUTOR.stage_executor, "execute_stage", execute_stage)
    monkeypatch.setattr(EXECUTOR, "_receipt", lambda **_kwargs: object())
    monkeypatch.setattr(
        EXECUTOR,
        "_open_gripper",
        lambda *_args, **_kwargs: pytest.fail("pick-hold released the object"),
    )
    monkeypatch.setattr(
        EXECUTOR.stage_executor,
        "execute_joint_path",
        lambda *_args, **_kwargs: pytest.fail("pick-hold started an implicit Home return"),
    )

    result = EXECUTOR.execute_workflow_phase(
        object(),
        object(),
        artifact(),
        workflow_phase="pick-hold",
        planning_session_id=SESSION_ID,
        receipt_dir=tmp_path / "pick",
        prior_receipt_dir=None,
        speed_percent=5,
        segment_timeout_s=12.0,
        gripper_force_n=1.0,
    )

    assert stages == ["pregrasp", "approach_close", "lift"]
    assert result["workflow"]["phase"] == "holding_at_lift"
    assert result["workflow"]["holding_object"] is True
    assert result["workflow"]["at_home"] is False


def test_http_staged_actions_are_distinct_and_never_call_legacy_full_chain(tmp_path):
    calls: list[tuple] = []

    class ControlBackend:
        def status(self):
            return {"available": True, "running": False, "state": "idle"}

    class GraspRunner:
        def status(self):
            return {
                "schema": "z_manip.grasp_action.v1",
                "available": True,
                "running": False,
                "workflow": {
                    "phase": "ready_at_home",
                    "artifact_id": ARTIFACT_ID,
                    "planning_session_id": SESSION_ID,
                    "holding_object": False,
                    "at_home": True,
                },
            }

        def start(self, *_args, **_kwargs):
            pytest.fail("a staged endpoint invoked the legacy full-chain action")

        def start_selected(self, *_args, **_kwargs):
            pytest.fail("a staged endpoint invoked Direct Perform")

        def start_pick_hold(self, target, speed_percent=5):
            calls.append(("pick_hold", target, speed_percent))
            return {"started": True, "grasp": self.status()}

        def start_return_home_holding(self, speed_percent=5):
            calls.append(("return_home_holding", speed_percent))
            return {"started": True, "grasp": self.status()}

        def start_place_back(self, speed_percent=5):
            calls.append(("place_back", speed_percent))
            return {"started": True, "grasp": self.status()}

    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "z_manip.debug_bundle.v1",
                "mode": {"read_only": True},
                "safety": {"motion_commands_published": 0},
                "stages": [],
                "artifacts": {},
                "visualization": {},
            },
        ),
        encoding="utf-8",
    )
    server = CONTROL.create_server(
        bundle,
        port=0,
        index_path=ROOT / "web" / "debug_dashboard" / "index.html",
        control_backend=ControlBackend(),
        runtime_state=None,
        grasp_runner=GraspRunner(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def post(path: str, action: str, body: dict[str, object]) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(
                "POST",
                path,
                body=json.dumps(body).encode("utf-8"),
                headers={
                    "Host": f"127.0.0.1:{port}",
                    "Origin": f"http://127.0.0.1:{port}",
                    "Content-Type": "application/json",
                    "X-Z-Manip-Action": action,
                },
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    try:
        assert post(
            "/api/grasp/pick-hold",
            CONTROL.PICK_HOLD_ACTION,
            {"target": "white adapter", "speed_percent": 7},
        ) == 202
        assert post(
            "/api/grasp/return-home-holding",
            CONTROL.RETURN_HOME_HOLDING_ACTION,
            {"speed_percent": 8},
        ) == 202
        assert post(
            "/api/grasp/place-back",
            CONTROL.PLACE_BACK_ACTION,
            {"speed_percent": 9},
        ) == 202
        assert calls == [
            ("pick_hold", "white adapter", 7),
            ("return_home_holding", 8),
            ("place_back", 9),
        ]

        before = list(calls)
        assert post(
            "/api/grasp/place-back",
            CONTROL.PICK_HOLD_ACTION,
            {"speed_percent": 9},
        ) == 403
        assert calls == before
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_runner_exposes_only_the_next_valid_physical_transition(monkeypatch):
    scheduled: list[str] = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            scheduled.append(self.name)

    monkeypatch.setattr(CONTROL.threading, "Thread", DeferredThread)
    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": False,
        "state": "idle",
        "phase": "idle",
        "outcome": None,
    }
    runner._workflow = {
        "phase": "ready_at_home",
        "artifact_id": None,
        "planning_session_id": None,
        "holding_object": False,
        "at_home": True,
        "receipt_dir": None,
        "planning_report": None,
        "planned_grasp": None,
    }

    blocked = runner.start_return_home_holding(5)
    assert blocked["started"] is False
    assert blocked["error"]["code"] == "WORKFLOW_PHASE_MISMATCH"
    assert scheduled == []

    pick = runner.start_pick_hold("white adapter", 5)
    assert pick["started"] is True
    assert scheduled == ["z-manip-pick_hold"]

    runner._status["running"] = False
    runner._workflow.update(
        phase="holding_at_lift",
        artifact_id=ARTIFACT_ID,
        planning_session_id=SESSION_ID,
        holding_object=True,
        at_home=False,
    )
    assert runner.start_place_back(5)["started"] is True
    assert scheduled[-1] == "z-manip-place_back"

    runner._status["running"] = False
    assert runner.start_return_home_holding(5)["started"] is True
    assert scheduled[-1] == "z-manip-return_home_holding"

    runner._status["running"] = False
    runner._workflow.update(phase="holding_at_home", at_home=True)
    assert runner.start_pick_hold("other object", 5)["started"] is False
    assert runner.start_place_back(5)["started"] is True
    assert scheduled[-1] == "z-manip-place_back"
