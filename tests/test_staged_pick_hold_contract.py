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
# J3 is limited to [-170, 0] degrees, so these fixtures keep a negative third
# joint: the carry polyline is now validated against the real PiPER envelope.
Q_HOME = np.asarray([0.0, 0.1, -0.2, 0.3, 0.4, 0.5])
Q_PRE = np.asarray([0.1, 0.2, -0.3, 0.4, 0.5, 0.6])
Q_GRASP = np.asarray([0.2, 0.3, -0.4, 0.5, 0.6, 0.7])
Q_LIFT = np.asarray([0.3, 0.4, -0.5, 0.6, 0.7, 0.8])


STAGE = EXECUTOR.stage_executor


def holding_return(
    *,
    lift: bool = True,
    reverse_approach: bool = True,
    carry: bool = True,
    return_transit: bool = True,
    carry_rejection_reason: str = "",
    carry_raw_waypoints: int = 2,
):
    return {
        "schema": STAGE.HOLDING_RETURN_SCHEMA,
        "collision_model": "closed_on_object_with_platform_fixtures",
        "attached_at": "grasp",
        "segments": {
            "lift": lift,
            "reverse_approach": reverse_approach,
            "carry": carry,
            "return_transit": return_transit,
        },
        "carry_raw_waypoints": carry_raw_waypoints,
        "carry_tip_height_m": {},
        "carry_rejection_reason": carry_rejection_reason,
    }


def artifact(
    *,
    artifact_id: str = ARTIFACT_ID,
    report: dict | None = None,
    with_carry: bool = True,
):
    arrays = {
        "approach_raw": np.asarray([Q_PRE, Q_GRASP]),
        "transit_raw": np.asarray([Q_HOME, Q_PRE]),
        "lift_raw": np.asarray([Q_GRASP, Q_LIFT]),
    }
    if with_carry:
        arrays["carry_raw"] = np.asarray([Q_LIFT, Q_PRE])
    return SimpleNamespace(
        artifact_id=artifact_id,
        report_sha256="b" * 64,
        npz_sha256="c" * 64,
        report={"holding_return": holding_return()} if report is None else report,
        arrays=arrays,
    )


def write_workflow(
    directory: Path,
    *,
    phase: str,
    artifact_id: str = ARTIFACT_ID,
    planning_session_id: str = SESSION_ID,
    holding_object: bool = True,
    final_joints: np.ndarray | None = None,
) -> Path:
    if final_joints is None:
        # Pick & Hold leaves the arm at the lift top; every other continuation
        # hands over at Home.
        final_joints = Q_LIFT if phase == "holding_at_lift" else Q_HOME
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
                "final_joints_rad": np.asarray(final_joints, dtype=float).tolist(),
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
    assert document["planning_report_sha256"] == "b" * 64
    assert document["planned_grasp_sha256"] == "c" * 64
    assert document["planning_session_id"] == SESSION_ID
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


def _return_home(tmp_path, monkeypatch, events, *, plan, prior_phase="holding_at_lift"):
    prior = tmp_path / "pick"
    prior_path = write_workflow(prior, phase=prior_phase)
    patch_paths(monkeypatch, events)
    monkeypatch.setattr(
        EXECUTOR,
        "_open_gripper",
        lambda *_args, **_kwargs: pytest.fail("return-home-holding opened the gripper"),
    )
    result = EXECUTOR.execute_workflow_phase(
        object(),
        object(),
        plan,
        workflow_phase="return-home-holding",
        planning_session_id=SESSION_ID,
        receipt_dir=tmp_path / "return",
        prior_receipt_dir=prior,
        speed_percent=5,
        segment_timeout_s=12.0,
        gripper_force_n=1.0,
    )
    return prior_path, result


def test_return_home_holding_carries_straight_home_without_putting_the_object_down(
    tmp_path,
    monkeypatch,
):
    events: list[tuple[str, np.ndarray]] = []
    prior_path, result = _return_home(
        tmp_path, monkeypatch, events, plan=artifact(),
    )

    # One continuous carry from the lift top to the pregrasp, then the checked
    # run Home.  No timed segment at all: the reverse lift -- the put-down --
    # is gone, and the object never revisits the grasp pose.
    assert [name for name, _path in events] == ["joint", "joint"]
    np.testing.assert_allclose(events[0][1], [Q_LIFT, Q_PRE])
    np.testing.assert_allclose(events[1][1], [Q_PRE, Q_HOME])
    assert not any(
        np.allclose(path[-1], Q_GRASP) for _name, path in events
    ), "the carry must never return the object to the grasp pose"
    workflow = result["workflow"]
    assert workflow["phase"] == "holding_at_home"
    assert workflow["holding_object"] is True
    assert workflow["at_home"] is True
    assert workflow["prior_workflow_sha256"] == hashlib.sha256(
        prior_path.read_bytes(),
    ).hexdigest()


def test_return_home_holding_falls_back_to_the_checked_replay_without_a_carry(
    tmp_path,
    monkeypatch,
):
    # A plan whose direct corner-cut was rejected still returns Home, on the
    # legacy reverse-lift corridor -- but only because that corridor itself
    # passed the loaded check.
    events: list[tuple[str, np.ndarray]] = []
    plan = artifact(
        report={
            "holding_return": holding_return(
                carry=False,
                carry_rejection_reason="collision",
            ),
        },
        with_carry=False,
    )
    prior_path, result = _return_home(tmp_path, monkeypatch, events, plan=plan)

    assert [name for name, _path in events] == ["timed", "joint", "joint"]
    np.testing.assert_allclose(events[0][1], [Q_LIFT, Q_GRASP])
    np.testing.assert_allclose(events[1][1], [Q_GRASP, Q_PRE])
    np.testing.assert_allclose(events[2][1], [Q_PRE, Q_HOME])
    assert result["workflow"]["phase"] == "holding_at_home"
    assert result["workflow"]["prior_workflow_sha256"] == hashlib.sha256(
        prior_path.read_bytes(),
    ).hexdigest()


@pytest.mark.parametrize(
    ("report", "message"),
    [
        # The defect: an artifact planned before attached-object validation
        # existed.  Its transit and approach were checked with an EMPTY
        # gripper and reversing them does not revalidate them.
        ({}, "no holding_return validation"),
        # The run Home is driven on every route, so it is never optional.
        (
            {"holding_return": holding_return(return_transit=False)},
            "run Home was rejected with the object attached",
        ),
        # Neither the corner-cut nor the replay survives the loaded check.
        (
            {
                "holding_return": holding_return(
                    carry=False,
                    lift=False,
                    carry_rejection_reason="collision",
                ),
            },
            "no return corridor is valid with the object attached",
        ),
        (
            {
                "holding_return": holding_return(
                    carry=False,
                    reverse_approach=False,
                    carry_rejection_reason="collision",
                ),
            },
            "no return corridor is valid with the object attached",
        ),
    ],
)
def test_return_home_holding_never_drives_an_unvalidated_loaded_corridor(
    tmp_path,
    monkeypatch,
    report,
    message,
):
    events: list[tuple[str, np.ndarray]] = []
    prior = tmp_path / "pick"
    write_workflow(prior, phase="holding_at_lift")
    patch_paths(monkeypatch, events)
    monkeypatch.setattr(
        EXECUTOR,
        "_verify_holding_object",
        lambda *_a, **_k: pytest.fail("the arm was touched before authorization"),
    )

    with pytest.raises(STAGE.SafetyError, match=message):
        EXECUTOR.execute_workflow_phase(
            object(),
            object(),
            artifact(report=report),
            workflow_phase="return-home-holding",
            planning_session_id=SESSION_ID,
            receipt_dir=tmp_path / "return",
            prior_receipt_dir=prior,
            speed_percent=5,
            segment_timeout_s=12.0,
            gripper_force_n=1.0,
        )

    assert events == [], "no motion may be commanded on an unvalidated corridor"


def test_return_home_holding_rejects_a_lift_top_that_is_not_where_the_arm_stopped(
    tmp_path,
    monkeypatch,
):
    events: list[tuple[str, np.ndarray]] = []
    prior = tmp_path / "pick"
    write_workflow(prior, phase="holding_at_lift", final_joints=Q_GRASP)
    patch_paths(monkeypatch, events)

    with pytest.raises(STAGE.SafetyError, match="does not match the planned carry start"):
        EXECUTOR.execute_workflow_phase(
            object(),
            object(),
            artifact(),
            workflow_phase="return-home-holding",
            planning_session_id=SESSION_ID,
            receipt_dir=tmp_path / "return",
            prior_receipt_dir=prior,
            speed_percent=5,
            segment_timeout_s=12.0,
            gripper_force_n=1.0,
        )

    assert events == []


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


def test_place_back_from_home_needs_the_loaded_corridor(tmp_path, monkeypatch):
    # Carrying the object back OUT to the grasp pose drives the same two
    # corridors as the return, the other way round, still loaded.
    prior = tmp_path / "home-holding"
    write_workflow(prior, phase="holding_at_home")
    events: list[tuple[str, np.ndarray]] = []
    patch_paths(monkeypatch, events)
    monkeypatch.setattr(
        EXECUTOR,
        "_verify_holding_object",
        lambda *_a, **_k: pytest.fail("the arm was touched before authorization"),
    )

    with pytest.raises(STAGE.SafetyError, match="valid with the object attached"):
        EXECUTOR.execute_workflow_phase(
            object(),
            object(),
            artifact(
                report={"holding_return": holding_return(reverse_approach=False)},
            ),
            workflow_phase="place-back",
            planning_session_id=SESSION_ID,
            receipt_dir=tmp_path / "placed",
            prior_receipt_dir=prior,
            speed_percent=5,
            segment_timeout_s=12.0,
            gripper_force_n=1.0,
        )

    assert events == []


@pytest.mark.parametrize(
    "report",
    [
        None,
        # The recovery limb must keep working on plans that carry no
        # holding_return block at all: the only loaded motion it makes is the
        # reverse lift, whose geometry the planner already checks attached.
        {},
        {"holding_return": holding_return(carry=False, return_transit=False)},
    ],
)
def test_place_back_from_lift_lowers_releases_and_reverses_home(
    tmp_path,
    monkeypatch,
    report,
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
        artifact(report=report, with_carry=report is None),
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
