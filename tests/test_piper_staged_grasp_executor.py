from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_staged_grasp_executor.py"
SPEC = importlib.util.spec_from_file_location("piper_staged_grasp_executor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXECUTOR
SPEC.loader.exec_module(EXECUTOR)


Q_START = np.asarray((0.0, 0.10, -0.10, 0.0, 0.0, 0.0))
Q_PRE = np.asarray((0.03, 0.12, -0.13, 0.01, 0.0, 0.0))
Q_GRASP = np.asarray((0.04, 0.14, -0.15, 0.01, 0.01, 0.0))
Q_LIFT = np.asarray((0.04, 0.17, -0.18, 0.01, 0.01, 0.0))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_artifact(
    tmp_path: Path,
    *,
    start: np.ndarray = Q_START,
    pregrasp: np.ndarray = Q_PRE,
    source_stamp_ns: int | None = None,
    plan_valid: bool = True,
    reconciliation: bool = False,
    measured: np.ndarray | None = None,
    embedded_hash: bool = True,
) -> tuple[Path, Path, str]:
    measured = np.asarray(start if measured is None else measured, dtype=float)
    transit = np.vstack((start, pregrasp))
    approach = np.vstack((pregrasp, Q_GRASP))
    lift = np.vstack((Q_GRASP, Q_LIFT))
    npz_path = tmp_path / "planned_grasp.npz"
    np.savez_compressed(
        npz_path,
        transit=transit,
        transit_times_s=np.asarray((0.0, 1.0)),
        approach=approach,
        approach_times_s=np.asarray((0.0, 1.0)),
        lift=lift,
        lift_times_s=np.asarray((0.0, 1.0)),
        transit_raw=transit,
        approach_raw=approach,
        lift_raw=lift,
        current_joints=np.asarray(start),
        measured_joints=measured,
    )
    digest = _sha256(npz_path)
    report = {
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "plan_valid": plan_valid,
        "source_stamp_ns": source_stamp_ns or time.time_ns(),
        "current_joints_rad": measured.tolist(),
        "measured_joints_rad": measured.tolist(),
        "planning_start_joints_rad": start.tolist(),
        "start_limit_projection_rad": (np.asarray(start) - measured).tolist(),
        "execution_start_requires_limit_reconciliation": reconciliation,
        "raw_paths_collision_validated": True,
        "transit_raw_waypoints": len(transit),
        "approach_raw_waypoints": len(approach),
        "lift_raw_waypoints": len(lift),
        "required_width_m": 0.03,
    }
    if embedded_hash:
        report["planned_grasp_sha256"] = digest
    report_path = tmp_path / "planning_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, npz_path, digest


def load_artifact(
    report: Path,
    archive: Path,
    *,
    stage: str = "pregrasp",
    digest: str | None = None,
    now_ns: int | None = None,
):
    return EXECUTOR.load_planning_artifact(
        report,
        archive,
        expected_npz_sha256=digest,
        stage=stage,
        now_ns=now_ns or time.time_ns(),
        max_source_age_s=15.0,
    )


def receipt_document(
    *,
    stage: str,
    artifact_id: str,
    final_joints: np.ndarray,
    finished_ns: int,
    nonempty: bool = False,
) -> dict:
    document = {
        "schema": "z_manip.piper_stage_receipt.v1",
        "stage": stage,
        "success": True,
        "artifact_id": artifact_id,
        "finished_unix_ns": finished_ns,
        "final_joints_rad": final_joints.tolist(),
    }
    if nonempty:
        document["gripper"] = {
            "aperture_m": 0.029,
            "force_n": 0.8,
            "commanded_target_m": 0.026,
            "timestamp": 5.0,
            "mode": "width",
            "healthy": True,
            "enabled": True,
            "homed": True,
            "nonempty_verified": True,
        }
    return document


def write_receipt(tmp_path: Path, document: dict, name: str = "receipt.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(duration, 0.001)


class FakeRobot:
    def __init__(self, joints: np.ndarray, events: list, *, move_succeeds: bool = True):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.events = events
        self.joint_stamp = 1.0
        self.move_succeeds = move_succeeds
        self.estops = 0
        self.ctrl_mode = 1
        self.OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J="j"))

    def get_joint_angles(self):
        self.joint_stamp += 1.0
        return SimpleNamespace(msg=self.joints.tolist(), timestamp=self.joint_stamp)

    def get_arm_status(self):
        return SimpleNamespace(
            msg=SimpleNamespace(
                arm_status=0,
                motion_status=0,
                err_code=0,
                ctrl_mode=self.ctrl_mode,
            ),
            timestamp=self.joint_stamp,
        )

    def set_motion_mode(self, mode):
        self.events.append(("mode", mode))
        self.ctrl_mode = 1

    def set_speed_percent(self, speed):
        self.events.append(("speed", speed))

    def enable(self):
        self.events.append(("enable",))
        return True

    def move_j(self, target):
        self.events.append(("move_j", tuple(target)))
        if self.move_succeeds:
            self.joints = np.asarray(target, dtype=float)

    def electronic_emergency_stop(self):
        self.events.append(("estop",))
        self.estops += 1


class EnableBeforeModeRobot(FakeRobot):
    """Model firmware that ignores control-mode requests until enabled."""

    def __init__(self, joints: np.ndarray, events: list):
        super().__init__(joints, events)
        self.ctrl_mode = 0
        self.enabled = False

    def enable(self):
        self.events.append(("enable",))
        self.enabled = True
        return True

    def set_motion_mode(self, mode):
        self.events.append(("mode", mode))
        if self.enabled:
            self.ctrl_mode = 1


class FakeEffector:
    def __init__(
        self,
        events: list,
        *,
        aperture_m: float = 0.03,
        force_n: float = 0.8,
        ignore_commands: bool = False,
    ):
        self.events = events
        self.aperture_m = aperture_m
        self.force_n = force_n
        self.timestamp = 1.0
        self.ignore_commands = ignore_commands

    def get_gripper_status(self):
        self.timestamp += 1.0
        foc = SimpleNamespace(
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=False,
            driver_overheating=False,
            sensor_status=False,
            driver_error_status=False,
            driver_enable_status=True,
            homing_status=True,
        )
        return SimpleNamespace(
            msg=SimpleNamespace(
                value=self.aperture_m,
                force=self.force_n,
                mode="width",
                foc_status=foc,
            ),
            timestamp=self.timestamp,
        )

    def move_gripper_m(self, *, value, force):
        self.events.append(("gripper", value, force))
        if self.ignore_commands:
            return
        if value >= 0.06:
            self.aperture_m = value
            self.force_n = 0.0
        else:
            self.aperture_m = 0.029
            self.force_n = 0.8


def test_can_control_mode_is_requested_after_enable_and_current_pose_is_held():
    events: list[tuple] = []
    robot = EnableBeforeModeRobot(Q_START, events)
    clock = FakeClock()
    guard = EXECUTOR.CommandGuard()

    EXECUTOR.enter_can_joint_control(
        robot,
        guard,
        timeout_s=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert events[0] == ("enable",)
    assert events[1] == ("mode", "j")
    assert ("speed", 1) in events
    hold = [event for event in events if event[0] == "move_j"]
    assert len(hold) == 1
    np.testing.assert_allclose(hold[0][1], Q_START)


def test_artifact_hash_plan_age_and_internal_start_are_validated(tmp_path):
    report, archive, digest = make_artifact(tmp_path, embedded_hash=False)
    artifact = load_artifact(report, archive, digest=digest)

    assert artifact.npz_sha256 == digest
    np.testing.assert_allclose(artifact.start_joints_rad, Q_START)

    with pytest.raises(EXECUTOR.SafetyError, match="trust anchor"):
        load_artifact(report, archive)
    with pytest.raises(EXECUTOR.SafetyError, match="mismatch"):
        load_artifact(report, archive, digest="0" * 64)


def test_executor_consumes_collision_checked_raw_polyline_not_resampled_path(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["transit"] = np.vstack((Q_START, (Q_START + Q_PRE) / 2.0, Q_PRE))
    arrays["transit_times_s"] = np.asarray((0.0, 0.5, 1.0))
    np.savez_compressed(archive, **arrays)
    document = json.loads(report.read_text())
    document["planned_grasp_sha256"] = _sha256(archive)
    report.write_text(json.dumps(document), encoding="utf-8")

    artifact = load_artifact(report, archive)
    execution_path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)

    assert len(artifact.arrays["transit"]) == 3
    assert len(execution_path) == 2
    np.testing.assert_allclose(execution_path, artifact.arrays["transit_raw"])


def test_execution_path_coalesces_dense_straight_samples_without_shortcutting() -> None:
    start = np.zeros(6)
    end = np.asarray((1.0, -0.5, 0.2, 0.0, 0.0, 0.0))
    dense = np.linspace(start, end, 72)

    path = EXECUTOR.coalesce_collinear_execution_path(
        dense,
        max_segment_rad=0.35,
    )

    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], end)
    assert len(path) == 4
    assert np.max(np.abs(np.diff(path, axis=0))) <= 0.35 + 1e-12


def test_execution_path_keeps_joint_space_corners() -> None:
    corner = np.asarray((0.4, 0.0, 0.0, 0.0, 0.0, 0.0))
    path = np.vstack((
        np.zeros(6),
        corner / 2.0,
        corner,
        corner + np.asarray((0.0, 0.2, 0.0, 0.0, 0.0, 0.0)),
    ))

    coalesced = EXECUTOR.coalesce_collinear_execution_path(path, max_segment_rad=1.0)

    assert len(coalesced) == 3
    np.testing.assert_allclose(coalesced[1], corner)


def test_slow_gripper_close_uses_monotonic_aperture_ramp() -> None:
    events: list[tuple] = []
    effector = FakeEffector(events, aperture_m=0.07, force_n=0.0)
    guard = EXECUTOR.CommandGuard()
    clock = FakeClock()

    EXECUTOR.command_slow_gripper_close(
        effector,
        guard,
        start_aperture_m=0.07,
        target_aperture_m=0.026,
        force_n=1.0,
        steps=6,
        interval_s=0.20,
        sleep=clock.sleep,
    )

    apertures = [event[1] for event in events if event[0] == "gripper"]
    assert len(apertures) == 6
    assert all(first > second for first, second in zip(apertures, apertures[1:]))
    assert apertures[-1] == pytest.approx(0.026)
    assert clock.value == pytest.approx(1.0)


def test_collision_checked_raw_edges_are_densified_without_shortcutting(tmp_path):
    start = Q_PRE.copy()
    wide_end = Q_GRASP.copy()
    wide_end[3] = start[3] + 0.40
    arrays = {"approach_raw": np.vstack((start, wide_end))}
    path = EXECUTOR._validate_raw_path(
        arrays,
        "approach",
        max_segment_rad=EXECUTOR.DEFAULT_MAX_SEGMENT_RAD,
    )

    assert len(path) > 2
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], wide_end)
    assert np.max(np.abs(np.diff(path, axis=0))) <= EXECUTOR.DEFAULT_MAX_SEGMENT_RAD


def test_missing_collision_checked_raw_polyline_fails_closed(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "transit_raw"
        }
    np.savez_compressed(archive, **arrays)
    document = json.loads(report.read_text())
    document["planned_grasp_sha256"] = _sha256(archive)
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(EXECUTOR.SafetyError, match="transit_raw"):
        load_artifact(report, archive)


def test_artifact_rejects_invalid_plan_and_stale_source(tmp_path):
    bad_plan = tmp_path / "bad_plan"
    bad_plan.mkdir()
    report, archive, _ = make_artifact(bad_plan, plan_valid=False)
    with pytest.raises(EXECUTOR.SafetyError, match="plan_valid"):
        load_artifact(report, archive)

    stale = tmp_path / "stale"
    stale.mkdir()
    now_ns = time.time_ns()
    report, archive, _ = make_artifact(
        stale,
        source_stamp_ns=now_ns - int(20e9),
    )
    with pytest.raises(EXECUTOR.SafetyError, match="source is stale"):
        load_artifact(report, archive, now_ns=now_ns)


def test_continuation_revalidates_old_hashed_artifact_without_live_freshness(tmp_path):
    now_ns = time.time_ns()
    report, archive, _ = make_artifact(
        tmp_path,
        source_stamp_ns=now_ns - int(120e9),
    )

    artifact = EXECUTOR.load_planning_artifact(
        report,
        archive,
        expected_npz_sha256=None,
        stage="pregrasp",
        now_ns=now_ns,
        max_source_age_s=12 * 60 * 60.0,
        require_fresh_source=False,
    )

    assert artifact.artifact_id



def test_small_pregrasp_start_reconciliation_is_explicit_and_bounded(tmp_path):
    planning_start = Q_START.copy()
    planning_start[2] = 0.0
    measured = planning_start.copy()
    measured[2] = 0.002967
    report, archive, _ = make_artifact(
        tmp_path,
        start=planning_start,
        measured=measured,
        reconciliation=True,
    )

    artifact = load_artifact(report, archive, stage="pregrasp")
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    np.testing.assert_allclose(path[0], measured)
    np.testing.assert_allclose(path[1], planning_start)

    with pytest.raises(EXECUTOR.SafetyError, match="only for pregrasp"):
        load_artifact(report, archive, stage="approach_close")

    too_far = tmp_path / "too_far"
    too_far.mkdir()
    measured[2] = 0.0065
    report, archive, _ = make_artifact(
        too_far,
        start=planning_start,
        measured=measured,
        reconciliation=True,
    )
    with pytest.raises(EXECUTOR.SafetyError, match="0.006rad"):
        load_artifact(report, archive, stage="pregrasp")


def test_approach_requires_same_home_planned_artifact_and_matching_pregrasp(tmp_path):
    now_ns = time.time_ns()
    finished_ns = now_ns - int(2e9)
    report, archive, _ = make_artifact(
        tmp_path,
        pregrasp=Q_PRE,
        source_stamp_ns=finished_ns - int(1e9),
    )
    artifact = load_artifact(report, archive, stage="approach_close", now_ns=now_ns)
    prior_path = write_receipt(
        tmp_path,
        receipt_document(
            stage="pregrasp",
            artifact_id=artifact.artifact_id,
            final_joints=Q_PRE,
            finished_ns=finished_ns,
        ),
    )
    prior = EXECUTOR.load_prior_receipt(
        prior_path,
        expected_stage="pregrasp",
        now_ns=now_ns,
    )

    path = EXECUTOR.validate_stage_context(artifact, "approach_close", prior)
    np.testing.assert_allclose(path[0], Q_PRE)

    wrong_receipt = receipt_document(
        stage="pregrasp",
        artifact_id="b" * 64,
        final_joints=Q_PRE,
        finished_ns=finished_ns,
    )
    wrong_path = write_receipt(tmp_path, wrong_receipt, "wrong-pregrasp.json")
    wrong_prior = EXECUTOR.load_prior_receipt(
        wrong_path,
        expected_stage="pregrasp",
        now_ns=now_ns,
    )
    with pytest.raises(EXECUTOR.SafetyError, match="differs"):
        EXECUTOR.validate_stage_context(artifact, "approach_close", wrong_prior)


def test_lift_requires_same_artifact_and_nonempty_receipt(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive, stage="lift")
    receipt = receipt_document(
        stage="approach_close",
        artifact_id=artifact.artifact_id,
        final_joints=Q_GRASP,
        finished_ns=time.time_ns() - int(1e9),
        nonempty=True,
    )
    path = write_receipt(tmp_path, receipt)
    prior = EXECUTOR.load_prior_receipt(
        path,
        expected_stage="approach_close",
        now_ns=time.time_ns(),
    )

    lift = EXECUTOR.validate_stage_context(artifact, "lift", prior)
    np.testing.assert_allclose(lift[0], Q_GRASP)

    receipt["artifact_id"] = "b" * 64
    bad = write_receipt(tmp_path, receipt, "wrong.json")
    prior = EXECUTOR.load_prior_receipt(
        bad,
        expected_stage="approach_close",
        now_ns=time.time_ns(),
    )
    with pytest.raises(EXECUTOR.SafetyError, match="differs"):
        EXECUTOR.validate_stage_context(artifact, "lift", prior)


def test_confirmation_tokens_are_stage_and_receipt_bound():
    artifact = "a" * 64
    receipt = "b" * 64

    pregrasp = EXECUTOR.confirmation_token("pregrasp", artifact)
    approach = EXECUTOR.confirmation_token("approach_close", artifact, receipt)
    lift = EXECUTOR.confirmation_token("lift", artifact, receipt)

    assert len({pregrasp, approach, lift}) == 3
    with pytest.raises(EXECUTOR.SafetyError, match="exact dry-run"):
        EXECUTOR.require_execution_authorization(
            execute=True,
            supplied_token=pregrasp,
            expected_token=approach,
        )


def test_close_target_and_nonempty_verification_are_width_based():
    assert EXECUTOR.close_target_m(0.03) == pytest.approx(0.026)
    good = EXECUTOR.GripperFeedback(0.027, 0.8, 1.0, "width", True, True, True)
    EXECUTOR.verify_nonempty_grasp(good, 0.03)

    empty = EXECUTOR.GripperFeedback(0.002, 0.8, 1.0, "width", True, True, True)
    with pytest.raises(EXECUTOR.SafetyError, match="empty-grasp"):
        EXECUTOR.verify_nonempty_grasp(empty, 0.03)
    weak = EXECUTOR.GripperFeedback(0.027, 0.0, 1.0, "width", True, True, True)
    with pytest.raises(EXECUTOR.SafetyError, match="force"):
        EXECUTOR.verify_nonempty_grasp(weak, 0.03)
    reached_target = EXECUTOR.GripperFeedback(
        0.0265,
        0.8,
        1.0,
        "width",
        True,
        True,
        True,
    )
    with pytest.raises(EXECUTOR.SafetyError, match="empty close target"):
        EXECUTOR.verify_nonempty_grasp(
            reached_target,
            0.03,
            commanded_close_target_m=0.026,
        )


def test_pregrasp_opens_and_confirms_gripper_before_segmented_move_j(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []
    robot = FakeRobot(Q_START, events)
    effector = FakeEffector(events)
    clock = FakeClock()

    final, feedback = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    names = [event[0] for event in events]
    move_indices = [index for index, name in enumerate(names) if name == "move_j"]
    assert move_indices[0] < names.index("gripper") < move_indices[-1]
    np.testing.assert_allclose(events[move_indices[0]][1], Q_START)
    assert events[names.index("gripper")][1] == pytest.approx(0.07)
    np.testing.assert_allclose(final, Q_PRE)
    assert feedback.aperture_m == pytest.approx(0.07)
    assert robot.estops == 0


def test_pregrasp_recovers_fault_free_disabled_gripper_after_reboot(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []
    robot = FakeRobot(Q_START, events)

    class RebootedEffector(FakeEffector):
        enabled = False

        def get_gripper_status(self):
            message = super().get_gripper_status()
            message.msg.foc_status.driver_enable_status = self.enabled
            return message

        def move_gripper_m(self, *, value, force):
            self.enabled = True
            super().move_gripper_m(value=value, force=force)

    effector = RebootedEffector(events)
    clock = FakeClock()

    final, feedback = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    names = [event[0] for event in events]
    assert names.index("move_j") < names.index("gripper")
    assert feedback is not None and feedback.enabled is True
    np.testing.assert_allclose(final, Q_PRE)
    assert robot.estops == 0


def test_pregrasp_does_not_command_a_faulted_disabled_gripper(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []
    robot = FakeRobot(Q_START, events)
    effector = FakeEffector(events)
    original_status = effector.get_gripper_status

    def faulted_status():
        message = original_status()
        message.msg.foc_status.driver_enable_status = False
        message.msg.foc_status.driver_error_status = True
        return message

    effector.get_gripper_status = faulted_status
    clock = FakeClock()

    with pytest.raises(EXECUTOR.SafetyError, match="hardware fault"):
        EXECUTOR.execute_stage(
            robot,
            effector,
            artifact,
            "pregrasp",
            path,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert "gripper" not in [event[0] for event in events]
    assert robot.estops == 0


def test_home_gripper_recovery_enables_without_changing_aperture():
    events: list[tuple] = []

    class RebootedEffector(FakeEffector):
        enabled = False

        def get_gripper_status(self):
            message = super().get_gripper_status()
            message.msg.foc_status.driver_enable_status = self.enabled
            return message

        def move_gripper_m(self, *, value, force):
            self.enabled = True
            self.events.append(("gripper", value, force))
            self.aperture_m = value

    effector = RebootedEffector(events, aperture_m=0.041)
    clock = FakeClock()
    feedback = EXECUTOR.restore_gripper_enable_at_current_aperture(
        effector,
        EXECUTOR.CommandGuard(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    command = next(event for event in events if event[0] == "gripper")
    assert command[1] == pytest.approx(0.041)
    assert feedback.enabled is True


def test_home_gripper_recovery_does_not_recommand_enabled_driver():
    events: list[tuple] = []
    effector = FakeEffector(events, aperture_m=0.041)

    feedback = EXECUTOR.restore_gripper_enable_at_current_aperture(
        effector,
        EXECUTOR.CommandGuard(),
    )

    assert feedback.enabled is True
    assert not events


def test_s_v188_false_homing_bit_does_not_override_verified_width_feedback(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []
    robot = FakeRobot(Q_START, events)
    effector = FakeEffector(events)
    effector.get_gripper_status = lambda: SimpleNamespace(
        msg=SimpleNamespace(
            value=effector.aperture_m,
            force=effector.force_n,
            mode="width",
            foc_status=SimpleNamespace(
                voltage_too_low=False,
                motor_overheating=False,
                driver_overcurrent=False,
                driver_overheating=False,
                sensor_status=False,
                driver_error_status=False,
                driver_enable_status=True,
                homing_status=False,
            ),
        ),
        timestamp=(setattr(effector, "timestamp", effector.timestamp + 1.0) or effector.timestamp),
    )
    clock = FakeClock()
    final, feedback = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    np.testing.assert_allclose(final, Q_PRE)
    assert feedback is not None and feedback.homed is False


def test_execution_waits_for_delayed_initial_sdk_feedback(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []

    class DelayedRobot(FakeRobot):
        missing = 3

        def get_arm_status(self):
            if self.missing:
                self.missing -= 1
                return None
            return super().get_arm_status()

    robot = DelayedRobot(Q_START, events)
    effector = FakeEffector(events)
    clock = FakeClock()
    final, _ = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    np.testing.assert_allclose(final, Q_PRE)
    assert robot.estops == 0


def test_motion_accepts_three_fresh_in_tolerance_samples_with_stale_failed_flag():
    events: list[tuple] = []

    class StaleMotionFlagRobot(FakeRobot):
        def get_arm_status(self):
            message = super().get_arm_status()
            message.msg.motion_status = 1
            return message

    robot = StaleMotionFlagRobot(Q_START, events)
    clock = FakeClock()
    target = Q_START.copy()
    target[0] += 0.02
    robot.move_j(target.tolist())
    actual = EXECUTOR.wait_for_motion(
        robot,
        target,
        after_timestamp=1.0,
        after_status_timestamp=1.0,
        timeout_s=1.0,
        tolerance_rad=np.deg2rad(0.35),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    np.testing.assert_allclose(actual, target)


def test_pregrasp_reconciliation_uses_two_percent_before_raw_transit(tmp_path):
    planning_start = Q_START.copy()
    planning_start[2] = 0.0
    measured = planning_start.copy()
    measured[2] = 0.002967
    report, archive, _ = make_artifact(
        tmp_path,
        start=planning_start,
        measured=measured,
        reconciliation=True,
    )
    artifact = load_artifact(report, archive)
    path = EXECUTOR.validate_stage_context(artifact, "pregrasp", None)
    events: list[tuple] = []
    robot = FakeRobot(measured, events)
    effector = FakeEffector(events)
    clock = FakeClock()

    final, _ = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        path,
        speed_percent=5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    speeds = [event[1] for event in events if event[0] == "speed"]
    moves = [event[1] for event in events if event[0] == "move_j"]
    assert speeds == [1, 2, 5]
    assert not any(np.allclose(move, planning_start) for move in moves)
    np.testing.assert_allclose(final, Q_PRE)
    move_indices = [index for index, event in enumerate(events) if event[0] == "move_j"]
    assert move_indices[0] < [event[0] for event in events].index("gripper") < move_indices[-1]


def test_approach_moves_then_closes_to_width_based_target(tmp_path):
    report, archive, _ = make_artifact(tmp_path, start=Q_PRE, pregrasp=Q_PRE)
    artifact = load_artifact(report, archive, stage="approach_close")
    path = np.asarray(artifact.arrays["approach_raw"])
    events: list[tuple] = []
    robot = FakeRobot(Q_PRE, events)
    effector = FakeEffector(events, aperture_m=0.07, force_n=0.0)
    clock = FakeClock()

    final, feedback = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "approach_close",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    names = [event[0] for event in events]
    assert names.index("move_j") < names.index("gripper")
    gripper_event = [event for event in events if event[0] == "gripper"][-1]
    assert gripper_event[1] == pytest.approx(0.026)
    np.testing.assert_allclose(final, Q_GRASP)
    assert feedback.force_n >= 0.2
    assert robot.estops == 0


def test_lift_rechecks_nonempty_grasp_before_and_after_motion(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive, stage="lift")
    path = np.asarray(artifact.arrays["lift_raw"])
    events: list[tuple] = []
    robot = FakeRobot(Q_GRASP, events)
    effector = FakeEffector(events, aperture_m=0.029, force_n=0.8)
    clock = FakeClock()

    final, feedback = EXECUTOR.execute_stage(
        robot,
        effector,
        artifact,
        "lift",
        path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    gripper_events = [event for event in events if event[0] == "gripper"]
    assert len(gripper_events) == 2
    assert all(event[1] == pytest.approx(0.026) for event in gripper_events)
    assert any(event[0] == "move_j" for event in events)
    np.testing.assert_allclose(final, Q_LIFT)
    assert feedback.aperture_m == pytest.approx(0.029)
    assert robot.estops == 0


def test_gripper_feedback_failure_keeps_arm_hold_instead_of_unloading(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = np.asarray(artifact.arrays["transit_raw"])
    events: list[tuple] = []
    robot = FakeRobot(Q_START, events)
    effector = FakeEffector(events, ignore_commands=True)
    clock = FakeClock()

    with pytest.raises(EXECUTOR.SafetyError, match="timed out"):
        EXECUTOR.execute_stage(
            robot,
            effector,
            artifact,
            "pregrasp",
            path,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert robot.estops == 0
    assert events[-1] != ("estop",)


def test_motion_feedback_failure_triggers_electronic_estop(tmp_path):
    report, archive, _ = make_artifact(tmp_path, start=Q_PRE, pregrasp=Q_PRE)
    artifact = load_artifact(report, archive, stage="approach_close")
    path = np.asarray(artifact.arrays["approach_raw"])
    events: list[tuple] = []
    robot = FakeRobot(Q_PRE, events, move_succeeds=False)
    effector = FakeEffector(events, aperture_m=0.07, force_n=0.0)
    clock = FakeClock()

    with pytest.raises(EXECUTOR.SafetyError, match="motion timed out"):
        EXECUTOR.execute_stage(
            robot,
            effector,
            artifact,
            "approach_close",
            path,
            segment_timeout_s=0.1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert robot.estops == 1
    assert events[-1] == ("estop",)


def test_precommand_start_failure_sends_no_estop(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    path = np.asarray(artifact.arrays["transit_raw"])
    events: list[tuple] = []
    robot = FakeRobot(Q_START + 0.2, events)
    effector = FakeEffector(events)

    with pytest.raises(EXECUTOR.SafetyError, match="authorized stage start"):
        EXECUTOR.execute_stage(robot, effector, artifact, "pregrasp", path)
    assert events == []
    assert robot.estops == 0


def test_dry_run_never_connects_or_writes_receipt(tmp_path, monkeypatch, capsys):
    report, archive, digest = make_artifact(tmp_path, embedded_hash=False)
    output = tmp_path / "must-not-exist.json"

    def forbidden_connect(*_args, **_kwargs):
        raise AssertionError("dry run attempted hardware connection")

    monkeypatch.setattr(EXECUTOR, "connect_real_arm", forbidden_connect)
    result = EXECUTOR.main([
        "--planning-report", str(report),
        "--planned-grasp", str(archive),
        "--stage", "pregrasp",
        "--expected-npz-sha256", digest,
        "--receipt-output", str(output),
    ])

    assert result == 0
    assert not output.exists()
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is True
    assert document["commands_sent"] == 0
    assert document["confirmation_token"].startswith("PIPER-PREGRASP-")


def test_wrong_execute_token_fails_before_hardware_connection(tmp_path, monkeypatch):
    report, archive, _ = make_artifact(tmp_path)
    called = False

    def forbidden_connect(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authorization failure reached hardware")

    monkeypatch.setattr(EXECUTOR, "connect_real_arm", forbidden_connect)
    result = EXECUTOR.main([
        "--planning-report", str(report),
        "--planned-grasp", str(archive),
        "--stage", "pregrasp",
        "--execute",
        "--confirm", "wrong",
    ])

    assert result == 2
    assert called is False


def test_receipt_is_atomic_and_carries_feedback_chain(tmp_path):
    report, archive, _ = make_artifact(tmp_path)
    artifact = load_artifact(report, archive)
    feedback = EXECUTOR.GripperFeedback(0.029, 0.8, 7.0, "width", True, True, True)
    document = EXECUTOR.build_receipt(
        artifact=artifact,
        stage="approach_close",
        prior=None,
        started_unix_ns=10,
        finished_unix_ns=20,
        final_joints_rad=Q_GRASP,
        gripper=feedback,
    )
    output = tmp_path / "receipt-output.json"

    EXECUTOR.atomic_write_json(output, document)

    loaded = json.loads(output.read_text())
    assert loaded["gripper"]["nonempty_verified"] is True
    assert loaded["planned_grasp_sha256"] == artifact.npz_sha256
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(EXECUTOR.SafetyError, match="refusing to overwrite"):
        EXECUTOR.atomic_write_json(output, document)


def test_real_sdk_import_is_deferred_and_no_subprocess_transport_exists():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_level_imports.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "pyAgxArm" not in top_level_imports
    assert "subprocess" not in top_level_imports
    # One explicit current-position hold before gripper motion plus the
    # collision-validated planned-path command site.
    assert source.count("robot.move_j(") == 2
    assert source.count("effector.move_gripper_m(") == 6
    assert "robot.electronic_emergency_stop()" in source
    assert "--execute" in source and "--confirm" in source
