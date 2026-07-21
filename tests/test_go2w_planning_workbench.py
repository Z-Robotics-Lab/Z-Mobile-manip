from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/go2w_planning_workbench.sh"
SESSION = ROOT / "scripts/runtime/go2w_planning_session.sh"
UNIT = ROOT / "configs/z-manip-planning-workbench.service"


def test_workbench_runs_session_then_serves_latest_read_only_bundle():
    source = SCRIPT.read_text(encoding="utf-8")
    unit = UNIT.read_text(encoding="utf-8")
    assert "go2w_planning_session.sh" in source
    assert "go2w_planning_control.py" in unit
    assert "go2w_debug_ui.py" in source
    assert "go2w_debug_safety_gate.py" in source
    assert "--bundle \"$bundle\"" in source
    assert "127.0.0.1:$PORT/api/health" in source
    assert "readlink -f -- \"$RUN_ROOT/latest\"" in source
    assert "--session-script" in unit
    assert "systemctl --user restart" in source
    assert "systemctl --user enable" in source
    assert "z-manip-planning-workbench.service" in source


def test_workbench_has_no_actuator_or_remote_transport():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "ssh ",
        "sudo ",
        "piper/cmd",
        "joint_trajectory",
        "gripper",
        "cmd_vel",
    ):
        assert forbidden not in source


def test_failed_attempt_does_not_replace_last_successful_bundle():
    source = SESSION.read_text(encoding="utf-8")
    attempt = 'ln -sfn "$RUN_ID" "$RUN_ROOT/latest_attempt"'
    failure_gate = 'if [[ "$perception_rc" -ne 0'
    success = 'ln -sfn "$RUN_ID" "$RUN_ROOT/latest"'
    assert source.count(attempt) == 1
    assert source.count(success) == 1
    assert source.index(attempt) < source.index(failure_gate) < source.index(success)
