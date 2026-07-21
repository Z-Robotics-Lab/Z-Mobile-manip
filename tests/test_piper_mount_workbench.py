from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_mount_workbench.sh"


def test_mount_workbench_separates_capture_solve_and_read_only_display():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Z_MANIP_CALIBRATION_CAPTURE_ONLY=1" in source
    assert "piper_mount_calibrate.py" in source
    assert "piper_mount_ui.py" in source
    assert "--restart unless-stopped" in source
    assert "platform_target_anchor.json" in source
    assert "piper_mount_anchor.example.json" in source
    assert "hand_eye_samples.json" in source


def test_mount_workbench_has_no_actuator_or_can_transport():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "piper/cmd",
        "joint_trajectory",
        "gripper",
        "cmd_vel",
        "candump",
        "cansend",
        "sudo ",
        "ssh ",
    ):
        assert forbidden not in source
