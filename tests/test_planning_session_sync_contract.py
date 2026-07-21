from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_perception_selects_only_a_valid_passive_window():
    source = (ROOT / "scripts/runtime/go2w_perception_dry_run.py").read_text()
    assert "validate_passive_capture(candidate_report)" in source
    assert "capture.start_unix_ns - 250_000_000" in source
    assert "selected_passive_window.write_text" in source
    assert "target_exclusion_mask(" in source
    assert "pixel_excluded_scene_points[~geometric_target_labels]" in source
    assert '"scene_target_geometric_excluded_points"' in source


def test_launcher_repeats_exact_least_privilege_gate_and_freezes_match():
    source = (ROOT / "scripts/runtime/go2w_planning_session.sh").read_text()
    assert "Z_MANIP_REQUIRE_PASSIVE_WINDOW=1" in source
    assert "while kill -0 \"$perception_pid\"" in source
    assert 'REMOTE_PASSIVE_PROBE="/usr/local/libexec/z-manip/piper_passive_probe.py"' in source
    assert '--interface can0' in source
    assert 'selected_passive_joint_report.json' in source
    assert "selected_passive_joint_report.json" in source
    assert "piper/cmd" not in source
    assert "joint_trajectory" not in source
