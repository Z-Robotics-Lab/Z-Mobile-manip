from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "manip"


def test_manip_cli_is_syntax_checked_and_has_fixed_operator_surface() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    source = SCRIPT.read_text(encoding="utf-8")

    for command in ("manip start", "manip stop", "manip bringup", "manip status"):
        assert command in help_result.stdout
    assert "go2w_component_manager.sh" in source
    assert 'SYSTEMCTL\" --user start' in source
    assert 'SYSTEMCTL\" --user stop' in source
    assert "/api/home/status" in source
    assert "/api/grasp/status" in source
    assert "/api/sessions/status" in source
    assert "argv[1]" in source
    assert "piper_full_grasp_remote\\.py" in source
    assert "cansend" not in source.lower()
    assert "move_j" not in source.lower()
    assert "motionenable" not in source.lower()


def test_manip_url_requires_no_service_or_hardware() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "url"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "http://127.0.0.1:8766/"
