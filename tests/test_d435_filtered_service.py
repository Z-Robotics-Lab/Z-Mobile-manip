from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "configs" / "d435i.service"


def test_d435_service_persistently_filters_the_aligned_source_depth():
    source = SERVICE.read_text(encoding="utf-8")

    assert "WantedBy=default.target" in source
    assert "Restart=always" in source
    assert "align_depth.enable:=true" in source
    assert "spatial_filter.enable:=true" in source
    assert "spatial_filter.filter_magnitude:=2" in source
    assert "spatial_filter.filter_smooth_alpha:=0.5" in source
    assert "spatial_filter.filter_smooth_delta:=20" in source
    assert "temporal_filter.enable:=true" in source
    assert "temporal_filter.filter_smooth_alpha:=0.4" in source
    assert "temporal_filter.filter_smooth_delta:=20" in source
    assert "rgb_camera.color_profile:=640x480x30" in source
    # The stereo module (depth + IR share one sensor readout) runs 15fps: the
    # PC and NUC are both on wifi and the full 30fps depth stream pushed the
    # ~7.5MB/s load past the airtime ceiling (measured 1.2s RTT bufferbloat);
    # the NUC depth stream is only the FFS fallback source now.  Color keeps
    # 30fps for tracking.
    assert "depth_module.depth_profile:=640x480x15" in source
    assert "depth_module.infra_profile:=640x480x15" in source


def test_d435_service_has_no_robot_or_can_command_surface():
    source = SERVICE.read_text(encoding="utf-8").lower()

    for forbidden in (
        "can0",
        "cansend",
        "piper",
        "cmd_vel",
        "joint_trajectory",
        "/dev/tty",
    ):
        assert forbidden not in source
