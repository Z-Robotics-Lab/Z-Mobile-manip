from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_live_perception.py"
SPEC = importlib.util.spec_from_file_location("go2w_live_perception_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)


def _encoded(extension: str, image: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(extension, image)
    assert ok
    return payload.tobytes()


def _anchor() -> dict[str, bytes]:
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:80, 65:95] = 255
    overlay = np.full((120, 160, 3), 25, dtype=np.uint8)
    overlay[40:80, 65:95] = (65, 180, 90)
    candidates = overlay.copy()
    cv2.line(candidates, (60, 60), (105, 60), (255, 255, 0), 3)
    return {
        "mask": _encoded(".png", mask),
        "overlay": _encoded(".png", overlay),
        "candidates": _encoded(".png", candidates),
    }


class _Camera:
    def __init__(self) -> None:
        frame = np.full((120, 160, 3), 25, dtype=np.uint8)
        frame[40:80, 65:95] = (225, 225, 225)
        for offset in range(0, 30, 6):
            cv2.line(frame, (65 + offset, 40), (65 + offset, 79), (70, 70, 70), 1)
        cv2.line(frame, (65, 40), (94, 79), (70, 70, 70), 1)
        self.payload = _encoded(".jpg", frame)
        self.etag = '"camera-1"'

    def snapshot(self):
        return "live", self.payload, self.etag, 0.01, ""

    def set_frame(self, frame: np.ndarray, etag: str = '"camera-2"') -> None:
        self.payload = _encoded(".jpg", frame)
        self.etag = etag


class _Artifacts:
    def __init__(self, timestamp_ns: int) -> None:
        self.timestamp_ns = timestamp_ns

    def perception_snapshot(self):
        return "20260720-120000", _anchor(), self.timestamp_ns


def test_fresh_anchor_produces_three_bounded_opencv_views() -> None:
    now_ns = 10_000_000_000
    renderer = LIVE.LivePerceptionRenderer(
        _Camera(),
        _Artifacts(now_ns - 100_000_000),
        clock_ns=lambda: now_ns,
    )
    for kind, magic in (("mask", b"\x89PNG"), ("overlay", b"\xff\xd8"), ("candidates", b"\xff\xd8")):
        state, payload, etag, camera_age_s, reference_age_s, detail = renderer.snapshot(kind)
        assert state == "fresh"
        assert payload is not None and payload.startswith(magic)
        assert etag is not None and etag.startswith('"live-perception-')
        assert camera_age_s == 0.01
        assert reference_age_s == 0.1
        assert "OpenCV affine" in detail


def test_affine_tracker_follows_translation_and_rotation_with_quality_evidence() -> None:
    now_ns = 10_000_000_000
    camera = _Camera()
    renderer = LIVE.LivePerceptionRenderer(
        camera,
        _Artifacts(now_ns - 100_000_000),
        clock_ns=lambda: now_ns,
    )
    state, *_ = renderer.snapshot("mask")
    assert state == "fresh"

    original = cv2.imdecode(np.frombuffer(camera.payload, np.uint8), cv2.IMREAD_COLOR)
    transform = cv2.getRotationMatrix2D((80, 60), 4.0, 1.03)
    transform[:, 2] += (7.0, -4.0)
    moved = cv2.warpAffine(original, transform, (160, 120), borderValue=(25, 25, 25))
    camera.set_frame(moved)

    state, payload, _etag, *_rest = renderer.snapshot("mask")
    assert state == "tracked"
    tracked = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    points = cv2.findNonZero(tracked)
    assert points is not None
    x, y, width, height = cv2.boundingRect(points)
    assert x > 68
    assert y < 40
    assert width >= 28 and height >= 38


def test_tracked_views_advance_once_per_camera_etag_and_all_three_keep_updating() -> None:
    now_ns = 10_000_000_000
    camera = _Camera()
    renderer = LIVE.LivePerceptionRenderer(
        camera,
        _Artifacts(now_ns - 100_000_000),
        clock_ns=lambda: now_ns,
    )
    kinds = ("mask", "overlay", "candidates")
    first = {kind: renderer.snapshot(kind) for kind in kinds}
    assert {value[0] for value in first.values()} == {"fresh"}

    original = cv2.imdecode(np.frombuffer(camera.payload, np.uint8), cv2.IMREAD_COLOR)
    previous = first
    for step in (3, 6):
        transform = np.float32([[1.0, 0.0, step], [0.0, 1.0, -step]])
        moved = cv2.warpAffine(original, transform, (160, 120), borderValue=(25, 25, 25))
        camera.set_frame(moved, etag=f'"camera-{step}"')
        with ThreadPoolExecutor(max_workers=3) as executor:
            values = list(executor.map(renderer.snapshot, kinds))
        current = dict(zip(kinds, values, strict=True))
        assert {value[0] for value in current.values()} == {"tracked"}
        for kind in kinds:
            assert current[kind][1] != previous[kind][1]
            assert current[kind][2] != previous[kind][2]
        previous = current


def test_affine_tracker_reports_stale_instead_of_drifting_to_unrelated_frame() -> None:
    now_ns = 10_000_000_000
    camera = _Camera()
    renderer = LIVE.LivePerceptionRenderer(
        camera,
        _Artifacts(now_ns - 100_000_000),
        clock_ns=lambda: now_ns,
    )
    assert renderer.snapshot("overlay")[0] == "fresh"
    camera.set_frame(np.full((120, 160, 3), 25, dtype=np.uint8))

    state, payload, etag, *_rest = renderer.snapshot("overlay")
    assert state == "stale"
    assert payload is None
    assert etag is None


def test_old_anchor_is_explicitly_stale_and_never_served_as_live() -> None:
    now_ns = 30_000_000_000
    renderer = LIVE.LivePerceptionRenderer(
        _Camera(),
        _Artifacts(now_ns - 20_000_000_000),
        clock_ns=lambda: now_ns,
        reference_max_age_s=15.0,
    )
    state, payload, etag, _camera_age_s, reference_age_s, detail = renderer.snapshot("overlay")
    assert state == "stale"
    assert payload is None
    assert etag is None
    assert reference_age_s == 20.0
    assert "run perception" in detail


def test_live_renderer_has_no_ros_can_subprocess_or_actuator_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = compile(source, str(SCRIPT), "exec")
    assert tree is not None
    for forbidden in ("rclpy", "socketcan", "subprocess", "Piper", "send_joint", "publish("):
        assert forbidden not in source


def test_dashboard_polls_distinct_live_views_and_labels_staleness() -> None:
    source = (ROOT / "web" / "debug_dashboard" / "index.html").read_text(encoding="utf-8")
    for route in (
        "/api/perception/live/mask.png",
        "/api/perception/live/overlay.jpg",
        "/api/perception/live/candidates.jpg",
    ):
        assert route in source
    assert "X-Z-Manip-Perception-State" in source
    assert "refreshPerceptionFeeds();" in source
    assert '"stale", "offline", "invalid"' in source
    assert 'tracking || perceptionStarting ? 500 : 1500' in source


def test_perception_remains_available_away_from_home() -> None:
    source = (ROOT / "web" / "debug_dashboard" / "index.html").read_text(encoding="utf-8")
    perception_gate = next(
        line for line in source.splitlines() if "perceptionButton.disabled =" in line
    )
    planning_gate = next(
        line for line in source.splitlines() if "planningButton.disabled =" in line
    )
    assert "!atHome" not in perception_gate
    assert "!atHome" in planning_gate
