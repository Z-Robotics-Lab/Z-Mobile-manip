import importlib.util
import json
from pathlib import Path
import threading

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "go2w_wrist_search",
    ROOT / "scripts/runtime/go2w_wrist_search.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration + 0.001


def test_shadow_search_never_invokes_motion_and_confirms_target():
    clock = Clock()
    motion_calls = []
    observations = iter(((False, None, "missing"), (True, 0.8, "charger"), (True, 0.82, "charger")))
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: next(observations),
        motion=lambda *args: motion_calls.append(args),
        config=MODULE.WristSearchConfig(
            settle_s=0.01,
            detector_hz=20.0,
            observations_per_view=3,
            confirmations_required=2,
        ),
        sleep=clock.sleep,
        clock=clock,
    )
    assert coordinator.run("charger", mode="shadow", speed_percent=5)
    assert motion_calls == []
    assert coordinator.status()["phase"] == "found"


def test_live_search_is_locked_without_operator_environment(monkeypatch):
    monkeypatch.delenv("Z_MANIP_ENABLE_WRIST_SEARCH", raising=False)
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: (True, 0.9, "target"),
        motion=lambda *_args: np.zeros(6),
    )
    assert not coordinator.run("charger", mode="live", speed_percent=5)
    assert coordinator.status()["phase"] == "locked"


def test_live_search_accepts_one_shot_operator_confirmation(monkeypatch):
    monkeypatch.delenv("Z_MANIP_ENABLE_WRIST_SEARCH", raising=False)
    clock = Clock()
    motion_calls = []
    config = _small_grid(observations_per_view=1, confirmations_required=1)
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: (True, 0.9, "target"),
        motion=_grid_motion(config, motion_calls),
        config=config,
        sleep=clock.sleep,
        clock=clock,
    )
    assert coordinator.run(
        "charger",
        mode="live",
        speed_percent=5,
        operator_present=True,
    )
    assert coordinator.status()["phase"] == "found"
    assert coordinator.status()["live_enabled"] is False


def test_cancel_stops_before_any_next_view_motion():
    clock = Clock()
    cancel = threading.Event()
    calls = []

    def detector(_target):
        cancel.set()
        return False, None, "missing"

    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6), detector, motion=lambda *args: calls.append(args),
        config=MODULE.WristSearchConfig(
            settle_s=0.01,
            detector_hz=20.0,
            observations_per_view=1,
            confirmations_required=1,
        ),
        sleep=clock.sleep,
        clock=clock,
    )
    assert not coordinator.run("charger", mode="shadow", speed_percent=5, cancel=cancel)
    assert calls == []
    assert coordinator.status()["phase"] == "stopped"


def test_stop_forwards_to_interruptible_motion():
    class Motion:
        def __init__(self):
            self.stopped = False

        def __call__(self, *_args):
            return np.zeros(6)

        def stop(self):
            self.stopped = True

    motion = Motion()
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6), lambda _target: (False, None, "missing"), motion=motion,
    )
    coordinator.stop()
    assert motion.stopped


def _grid_motion(config, motion_calls):
    grid = MODULE.BoundedWristSearch(config)

    def motion(view_index, speed_percent):
        motion_calls.append(view_index)
        view = grid.views[view_index]
        joints = np.zeros(6)
        joints[config.yaw_joint_index] = view.yaw_offset_rad
        joints[config.pitch_joint_index] = view.pitch_offset_rad
        return joints

    return motion


def _small_grid(**overrides):
    parameters = dict(
        settle_s=0.01,
        detector_hz=20.0,
        observations_per_view=1,
        confirmations_required=1,
        yaw_step_rad=0.1,
        max_yaw_offset_rad=0.1,
        pitch_step_rad=0.1,
        max_pitch_offset_rad=0.1,
    )
    parameters.update(overrides)
    return MODULE.WristSearchConfig(**parameters)


def test_exhausted_live_search_restores_home_anchor(monkeypatch):
    monkeypatch.setenv("Z_MANIP_ENABLE_WRIST_SEARCH", "1")
    clock = Clock()
    motion_calls = []
    config = _small_grid()
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: (False, None, "missing"),
        motion=_grid_motion(config, motion_calls),
        config=config,
        sleep=clock.sleep,
        clock=clock,
    )
    assert not coordinator.run("charger", mode="live", speed_percent=5)
    assert any(view != 0 for view in motion_calls[:-1])
    assert motion_calls[-1] == 0
    assert "returned" in coordinator.status()["message"]


def test_found_live_search_stays_on_target_view(monkeypatch):
    monkeypatch.setenv("Z_MANIP_ENABLE_WRIST_SEARCH", "1")
    clock = Clock()
    motion_calls = []
    seen = {"count": 0}

    def detector(_target):
        seen["count"] += 1
        if seen["count"] <= 2:
            return (False, None, "missing")
        return (True, 0.8, "charger")

    config = _small_grid(observations_per_view=2, confirmations_required=2)
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        detector,
        motion=_grid_motion(config, motion_calls),
        config=config,
        sleep=clock.sleep,
        clock=clock,
    )
    assert coordinator.run("charger", mode="live", speed_percent=5)
    assert motion_calls and motion_calls[-1] != 0


def test_detector_probe_rejects_stale_premove_and_invalid_frames(tmp_path):
    valid = tmp_path / "camera-latest.jpg"
    valid.write_bytes(b"\xff\xd8" + b"\x00" * 128 + b"\xff\xd9")
    mtime = valid.stat().st_mtime

    stale = MODULE.DetectorProbe(valid, max_age_s=1.0, wall_clock=lambda: mtime + 5.0)
    visible, _conf, detail = stale("charger")
    assert visible is False and "stale" in detail

    premove = MODULE.DetectorProbe(valid, max_age_s=100.0, wall_clock=lambda: mtime + 0.1)
    premove.require_fresh_after(mtime + 10.0)
    visible, _conf, detail = premove("charger")
    assert visible is False and "before the view move" in detail

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg at all")
    bmtime = bad.stat().st_mtime
    invalid = MODULE.DetectorProbe(bad, max_age_s=100.0, wall_clock=lambda: bmtime + 0.1)
    visible, _conf, detail = invalid("charger")
    assert visible is False and "JPEG" in detail

    missing = MODULE.DetectorProbe(tmp_path / "missing.jpg", wall_clock=lambda: 0.0)
    visible, _conf, detail = missing("charger")
    assert visible is False and "unavailable" in detail


def test_per_view_vlm_fallback_confirms_a_blind_yoloe_view(tmp_path):
    clock = Clock()
    vlm_calls = {"n": 0}

    def yoloe(_target):
        return (False, None, "target not detected")

    def vlm(_target):
        vlm_calls["n"] += 1
        return (True, 0.8, "vlm charger")

    records = tmp_path / "records.jsonl"
    config = MODULE.WristSearchConfig(
        settle_s=0.01,
        detector_hz=20.0,
        observations_per_view=3,
        confirmations_required=2,
    )
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        yoloe,
        config=config,
        sleep=clock.sleep,
        clock=clock,
        vlm_detector=vlm,
        per_view_vlm_calls=2,
        search_records_path=records,
    )
    assert coordinator.run("charger", mode="shadow", speed_percent=5)
    assert coordinator.status()["phase"] == "found"
    # Two per-view VLM rescues supply the two confirmations yoloe could not.
    assert vlm_calls["n"] == 2
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    assert lines[0]["detector"] == "vlm"
    assert lines[-1]["decision"] == "found"


def test_per_view_vlm_budget_is_bounded():
    clock = Clock()
    vlm_calls = {"n": 0}

    def vlm(_target):
        vlm_calls["n"] += 1
        return (False, None, "vlm also missed")

    config = _small_grid(observations_per_view=3, confirmations_required=2)
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: (False, None, "target not detected"),
        config=config,
        sleep=clock.sleep,
        clock=clock,
        vlm_detector=vlm,
        per_view_vlm_calls=1,
    )
    assert not coordinator.run("charger", mode="shadow", speed_percent=5)
    # Exactly one VLM call per observed view (budget of 1, never repeated within
    # a view); the redundant anchor view 0 is skipped by the coordinator.
    observed_views = len(MODULE.BoundedWristSearch(config).views) - 1
    assert vlm_calls["n"] == observed_views


def test_search_records_persist_per_observation(tmp_path):
    clock = Clock()
    records = tmp_path / "search.jsonl"
    observations = iter((
        (False, None, "missing"),
        (True, 0.8, "charger"),
        (True, 0.82, "charger"),
    ))
    coordinator = MODULE.WristSearchCoordinator(
        np.zeros(6),
        lambda _target: next(observations),
        config=MODULE.WristSearchConfig(
            settle_s=0.01,
            detector_hz=20.0,
            observations_per_view=3,
            confirmations_required=2,
        ),
        sleep=clock.sleep,
        clock=clock,
        search_records_path=records,
    )
    assert coordinator.run("charger", mode="shadow", speed_percent=5)
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    required = {"view_index", "joints_rad", "frame_age_s", "detector", "score", "decision"}
    assert lines and all(required <= set(line) for line in lines)
    assert lines[-1]["decision"] == "found"
    assert lines[0]["detector"] == "yoloe"
    # The coordinator skips the redundant anchor view 0; the first observed view
    # is the first genuinely new viewpoint.
    assert lines[0]["view_index"] == 1
