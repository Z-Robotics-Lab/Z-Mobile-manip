from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "perception_bag_benchmark.py"
SPEC = importlib.util.spec_from_file_location("perception_bag_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates_and_empty_is_none() -> None:
    assert MODULE.percentile([], 50) is None
    assert MODULE.percentile([1.0], 95) == 1.0
    assert MODULE.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert MODULE.percentile([1.0, 2.0, 3.0], 95) == 2.9


def test_nearest_stamp_deltas_are_absolute_seconds() -> None:
    assert MODULE.nearest_stamp_deltas_s([], [1]) == []
    assert MODULE.nearest_stamp_deltas_s([100, 220, 400], []) == []
    assert MODULE.nearest_stamp_deltas_s(
        [1_000_000_000, 2_020_000_000],
        [1_010_000_000, 2_000_000_000],
    ) == pytest.approx([0.01, 0.02])


def test_request_and_manifest_parsers_fail_closed() -> None:
    class Message:
        def __init__(self, data: str) -> None:
            self.data = data

    assert MODULE._request_document(Message("not json")) is None
    assert MODULE._request_document(Message('{"schema":"wrong"}')) is None
    request = MODULE._request_document(
        Message(
            '{"schema":"z_manip.grounding_request.v2",'
            '"request_id":"r1","instruction":"charger"}'
        )
    )
    assert request is not None and request["request_id"] == "r1"

    assert MODULE._manifest_document(Message("{}")) is None
    manifest = MODULE._manifest_document(
        Message(
            '{"schema":"z_manip.tracker_frame.v1",'
            '"result_stamp_ns":123,"track_id":"t1"}'
        )
    )
    assert manifest is not None and manifest["result_stamp_ns"] == 123


def test_bundle_completion_requires_all_six_artifacts() -> None:
    slot = {"messages": {topic: object() for topic in MODULE.BUNDLE_TOPICS}}
    assert MODULE._bundle_complete(slot)
    slot["messages"].pop(MODULE.SCENE_CLOUD_TOPIC)
    assert not MODULE._bundle_complete(slot)


def test_summary_rejects_no_samples_without_inventing_latency() -> None:
    summary = MODULE.summarize_seconds([])
    assert summary == {
        "count": 0,
        "min_s": None,
        "p50_s": None,
        "p95_s": None,
        "max_s": None,
    }


def test_evict_oldest_bounds_ordereddict_like_bundles_already_does() -> None:
    from collections import OrderedDict

    cache: "OrderedDict[int, str]" = OrderedDict()
    for key in range(5):
        cache[key] = f"v{key}"
        MODULE._evict_oldest(cache, key, maximum=3)
        assert len(cache) <= 3

    # Only the three most-recently-inserted keys survive.
    assert list(cache.keys()) == [2, 3, 4]

    # Re-touching an existing key must bump it to most-recently-used instead
    # of being treated as a fresh insert that could exceed the cap.
    MODULE._evict_oldest(cache, 2, maximum=3)
    assert list(cache.keys()) == [3, 4, 2]


def _install_fake_perception_bridges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the ROS-only cv_bridge/sensor_msgs_py imports.

    `_benchmark_bundle` imports these lazily specifically so unit tests and
    report parsing stay ROS-independent (see the module docstring). These
    fakes decode a message's plain ndarray payload the same way the real
    bridge/point_cloud2 helpers would, without requiring the ROS runtime
    image `cv_bridge`/`sensor_msgs_py` are only installed in.
    """

    fake_cv_bridge = types.ModuleType("cv_bridge")

    class _FakeCvBridge:
        def imgmsg_to_cv2(self, msg: object, desired_encoding: str = "passthrough") -> np.ndarray:
            # A fresh copy each call, matching real decode semantics -- the
            # benchmark mutates its decoded mask array in place downstream.
            return np.array(msg.array, copy=True)  # type: ignore[attr-defined]

    fake_cv_bridge.CvBridge = _FakeCvBridge
    monkeypatch.setitem(sys.modules, "cv_bridge", fake_cv_bridge)

    fake_point_cloud2 = types.ModuleType("sensor_msgs_py.point_cloud2")

    def _read_points_numpy(msg: object, field_names=None, skip_nans: bool = True) -> np.ndarray:
        return np.array(msg.array, copy=True)  # type: ignore[attr-defined]

    fake_point_cloud2.read_points_numpy = _read_points_numpy
    fake_sensor_msgs_py = types.ModuleType("sensor_msgs_py")
    fake_sensor_msgs_py.point_cloud2 = fake_point_cloud2
    monkeypatch.setitem(sys.modules, "sensor_msgs_py", fake_sensor_msgs_py)
    monkeypatch.setitem(sys.modules, "sensor_msgs_py.point_cloud2", fake_point_cloud2)


class _Header:
    def __init__(self, frame_id: str) -> None:
        self.frame_id = frame_id


class _ArrayMessage:
    def __init__(self, array: np.ndarray, frame_id: str | None = None) -> None:
        self.array = np.asarray(array)
        if frame_id is not None:
            self.header = _Header(frame_id)


def _cylinder_points(
    radius: float = 0.020,
    height: float = 0.14,
    n_angle: int = 48,
    n_height: int = 10,
    cx: float = 0.48,
    cy: float = 0.0,
    z0: float = 0.08,
) -> np.ndarray:
    """A small upright cylinder cloud -- same shape z_manip's antipodal-grasp
    tests use, sized (0.04 m diameter) to clear the benchmark's fixed
    0.068 m aperture minus its 0.010 m default clearance."""

    angles = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)
    zs = np.linspace(z0, z0 + height, n_height)
    side = np.array([
        [cx + radius * np.cos(a), cy + radius * np.sin(a), z]
        for a in angles for z in zs
    ])
    rings = np.linspace(0.0, radius, 7)
    caps = np.array([
        [cx + r * np.cos(a), cy + r * np.sin(a), z]
        for z in (z0, z0 + height) for r in rings for a in angles
    ])
    return np.vstack((side, caps)).astype(np.float32)


def _background_plane(n: int = 12) -> np.ndarray:
    """A flat scene surface well below the cylinder, outside the exclusion
    radius, so `collision_scene` in `_benchmark_bundle` stays non-trivial."""

    xs = np.linspace(0.30, 0.66, n)
    ys = np.linspace(-0.20, 0.20, n)
    return np.array(
        [[x, y, 0.0] for x in xs for y in ys], dtype=np.float32
    )


def test_benchmark_bundle_builds_grasp_source_once_not_once_per_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `_benchmark_bundle` must construct its
    `AntipodalGraspSource` once per bundle and reuse it across every repeat
    of the timed loop, not rebuild it (paying its __init__ coercion/math.cos
    setup cost) on every one of `repeats` iterations.

    Also proves the hoist is behavior-preserving: every repeat must call
    `.generate()` (not skip it) and produce byte-identical candidates, since
    the source's fixed kwargs and the per-call `context` never change.
    """

    _install_fake_perception_bridges(monkeypatch)

    from z_manip.models import antipodal_grasp as antipodal_grasp_module

    construction_count = 0
    generate_call_count = 0
    generated_signatures: list[tuple] = []

    class _CountingAntipodalGraspSource(antipodal_grasp_module.AntipodalGraspSource):
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal construction_count
            construction_count += 1
            super().__init__(*args, **kwargs)

        def generate(self, context):
            nonlocal generate_call_count
            generate_call_count += 1
            candidates = super().generate(context)
            generated_signatures.append((
                int(len(candidates.grasps)),
                tuple(np.round(np.asarray(candidates.grasps).ravel(), 6).tolist()),
                tuple(np.round(np.asarray(candidates.scores).ravel(), 6).tolist()),
            ))
            return candidates

    monkeypatch.setattr(
        antipodal_grasp_module, "AntipodalGraspSource", _CountingAntipodalGraspSource
    )

    target_points = _cylinder_points()
    scene_points = np.vstack((target_points, _background_plane()))
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:10, 3:13] = 1
    overlay = np.zeros((12, 16, 3), dtype=np.uint8)

    slot = {
        "messages": {
            MODULE.OVERLAY_TOPIC: _ArrayMessage(overlay),
            MODULE.MASK_TOPIC: _ArrayMessage(mask),
            MODULE.TARGET_CLOUD_TOPIC: _ArrayMessage(
                target_points, frame_id="base_link"
            ),
            MODULE.SCENE_CLOUD_TOPIC: _ArrayMessage(scene_points),
        }
    }

    repeats = 4
    result = MODULE._benchmark_bundle(slot, repeats)

    assert construction_count == 1, (
        "AntipodalGraspSource must be built once before the repeat loop, "
        f"not {construction_count} times across {repeats} repeats"
    )
    assert generate_call_count == repeats
    assert result["repeats"] == repeats
    assert result["grasp_generation"]["count"] == repeats
    assert result["sample"]["grasp_error"] is None
    assert result["sample"]["grasp_candidates"] > 0

    # Every repeat re-runs `.generate()` against the identical fixed context,
    # so the reused solver must produce byte-identical candidates each time.
    assert len(generated_signatures) == repeats
    assert len(set(generated_signatures)) == 1
