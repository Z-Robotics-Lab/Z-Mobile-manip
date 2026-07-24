"""Unit + contract tests for the FFS depth noise filter.

The filter core (scripts/runtime/ffs_depth_filter.py) is pure numpy/cv2, so
these run on a bare host with no ROS/torch/GPU.  Synthetic 640x480 mm-depth
frames reproduce the three live noise modes (free-space specks, flying pixels at
discontinuities, low-texture banding) plus a clean box whose geometry must
survive.  Contract tests pin the relay wiring + the stack env interface.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
FILTER_PATH = ROOT / "scripts" / "runtime" / "ffs_depth_filter.py"
RELAY_PATH = ROOT / "scripts" / "runtime" / "ffs_depth_relay.py"
STACK_PATH = ROOT / "scripts" / "runtime" / "ffs_depth_stack.sh"

_SPEC = importlib.util.spec_from_file_location("ffs_depth_filter", FILTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ffs = importlib.util.module_from_spec(_SPEC)
# Register before exec: the module uses `from __future__ import annotations`, so
# @dataclass resolves its (stringised) field types via sys.modules -- exactly as
# the relay's normal `import ffs_depth_filter` would.
sys.modules["ffs_depth_filter"] = ffs
_SPEC.loader.exec_module(ffs)

H, W = 480, 640


# --------------------------------------------------------------------------- #
# synthetic scene helpers
# --------------------------------------------------------------------------- #
def make_floor(band_amp_mm: float = 0.0) -> np.ndarray:
    """Obliquely-viewed floor: depth ramps top->bottom, optional banding ripple."""
    rows = np.linspace(700.0, 2600.0, H, dtype=np.float32)[:, None]
    depth = np.repeat(rows, W, axis=1)
    if band_amp_mm:
        # vertical striping: sub-pixel disparity quantisation shows up as a
        # short-period ripple across columns.
        ripple = band_amp_mm * np.sign(np.sin(np.arange(W) * (np.pi / 1.5)))
        depth += ripple[None, :]
    return depth.astype(np.uint16)


def add_box(depth: np.ndarray, top=150, left=250, size=140,
            face_mm=600, ramp_mm=200) -> tuple:
    """Stamp a foreground box: flat front face + a slanted top (side-face proxy).

    Returns (depth, interior_slice) where interior excludes a 3 px border so
    tests can assert the object body is untouched by edge/median stages.
    """
    d = depth.copy()
    r0, r1 = top, top + size
    c0, c1 = left, left + size
    # front face: flat plane; top third slanted to emulate a real side face.
    face = np.full((size, size), face_mm, np.float32)
    ramp_rows = size // 3
    face[:ramp_rows, :] += np.linspace(ramp_mm, 0, ramp_rows, dtype=np.float32)[:, None]
    d[r0:r1, c0:c1] = face.astype(np.uint16)
    interior = (slice(r0 + 3, r1 - 3), slice(c0 + 3, c1 - 3))
    return d, interior


def add_specks(depth: np.ndarray, n=30, seed=0) -> tuple:
    """Sprinkle small isolated free-space blobs at wrong depths on invalid gaps."""
    rng = np.random.default_rng(seed)
    d = depth.copy()
    coords = []
    for _ in range(n):
        r = int(rng.integers(20, H - 20))
        c = int(rng.integers(20, W - 20))
        s = int(rng.integers(1, 4))            # 1..3 px radius blobs
        val = int(rng.integers(300, 3000))
        # punch a hole around it so the speck is a disconnected component
        d[r - s - 2:r + s + 3, c - s - 2:c + s + 3] = 0
        d[r - s:r + s + 1, c - s:c + s + 1] = val
        coords.append((r, c))
    return d, coords


def add_flying_pixels(depth: np.ndarray, box_slice) -> tuple:
    """Smear a thin intermediate-depth streak across a box silhouette (edge bleed).

    Real flying pixels are a 1-2 px smear straddling the fg/bg jump; use 1 px so
    every band pixel genuinely sits on a discontinuity.  Returns (depth, region).
    """
    d = depth.copy()
    rs, cs = box_slice
    region = (slice(rs.start - 1, rs.start), cs)   # 1 px band just above the box
    d[region] = 1050                               # mid-depth (600 fg / ~1300 bg)
    return d, region


# --------------------------------------------------------------------------- #
# stage: speckle removal
# --------------------------------------------------------------------------- #
def _small_components(depth: np.ndarray, thr=100) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        (depth > 0).astype(np.uint8), connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] < thr).sum())


def test_speckle_removes_free_space_specks():
    floor = make_floor()
    noisy, _ = add_specks(floor, n=30, seed=1)
    before = _small_components(noisy)
    assert before >= 20, "test fixture should inject many specks"
    cleaned = ffs.remove_speckles(noisy, max_size=50, max_diff_mm=24)
    assert _small_components(cleaned) == 0
    # the big floor component must survive untouched
    assert (cleaned > 0).sum() > 0.9 * (floor > 0).sum()


def test_speckle_keeps_a_real_object():
    floor = make_floor()
    scene, interior = add_box(floor)
    cleaned = ffs.remove_speckles(scene, max_size=50, max_diff_mm=24)
    # a 140x140 box is far larger than max_size and must be fully retained
    assert np.count_nonzero(cleaned[interior]) == np.count_nonzero(scene[interior])


# --------------------------------------------------------------------------- #
# stage: flying-pixel / edge removal
# --------------------------------------------------------------------------- #
def test_edge_removes_flying_pixels_at_discontinuity():
    floor = make_floor()
    scene, box_slice = add_box(floor)
    with_fly, fly_region = add_flying_pixels(scene, box_slice)
    assert np.count_nonzero(with_fly[fly_region]) > 0
    cleaned = ffs.remove_flying_pixels(with_fly, max_grad_mm=120)
    # the smeared mid-depth band on the silhouette is gone
    assert np.count_nonzero(cleaned[fly_region]) == 0


def test_edge_preserves_object_body_and_floor_interior():
    floor = make_floor()
    scene, interior = add_box(floor)
    cleaned = ffs.remove_flying_pixels(scene, max_grad_mm=120)
    # object interior (away from silhouette) fully retained, values unchanged
    assert np.array_equal(cleaned[interior], scene[interior])
    # gentle floor ramp (no discontinuity) keeps essentially all pixels
    floor_only = ffs.remove_flying_pixels(floor, max_grad_mm=120)
    assert (floor_only > 0).sum() >= 0.999 * (floor > 0).sum()


# --------------------------------------------------------------------------- #
# stage: banding median
# --------------------------------------------------------------------------- #
def test_median_reduces_banding_std():
    banded = make_floor(band_amp_mm=6.0)
    flat = make_floor(band_amp_mm=0.0).astype(np.float32)
    before = float((banded.astype(np.float32) - flat).std())
    smoothed = ffs.smooth_banding(banded, ksize=5).astype(np.float32)
    after = float((smoothed - flat).std())
    assert after < 0.5 * before, f"banding not reduced: {before:.2f}->{after:.2f}"


def test_median_preserves_plane_mean_within_2mm():
    banded = make_floor(band_amp_mm=6.0)
    smoothed = ffs.smooth_banding(banded, ksize=5)
    # interior rows only (avoid the top/bottom border where the ramp clips)
    core = (slice(40, H - 40), slice(40, W - 40))
    assert abs(float(smoothed[core].mean()) - float(banded[core].mean())) < 2.0


def test_median_never_invalidates_or_fills():
    floor = make_floor()
    scene, _ = add_box(floor)
    scene, _ = add_specks(scene, n=15, seed=3)   # introduces invalid holes
    smoothed = ffs.smooth_banding(scene, ksize=5)
    # a pixel invalid before stays invalid; a valid pixel stays valid
    assert np.array_equal((smoothed > 0), (scene > 0))


# --------------------------------------------------------------------------- #
# object-extent preservation through the FULL chain
# --------------------------------------------------------------------------- #
def test_full_chain_does_not_erode_object_extent_beyond_2mm():
    floor = make_floor(band_amp_mm=4.0)
    # specks first, then stamp the box on top so its body starts fully valid
    floor, _ = add_specks(floor, n=25, seed=5)
    scene, interior = add_box(floor)
    cfg = ffs.FilterConfig()   # all defaults, temporal off
    out, report = ffs.filter_depth(scene, cfg)

    body_before = scene[interior].astype(np.float32)
    body_after = out[interior].astype(np.float32)
    # object body not hollowed out by any stage
    assert np.count_nonzero(body_after) == body_before.size
    # near-face depth (min) and mid/side geometry (mean) preserved <=2 mm:
    # the mid-plane TCP depends on true side-face depth, must not shift.
    assert abs(body_after.min() - body_before.min()) <= 2.0
    assert abs(body_after.mean() - body_before.mean()) <= 2.0
    assert abs(body_after.max() - body_before.max()) <= 2.0
    # and the specks are gone
    assert _small_components(out) == 0


def test_full_chain_removes_specks_and_flying_pixels_together():
    floor = make_floor(band_amp_mm=4.0)
    scene, box_slice = add_box(floor)
    scene, _ = add_flying_pixels(scene, box_slice)
    scene, _ = add_specks(scene, n=30, seed=7)
    cfg = ffs.FilterConfig()
    out, report = ffs.filter_depth(scene, cfg)
    assert _small_components(out) == 0
    assert report["stages"] == ["edge", "speckle", "median"]
    assert report["removed"] >= 0


# --------------------------------------------------------------------------- #
# temporal EMA (opt-in)
# --------------------------------------------------------------------------- #
def test_temporal_ema_blends_static_and_resets_moved():
    prev = np.full((H, W), 1000, np.uint16)
    curr = np.full((H, W), 1000, np.uint16)
    curr[100:110, 100:110] = 1004          # tiny jitter (static)
    curr[200:210, 200:210] = 1400          # big change (motion)
    out = ffs.temporal_ema(prev, curr, alpha=0.5, change_thresh_mm=40)
    assert out[105, 105] == 1002           # blended 0.5*1004 + 0.5*1000
    assert out[205, 205] == 1400           # motion pixel takes current frame
    # invalid-in-either stays current
    prev2 = prev.copy(); prev2[300, 300] = 0
    out2 = ffs.temporal_ema(prev2, curr, alpha=0.5, change_thresh_mm=40)
    assert out2[300, 300] == curr[300, 300]


def test_temporal_ema_handles_missing_prev():
    curr = make_floor()
    out = ffs.temporal_ema(None, curr, alpha=0.5, change_thresh_mm=40)
    assert np.array_equal(out, curr)


# --------------------------------------------------------------------------- #
# config + escape hatch
# --------------------------------------------------------------------------- #
def test_disabled_is_passthrough_same_object():
    scene, _ = add_box(make_floor())
    cfg = ffs.FilterConfig(enabled=False)
    out, report = ffs.filter_depth(scene, cfg)
    assert out is scene              # zero-copy escape hatch
    assert report["stages"] == []


def test_from_env_defaults():
    cfg = ffs.FilterConfig.from_env(env={})
    assert cfg.enabled and cfg.edge and cfg.speckle and cfg.median
    assert cfg.temporal is False
    assert cfg.max_grad_mm == 120.0
    assert cfg.speckle_max_size == 50
    assert cfg.median_ksize == 5
    assert cfg.active_stages() == ["edge", "speckle", "median"]


def test_from_env_overrides_and_master_switch():
    off = ffs.FilterConfig.from_env(env={"FFS_FILTER": "0"})
    assert off.enabled is False
    assert off.active_stages() == []
    tuned = ffs.FilterConfig.from_env(env={
        "FFS_FILTER_EDGE": "0",
        "FFS_FILTER_SPECKLE_MAX_SIZE": "80",
        "FFS_FILTER_MEDIAN_KSIZE": "3",
        "FFS_FILTER_TEMPORAL": "1",
    })
    assert tuned.edge is False
    assert tuned.speckle_max_size == 80
    assert tuned.median_ksize == 3
    assert tuned.temporal is True
    assert tuned.active_stages() == ["speckle", "median", "temporal"]


def test_per_stage_toggle_via_config():
    scene, _ = add_box(make_floor())
    only_speckle = ffs.FilterConfig(edge=False, median=False)
    out, report = ffs.filter_depth(scene, only_speckle)
    assert report["stages"] == ["speckle"]


def test_median_ksize_clamped_to_supported_aperture():
    floor = make_floor(band_amp_mm=6.0)
    # any >=5 request clamps to 5, <5 to 3; both are valid CV_16U apertures
    assert ffs.smooth_banding(floor, ksize=9).shape == floor.shape
    assert ffs.smooth_banding(floor, ksize=1).shape == floor.shape


def test_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        ffs.remove_speckles(np.zeros((H, W), np.float32))


# --------------------------------------------------------------------------- #
# timing budget (generous CI margin over the ~3.3 ms measured on the relay host)
# --------------------------------------------------------------------------- #
def test_full_chain_within_timing_budget():
    import time
    floor = make_floor(band_amp_mm=4.0)
    scene, box_slice = add_box(floor)
    scene, _ = add_flying_pixels(scene, box_slice)
    scene, _ = add_specks(scene, n=30, seed=9)
    cfg = ffs.FilterConfig()
    ffs.filter_depth(scene, cfg)     # warm cv2
    ts = []
    for _ in range(25):
        t0 = time.perf_counter()
        ffs.filter_depth(scene, cfg)
        ts.append((time.perf_counter() - t0) * 1e3)
    median_ms = float(np.median(ts))
    # real budget is <=5 ms; assert a generous 15 ms so CI hardware jitter
    # can't flake while still catching a gross algorithmic regression.
    assert median_ms < 15.0, f"chain median {median_ms:.2f} ms exceeds budget"


# --------------------------------------------------------------------------- #
# contract: relay wiring + stack env interface
# --------------------------------------------------------------------------- #
def test_relay_applies_filter_at_the_source():
    src = RELAY_PATH.read_text(encoding="utf-8")
    assert "from ffs_depth_filter import FilterConfig, filter_depth" in src
    assert "FilterConfig.from_env()" in src
    assert "filter_depth(depth, self._filter_cfg" in src
    # escape hatch: raw bytes pass straight through when disabled
    assert "if self._filter_cfg.enabled:" in src


def test_stack_mounts_filter_and_documents_env():
    src = STACK_PATH.read_text(encoding="utf-8")
    assert "ffs_depth_filter.py:/usr/local/bin/ffs_depth_filter.py:ro" in src
    for var in (
        "FFS_FILTER",
        "FFS_FILTER_EDGE",
        "FFS_FILTER_EDGE_MAX_GRAD_MM",
        "FFS_FILTER_SPECKLE",
        "FFS_FILTER_SPECKLE_MAX_SIZE",
        "FFS_FILTER_SPECKLE_MAX_DIFF_MM",
        "FFS_FILTER_MEDIAN",
        "FFS_FILTER_MEDIAN_KSIZE",
        "FFS_FILTER_TEMPORAL",
    ):
        assert var in src, f"{var} not documented/passed in stack script"
    assert "${FFS_FILTER:-1}" in src   # default-on, overridable
