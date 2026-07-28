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
    assert report["stages"] == ["clamp", "edge", "speckle", "median"]
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
    assert cfg.enabled and cfg.edge and cfg.speckle and cfg.median and cfg.clamp
    assert cfg.temporal is False
    assert cfg.max_grad_mm == 120.0
    assert cfg.speckle_max_size == 50
    assert cfg.median_ksize == 5
    # thresholds live in disparity px, and the near clip matches the 0.28 m
    # contract in z_manip/adapters and edgetam.yaml
    assert cfg.edge_mode == "bleed"
    assert (cfg.edge_radius, cfg.edge_near_radius) == (7, 2)
    assert cfg.edge_margin_px == 1.0
    assert cfg.speckle_max_diff_px == 0.5
    assert cfg.min_depth_mm == 280
    assert cfg.active_stages() == ["clamp", "edge", "speckle", "median"]


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
    assert tuned.active_stages() == ["clamp", "speckle", "median", "temporal"]


def test_per_stage_toggle_via_config():
    scene, _ = add_box(make_floor())
    only_speckle = ffs.FilterConfig(edge=False, median=False)
    out, report = ffs.filter_depth(scene, only_speckle)
    assert report["stages"] == ["clamp", "speckle"]


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
        "FFS_FILTER_SPECKLE_MAX_DIFF_PX",
        "FFS_FILTER_CLAMP",
        "FFS_FILTER_MIN_DEPTH_MM",
        "FFS_FILTER_EDGE_MODE",
        "FFS_FILTER_EDGE_RADIUS",
        "FFS_FILTER_EDGE_MARGIN_PX",
        "FFS_FILTER_MEDIAN",
        "FFS_FILTER_MEDIAN_KSIZE",
        "FFS_FILTER_TEMPORAL",
    ):
        assert var in src, f"{var} not documented/passed in stack script"
    assert "${FFS_FILTER:-1}" in src   # default-on, overridable


# --------------------------------------------------------------------------- #
# LIVE-DATA regression suite
#
# tests/data/ffs_depth_live_pairs.npz holds 4 exact-stamp-matched
# (raw D435 aligned depth, published FFS depth) pairs captured read-only off the
# running stack on 2026-07-28 (topics /camera/aligned_depth_to_color/image_raw
# and /camera/ffs_depth_aligned/image_raw, both 640x480 16UC1 mm in
# camera_color_optical_frame).  Every acceptance number the FFS chain had before
# was a PRECISION metric -- plane-fit RMS, temporal std, hole %, mad_p95 -- and
# none of them can see a bias.  These check FFS against a real reference.
#
# Scene caveat baked into these bounds: the captured scenes span 0.85-3.0 m and
# contain essentially NO pixels in [0.30, 0.60) m, so nothing here measures the
# grasp band.  The grasp-band guarantee is carried by
# test_edge_bleed_is_plane_invariant_across_range instead, which pins the
# mathematical invariance the thresholds rest on.
# --------------------------------------------------------------------------- #
LIVE_PAIRS = ROOT / "tests" / "data" / "ffs_depth_live_pairs.npz"


def _live_pairs():
    if not LIVE_PAIRS.exists():                      # pragma: no cover
        pytest.skip(f"live capture fixture missing: {LIVE_PAIRS}")
    d = np.load(LIVE_PAIRS)
    n = int(d["n"])
    return [(d[f"raw_{i}"], d[f"ffs_{i}"]) for i in range(n)], float(
        d["disparity_const_mm_px"])


def _live_cfg(disp_const, **kw):
    return ffs.FilterConfig(disparity_const_mm_px=disp_const, **kw)


def _legacy_chain(depth):
    """The chain exactly as it shipped: 3x3/120 mm gate, mm speckle, naive median."""
    a = ffs.remove_flying_pixels(depth, 120.0)
    b = ffs.remove_speckles(a, 50, 24)
    med = cv2.medianBlur(b, 5)
    return np.where((b > 0) & (med > 0), med, b).astype(np.uint16)


def _edge_distance(raw):
    """Distance in px to the nearest strong depth discontinuity in the reference."""
    v = raw > 0
    lo = cv2.erode(np.where(v, raw.astype(np.float32), 1e6), ffs._KERNEL3)
    hi = cv2.dilate(np.where(v, raw.astype(np.float32), 0.0), ffs._KERNEL3)
    edge = (v & (hi - lo > 100) & (hi - lo < 1e5)).astype(np.uint8)
    return cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)


# --- 1. the silhouette bleed the filter is supposed to target ---------------- #
def test_edge_bleed_gate_cuts_the_near_silhouette_overshoot_on_live_frames():
    """The p90 of (FFS - raw) inside 4 px of a real silhouette must come down.

    The bleed is a TAIL, not a shift: the median excess inside 4 px is only
    +6..+25 mm but the p90 reaches +180..+210 mm, and those are the points that
    hang in front of / behind object outlines in the 3D view.  A 3x3/120 mm
    gradient gate leaves the p90 untouched (it moves it by ~1 mm) because a
    200 mm excess spread over a 7 px ramp is under 30 mm/px.
    """
    pairs, dc = _live_pairs()
    before, legacy, after = [], [], []
    for raw, ffs_depth in pairs:
        dist = _edge_distance(raw)
        band = (dist < 4) & (raw >= 280) & (raw <= 2500)
        new, _ = ffs.filter_depth(ffs_depth, _live_cfg(dc))
        old = _legacy_chain(ffs_depth)
        for sink, img in ((before, ffs_depth), (legacy, old), (after, new)):
            sel = band & (img >= 280) & (img <= 2500)
            sink.append((img.astype(np.float32) - raw.astype(np.float32))[sel])
    p90 = lambda xs: float(np.percentile(np.concatenate(xs), 90))
    b, l, a = p90(before), p90(legacy), p90(after)
    assert b > 100.0, f"fixture should contain a real bleed tail, got p90 {b:.0f} mm"
    assert l > 0.95 * b, (
        f"legacy 3x3 gate unexpectedly helped ({b:.0f} -> {l:.0f} mm); if the "
        "fixture changed, re-derive the ramp profile before re-tuning")
    assert a < 0.88 * b, f"edge-bleed gate did not cut the overshoot: {b:.0f} -> {a:.0f} mm"


def test_edge_bleed_gate_reduces_points_beyond_the_reference_background():
    """Points that sit BEHIND everything real in their neighbourhood must shrink.

    This is the unambiguous failure: not "a bit far" but past the local
    background the reference sensor reports, i.e. floating in free space.
    """
    pairs, dc = _live_pairs()
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tally = {"before": [0, 0], "after": [0, 0]}
    for raw, ffs_depth in pairs:
        rv = raw > 0
        rf = raw.astype(np.float32)
        bg = cv2.dilate(np.where(rv, rf, 0.0), k15)
        fg = cv2.erode(np.where(rv, rf, 1e6), k15)
        strong = rv & (bg - fg > 200) & (fg < 1e5)
        new, _ = ffs.filter_depth(ffs_depth, _live_cfg(dc))
        for name, img in (("before", ffs_depth), ("after", new)):
            sel = strong & (img >= 280) & (img <= 2500)
            tally[name][0] += int(sel.sum())
            tally[name][1] += int((sel & (img.astype(np.float32) > bg + 10)).sum())
    rate = lambda k: tally[k][1] / max(1, tally[k][0])
    assert rate("before") > 0.005, "fixture should contain beyond-background points"
    assert rate("after") < 0.85 * rate("before"), (
        f'beyond-background rate {rate("before"):.4f} -> {rate("after"):.4f}')


def test_edge_bleed_is_plane_invariant_across_range():
    """A plane at ANY slant and ANY range must survive the gate intact.

    This is the property that lets the 0.6-2.5 m measurements set a threshold
    for the UNMEASURED 0.3-0.6 m grasp band.  In disparity a plane is exactly
    affine in (u, v), so its max-disparity excess over a square window scales
    linearly with the window radius and the (R/r)*step_r term cancels it
    identically -- independent of slant and of range.  A mm-domain slope test
    cannot do this: an 80-degree floor and a silhouette ramp both run at about
    30 mm/px.
    """
    v_idx, u_idx = np.mgrid[0:H, 0:W].astype(np.float32)
    C = ffs.DISPARITY_CONST_MM_PX
    for z0_mm in (300.0, 350.0, 450.0, 700.0, 1200.0, 2400.0):
        for slant in (0.4, 0.8, 1.2, 1.4):
            # Build the plane in DISPARITY -- that is what a real plane IS --
            # then invert to uint16 mm, so the fixture is a physically
            # realisable surface complete with the 1 mm quantisation that
            # terraces close-range geometry.  slant 1.4 spans roughly 0.17x to
            # 3x z0 across the frame: grazing incidence.
            disp = (C / z0_mm) * (1.0 + slant * (u_idx - W / 2) / W
                                  + 0.5 * slant * (v_idx - H / 2) / H)
            depth = np.clip(np.rint(C / disp), 0, 65535).astype(np.uint16)
            lost = int(((depth > 0) & (ffs.remove_edge_bleed(depth, C) == 0)).sum())
            assert lost == 0, (
                f"oblique plane at {z0_mm:.0f} mm (slant {slant}, depth "
                f"{depth.min()}-{depth.max()} mm) lost {lost} px")

    # ...while a real standoff INSIDE the grasp band is still caught: a 7 cm
    # step at 0.45 m is a 7.2 disparity px jump, far above the 1.0 px floor.
    step = np.full((H, W), 450, np.uint16)
    step[150:330, 200:440] = 380
    dropped = int(((step > 0) & (ffs.remove_edge_bleed(step, C) == 0)).sum())
    assert dropped > 2000, f"grasp-band silhouette not detected ({dropped} px)"


def test_edge_bleed_never_erodes_the_foreground_side():
    """The gate is one-sided: nothing with an empty foreground is ever dropped.

    A pixel on the NEAR side of a discontinuity has step_R = 0 by construction,
    so object silhouettes are never eaten -- only the unsupported background
    ribbon behind them is.  Checked on a synthetic step (exact) and on the live
    frames (the near-side rim of every real silhouette).
    """
    floor = make_floor()
    scene, interior = add_box(floor)                 # 600 mm box on a ~1300 mm floor
    keep = ffs.remove_edge_bleed(scene, ffs.DISPARITY_CONST_MM_PX)
    box = scene == scene[interior].min()
    assert np.array_equal(keep[interior], scene[interior])
    # every pixel at the box's own depth survives, including its outermost row
    assert int((box & (keep == 0)).sum()) == 0

    pairs, dc = _live_pairs()
    for raw, ffs_depth in pairs:
        keep = ffs.remove_edge_bleed(ffs_depth, dc)
        dropped = (ffs_depth > 0) & (keep == 0)
        if not dropped.any():
            continue
        # a dropped pixel always has a strictly nearer surface within 7 px
        disp = ffs.to_disparity(ffs_depth, dc)
        wide = cv2.dilate(disp, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        assert float((wide - disp)[dropped].min()) > 1.0


# --- 2. hole-aware median ---------------------------------------------------- #
def test_median_is_hole_aware_and_never_pulls_a_pixel_nearer_through_a_hole():
    """cv2.medianBlur sorts the 0s, so k holes bias the result k/2 ranks NEAR.

    On the live frames the old stage moved 16k px/frame by -2.8 mm on average
    but as much as -2496 mm on a single pixel -- a near-side flying pixel
    manufactured by the smoother itself, exactly where the map is least
    trustworthy.  The hole-aware stage must be exactly unbiased.
    """
    pairs, _ = _live_pairs()
    naive_worst, new_worst = 0, 0
    naive_touched = new_touched = 0
    for _raw, ffs_depth in pairs:
        med = cv2.medianBlur(ffs_depth, 5)
        naive = np.where((ffs_depth > 0) & (med > 0), med, ffs_depth).astype(np.uint16)
        new = ffs.smooth_banding(ffs_depth, 5)
        for img, key in ((naive, "naive"), (new, "new")):
            ch = img.astype(np.int32) - ffs_depth.astype(np.int32)
            touched = int(np.count_nonzero(ch))
            if key == "naive":
                naive_worst = min(naive_worst, int(ch.min())); naive_touched += touched
            else:
                new_worst = min(new_worst, int(ch.min())); new_touched += touched
    assert naive_worst < -500, (
        f"fixture should exhibit the hole bias; naive worst {naive_worst} mm")
    # the new stage only ever substitutes a rank drawn from a hole-free window,
    # so a change larger than the local surface variation is impossible
    assert new_worst > -200, f"hole-aware median still swings {new_worst} mm"
    assert new_touched > 0.2 * naive_touched, (
        "hole-awareness must not disable the smoothing wholesale "
        f"({new_touched} vs {naive_touched} px touched)")


def test_median_only_substitutes_from_a_hole_free_window():
    """Exact contract: any pixel whose k x k window touches a hole is untouched."""
    scene, _ = add_box(make_floor(band_amp_mm=6.0))
    scene = scene.copy()
    scene[200:206, 300:306] = 0                       # a hole
    out = ffs.smooth_banding(scene, 5)
    invalid = (scene == 0).astype(np.uint8)
    # a pixel adjacent to the hole must keep its original value: neither the
    # 5x5 nor the 3x3 window around it is hole-free
    assert out[199, 302] == scene[199, 302]
    assert out[206, 302] == scene[206, 302]
    # far from any hole the median is applied as before
    assert out[100, 100] == cv2.medianBlur(scene, 5)[100, 100]
    assert np.array_equal(out > 0, scene > 0)


# --- 3. coverage: which stage costs what ------------------------------------- #
def test_new_chain_keeps_far_more_coverage_than_the_legacy_chain():
    """Equal-start comparison on an UNFILTERED real depth map.

    The published FFS frame has already been carved by the shipped filter, so
    comparing chains on it flatters the legacy one.  The raw D435 aligned frame
    in the same color frame is an unfiltered real depth map, which is the only
    honest common starting point available read-only.
    """
    pairs, dc = _live_pairs()
    legacy_valid = new_valid = in_valid = 0
    stage_drops = {}
    for raw, _ffs in pairs:
        in_valid += int(np.count_nonzero(raw))
        legacy_valid += int(np.count_nonzero(_legacy_chain(raw)))
        out, rep = ffs.filter_depth(raw, _live_cfg(dc))
        new_valid += int(np.count_nonzero(out))
        for k, v in rep["dropped"].items():
            stage_drops[k] = stage_drops.get(k, 0) + v
    legacy_loss = 1.0 - legacy_valid / in_valid
    new_loss = 1.0 - new_valid / in_valid
    assert legacy_loss > 0.10, f"legacy chain loss {legacy_loss:.4f}"
    assert new_loss < 0.05, f"new chain loss {new_loss:.4f}"
    # and the report must attribute it, which is the whole point of the split
    assert set(stage_drops) == {"clamp", "edge", "speckle", "median"}
    assert stage_drops["median"] == 0     # the median never changes validity


def test_speckle_in_disparity_stops_deleting_the_far_field():
    """A fixed mm join tolerance shreds the far field; depth noise grows as z^2.

    On the raw D435 frames a 24 mm tolerance deletes over 40% of everything
    beyond 2.5 m, because at that range adjacent pixels of a flat surface
    already differ by more than 24 mm, so the surface fragments into
    sub-max_size components and the stage removes it wholesale.
    """
    pairs, dc = _live_pairs()
    far_pop = mm_lost = disp_lost = 0
    for raw, _ffs in pairs:
        far = (raw > 2500) & (raw < 60000)
        far_pop += int(far.sum())
        mm_lost += int((far & (ffs.remove_speckles(raw, 50, 24) == 0)).sum())
        disp_lost += int((far & (ffs.remove_speckles(
            raw, 50, disparity_const_mm_px=dc, max_diff_px=0.5) == 0)).sum())
    assert far_pop > 10000, "fixture should contain a far field"
    assert mm_lost / far_pop > 0.30, f"mm speckle far-field loss {mm_lost/far_pop:.3f}"
    assert disp_lost / far_pop < 0.05, (
        f"disparity speckle far-field loss {disp_lost/far_pop:.3f}")


def test_new_chain_does_not_manufacture_validity_flicker():
    """A static scene must not shimmer.

    Over the captured static run the legacy chain nearly TRIPLES the fraction of
    pixels that toggle between valid and invalid frame to frame (9% -> 26% on
    the raw stream).  That shimmer is what reads as a sparse, banded cloud in a
    live 3D view; the depth values themselves are barely noisier than the raw
    sensor's.
    """
    pairs, dc = _live_pairs()
    seq = {"input": [], "legacy": [], "new": []}
    for raw, _ffs in pairs:
        seq["input"].append(raw > 0)
        seq["legacy"].append(_legacy_chain(raw) > 0)
        seq["new"].append(ffs.filter_depth(raw, _live_cfg(dc))[0] > 0)
    def toggling(masks):
        st = np.stack(masks)
        frac = st.mean(axis=0)
        return float(((frac > 0.01) & (frac < 0.99)).mean())
    base, legacy, new = (toggling(seq[k]) for k in ("input", "legacy", "new"))
    assert legacy > 1.8 * base, f"fixture: legacy flicker {base:.4f} -> {legacy:.4f}"
    assert new < 1.35 * base, f"new chain added flicker {base:.4f} -> {new:.4f}"


# --- 4. near-range clamp ------------------------------------------------------ #
def test_clamp_enforces_the_028_m_contract_consumers_are_written_against():
    """z_manip/adapters pins DEPTH_NEAR_CLIP_M = 0.28 and edgetam.yaml
    min_depth_m: 0.28 for this very topic; the raw D435 stream never emits
    below ~0.24 m.  FFS has no hardware near limit and was publishing ~1900
    px/frame at 0.08-0.15 m -- points inside the gripper, on the robot's own
    body, that no consumer's near clip is expecting to have to reject.
    """
    pairs, dc = _live_pairs()
    before = after = 0
    for _raw, ffs_depth in pairs:
        before += int(((ffs_depth > 0) & (ffs_depth < 280)).sum())
        out, _ = ffs.filter_depth(ffs_depth, _live_cfg(dc))
        after += int(((out > 0) & (out < 280)).sum())
    assert before > 1000, f"fixture should contain the sub-clip points ({before})"
    assert after == 0

    adapters = (ROOT / "z_manip" / "adapters" / "__init__.py").read_text(encoding="utf-8")
    assert "DEPTH_NEAR_CLIP_M = 0.28" in adapters, (
        "the clamp default is pinned to the consumer contract; if that moved, "
        "move FilterConfig.min_depth_mm with it")
    assert ffs.FilterConfig().min_depth_mm == 280


def test_clamp_is_a_pure_range_mask():
    scene = np.array([[0, 100, 279, 280, 2000, 9999, 10000, 10001]], np.uint16)
    out = ffs.clamp_range(scene, 280, 10000)
    assert out.tolist() == [[0, 0, 0, 280, 2000, 9999, 10000, 0]]


def test_report_attributes_drops_per_stage():
    scene, _ = add_box(make_floor(band_amp_mm=4.0))
    scene, _ = add_specks(scene, n=25, seed=11)
    out, report = ffs.filter_depth(scene, ffs.FilterConfig())
    assert set(report["dropped"]) == {"clamp", "edge", "speckle", "median"}
    assert sum(report["dropped"].values()) == report["removed"]


def test_relay_logs_the_per_stage_split_and_uses_live_calibration():
    src = RELAY_PATH.read_text(encoding="utf-8")
    # the filter's thresholds are in disparity px, so it needs fx_ir*B from the
    # calibration the relay already fail-closed verifies against camera_info
    assert "disparity_const_mm_px" in src
    assert "self.calib['K_ir1']['fx']" in src
    assert "report.get('dropped'" in src
    assert "filt_drop~" in src
