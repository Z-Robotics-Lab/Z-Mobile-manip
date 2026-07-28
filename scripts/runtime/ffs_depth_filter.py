#!/usr/bin/env python3
"""Lightweight noise filtering for the Fast-FoundationStereo depth chain.

Pure, dependency-light (numpy + OpenCV only) filter core, applied in the FFS
relay right after the service returns the COLOR-frame depth image and before it
is published on ``/camera/ffs_depth_aligned/image_raw`` -- i.e. ONE place,
upstream of every consumer (EdgeTAM depth, grasp scene_points, UI cloud,
collision).  No ROS / torch imports, so the whole module is unit-testable on a
bare host.

WHY THE STAGES WORK IN DISPARITY, NOT MILLIMETRES
-------------------------------------------------
Every threshold that used to be expressed in mm was wrong by a factor of z^2.
Stereo depth noise is sigma_z = z^2 * sigma_disp / (fx*B); with the wrist D435's
fx*B = 19375 mm*px and a ~0.25 px matching noise that is 2 mm at 0.4 m but
80 mm at 2.5 m.  A single mm threshold therefore either shreds the far field or
ignores the near field.  Measured on 24 live stamp-matched frames (2026-07-28),
running the WHOLE mm chain on the raw D435 aligned stream -- an equal-start
comparison against an unfiltered real depth map in the same color frame --
it deleted 14.6% of the frame (mm speckle alone removed 52.2% of everything
beyond 2.5 m and 8.6% of 1.8-2.5 m) and TRIPLED temporal instability: over a
24-frame static run the fraction of pixels toggling between valid and invalid
went 9.0% -> 25.8%.  The disparity chain on the same input drops 1.8% and
leaves flicker at 9.6%.  That shimmer is what a live 3D view shows as a sparse,
banded, hole-ridden cloud.

(The mm speckle stage is not currently the dominant hole source for FFS itself:
FFS output is ~3.6x smoother per pixel than the raw D435 stream -- mean
|Laplacian| 6.1 vs 22.0 mm on hole-free 3x3 support -- so its far field stays
inside a 24 mm join tolerance.  It is a latent range-dependent trap that fires
the moment the depth gets noisier, and it is the wrong unit either way.)

In disparity D = fx*B/z the picture is clean and range-free:
  * matching noise is ~constant in px at every range;
  * a PLANE is exactly affine in (u, v) at any slant, so a plane's disparity
    spread over a window of radius R is exactly R/r times its spread over
    radius r -- an oblique floor and a fronto-parallel wall look identical;
  * a real depth discontinuity is a step, which breaks that ratio.
So every threshold below is in disparity px, and the same numbers hold in the
0.3-0.6 m grasp band as at 2.5 m (see remove_edge_bleed for the caveat).

STAGES (all vectorised, on the uint16 mm depth map, 0 == invalid)
  0. clamp_range        -- near/far clamp.  Consumers are documented against a
     0.28 m near-clipped stream (z_manip/adapters DEPTH_NEAR_CLIP_M,
     edgetam.yaml min_depth_m); FFS was emitting ~0.6% of the image at
     0.08-0.15 m (the robot's own body, below the D435's optical minimum and
     below anything the raw stream can produce).  Runs first so the bogus near
     pixels cannot seed the later neighbourhood statistics.

  1. remove_edge_bleed  -- wide-support silhouette gate.  FFS reports depth
     systematically FARTHER near depth discontinuities, and the excess ramps
     over 7-12 px (measured bias by distance to the nearest raw depth edge:
     0-1 px +10 mm / 1-2 +28 / 2-4 +25 / 4-7 +15 / 7-12 +4 / 12-25 -4 /
     >25 -6 mm, with a p90 tail of +180..+210 mm inside 4 px).  A 3x3 gradient
     gate cannot see a ramp that gentle -- 200 mm spread over 7 px is under
     30 mm/px, half of what the old 120 mm/3x3 gate needed -- so this stage
     uses a radius-7 window (the measured ramp length) and a two-scale,
     plane-invariant test.  See the function docstring.

  2. remove_speckles    -- cv2.filterSpeckles on the DISPARITY map (see above).
     Kills the isolated free-space specks that would otherwise fill the grasp
     finger-sweep corridor and veto good approaches, without the range-
     dependent mass deletion of the mm formulation.

  3. smooth_banding     -- HOLE-AWARE small median blur that flattens the
     sub-pixel disparity-quantisation ripple on low-texture surfaces.
     cv2.medianBlur treats 0 as a value, so with k zeros in a 5x5 window it
     returns the (13-k)-th smallest NON-zero sample -- the window minimum at
     k=12 -- and the old ``med > 0`` re-mask only caught k>=13.  On live frames
     that biased 16292 px/frame NEAR by -2.8 mm mean, -36 mm at p1 and -2496 mm
     worst case: a near-side flying pixel manufactured by the smoother itself.
     The stage now only substitutes the median where the window is hole-free,
     falling back to a hole-free 3x3 window, so the result is exactly unbiased.

Optional stage 4 (OFF by default): temporal_ema, a RealSense-style per-pixel
exponential moving average gated by a per-pixel change threshold so it does not
ghost while the base/arm move.  Only blends pixels that are valid in both frames
and changed by less than the threshold; everything else takes the current frame.

Measured on live 640x480 frames (host cv2 4.6.0, same build as the relay
container): clamp 0.3 + bleed 1.6 + speckle 1.7 + median 1.3 + the per-stage
valid-pixel counts = 3.9-5.0 ms per frame, at the <=5 ms/frame CPU budget
(the old chain measured 3.0 ms on the same host and frames).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# int16 ceiling for cv2.filterSpeckles (it wants CV_16SC1).
_INT16_MAX = 32767
_KERNEL3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_KERNEL_CACHE: dict = {}

# fx_ir * baseline_m * 1000, in mm*px, for the wrist D435i in
# configs/ffs_d435_calibration.json (387.3583679 px * 0.05001901 m).  Disparity
# in px is DISPARITY_CONST_MM_PX / depth_mm.  The relay overrides this from the
# calibration it already loads, so a camera swap cannot silently mistune the
# filter; this default only matters for bare-host tests.
DISPARITY_CONST_MM_PX = 19375.3

# Sub-px fixed-point scale used to push disparity through cv2.filterSpeckles
# (CV_16SC1).  At the 0.10 m floor below, disparity tops out at 193.8 px ->
# 24 803 < 32767, so the fixed-point map can never overflow.
_DISP_FIXED_SCALE = 128.0
_DISP_FIXED_MIN_MM = 100.0


def _kernel(size: int) -> np.ndarray:
    k = _KERNEL_CACHE.get(size)
    if k is None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        _KERNEL_CACHE[size] = k
    return k


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class FilterConfig:
    """Filter switches + parameters.  Built from env vars in the relay."""

    enabled: bool = True
    # geometry: disparity px = disparity_const_mm_px / depth_mm.  Set by the
    # relay from the live calibration (fx_ir * baseline_m * 1000).
    disparity_const_mm_px: float = DISPARITY_CONST_MM_PX
    # stage 0: range clamp (mm).  Default matches the 0.28 m near clip every
    # consumer is documented against.
    clamp: bool = True
    min_depth_mm: int = 280
    max_depth_mm: int = 10000
    # stage 1: silhouette / edge-bleed gate (disparity px)
    edge: bool = True
    edge_radius: int = 7
    edge_near_radius: int = 2
    edge_margin_px: float = 1.0
    # legacy mm gradient gate, kept for A/B on hardware: FFS_FILTER_EDGE_MODE=grad
    edge_mode: str = 'bleed'
    max_grad_mm: float = 120.0
    # stage 2: free-space speckle removal (disparity px)
    speckle: bool = True
    speckle_max_size: int = 50
    speckle_max_diff_px: float = 0.5
    # stage 3: banding / ripple median
    median: bool = True
    median_ksize: int = 5
    # stage 4 (opt-in): temporal EMA, gated by per-pixel motion
    temporal: bool = False
    temporal_alpha: float = 0.5
    temporal_change_mm: int = 40

    @classmethod
    def from_env(cls, env=None) -> 'FilterConfig':
        env = os.environ if env is None else env
        return cls(
            enabled=_as_bool(env.get('FFS_FILTER'), True),
            disparity_const_mm_px=float(env.get(
                'FFS_FILTER_DISPARITY_CONST', str(DISPARITY_CONST_MM_PX))),
            clamp=_as_bool(env.get('FFS_FILTER_CLAMP'), True),
            min_depth_mm=int(env.get('FFS_FILTER_MIN_DEPTH_MM', '280')),
            max_depth_mm=int(env.get('FFS_FILTER_MAX_DEPTH_MM', '10000')),
            edge=_as_bool(env.get('FFS_FILTER_EDGE'), True),
            edge_mode=str(env.get('FFS_FILTER_EDGE_MODE', 'bleed')).strip().lower(),
            edge_radius=int(env.get('FFS_FILTER_EDGE_RADIUS', '7')),
            edge_near_radius=int(env.get('FFS_FILTER_EDGE_NEAR_RADIUS', '2')),
            edge_margin_px=float(env.get('FFS_FILTER_EDGE_MARGIN_PX', '1.0')),
            max_grad_mm=float(env.get('FFS_FILTER_EDGE_MAX_GRAD_MM', '120')),
            speckle=_as_bool(env.get('FFS_FILTER_SPECKLE'), True),
            speckle_max_size=int(env.get('FFS_FILTER_SPECKLE_MAX_SIZE', '50')),
            speckle_max_diff_px=float(
                env.get('FFS_FILTER_SPECKLE_MAX_DIFF_PX', '0.5')),
            median=_as_bool(env.get('FFS_FILTER_MEDIAN'), True),
            median_ksize=int(env.get('FFS_FILTER_MEDIAN_KSIZE', '5')),
            temporal=_as_bool(env.get('FFS_FILTER_TEMPORAL'), False),
            temporal_alpha=float(env.get('FFS_FILTER_TEMPORAL_ALPHA', '0.5')),
            temporal_change_mm=int(env.get('FFS_FILTER_TEMPORAL_CHANGE_MM', '40')),
        )

    def active_stages(self) -> list:
        if not self.enabled:
            return []
        stages = []
        if self.clamp:
            stages.append('clamp')
        if self.edge:
            stages.append('edge')
        if self.speckle:
            stages.append('speckle')
        if self.median:
            stages.append('median')
        if self.temporal:
            stages.append('temporal')
        return stages


def _validate_depth(depth: np.ndarray) -> np.ndarray:
    if depth.dtype != np.uint16:
        raise TypeError(f'depth must be uint16 mm, got {depth.dtype}')
    if depth.ndim != 2:
        raise ValueError(f'depth must be HxW, got shape {depth.shape}')
    return depth


def to_disparity(depth: np.ndarray,
                 disparity_const_mm_px: float = DISPARITY_CONST_MM_PX
                 ) -> np.ndarray:
    """float32 disparity px (0 where invalid).  ``D = fx*B*1000 / depth_mm``."""
    out = depth.astype(np.float32)
    np.maximum(out, np.float32(1.0), out=out)
    np.divide(np.float32(disparity_const_mm_px), out, out=out)
    out[depth == 0] = np.float32(0.0)
    return out


def clamp_range(depth: np.ndarray, min_depth_mm: int = 280,
                max_depth_mm: int = 10000) -> np.ndarray:
    """Invalidate pixels outside the range consumers are documented against.

    z_manip/adapters/__init__.py pins DEPTH_NEAR_CLIP_M = 0.28 and
    ros2/z_manip_edgetam/config/edgetam.yaml pins min_depth_m: 0.28 for the
    stream this filter feeds; the raw D435 aligned stream never emits below
    ~0.24 m.  FFS, having no hardware near limit, emitted 0.63% of the image at
    0.08-0.15 m on the live scene (concentrated on the robot's own body in the
    bottom-right) where the raw stream emits nothing at all -- points that are
    physically impossible for this camera and land inside the gripper.
    """
    _validate_depth(depth)
    out = depth.copy()
    out[(depth > 0) & ((depth < min_depth_mm) | (depth > max_depth_mm))] = 0
    return out


def remove_edge_bleed(depth: np.ndarray,
                      disparity_const_mm_px: float = DISPARITY_CONST_MM_PX,
                      radius: int = 7, near_radius: int = 2,
                      margin_px: float = 1.0) -> np.ndarray:
    """Invalidate pixels that sit behind a nearby foreground with no support.

    Two-scale, plane-invariant test in disparity space.  With
    ``step_R(p) = max_{q in R-square} D(q) - D(p)`` (how much nearer the closest
    surface within R px is, in disparity px):

        drop  <=>  step_R  >  (R / r) * step_r  +  margin_px

    WHY THIS SHAPE, FROM THE MEASURED PROFILE
      * ``R = 7``: the bleed is a multi-pixel RAMP.  The measured median bias
        by distance to the nearest raw depth edge decays +28 mm (1-2 px) ->
        +15 (4-7) -> +4 (7-12) -> -4 (12-25), so a radius-7 window is the
        smallest one that reaches clean foreground from anywhere inside the
        ramp.  The old 3x3 window saw at most 2/7 of the ramp, which is why a
        120 mm/3x3 gate dropped 6.2% of the frame and left the bias intact.
      * ``(R / r) * step_r``: for ANY plane, at any slant, the max-disparity
        excess over a square window scales exactly linearly with the window
        radius, so this term cancels the surface's own slope and leaves zero
        response.  A real discontinuity inside R but outside r does not cancel.
        This is what a mm-domain slope test cannot do: an 80-degree floor and a
        silhouette ramp both run at ~30 mm/px, and no mm threshold separates
        them.
      * ``r = 2``: smallest window that still averages the per-pixel matching
        noise; a radius-1 window is dominated by it.
      * ``margin_px = 1.0``: the disparity matching-noise floor.  Measured over
        24 live frames on pixels agreeing with the raw stream to <=30 mm, the
        15x15 disparity step is p90 0.47 px, p99 0.68 px -- so 1.0 px sits
        ~1.5x above the 99th percentile of clean geometry.

    RANGE VALIDITY.  Matching noise in disparity does not depend on range, so
    ``margin_px`` is the same number at 0.4 m as at 2.5 m, and the slope term
    is scale-free by construction.  That is the whole reason for working in
    disparity: the parameters are justified in the 0.6-2.5 m band where we have
    data AND carry to the 0.3-0.6 m grasp band without re-tuning.  What does
    NOT carry is the SELECTIVITY: at 0.4 m a 5 cm standoff is a 6.1 px
    disparity step versus 0.24 px at 2.0 m, so the gate is far more sensitive
    close in and will trim more of the grasp-band silhouette.  The captured
    scenes contained no pixels in [0.30, 0.60) m, so that false-positive rate
    is UNMEASURED; ``FFS_FILTER_EDGE_MARGIN_PX`` raises the floor if the grasp
    band turns out to over-trim.

    Note the gate is one-sided: a pixel with nothing nearer in reach has
    ``step_R = 0`` and is never dropped, so a foreground silhouette is never
    eroded and a smooth surface is untouched no matter how oblique.
    """
    _validate_depth(depth)
    if radius < 1 or near_radius < 1 or near_radius >= radius:
        raise ValueError('need 1 <= near_radius < radius')
    disp = to_disparity(depth, disparity_const_mm_px)
    # invalid -> 0, which can never raise a max, so holes are neutral.
    near = cv2.dilate(disp, _kernel(2 * near_radius + 1))
    wide = cv2.dilate(disp, _kernel(2 * radius + 1))
    gain = np.float32(float(radius) / float(near_radius))
    near -= disp                    # step_r
    near *= gain
    near += np.float32(margin_px)
    # depth-quantisation allowance.  The map is uint16 MILLIMETRES, so one
    # quantum is q = C/z^2 disparity px -- 0.003 px at 2.5 m but 0.215 px at
    # 0.30 m, where it exceeds a grazing plane's own disparity gradient and
    # terraces the surface.  Without this the two-scale ratio stops being exact
    # exactly in the grasp band; with it, planes survive intact at every range
    # and slant the camera can resolve.  Worst-case error is (1 + gain)*q.
    quant = disp * disp * np.float32((1.0 + float(gain)) / disparity_const_mm_px)
    near += quant
    wide -= disp                    # step_R
    out = depth.copy()
    out[(depth > 0) & (wide > near)] = 0
    return out


def remove_flying_pixels(depth: np.ndarray, max_grad_mm: float = 120.0) -> np.ndarray:
    """LEGACY 3x3 mm gradient gate (``FFS_FILTER_EDGE_MODE=grad``).

    Kept only so the operator can A/B the new gate against the shipped
    behaviour without a rebuild.  It cannot see the multi-pixel silhouette ramp
    that ``remove_edge_bleed`` targets: a 200 mm excess spread over 7 px is
    under 30 mm/px, so its 3x3 spread stays well below 120 mm.
    """
    _validate_depth(depth)
    valid = depth > 0
    d = depth.astype(np.float32)
    # local min over valid neighbours (invalid -> large so it can't lower min)
    dmin = cv2.erode(np.where(valid, d, np.float32(1e6)), _KERNEL3)
    # local max over valid neighbours (invalid -> 0 so it can't raise max)
    dmax = cv2.dilate(np.where(valid, d, np.float32(0.0)), _KERNEL3)
    drop = valid & ((dmax - dmin) > max_grad_mm)
    out = depth.copy()
    out[drop] = 0
    return out


def remove_speckles(depth: np.ndarray, max_size: int = 50,
                    max_diff_mm: int = 24,
                    disparity_const_mm_px: 'float | None' = None,
                    max_diff_px: float = 0.5) -> np.ndarray:
    """Drop small inconsistent connected components (isolated free-space specks).

    With ``disparity_const_mm_px`` set (the relay always sets it) the connected-
    component test runs on the DISPARITY map in 1/128 px fixed point and
    ``max_diff_px`` is the join tolerance; the mm path is the legacy fallback.

    The unit matters more than the threshold.  Depth noise grows as z^2, so a
    fixed mm tolerance fragments the far field into sub-``max_size`` components
    and deletes it wholesale.  Measured over 24 live frames on the raw D435
    aligned stream, ``max_diff_mm=24`` removed 52.2% of pixels beyond 2.5 m and
    8.6% of the 1.8-2.5 m band (13.9% of the whole frame); the same stage at
    ``max_diff_px=0.5`` removed 0.2% and 0.2% (0.11% of the frame) while still
    clearing every isolated speck.
    """
    _validate_depth(depth)
    if disparity_const_mm_px:
        disp = to_disparity(depth, disparity_const_mm_px)
        # Guard the fixed-point range: disparity is monotone-decreasing in
        # depth, so this floor bounds the largest representable value.  Pixels
        # under it are dropped rather than saturated (saturating would fuse
        # them into one bogus component).  With the clamp stage on -- the
        # default, floor 280 mm -- nothing ever reaches this branch.
        disp = np.where(depth >= _DISP_FIXED_MIN_MM, disp, np.float32(0.0))
        work = np.clip(disp * _DISP_FIXED_SCALE, 0, _INT16_MAX).astype(np.int16)
        thresh = max(1, int(round(float(max_diff_px) * _DISP_FIXED_SCALE)))
    else:
        work = np.clip(depth, 0, _INT16_MAX).astype(np.int16)
        thresh = int(max_diff_mm)
    # filterSpeckles mutates `work` in place, setting removed pixels to newVal=0.
    cv2.filterSpeckles(work, 0, int(max_size), thresh)
    out = depth.copy()
    out[work == 0] = 0  # covers originally-invalid and newly-removed specks
    return out


def smooth_banding(depth: np.ndarray, ksize: int = 5) -> np.ndarray:
    """HOLE-AWARE median blur to flatten low-texture banding/ripple.

    cv2.medianBlur has no notion of an invalid pixel: it sorts the 0s with
    everything else, so a 5x5 window holding k zeros returns the (13-k)-th
    smallest VALID sample instead of the 13th -- the window minimum once k
    reaches 12 -- and the old ``med > 0`` re-mask only rejected k>=13.  That
    made the smoother a NEAR-biased flying-pixel generator exactly where the
    depth map is least trustworthy (around holes): measured over 24 live FFS
    frames it shifted 16292 px/frame by -2.83 mm mean, -36 mm at p1 and
    -2496 mm at worst.

    The fix is to substitute the median only where the window is hole-free (an
    exact, zero-bias rank), falling back to a hole-free 3x3 window so most of
    the smoothing near a hole is still recovered.  A valid pixel is never
    invalidated and an invalid pixel is never filled, so object extents are
    untouched.
    """
    _validate_depth(depth)
    k = 5 if int(ksize) >= 5 else 3  # clamp to the only uint16-supported sizes
    valid = (depth > 0).view(np.uint8)
    out = depth.copy()
    # eroding the validity mask by a kxk rect is 1 exactly where the whole
    # window is valid -- the hole-free windows, and much cheaper than counting.
    whole = cv2.erode(valid, _kernel(k)).astype(bool)
    np.copyto(out, cv2.medianBlur(depth, k), where=whole)
    if k > 3:
        whole3 = cv2.erode(valid, _KERNEL3).astype(bool)
        np.copyto(out, cv2.medianBlur(depth, 3), where=whole3 & ~whole)
    return out


def temporal_ema(prev: np.ndarray, curr: np.ndarray, alpha: float = 0.5,
                change_thresh_mm: int = 40) -> np.ndarray:
    """Per-pixel EMA gated by motion; static pixels blend, moved pixels reset.

    ``out = alpha*curr + (1-alpha)*prev`` only where both frames are valid and
    changed by <= ``change_thresh_mm``; everywhere else ``out = curr`` so moving
    geometry (base/arm motion) does not ghost.  OFF by default.
    """
    _validate_depth(curr)
    if prev is None or prev.shape != curr.shape:
        return curr.copy()
    p = prev.astype(np.int32)
    c = curr.astype(np.int32)
    both = (p > 0) & (c > 0)
    static = both & (np.abs(c - p) <= change_thresh_mm)
    out = c.copy()
    blend = (alpha * c + (1.0 - alpha) * p)
    out[static] = blend[static].astype(np.int32)
    return np.clip(out, 0, 65535).astype(np.uint16)


def filter_depth(depth: np.ndarray, cfg: FilterConfig,
                prev: 'np.ndarray | None' = None):
    """Apply the enabled stages in order; return ``(filtered, report)``.

    Order is clamp -> edge -> speckle -> median -> temporal.  The clamp runs
    first so impossible near pixels cannot seed any neighbourhood statistic;
    the edge carve runs before speckle so speckle mops up the fragments it
    leaves; the median runs last on the cleaned map.  ``report`` carries the
    per-stage drop counts (``dropped``) so the relay can log which stage is
    actually costing coverage -- the question that was unanswerable from a
    single aggregate counter.  When ``cfg.enabled`` is False the input array is
    returned untouched (zero-copy escape hatch).
    """
    _validate_depth(depth)
    if not cfg.enabled:
        return depth, {'stages': [], 'in_valid': None, 'out_valid': None,
                       'removed': 0, 'dropped': {}}

    in_valid = int(np.count_nonzero(depth))
    out = depth
    dropped = {}

    def _step(name, new):
        nonlocal out
        before = int(np.count_nonzero(out))
        out = new
        dropped[name] = before - int(np.count_nonzero(out))

    if cfg.clamp:
        _step('clamp', clamp_range(out, cfg.min_depth_mm, cfg.max_depth_mm))
    if cfg.edge:
        if cfg.edge_mode == 'grad':
            _step('edge', remove_flying_pixels(out, cfg.max_grad_mm))
        else:
            _step('edge', remove_edge_bleed(
                out, cfg.disparity_const_mm_px, cfg.edge_radius,
                cfg.edge_near_radius, cfg.edge_margin_px))
    if cfg.speckle:
        _step('speckle', remove_speckles(
            out, cfg.speckle_max_size,
            disparity_const_mm_px=cfg.disparity_const_mm_px,
            max_diff_px=cfg.speckle_max_diff_px))
    if cfg.median:
        _step('median', smooth_banding(out, cfg.median_ksize))
    if cfg.temporal:
        _step('temporal', temporal_ema(prev, out, cfg.temporal_alpha,
                                       cfg.temporal_change_mm))

    out_valid = int(np.count_nonzero(out))
    report = {
        'stages': cfg.active_stages(),
        'in_valid': in_valid,
        'out_valid': out_valid,
        'removed': in_valid - out_valid,
        'dropped': dropped,
    }
    return out, report
