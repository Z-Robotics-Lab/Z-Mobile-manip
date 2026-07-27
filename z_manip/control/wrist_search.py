"""Pure, transport-free policy for bounded wrist-camera target search.

This module deliberately cannot open ROS, CAN, SSH, or a robot SDK.  It only
turns measured joint feedback and detector observations into a finite sequence
of fixed camera viewpoints.  A runtime adapter must separately authorize and
execute each returned target.

AXIS CHOICE (2026-07-27).  The horizontal sweep is driven by PiPER **J1**, the
base yaw, and the vertical sweep by **J5**, the wrist elevation.  It used to be
driven by J4, a rotation about the forearm, and that was wrong: forward
kinematics on the deployed URDF (``go2w_sensored.urdf``) composed with the
measured hand-eye calibration shows J4's axis sits 20.8 deg off the camera's
own optical axis at the measured anchor, so it buys 1.112 deg of HORIZON ROLL
for every 0.660 deg of azimuth.  At the old +/-36 deg envelope that rolled the
picture +/-38.3 deg while panning only +/-21.6 deg - the operator saw the view
go oblique instead of shaking left and right.  J1's axis is [0, 0, 1] in
``piper_base_link``, which is exactly the vector the world-up direction is
invariant under, so rotating about it cannot change the angle between world-up
and any camera axis: FK gives 0.000 deg of horizon-roll change and 1.000 deg of
optical-axis azimuth per degree of J1, verified numerically over +/-90 deg.
``tests/test_wrist_search_camera_axes.py`` re-derives this from the URDF and
fails if the sweep axis is ever moved back to a joint that tilts the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


# Camera-space gain of the pan joint, measured by forward kinematics on the
# deployed URDF plus the measured hand-eye calibration at the search anchor
# (``configs/piper_home.json``, joints [0.203, 4.582, -8.233, -1.814, 15.084,
# 0.0] deg).  The camera looks 27.25 deg DOWN there, so as J1 turns the optical
# axis traces a cone rather than a great circle: 15 deg of J1 moves the optical
# axis 13.327 deg (gain 0.8885) and 30 deg of J1 moves it 26.605 deg (gain
# 0.8868).  J5 by contrast moves the optical axis 13.997 deg per 14 deg (gain
# 0.9998).  The raster walk below compares a pan step against a pitch step in
# this CAMERA space; comparing raw joint radians would silently treat one
# degree of base yaw as interchangeable with one degree of wrist elevation,
# which they are not.
PAN_VIEW_GAIN: float = 0.888
PITCH_VIEW_GAIN: float = 1.000


class WristSearchPhase(str, Enum):
    IDLE = "idle"
    MOVE = "move"
    SETTLE = "settle"
    OBSERVE = "observe"
    FOUND = "found"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WristSearchConfig:
    """Finite search envelope around a measured, known-clear anchor pose."""

    # PiPER J1: base yaw.  The whole kinematic chain turns rigidly about the
    # vertical base axis, so the camera PANS left and right with its horizon
    # level.  FK on the measured anchor: 1.000 deg of optical-axis azimuth and
    # 0.000 deg of horizon roll per degree.  Do NOT move this back to J4
    # (index 3) - that axis rotates about the forearm and tilts the picture.
    pan_joint_index: int = 0
    # PiPER J5: wrist elevation.  Rotates about a horizontal axis, so
    # look-up/look-down keeps the horizon level (|d tilt| <= 0.37 deg over the
    # whole +/-28 deg envelope).  The operator is explicitly happy with this
    # axis; leave it alone.
    pitch_joint_index: int = 4
    # Derived from the PUBLISHED colour intrinsics (fx 605.869, cx 331.779,
    # 640x480 -> 55.669 deg HFOV, cross-checked against
    # configs/ffs_d435_calibration.json and the recorded /camera/color/
    # camera_info) and the MEASURED median working distance of 0.401 m over
    # 249 planning sessions.  Projecting the neighbour view's optical-axis
    # point into this view - so both the rotation and the 1.72 mm/deg camera
    # translation are included - gives 71.2% frame overlap at 0.401 m, and
    # 69.8%/72.4% at the measured 0.33 m/0.50 m tails, i.e. ~30% genuinely new
    # scene per step.  25 deg was rejected: 49.7% overlap opens a blind gap
    # wider than the border-gated usable field.
    pan_step_rad: float = math.radians(15.0)
    pitch_step_rad: float = math.radians(14.0)
    # Two pan levels each side.  Camera azimuth reaches +/-26.6 deg (the 0.888
    # cone gain above), so the search covers 55.669 + 2*26.605 = 108.9 deg of
    # horizontal field, 1.96x what the anchor view alone sees, while keeping
    # the grid at 5 pan x 5 pitch = 25 views.  Whole-arm proof for this
    # envelope: joint-limit margins [119.8, 4.58, 8.23, 98.19, 26.92, 120.0]
    # deg (the J2/J3 margins are properties of the anchor and are not touched
    # by the search, which only writes indices 0 and 4); pinocchio+hpp-fcl mesh
    # self-collision over the full rectangle reports 0 collisions in 1681
    # states, and 0 out to a +/-60 deg stress envelope.  J1 is exactly
    # collision-NEUTRAL for every arm-arm capsule pair (it rotates them
    # rigidly about their own base axis) and monotonically INCREASES clearance
    # to the Go2-W trunk: worst body-capsule margin improves from +0.0194 m
    # under the old J4 envelope to +0.0232 m under this one.
    max_pan_offset_rad: float = math.radians(30.0)
    max_pitch_offset_rad: float = math.radians(28.0)
    settle_s: float = 0.35
    detector_hz: float = 5.0
    observations_per_view: int = 3
    # The resident grounding service already rejects candidates below 0.15
    # and applies finite/border/area gates. Search adds temporal confirmation,
    # not a contradictory second confidence gate. 0.15 keeps the measured
    # white-charger operating point (0.19 on a plainly visible target) inside
    # the gate while staying far above the 0.05 that the offline seed
    # benchmark rejected for false positives.
    confidence_threshold: float = 0.15
    confirmations_required: int = 2
    # Confirmation counts qualified frames within a view (2-of-N), not a
    # consecutive streak: a frame the detector cannot score (a 422, a border or
    # blur reject) ABSTAINS and leaves the count intact, so a single transient
    # miss no longer vetoes a plainly visible target. Only a confident negative
    # (a real detection scoring below the threshold) resets the count.
    reset_on_confident_negative: bool = True
    # The sweep is only entered after the stationary flow already checked the
    # current (anchor) view with the full detector; re-observing view 0 wastes a
    # whole remote fixed-view command (~5.9 s, see max_search_s below; it was
    # ~12 s before the passive-CAN gate collapsed, and skipping still earns its
    # keep at the lower cost). The runtime enables this to begin the
    # raster at the first genuinely new viewpoint. The anchor remains view 0 in
    # the grid so the executor can still restore it after a failed search.
    skip_anchor_view: bool = False
    joint_tolerance_rad: float = math.radians(1.0)
    # Re-derived 2026-07-27 after the passive-CAN gate around each bounded move
    # collapsed from ~9.1 s to ~1.57 s.  The superseded comment here claimed
    # ~12 s per view and produced the old 360 s budget; that measurement named
    # the passive-feedback service stop/start as its dominant term, which is
    # exactly the term that collapsed.  Bottom-up per view now: one remote
    # fixed-view command ~5.9 s end to end (ssh, with the control-master and
    # manifest-sha upload skip in piper_wrist_search_remote.sh amortising
    # staging across views 2..N; the measured bounded move itself is 1.755 s
    # per the recorded piper-wrist-search.log) + settle_s 0.35 s +
    # observations_per_view 3 at detector_hz 5.0 = 0.6 s => ~6.8 s.  25 views x
    # 6.8 s = 170.5 s; x1.25 for detector latency and ssh jitter => 210 s.
    # ESTIMATE: the per-view wall time is NOT recorded in any artifact under
    # artifacts/go2w_real, so this takes the conservative arm (5.87 s/view from
    # 13.4 - 9.1 + 1.57) rather than the optimistic 4.8 s/view (which would
    # give 180 s).  This is a HARD deadline that raises TimeoutError and aborts
    # the search, so it must not be cut to the bone; re-time one live view
    # before lowering it further.  It still forbids an unbounded scan.
    max_search_s: float = 210.0

    def __post_init__(self) -> None:
        if self.pan_joint_index == self.pitch_joint_index:
            raise ValueError("wrist search joints must be distinct")
        if min(self.pan_joint_index, self.pitch_joint_index) < 0:
            raise ValueError("wrist search joint indices must be non-negative")
        positive = (
            self.pan_step_rad,
            self.pitch_step_rad,
            self.max_pan_offset_rad,
            self.max_pitch_offset_rad,
            self.settle_s,
            self.detector_hz,
            self.joint_tolerance_rad,
            self.max_search_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("wrist search limits must be finite and positive")
        if self.pan_step_rad > self.max_pan_offset_rad:
            raise ValueError("pan step exceeds the search envelope")
        if self.pitch_step_rad > self.max_pitch_offset_rad:
            raise ValueError("pitch step exceeds the search envelope")
        if self.observations_per_view < 1 or self.confirmations_required < 1:
            raise ValueError("wrist search observation counts must be positive")
        if self.confirmations_required > self.observations_per_view:
            raise ValueError("confirmations cannot exceed observations per view")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in (0, 1]")

    @property
    def observation_period_s(self) -> float:
        return 1.0 / self.detector_hz


@dataclass(frozen=True)
class WristView:
    index: int
    #: Base-yaw (J1) offset from the anchor.  Positive pans the camera left in
    #: ``piper_base_link``; the horizon stays level.
    pan_offset_rad: float
    #: Wrist-elevation (J5) offset from the anchor.
    pitch_offset_rad: float


@dataclass(frozen=True)
class WristSearchDecision:
    phase: WristSearchPhase
    view: WristView | None
    target_joints_rad: tuple[float, ...] | None
    confidence: float | None = None
    confirmations: int = 0
    message: str = ""


class BoundedWristSearch:
    """Deterministic raster search with multi-frame detector confirmation."""

    def __init__(self, config: WristSearchConfig | None = None) -> None:
        self.config = config or WristSearchConfig()
        self._views = self._make_views()
        self.reset()

    @staticmethod
    def _view_space(pan_rad: float, pitch_rad: float) -> tuple[float, float]:
        """Map a (pan, pitch) joint offset into approximate camera angles.

        The two sweep joints are not interchangeable: a degree of base yaw
        moves the optical axis ``PAN_VIEW_GAIN`` degrees because the camera is
        pitched down, while a degree of wrist elevation moves it
        ``PITCH_VIEW_GAIN`` degrees.  Ordering the raster in raw joint radians
        therefore compared two different things and, with the derived steps,
        buried the first pan behind two pure-pitch views - the exact motion the
        operator asked for would not have appeared in a short search.
        """

        return pan_rad * PAN_VIEW_GAIN, pitch_rad * PITCH_VIEW_GAIN

    def _make_views(self) -> tuple[WristView, ...]:
        # Near-center views are intentionally visited first.  A deterministic
        # nearest-neighbour walk reduces large cross-sweeps; the executor still
        # interpolates every selected edge into small measured increments.  The
        # walk runs in camera-view space (see ``_view_space``) so a pan step is
        # compared against a pitch step by how far each actually moves the
        # picture, not by raw joint radians.
        pan_levels = [0.0]
        radius = self.config.pan_step_rad
        while radius <= self.config.max_pan_offset_rad + 1e-9:
            pan_levels.extend((radius, -radius))
            radius += self.config.pan_step_rad
        pitch_levels = [0.0]
        radius = self.config.pitch_step_rad
        while radius <= self.config.max_pitch_offset_rad + 1e-9:
            pitch_levels.extend((radius, -radius))
            radius += self.config.pitch_step_rad
        remaining = {
            (pan, pitch)
            for pitch in pitch_levels
            for pan in pan_levels
            if not (pan == 0.0 and pitch == 0.0)
        }
        space = {offset: self._view_space(*offset) for offset in remaining}
        space[(0.0, 0.0)] = (0.0, 0.0)
        offsets: list[tuple[float, float]] = [(0.0, 0.0)]
        while remaining:
            current_pan, current_pitch = space[offsets[-1]]
            next_offset = min(
                remaining,
                key=lambda value: (
                    math.hypot(
                        space[value][0] - current_pan,
                        space[value][1] - current_pitch,
                    ),
                    math.hypot(*space[value]),
                    abs(space[value][1]),
                    abs(space[value][0]),
                    space[value][1],
                    space[value][0],
                ),
            )
            remaining.remove(next_offset)
            offsets.append(next_offset)
        return tuple(
            WristView(index=index, pan_offset_rad=pan, pitch_offset_rad=pitch)
            for index, (pan, pitch) in enumerate(offsets)
        )

    @property
    def views(self) -> tuple[WristView, ...]:
        return self._views

    def reset(self) -> None:
        self.phase = WristSearchPhase.IDLE
        self.anchor_joints_rad: np.ndarray | None = None
        self.started_at_s: float | None = None
        self.view_started_at_s: float | None = None
        self.last_observation_at_s: float | None = None
        self.view_index = 0
        self.observation_count = 0
        self.confirmation_count = 0
        self.last_confidence: float | None = None

    def start(self, joints_rad: Sequence[float], *, now_s: float) -> WristSearchDecision:
        joints = np.asarray(joints_rad, dtype=float)
        required = max(self.config.pan_joint_index, self.config.pitch_joint_index) + 1
        if joints.ndim != 1 or len(joints) < required or not np.isfinite(joints).all():
            raise ValueError("wrist search anchor must contain finite measured joints")
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("wrist search timestamp must be finite")
        self.reset()
        self.anchor_joints_rad = joints.copy()
        self.started_at_s = now
        self.view_started_at_s = now
        if self.config.skip_anchor_view and len(self._views) > 1:
            # The stationary flow already checked the anchor view; start the
            # raster at the first genuinely new viewpoint.
            self.view_index = 1
        self.phase = WristSearchPhase.MOVE
        return self._decision(message="move to the first bounded search view")

    def stop(self) -> WristSearchDecision:
        self.phase = WristSearchPhase.STOPPED
        return self._decision(message="wrist search stopped")

    def _target(self) -> np.ndarray:
        if self.anchor_joints_rad is None:
            raise RuntimeError("wrist search has no measured anchor")
        view = self._views[self.view_index]
        target = self.anchor_joints_rad.copy()
        target[self.config.pan_joint_index] += view.pan_offset_rad
        target[self.config.pitch_joint_index] += view.pitch_offset_rad
        return target

    def _decision(
        self,
        *,
        confidence: float | None = None,
        message: str = "",
    ) -> WristSearchDecision:
        view = None
        target = None
        if self.anchor_joints_rad is not None and self.view_index < len(self._views):
            view = self._views[self.view_index]
            target = tuple(float(value) for value in self._target())
        return WristSearchDecision(
            phase=self.phase,
            view=view,
            target_joints_rad=target,
            confidence=confidence,
            confirmations=self.confirmation_count,
            message=message,
        )

    def update_motion(self, joints_rad: Sequence[float], *, now_s: float) -> WristSearchDecision:
        if self.phase not in {WristSearchPhase.MOVE, WristSearchPhase.SETTLE}:
            raise RuntimeError("wrist search is not waiting for motion")
        now = self._validate_time(now_s)
        joints = np.asarray(joints_rad, dtype=float)
        target = self._target()
        if joints.shape != target.shape or not np.isfinite(joints).all():
            raise ValueError("wrist search feedback does not match the anchor")
        if float(np.max(np.abs(joints - target))) > self.config.joint_tolerance_rad:
            self.phase = WristSearchPhase.MOVE
            return self._decision(message="waiting for measured wrist viewpoint")
        if self.phase is WristSearchPhase.MOVE:
            self.phase = WristSearchPhase.SETTLE
            self.view_started_at_s = now
        if now - float(self.view_started_at_s) < self.config.settle_s:
            return self._decision(message="letting the wrist camera settle")
        self.phase = WristSearchPhase.OBSERVE
        self.observation_count = 0
        self.confirmation_count = 0
        self.last_observation_at_s = None
        return self._decision(message="sampling detector confidence")

    def observe(
        self,
        *,
        visible: bool,
        confidence: float | None,
        now_s: float,
    ) -> WristSearchDecision:
        if self.phase is not WristSearchPhase.OBSERVE:
            raise RuntimeError("wrist search is not accepting observations")
        now = self._validate_time(now_s)
        if (
            self.last_observation_at_s is not None
            and now - self.last_observation_at_s + 1e-9
            < self.config.observation_period_s
        ):
            return self._decision(
                confidence=self.last_confidence,
                message="waiting for the next detector cadence",
            )
        value = None if confidence is None else float(confidence)
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("detector confidence must be finite and within [0, 1]")
        self.last_observation_at_s = now
        self.last_confidence = value
        self.observation_count += 1
        qualified = visible and value is not None and value >= self.config.confidence_threshold
        # A confident negative is a real detection that scored below the
        # threshold; a 422 / border / blur frame reports visible=False (or no
        # confidence) and ABSTAINS so it neither confirms nor vetoes.
        confident_negative = (
            visible and value is not None and value < self.config.confidence_threshold
        )
        if qualified:
            self.confirmation_count += 1
        elif confident_negative and self.config.reset_on_confident_negative:
            self.confirmation_count = 0
        if self.confirmation_count >= self.config.confirmations_required:
            self.phase = WristSearchPhase.FOUND
            return self._decision(confidence=value, message="target confirmed")
        if self.observation_count < self.config.observations_per_view:
            return self._decision(confidence=value, message="collecting detector confirmations")
        self.view_index += 1
        self.observation_count = 0
        self.confirmation_count = 0
        if self.view_index >= len(self._views):
            self.phase = WristSearchPhase.EXHAUSTED
            return WristSearchDecision(
                phase=self.phase,
                view=None,
                target_joints_rad=None,
                confidence=value,
                message="bounded wrist search exhausted",
            )
        self.phase = WristSearchPhase.MOVE
        self.view_started_at_s = now
        return self._decision(confidence=value, message="advance to the next search view")

    def _validate_time(self, now_s: float) -> float:
        now = float(now_s)
        if self.started_at_s is None or not math.isfinite(now) or now < self.started_at_s:
            raise ValueError("wrist search time must be finite and monotonic")
        if now - self.started_at_s > self.config.max_search_s:
            self.phase = WristSearchPhase.EXHAUSTED
            raise TimeoutError("bounded wrist search exceeded its hard deadline")
        return now
