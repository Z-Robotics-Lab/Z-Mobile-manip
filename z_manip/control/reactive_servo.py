"""Transport-free 3-D reactive target keeping for mobile manipulation.

The policy in this module deliberately returns *intents*.  It cannot import
ROS, CAN, WebRTC, or a robot SDK and therefore cannot move hardware.  Runtime
adapters remain responsible for command ownership, joint/body limits, measured
settling, and converting the intents into platform-specific commands.

Frame conventions are explicit:

* camera optical: ``x`` right, ``y`` down, ``z`` forward;
* mobile base: ``x`` forward, ``y`` left, ``z`` up;
* arm base: whatever Cartesian convention is encoded by ``T_arm_camera``.

Base translation is controlled with Euclidean distance on the ground plane.
Camera posture and manipulation handoff use full 3-D range plus target height.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _point3(value: Sequence[float], *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must contain exactly three finite values")
    return point


def transform_point(
    target_from_source: Sequence[Sequence[float]],
    point_source: Sequence[float],
) -> tuple[float, float, float]:
    """Transform one 3-D point with an explicit 4x4 rigid transform."""

    transform = np.asarray(target_from_source, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("point transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("point transform must have a homogeneous final row")
    point = _point3(point_source, label="source point")
    result = transform @ np.append(point, 1.0)
    if not np.isfinite(result).all() or abs(float(result[3])) < 1e-12:
        raise ValueError("point transform produced an invalid homogeneous point")
    xyz = result[:3] / result[3]
    return tuple(float(value) for value in xyz)


@dataclass(frozen=True)
class TargetGeometry:
    """One synchronized target expressed in camera, base, and arm frames."""

    camera_xyz_m: tuple[float, float, float]
    base_xyz_m: tuple[float, float, float]
    arm_xyz_m: tuple[float, float, float]
    camera_range_m: float
    camera_elevation_rad: float
    base_planar_distance_m: float
    base_range_m: float
    base_bearing_rad: float
    target_height_m: float
    arm_range_m: float

    @classmethod
    def from_frames(
        cls,
        camera_xyz_m: Sequence[float],
        *,
        base_xyz_m: Sequence[float],
        arm_xyz_m: Sequence[float],
    ) -> "TargetGeometry":
        """Build synchronized metrics from already transformed points."""

        camera = _point3(camera_xyz_m, label="camera target")
        base = _point3(base_xyz_m, label="base target")
        arm = _point3(arm_xyz_m, label="arm target")
        if camera[2] <= 0.0:
            raise ValueError("camera target must be in front of the optical frame")
        planar = float(np.hypot(base[0], base[1]))
        return cls(
            camera_xyz_m=tuple(float(value) for value in camera),
            base_xyz_m=tuple(float(value) for value in base),
            arm_xyz_m=tuple(float(value) for value in arm),
            camera_range_m=float(np.linalg.norm(camera)),
            camera_elevation_rad=math.atan2(-float(camera[1]), float(camera[2])),
            base_planar_distance_m=planar,
            base_range_m=float(np.linalg.norm(base)),
            base_bearing_rad=math.atan2(float(base[1]), float(base[0])),
            target_height_m=float(base[2]),
            arm_range_m=float(np.linalg.norm(arm)),
        )

    @classmethod
    def from_camera(
        cls,
        camera_xyz_m: Sequence[float],
        *,
        T_base_camera: Sequence[Sequence[float]],
        T_arm_camera: Sequence[Sequence[float]],
    ) -> "TargetGeometry":
        camera = _point3(camera_xyz_m, label="camera target")
        if camera[2] <= 0.0:
            raise ValueError("camera target must be in front of the optical frame")
        return cls.from_frames(
            camera,
            base_xyz_m=transform_point(T_base_camera, camera),
            arm_xyz_m=transform_point(T_arm_camera, camera),
        )


class ReactivePhase(str, Enum):
    WAITING_TARGET = "waiting_target"
    TRANSFORM_UNAVAILABLE = "transform_unavailable"
    BASE_APPROACH = "base_approach"
    POSTURE_ADJUST = "posture_adjust"
    REACQUIRE = "reacquire"
    TRACKING_HOLD = "tracking_hold"
    VIEW_RECOVERY = "view_recovery"
    SEARCH_REQUIRED = "search_required"
    HANDOFF_PROBE = "handoff_probe"
    HANDOFF_SETTLE = "handoff_settle"
    HANDOFF_READY = "handoff_ready"


class ArmViewMode(str, Enum):
    HOLD = "hold"
    TRACK = "track"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    SEARCH = "search"


@dataclass(frozen=True)
class BaseMotionIntent:
    linear_x_mps: float = 0.0
    angular_z_rps: float = 0.0


@dataclass(frozen=True)
class PostureIntent:
    """Relative body pose request; a runtime must clamp against live limits."""

    body_height_delta_m: float = 0.0
    pitch_delta_rad: float = 0.0


@dataclass(frozen=True)
class ArmViewIntent:
    """Semantic camera-keeping request for a measured arm controller."""

    mode: ArmViewMode = ArmViewMode.HOLD
    yaw_error_rad: float = 0.0
    pitch_error_rad: float = 0.0
    extension_fraction: float = 0.0


@dataclass(frozen=True)
class ReactiveServoDecision:
    phase: ReactivePhase
    base: BaseMotionIntent
    posture: PostureIntent
    arm_view: ArmViewIntent
    geometry: TargetGeometry | None
    handoff_ready: bool = False
    needs_ik_probe: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ReactiveServoConfig:
    """Coarse geometry policy; exact body and arm limits stay in adapters."""

    desired_planar_standoff_m: float = 0.52
    posture_entry_planar_m: float = 0.85
    handoff_planar_max_m: float = 0.62
    handoff_lateral_tolerance_m: float = 0.07
    handoff_arm_min_range_m: float = 0.20
    handoff_arm_max_range_m: float = 0.80
    handoff_arm_min_height_m: float = -0.40
    handoff_arm_max_height_m: float = 0.65
    # The D435 is mounted on the wrist.  Do not keep extending the viewing
    # pose until the target enters the camera's unreliable near field.  The
    # soft limit starts close-range planning while depth is still usable; the
    # hard limit stops posture/view motion even if another corridor check has
    # not converged yet.
    camera_handoff_depth_m: float = 0.52
    # The wrist camera can be materially closer to the target than base_link.
    # Allow a bounded amount of base-planar slack for the soft camera handoff,
    # while retaining the arm range/height and lateral corridor checks below.
    camera_handoff_planar_slack_m: float = 0.15
    camera_hard_min_depth_m: float = 0.38
    preferred_target_height_m: float = -0.10
    target_height_deadband_m: float = 0.08
    desired_camera_elevation_rad: float = math.radians(-8.0)
    camera_elevation_soft_limit_rad: float = math.radians(16.0)
    camera_elevation_hard_limit_rad: float = math.radians(26.0)
    linear_gain: float = 0.65
    yaw_gain: float = 0.70
    max_forward_mps: float = 0.18
    max_yaw_rps: float = 0.12
    yaw_deadband_rad: float = math.radians(7.0)
    # Couple base approach speed to camera-elevation view risk.  Once the
    # elevation error exceeds the soft limit, the forward command is ramped
    # down linearly, reaching this fraction of its nominal value at the hard
    # limit (and held there beyond it).  A non-holonomic base cannot reduce a
    # high target's elevation error by advancing -- getting closer makes it
    # steeper -- so at full speed the target leaves the wrist-camera FOV before
    # the arm view can compensate.  This gate is independent of body posture,
    # which the Go2W SPORT service does not actuate.  1.0 disables the slowdown.
    elevation_approach_speed_floor: float = 0.25
    posture_height_gain: float = 0.55
    posture_pitch_gain: float = 0.70
    max_height_step_m: float = 0.10
    max_pitch_step_rad: float = math.radians(10.0)
    preferred_arm_range_m: float = 0.55
    max_arm_view_extension_fraction: float = 0.65
    tracking_loss_grace_s: float = 0.75
    # Zero preserves the controller's transport-agnostic legacy behaviour;
    # real stepping-base deployments opt into a short hold explicitly.
    tracking_hold_s: float = 0.0
    reacquire_stable_s: float = 0.25
    handoff_settle_s: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.desired_planar_standoff_m,
            self.posture_entry_planar_m,
            self.handoff_planar_max_m,
            self.handoff_lateral_tolerance_m,
            self.handoff_arm_min_range_m,
            self.handoff_arm_max_range_m,
            self.camera_handoff_depth_m,
            self.camera_handoff_planar_slack_m,
            self.camera_hard_min_depth_m,
            self.target_height_deadband_m,
            self.camera_elevation_soft_limit_rad,
            self.camera_elevation_hard_limit_rad,
            self.linear_gain,
            self.yaw_gain,
            self.max_forward_mps,
            self.max_yaw_rps,
            self.posture_height_gain,
            self.posture_pitch_gain,
            self.max_height_step_m,
            self.max_pitch_step_rad,
            self.preferred_arm_range_m,
            self.tracking_loss_grace_s,
            self.reacquire_stable_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("reactive-servo limits must be finite and positive")
        if self.posture_entry_planar_m <= self.handoff_planar_max_m:
            raise ValueError("posture entry must be outside the handoff corridor")
        if self.handoff_arm_min_range_m >= self.handoff_arm_max_range_m:
            raise ValueError("invalid arm range corridor")
        if self.handoff_arm_min_height_m >= self.handoff_arm_max_height_m:
            raise ValueError("invalid arm height corridor")
        if self.camera_hard_min_depth_m >= self.camera_handoff_depth_m:
            raise ValueError("camera hard depth limit must precede handoff depth")
        if self.camera_elevation_soft_limit_rad >= self.camera_elevation_hard_limit_rad:
            raise ValueError("camera soft elevation limit must precede hard limit")
        if not 0.0 <= self.max_arm_view_extension_fraction <= 1.0:
            raise ValueError("arm extension fraction must be within [0, 1]")
        if not 0.0 <= self.elevation_approach_speed_floor <= 1.0:
            raise ValueError("elevation approach speed floor must be within [0, 1]")
        if (
            not math.isfinite(self.tracking_hold_s)
            or self.tracking_hold_s < 0.0
            or self.tracking_hold_s >= self.tracking_loss_grace_s
        ):
            raise ValueError("tracking hold must be shorter than loss grace")
        if not math.isfinite(self.handoff_settle_s) or self.handoff_settle_s < 0.0:
            raise ValueError("handoff settle time must be finite and non-negative")


class ReactiveTargetController:
    """Coordinate base approach, view keeping, posture, and arm handoff.

    The controller never declares handoff from distance alone.  ``ik_feasible``
    must explicitly be true, so the downstream runtime can probe grasp and
    pregrasp IK before relinquishing base/posture control.
    """

    def __init__(self, config: ReactiveServoConfig | None = None) -> None:
        self.config = config or ReactiveServoConfig()
        self.reset()

    def reset(self) -> None:
        self._body_posture_actionable = True
        self.phase = ReactivePhase.WAITING_TARGET
        self._last_geometry: TargetGeometry | None = None
        self._last_seen_s: float | None = None
        self._posture_requested = False
        self._reacquire_since_s: float | None = None
        self._handoff_since_s: float | None = None

    def _arm_view(self, geometry: TargetGeometry, *, search: bool = False) -> ArmViewIntent:
        error = geometry.camera_elevation_rad - self.config.desired_camera_elevation_rad
        if search:
            mode = ArmViewMode.SEARCH
        elif error > self.config.camera_elevation_soft_limit_rad:
            mode = ArmViewMode.LOOK_UP
        elif error < -self.config.camera_elevation_soft_limit_rad:
            mode = ArmViewMode.LOOK_DOWN
        else:
            mode = ArmViewMode.TRACK
        extension = _clamp(
            (geometry.arm_range_m - self.config.preferred_arm_range_m) / 0.35,
            0.0,
            self.config.max_arm_view_extension_fraction,
        )
        return ArmViewIntent(
            mode=mode,
            yaw_error_rad=geometry.base_bearing_rad,
            pitch_error_rad=error,
            extension_fraction=extension,
        )

    def _posture(self, geometry: TargetGeometry) -> PostureIntent:
        height_error = geometry.target_height_m - self.config.preferred_target_height_m
        elevation_error = (
            geometry.camera_elevation_rad - self.config.desired_camera_elevation_rad
        )
        return PostureIntent(
            body_height_delta_m=_clamp(
                self.config.posture_height_gain * height_error,
                -self.config.max_height_step_m,
                self.config.max_height_step_m,
            ),
            pitch_delta_rad=_clamp(
                self.config.posture_pitch_gain * elevation_error,
                -self.config.max_pitch_step_rad,
                self.config.max_pitch_step_rad,
            ),
        )

    def _handoff_geometry_ok(
        self,
        geometry: TargetGeometry,
        *,
        desired_target_lateral_m: float,
    ) -> bool:
        arm_z = geometry.arm_xyz_m[2]
        return (
            geometry.base_planar_distance_m <= self.config.handoff_planar_max_m
            and abs(geometry.base_xyz_m[1] - desired_target_lateral_m)
            <= self.config.handoff_lateral_tolerance_m
            and self.config.handoff_arm_min_range_m
            <= geometry.arm_range_m
            <= self.config.handoff_arm_max_range_m
            and self.config.handoff_arm_min_height_m
            <= arm_z
            <= self.config.handoff_arm_max_height_m
            and abs(geometry.camera_elevation_rad)
            <= self.config.camera_elevation_hard_limit_rad
        )

    def _lost(self, *, now_s: float) -> ReactiveServoDecision:
        geometry = self._last_geometry
        if geometry is None or self._last_seen_s is None:
            self.phase = ReactivePhase.SEARCH_REQUIRED
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=ArmViewIntent(mode=ArmViewMode.SEARCH),
                geometry=None,
                reason="no 3-D target is available; start bounded search",
            )
        age_s = max(0.0, now_s - self._last_seen_s)
        if age_s <= self.config.tracking_hold_s:
            self.phase = ReactivePhase.TRACKING_HOLD
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=ArmViewIntent(mode=ArmViewMode.HOLD),
                geometry=geometry,
                reason=(
                    "brief tracker gap; freeze base/posture commands and retain "
                    "the last filtered viewing ray"
                ),
            )
        if age_s <= self.config.tracking_loss_grace_s:
            self._handoff_since_s = None
            self.phase = ReactivePhase.VIEW_RECOVERY
            recovery_posture = (
                self._posture(geometry)
                if self._body_posture_actionable
                else PostureIntent()
            )
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=recovery_posture,
                arm_view=self._arm_view(geometry, search=True),
                geometry=geometry,
                reason="hold the base and recover the last observed viewing ray",
            )
        self.phase = ReactivePhase.SEARCH_REQUIRED
        self._handoff_since_s = None
        return ReactiveServoDecision(
            phase=self.phase,
            base=BaseMotionIntent(),
            posture=PostureIntent(),
            arm_view=self._arm_view(geometry, search=True),
            geometry=geometry,
            reason="tracking-loss grace expired; start bounded wrist/body search",
        )

    def update(
        self,
        geometry: TargetGeometry | None,
        *,
        now_s: float,
        tracking: bool,
        body_settled: bool,
        ik_feasible: bool | None = None,
        desired_target_lateral_m: float = 0.0,
        body_posture_actionable: bool = True,
    ) -> ReactiveServoDecision:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("reactive-servo timestamp must be finite")
        desired_lateral = float(desired_target_lateral_m)
        if not math.isfinite(desired_lateral):
            raise ValueError("desired target lateral offset must be finite")
        if not tracking or geometry is None:
            self._reacquire_since_s = None
            # The loss stair must obey the same structural-capability verdict
            # as live tracking: VIEW_RECOVERY on an Euler-less platform must
            # not command body posture the service will only reject.
            self._body_posture_actionable = bool(body_posture_actionable)
            return self._lost(now_s=now)

        self._last_geometry = geometry
        self._last_seen_s = now
        arm_view = self._arm_view(geometry)
        camera_depth_m = geometry.camera_xyz_m[2]
        lateral_error_m = geometry.base_xyz_m[1] - desired_lateral
        lateral_aligned = (
            abs(lateral_error_m) <= self.config.handoff_lateral_tolerance_m
        )
        arm_corridor_ok = (
            self.config.handoff_arm_min_range_m
            <= geometry.arm_range_m
            <= self.config.handoff_arm_max_range_m
            and self.config.handoff_arm_min_height_m
            <= geometry.arm_xyz_m[2]
            <= self.config.handoff_arm_max_height_m
        )
        near_field_handoff = (
            camera_depth_m <= self.config.camera_handoff_depth_m
            and geometry.base_planar_distance_m
            <= (
                self.config.handoff_planar_max_m
                + self.config.camera_handoff_planar_slack_m
            )
            and lateral_aligned
            and arm_corridor_ok
        )
        hard_near_field = camera_depth_m <= self.config.camera_hard_min_depth_m

        handoff_geometry_ok = self._handoff_geometry_ok(
            geometry,
            desired_target_lateral_m=desired_lateral,
        )
        handoff_candidate = (
            near_field_handoff or hard_near_field or handoff_geometry_ok
        )
        if handoff_candidate:
            if self._handoff_since_s is None:
                self._handoff_since_s = now
            if now - self._handoff_since_s + 1e-9 < self.config.handoff_settle_s:
                self._posture_requested = False
                self._reacquire_since_s = None
                self.phase = ReactivePhase.HANDOFF_SETTLE
                return ReactiveServoDecision(
                    phase=self.phase,
                    base=BaseMotionIntent(),
                    posture=PostureIntent(),
                    arm_view=ArmViewIntent(mode=ArmViewMode.HOLD),
                    geometry=geometry,
                    reason=(
                        "handoff corridor reached; wait for stepping/image motion "
                        "to settle before the close-range capture"
                    ),
                )
        else:
            self._handoff_since_s = None

        # Handoff must precede another posture increment.  Once the wrist
        # camera reaches its near-field boundary, continuing to move the arm
        # destroys the depth evidence that the grasp planner needs.  HOLD is
        # intentional: the planning supervisor captures at the last reliable
        # pose and owns any subsequent pregrasp motion.
        if near_field_handoff or hard_near_field:
            self._posture_requested = False
            self._reacquire_since_s = None
            hold_view = ArmViewIntent(mode=ArmViewMode.HOLD)
            corridor_ok = handoff_geometry_ok
            if corridor_ok and ik_feasible is True:
                self.phase = ReactivePhase.HANDOFF_READY
                return ReactiveServoDecision(
                    phase=self.phase,
                    base=BaseMotionIntent(),
                    posture=PostureIntent(),
                    arm_view=hold_view,
                    geometry=geometry,
                    handoff_ready=True,
                    reason="D435 near-field handoff reached with an IK-feasible grasp",
                )
            self.phase = ReactivePhase.HANDOFF_PROBE
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=hold_view,
                geometry=geometry,
                needs_ik_probe=ik_feasible is None,
                reason=(
                    "D435 hard near-field floor reached; freeze the viewing pose "
                    "and start close-range planning"
                    if hard_near_field
                    else "D435 soft near-field boundary reached; start close-range "
                    "planning before depth is lost"
                ),
            )
        height_error = geometry.target_height_m - self.config.preferred_target_height_m
        elevation_error = (
            geometry.camera_elevation_rad - self.config.desired_camera_elevation_rad
        )
        view_at_risk = (
            abs(elevation_error) > self.config.camera_elevation_soft_limit_rad
            or abs(height_error) > self.config.target_height_deadband_m
        )
        in_posture_zone = (
            geometry.base_planar_distance_m <= self.config.posture_entry_planar_m
        )

        if not body_posture_actionable:
            # The body attitude actuator cannot move the wrist-camera view
            # (e.g. the Go2W ai-w service rejects Euler(1007) for the epoch).
            # Never latch a body-posture request the platform structurally
            # cannot satisfy: the wrist arm view keeps the target in frame
            # while the FSM proceeds toward the handoff corridor / IK probe,
            # instead of oscillating between POSTURE_ADJUST and REACQUIRE.
            self._posture_requested = False
            self._reacquire_since_s = None

        if self._posture_requested:
            if not body_settled:
                self.phase = ReactivePhase.POSTURE_ADJUST
                return ReactiveServoDecision(
                    phase=self.phase,
                    base=BaseMotionIntent(),
                    posture=self._posture(geometry),
                    arm_view=arm_view,
                    geometry=geometry,
                    reason="waiting for measured body/arm viewing pose to settle",
                )
            if self._reacquire_since_s is None:
                self._reacquire_since_s = now
            if now - self._reacquire_since_s + 1e-9 < self.config.reacquire_stable_s:
                self.phase = ReactivePhase.REACQUIRE
                return ReactiveServoDecision(
                    phase=self.phase,
                    base=BaseMotionIntent(),
                    posture=PostureIntent(),
                    arm_view=arm_view,
                    geometry=geometry,
                    reason="rebuilding a stable 3-D track after posture motion",
                )
            self._posture_requested = False
            self._reacquire_since_s = None

        if in_posture_zone and view_at_risk and body_posture_actionable:
            self._posture_requested = True
            self.phase = ReactivePhase.POSTURE_ADJUST
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=self._posture(geometry),
                arm_view=arm_view,
                geometry=geometry,
                reason="target height/elevation risks leaving the wrist-camera view",
            )

        if handoff_geometry_ok:
            if ik_feasible is True:
                self.phase = ReactivePhase.HANDOFF_READY
                return ReactiveServoDecision(
                    phase=self.phase,
                    base=BaseMotionIntent(),
                    posture=PostureIntent(),
                    arm_view=arm_view,
                    geometry=geometry,
                    handoff_ready=True,
                    reason="3-D arm corridor and explicit IK probe both passed",
                )
            self.phase = ReactivePhase.HANDOFF_PROBE
            return ReactiveServoDecision(
                phase=self.phase,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=arm_view,
                geometry=geometry,
                needs_ik_probe=ik_feasible is None,
                reason=(
                    "3-D corridor reached; waiting for downstream IK probe"
                    if ik_feasible is None
                    else "IK probe rejected this work pose; base remains stopped"
                ),
            )

        self.phase = ReactivePhase.BASE_APPROACH
        desired_forward_m = math.sqrt(max(
            self.config.desired_planar_standoff_m**2 - desired_lateral**2,
            0.05**2,
        ))
        distance_error = max(0.0, geometry.base_xyz_m[0] - desired_forward_m)
        linear = min(
            self.config.linear_gain * distance_error,
            self.config.max_forward_mps,
        )
        # Ramp the forward command down when the camera elevation error is at
        # risk.  ``elevation_error`` was computed above from the current
        # geometry.  Advancing on a high target only steepens the elevation, so
        # slowing here keeps the target inside the wrist-camera FOV long enough
        # for the arm view to compensate before a VIEW_RECOVERY/SEARCH loss.
        elevation_excess = (
            abs(elevation_error) - self.config.camera_elevation_soft_limit_rad
        )
        if elevation_excess > 0.0:
            span = (
                self.config.camera_elevation_hard_limit_rad
                - self.config.camera_elevation_soft_limit_rad
            )
            ramp = 1.0 - (elevation_excess / span) * (
                1.0 - self.config.elevation_approach_speed_floor
            )
            linear *= _clamp(ramp, self.config.elevation_approach_speed_floor, 1.0)
        # Steer toward a side work pose, not the platform centreline.  The
        # lateral error is expressed in the base frame; a non-holonomic base
        # removes it by following this shifted line of sight while advancing.
        bearing = math.atan2(
            lateral_error_m,
            max(geometry.base_xyz_m[0], 0.05),
        )
        angular = _clamp(
            self.config.yaw_gain * bearing,
            -self.config.max_yaw_rps,
            self.config.max_yaw_rps,
        )
        if abs(bearing) <= self.config.yaw_deadband_rad:
            angular = 0.0
        # Keep moving coarsely aligned instead of demanding camera-perfect
        # yaw from a stepping legged base.  A hard view-risk condition above
        # stops translation and requests camera/body tracking instead.
        return ReactiveServoDecision(
            phase=self.phase,
            base=BaseMotionIntent(linear_x_mps=linear, angular_z_rps=angular),
            posture=PostureIntent(),
            arm_view=arm_view,
            geometry=geometry,
            reason=(
                "inside the coarse arm corridor; waiting for an IK-feasible grasp"
                if self._handoff_geometry_ok(
                    geometry,
                    desired_target_lateral_m=desired_lateral,
                )
                else "approaching with ground-plane Euclidean distance"
            ),
        )
