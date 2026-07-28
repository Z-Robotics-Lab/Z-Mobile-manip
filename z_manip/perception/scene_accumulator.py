"""Multi-view accumulation of the arm's collision scene in the base frame.

A wrist-mounted D435 sees roughly 69 x 42 deg and is aimed at the grasp target,
so a single back-projected frame cannot contain the shelf lip beside the target,
the cabinet side wall, or the neighbouring clutter the elbow sweeps through --
let alone geometry the arm itself occludes once it enters the frustum.  Planning
against one frame therefore plans confidently through obstacles that were simply
never measured.  This module keeps the geometry earlier viewpoints did measure.

The hard constraint is that merging views must not manufacture obstacles.  On
this robot two views of the same physical surface, taken from wrist poses about
14 deg apart and registered through hand-eye plus forward kinematics, disagree
by roughly 18 mm at p90 (measured over 44 same-scene pairs in the recorded
corpus; the same-view point-spacing floor is 2 mm, so nearly all of it is real
misalignment).  The collision checker inflates every scene point by
``clearance + point_radius``, 25 mm by default.  Naively concatenating views
would therefore smear each surface by most of the clearance budget and turn
previously reachable grasps into false collisions.

So accumulation here is deliberately asymmetric: **the newest frame is always
authoritative wherever it has an opinion.**  A retained point survives only
where the newest frame is blind -- outside its frustum, or behind what it
measured.  Anywhere the newest frame sees a surface, the retained copy of that
surface is discarded rather than averaged, so every surface stays exactly as
thick as it is today.  The output can only add geometry where the current frame
has none, which is precisely the coverage bug and nothing else.

Because the Go2-W publishes no usable base odometry (``/state_estimation``
carries zero messages in every recorded bag), the buffer cannot be invalidated
from measured base motion.  It is invalidated by *disagreement* instead: if the
retained cloud and the newest frame contradict each other where both can see,
the world moved under us and the whole buffer is dropped.  Callers that do have
a trustworthy motion signal force the same reset via ``base_moved=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SceneAccumulatorConfig:
    """Bounds and tolerances for multi-view scene accumulation.

    ``authoritative_radius_m`` is the one value tied to measured hardware error:
    it must exceed the cross-view registration disagreement, or retained copies
    of an already-visible surface survive and thicken it.
    """

    voxel_size_m: float = 0.01
    max_points: int = 120_000
    max_frames: int = 8
    max_frame_age_s: float = 3.0
    authoritative_radius_m: float = 0.025
    carve_tangent_bin: float = 0.01
    carve_depth_margin_m: float = 0.04
    min_frame_points: int = 32
    # Keyframe admission.  Scene clouds arrive at 10-30 Hz, so a plain ring
    # buffer of N frames spans well under a second and every frame in it shares
    # one viewpoint -- useless for coverage.  Retaining only poses that differ
    # lets the same N slots span an entire wrist search pan.
    keyframe_rotation_rad: float = 0.087
    keyframe_translation_m: float = 0.02
    # Consistency gate.  Thresholds measured over 38 same-scene and 540
    # scene-changed steps in the recorded corpus: same-scene agreement runs
    # p10 0.88 / p50 0.99, scene-changed p50 0.50.  At 0.80 / 0.10 the gate
    # rejects 77% of scene-changed steps while keeping 95% of genuine ones.
    consistency_min_samples: int = 200
    consistency_min_agreement: float = 0.80
    consistency_min_overlap: float = 0.10

    def __post_init__(self) -> None:
        positive = (
            self.voxel_size_m,
            self.max_frame_age_s,
            self.authoritative_radius_m,
            self.carve_tangent_bin,
            self.carve_depth_margin_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("accumulator distances and ages must be finite and positive")
        nonnegative = (self.keyframe_rotation_rad, self.keyframe_translation_m)
        if not all(np.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("keyframe thresholds must be finite and non-negative")
        if self.max_points < 1:
            raise ValueError("max_points must be positive")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.min_frame_points < 1:
            raise ValueError("min_frame_points must be positive")
        if self.consistency_min_samples < 1:
            raise ValueError("consistency_min_samples must be positive")
        if not 0.0 <= self.consistency_min_agreement <= 1.0:
            raise ValueError("consistency_min_agreement must be within [0, 1]")
        if not 0.0 <= self.consistency_min_overlap <= 1.0:
            raise ValueError("consistency_min_overlap must be within [0, 1]")


@dataclass(frozen=True)
class AccumulationReport:
    """Per-update diagnostics, suitable for the runtime perception card."""

    accepted: bool
    reason: str
    reset_reason: str
    superseded_viewpoint: bool
    frame_points: int
    retained_points: int
    total_points: int
    frames: int
    newest_stamp_s: float | None
    oldest_stamp_s: float | None
    carved_authoritative: int
    carved_free_space: int
    consistency_samples: int
    consistency_agreement: float

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reset_reason": self.reset_reason,
            "superseded_viewpoint": self.superseded_viewpoint,
            "frame_points": self.frame_points,
            "retained_points": self.retained_points,
            "total_points": self.total_points,
            "frames": self.frames,
            "newest_stamp_s": self.newest_stamp_s,
            "oldest_stamp_s": self.oldest_stamp_s,
            "carved_authoritative": self.carved_authoritative,
            "carved_free_space": self.carved_free_space,
            "consistency_samples": self.consistency_samples,
            "consistency_agreement": self.consistency_agreement,
        }


@dataclass(frozen=True)
class _Verdict:
    """Per-point classification of one retained frame against the newest one."""

    keep: np.ndarray
    authoritative: np.ndarray
    free_space: np.ndarray
    comparable: np.ndarray


@dataclass
class _Frame:
    """One accepted viewpoint, already expressed in the arm base frame."""

    points: np.ndarray
    stamp_s: float
    sequence: int
    origin: np.ndarray
    rotation: np.ndarray

    def differs_from(self, other: "_Frame", *, rotation_rad: float, translation_m: float) -> bool:
        """True when this viewpoint is far enough from ``other`` to be worth keeping."""

        if float(np.linalg.norm(self.origin - other.origin)) >= translation_m:
            return True
        relative = self.rotation.T @ other.rotation
        cosine = (float(np.trace(relative)) - 1.0) / 2.0
        return math.acos(max(-1.0, min(1.0, cosine))) >= rotation_rad


def _validate_transform(transform: object) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("base_from_camera must be a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise ValueError("base_from_camera rotation is not orthonormal")
    return matrix


def _finite_points(points: object) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("scene points must have shape (N, 3)")
    return np.ascontiguousarray(cloud[np.all(np.isfinite(cloud), axis=1)])


def voxel_downsample(
    points: np.ndarray,
    leaf_m: float,
    *,
    priority: np.ndarray | None = None,
) -> np.ndarray:
    """Keep one point per ``leaf_m`` cube, preferring the highest ``priority``.

    Priority carries frame recency, so a voxel occupied by both a retained and a
    fresh observation resolves to the fresh one rather than to an arbitrary
    array-order winner.  Deterministic output keeps offline replay diffs
    meaningful.
    """

    cloud = np.asarray(points, dtype=float)
    if len(cloud) == 0:
        return np.zeros((0, 3), dtype=float)
    if not np.isfinite(leaf_m) or leaf_m <= 0.0:
        raise ValueError("voxel leaf size must be finite and positive")
    weights = (
        np.zeros(len(cloud), dtype=float)
        if priority is None
        else np.asarray(priority, dtype=float)
    )
    if weights.shape != (len(cloud),):
        raise ValueError("priority must be aligned with the point array")

    keys = np.floor(cloud / leaf_m).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0).astype(np.int64) + 1
    if float(span[0]) * float(span[1]) * float(span[2]) >= 2.0**62:
        raise ValueError("scene extent overflows the voxel index space")
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    # Sort by voxel, then by descending priority, so each run's first row wins.
    order = np.lexsort((-weights, flat))
    ordered = flat[order]
    first = np.empty(len(ordered), dtype=bool)
    first[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=first[1:])
    return np.ascontiguousarray(cloud[order[first]])


class SceneCloudAccumulator:
    """Bounded multi-view scene cloud with a newest-frame-authoritative merge."""

    def __init__(self, config: SceneAccumulatorConfig | None = None) -> None:
        self.config = config or SceneAccumulatorConfig()
        self._frames: list[_Frame] = []
        self._sequence = 0
        self._merged: np.ndarray | None = None

    # -- state ---------------------------------------------------------------

    def reset(self, reason: str = "explicit reset") -> str:
        self._frames.clear()
        self._merged = None
        return reason

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def newest_stamp_s(self) -> float | None:
        """Stamp of the newest contributing frame.

        The freshness gates downstream -- ``PointCloudCollisionConfig`` and the
        node's exact-stamp identity check -- must keep measuring the age of the
        *newest* evidence.  An accumulated cloud whose newest frame is 10 ms old
        is exactly as fresh as today's single-frame cloud; older contributions
        are bounded separately by ``max_frame_age_s``.
        """

        return self._frames[-1].stamp_s if self._frames else None

    @property
    def oldest_stamp_s(self) -> float | None:
        return self._frames[0].stamp_s if self._frames else None

    @property
    def points(self) -> np.ndarray:
        """The merged, voxel-downsampled scene cloud in the arm base frame."""

        if self._merged is None:
            self._merged = self._merge()
        return self._merged

    # -- update --------------------------------------------------------------

    def update(
        self,
        points: object,
        *,
        stamp_s: float,
        base_from_camera: object,
        base_moved: bool = False,
    ) -> AccumulationReport:
        """Fold one base-frame viewpoint into the buffer.

        ``points`` are already in the arm base frame -- the node transforms on
        receipt -- and ``base_from_camera`` is the pose that produced them,
        which is what makes the occlusion reasoning possible.
        """

        transform = _validate_transform(base_from_camera)
        cloud = _finite_points(points)
        if not np.isfinite(stamp_s):
            self.reset("non-finite scene stamp")
            raise ValueError("scene timestamp must be finite")
        stamp = float(stamp_s)

        if len(cloud) < self.config.min_frame_points:
            # Do not reset here: one thin frame is far more likely to be a depth
            # dropout than evidence that the accumulated world is wrong.
            return self._report(
                accepted=False,
                reason=(
                    f"frame carries {len(cloud)} usable points; "
                    f"requires {self.config.min_frame_points}"
                ),
            )

        reset_reason = ""
        if base_moved:
            reset_reason = "caller reported base motion"
        elif self._frames and stamp < self._frames[-1].stamp_s:
            reset_reason = "scene stamp went backwards"
        if reset_reason:
            self.reset(reset_reason)

        self._expire(stamp)

        origin = np.ascontiguousarray(transform[:3, 3])
        rotation = np.ascontiguousarray(transform[:3, :3])
        # Captured before the carve, which can delete the frame it refers to.
        previous_newest = self._frames[-1] if self._frames else None

        carved_authoritative = 0
        carved_free_space = 0
        samples = 0
        agreeing = 0
        agreement = 1.0

        if self._frames:
            fresh_tree = cKDTree(cloud)
            depth_buffer = _RangeBuffer.build(cloud, origin, rotation, self.config)

            verdicts = [
                self._classify(frame.points, fresh_tree, depth_buffer)
                for frame in self._frames
            ]
            for verdict in verdicts:
                carved_authoritative += int(np.count_nonzero(verdict.authoritative))
                carved_free_space += int(np.count_nonzero(verdict.free_space))
                samples += int(np.count_nonzero(verdict.comparable))
                agreeing += int(
                    np.count_nonzero(verdict.comparable & verdict.authoritative)
                )

            if samples >= self.config.consistency_min_samples:
                agreement = agreeing / samples
                if agreement < self.config.consistency_min_agreement:
                    reset_reason = (
                        f"retained cloud disagrees with the newest frame "
                        f"({agreement:.0%} of {samples} comparable points agree); "
                        f"assuming the base or the scene moved"
                    )
                    self.reset(reset_reason)
                    carved_authoritative = 0
                    carved_free_space = 0

            if not reset_reason:
                survivors: list[_Frame] = []
                for frame, verdict in zip(self._frames, verdicts):
                    kept = frame.points[verdict.keep]
                    if len(kept):
                        survivors.append(
                            _Frame(
                                points=np.ascontiguousarray(kept),
                                stamp_s=frame.stamp_s,
                                sequence=frame.sequence,
                                origin=frame.origin,
                                rotation=frame.rotation,
                            )
                        )
                self._frames = survivors

        self._sequence += 1
        fresh = _Frame(
            points=cloud,
            stamp_s=stamp,
            sequence=self._sequence,
            origin=origin,
            rotation=rotation,
        )
        # Keyframe admission.  A viewpoint that has not moved carries no coverage
        # the previous frame lacked, so it SUPERSEDES that frame instead of
        # consuming a slot.  The carve usually erases a co-located predecessor
        # anyway -- it is authoritative over everything it can still see -- but
        # not the slivers it occludes or crops, and at 10-30 Hz those slivers
        # would fill every slot with one viewpoint in under a second.  That is
        # precisely the coverage failure this class exists to fix.
        superseded = previous_newest is not None and not fresh.differs_from(
            previous_newest,
            rotation_rad=self.config.keyframe_rotation_rad,
            translation_m=self.config.keyframe_translation_m,
        )
        # The carve rebuilds survivors, so identity is carried by ``sequence``.
        if (
            superseded
            and self._frames
            and self._frames[-1].sequence == previous_newest.sequence
        ):
            self._frames[-1] = fresh
        else:
            self._frames.append(fresh)
        self._trim()
        self._merged = None

        return self._report(
            accepted=True,
            reset_reason=reset_reason,
            superseded_viewpoint=superseded,
            carved_authoritative=carved_authoritative,
            carved_free_space=carved_free_space,
            consistency_samples=samples,
            consistency_agreement=agreement,
        )

    # -- internals -----------------------------------------------------------

    def _classify(
        self,
        retained: np.ndarray,
        fresh_tree: cKDTree,
        depth_buffer: "_RangeBuffer",
    ) -> _Verdict:
        """Decide, per retained point, whether the newest frame overrules it."""

        # 1. The newest frame already measured this surface.  Drop the retained
        #    copy rather than averaging: registration error between views is
        #    comparable to the collision clearance, so keeping both thickens the
        #    obstacle by most of the planner's margin.
        distance, _ = fresh_tree.query(retained, workers=-1)
        authoritative = distance <= self.config.authoritative_radius_m

        # 2. The newest frame looked along this ray and measured something
        #    further away, so the retained point sits in observed free space.
        observed, depth, surface = depth_buffer.probe(retained)
        margin = self.config.carve_depth_margin_m
        free_space = observed & (depth < surface - margin)

        # 3. A point is "comparable" when the newest frame has any opinion about
        #    it -- in front of or on the measured surface.  Points behind the
        #    surface are occluded, so silence there is not disagreement.
        comparable = observed & (depth <= surface + margin)

        keep = ~(authoritative | free_space)
        return _Verdict(
            keep=keep,
            authoritative=authoritative,
            free_space=free_space,
            comparable=comparable,
        )

    def _expire(self, now_s: float) -> None:
        horizon = now_s - self.config.max_frame_age_s
        self._frames = [frame for frame in self._frames if frame.stamp_s >= horizon]

    def _trim(self) -> None:
        """Enforce the frame and point budgets, evicting the oldest first.

        The newest frame is never evicted: dropping it would leave the planner
        with a cloud that is strictly worse than the single-frame behaviour this
        replaces.
        """

        if len(self._frames) > self.config.max_frames:
            self._frames = self._frames[-self.config.max_frames:]
        while len(self._frames) > 1 and self._total_points() > self.config.max_points:
            self._frames.pop(0)

    def _total_points(self) -> int:
        return sum(len(frame.points) for frame in self._frames)

    def _merge(self) -> np.ndarray:
        if not self._frames:
            return np.zeros((0, 3), dtype=float)
        if len(self._frames) == 1:
            frame = self._frames[0]
            return voxel_downsample(frame.points, self.config.voxel_size_m)
        stacked = np.concatenate([frame.points for frame in self._frames], axis=0)
        priority = np.concatenate(
            [
                np.full(len(frame.points), float(frame.sequence))
                for frame in self._frames
            ]
        )
        return voxel_downsample(stacked, self.config.voxel_size_m, priority=priority)

    def _report(
        self,
        *,
        accepted: bool,
        reason: str = "",
        reset_reason: str = "",
        superseded_viewpoint: bool = False,
        carved_authoritative: int = 0,
        carved_free_space: int = 0,
        consistency_samples: int = 0,
        consistency_agreement: float = 1.0,
    ) -> AccumulationReport:
        newest = self._frames[-1] if self._frames else None
        retained = self._total_points() - (len(newest.points) if newest else 0)
        return AccumulationReport(
            accepted=accepted,
            reason=reason,
            reset_reason=reset_reason,
            superseded_viewpoint=superseded_viewpoint,
            frame_points=len(newest.points) if newest else 0,
            retained_points=retained,
            total_points=len(self.points),
            frames=len(self._frames),
            newest_stamp_s=self.newest_stamp_s,
            oldest_stamp_s=self.oldest_stamp_s,
            carved_authoritative=carved_authoritative,
            carved_free_space=carved_free_space,
            consistency_samples=consistency_samples,
            consistency_agreement=consistency_agreement,
        )


class _RangeBuffer:
    """A depth buffer for the newest frame, rasterized without camera intrinsics.

    Points are binned by their pinhole tangent coordinates ``(x/z, y/z)`` in the
    camera frame, which is the same parameterization an image grid uses, with an
    implicit focal length of ``1 / carve_tangent_bin``.  Deriving the grid extent
    from the frame's own points means the buffer covers exactly what the camera
    actually returned -- no assumed field of view, no CameraInfo dependency, and
    depth dropouts correctly read as "not observed" rather than "empty".
    """

    __slots__ = ("_rotation_t", "_origin", "_bin", "_low", "_shape", "_surface")

    def __init__(
        self,
        rotation_t: np.ndarray,
        origin: np.ndarray,
        bin_size: float,
        low: np.ndarray,
        shape: tuple[int, int],
        surface: np.ndarray,
    ) -> None:
        self._rotation_t = rotation_t
        self._origin = origin
        self._bin = bin_size
        self._low = low
        self._shape = shape
        self._surface = surface

    @classmethod
    def build(
        cls,
        cloud: np.ndarray,
        origin: np.ndarray,
        rotation: np.ndarray,
        config: SceneAccumulatorConfig,
    ) -> "_RangeBuffer":
        rotation_t = rotation.T
        camera = (cloud - origin) @ rotation_t.T
        depth = camera[:, 2]
        forward = depth > 1e-6
        if not np.any(forward):
            empty = np.full((1, 1), np.inf)
            return cls(rotation_t, origin, config.carve_tangent_bin,
                       np.zeros(2), (1, 1), empty)
        tangent = camera[forward, :2] / depth[forward, None]
        low = tangent.min(axis=0)
        index = np.floor((tangent - low) / config.carve_tangent_bin).astype(np.int64)
        shape = (int(index[:, 0].max()) + 1, int(index[:, 1].max()) + 1)
        surface = np.full(shape, np.inf)
        # Nearest surface per bin.  Taking the minimum makes the free-space carve
        # conservative: a bin holding both a near object and a far wall reports
        # the near one, so retained points beyond it read as occluded and stay.
        np.minimum.at(surface, (index[:, 0], index[:, 1]), depth[forward])
        return cls(rotation_t, origin, config.carve_tangent_bin, low, shape, surface)

    def probe(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(observed, depth, surface_depth)`` for each queried point."""

        camera = (points - self._origin) @ self._rotation_t.T
        depth = camera[:, 2]
        observed = depth > 1e-6
        surface = np.full(len(points), np.inf)
        if not np.any(observed):
            return observed, depth, surface
        tangent = camera[observed, :2] / depth[observed, None]
        index = np.floor((tangent - self._low) / self._bin).astype(np.int64)
        inside = (
            (index[:, 0] >= 0)
            & (index[:, 0] < self._shape[0])
            & (index[:, 1] >= 0)
            & (index[:, 1] < self._shape[1])
        )
        sampled = np.full(len(index), np.inf)
        sampled[inside] = self._surface[index[inside, 0], index[inside, 1]]
        surface[observed] = sampled
        # A bin the newest frame never filled is unobserved, not empty.
        observed = observed.copy()
        observed[np.flatnonzero(observed)] = inside & np.isfinite(sampled)
        return observed, depth, surface
