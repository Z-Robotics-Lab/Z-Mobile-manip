"""ROS-independent contracts for synchronized observed placement planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math

import numpy as np

from z_manip.models.planner import PlanningError
from z_manip.planning import (
    NormalizedPlacementRegion,
    ObservedPlacementInput,
    ObservedPlacementPlanner,
    PlacementCandidate,
    PlacementConstraints,
    PlannedPlacement,
)
from z_manip.planning_control import (
    checkpoint,
    PlanningAborted,
    PlanningControl,
)


class PlacementContractError(ValueError):
    """An input snapshot or output trajectory is unsafe to consume."""


POST_RELEASE_VERIFICATION_SCHEMA = 'z_manip.post_release_verification.v2'
POST_RELEASE_VERIFICATION_RESULT = 'post_release_target_stable_in_region'
POST_RELEASE_OBSERVATION_SOURCE = 'synchronized_rgbd_pointcloud'
MIN_OBJECT_REFERENCE_POINTS = 40
MAX_OBJECT_REFERENCE_POINTS = 512
OBJECT_REFERENCE_EXTENT_RELATIVE_TOLERANCE = 1e-6
OBJECT_REFERENCE_EXTENT_ABSOLUTE_TOLERANCE_M = 1e-9


def _finite_vector(values: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise PlacementContractError(f'{label} must be a finite {length}-vector')
    return result


def _finite_transform(values: object, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise PlacementContractError(f'{label} must be a finite 4x4 transform')
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise PlacementContractError(f'{label} has an invalid homogeneous row')
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise PlacementContractError(f'{label} rotation is not orthonormal')
    if np.linalg.det(rotation) < 0.999:
        raise PlacementContractError(f'{label} rotation is not right handed')
    return result


def _positive_stamp(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value < 2**63
    ):
        raise PlacementContractError(f'{label} must be a positive integer ns stamp')
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlacementContractError(f'placement request repeats field {key!r}')
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise PlacementContractError(f'placement request contains non-finite {value}')


def _contains_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Mapping):
        return any(_contains_boolean(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _nearest_planar_distances(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return bounded-memory nearest distances between two planar point sets."""
    if len(queries) == 0 or len(references) == 0:
        return np.full(len(queries), np.inf, dtype=float)
    result = np.empty(len(queries), dtype=float)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        difference = queries[start:stop, None, :] - references[None, :, :]
        result[start:stop] = np.sqrt(
            np.min(np.einsum('ijk,ijk->ij', difference, difference), axis=1),
        )
    return result


def _deterministic_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=int)
    return points[indices]


def _nearest_spatial_distances(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return bounded-memory nearest distances between 3-D point sets."""
    if len(queries) == 0 or len(references) == 0:
        return np.full(len(queries), np.inf, dtype=float)
    result = np.empty(len(queries), dtype=float)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        difference = queries[start:stop, None, :] - references[None, :, :]
        result[start:stop] = np.sqrt(
            np.min(np.einsum('ijk,ijk->ij', difference, difference), axis=1),
        )
    return result


def _unit_vector(values: object, label: str) -> np.ndarray:
    vector = _finite_vector(values, 3, label)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise PlacementContractError(f'{label} is degenerate')
    return vector / norm


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _signed_axis_profile(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Return dimensionless odd shape evidence along one directed axis.

    Every component changes sign when ``axis`` is reversed.  The profile uses
    both axial sample distribution and cross-section variation, so direction
    can be recovered for generic asymmetric geometry without a class-specific
    top/bottom heuristic.  A zero profile is intentionally unobservable.
    """
    cloud = np.asarray(points, dtype=float)
    direction = _unit_vector(axis, 'signed profile axis')
    if (
        cloud.ndim != 2
        or cloud.shape[1] != 3
        or len(cloud) < 3
        or not np.all(np.isfinite(cloud))
    ):
        raise PlacementContractError('signed upright profile cloud is malformed')
    centered = cloud - np.mean(cloud, axis=0)
    axial = centered @ direction
    axial_scale = float(np.sqrt(np.mean(axial * axial)))
    if not math.isfinite(axial_scale) or axial_scale < 1e-8:
        raise PlacementContractError('signed upright profile has no axial extent')
    radial_vectors = centered - axial[:, None] * direction
    radial = np.linalg.norm(radial_vectors, axis=1)
    radial_scale = float(np.sqrt(np.mean(radial * radial)))
    if not math.isfinite(radial_scale) or radial_scale < 1e-8:
        raise PlacementContractError('signed upright profile has no radial extent')
    normalized_axial = axial / axial_scale
    normalized_radial = radial / radial_scale
    low = normalized_axial <= np.quantile(normalized_axial, 0.25)
    high = normalized_axial >= np.quantile(normalized_axial, 0.75)
    if np.count_nonzero(low) < 2 or np.count_nonzero(high) < 2:
        raise PlacementContractError('signed upright endpoint evidence is insufficient')
    profile = np.asarray((
        np.mean(normalized_axial ** 3),
        np.mean(normalized_axial * normalized_radial),
        np.mean(normalized_axial * normalized_radial ** 2),
        np.mean(normalized_radial[high]) - np.mean(normalized_radial[low]),
        np.quantile(normalized_radial[high], 0.80)
        - np.quantile(normalized_radial[low], 0.80),
    ), dtype=float)
    if not np.all(np.isfinite(profile)):
        raise PlacementContractError('signed upright profile is non-finite')
    return profile


@dataclass(frozen=True, eq=False)
class PlacementOrientationVerification:
    """Task-provided semantic orientation and symmetry contract."""

    require_upright: bool = True
    upright_axis_object: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orientation_symmetry: str = 'auto'
    symmetry_axis_object: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.require_upright, bool):
            raise PlacementContractError('require_upright must be boolean')
        axis = _finite_vector(
            self.upright_axis_object,
            3,
            'upright_axis_object',
        )
        if np.linalg.norm(axis) < 1e-9:
            raise PlacementContractError('upright_axis_object is degenerate')
        if self.orientation_symmetry not in {'auto', 'none', 'axial'}:
            raise PlacementContractError(
                'orientation_symmetry must be auto, none, or axial',
            )
        if self.orientation_symmetry == 'axial':
            if self.symmetry_axis_object is None:
                raise PlacementContractError(
                    'axial symmetry requires symmetry_axis_object',
                )
            symmetry_axis = _finite_vector(
                self.symmetry_axis_object,
                3,
                'symmetry_axis_object',
            )
            if np.linalg.norm(symmetry_axis) < 1e-9:
                raise PlacementContractError('symmetry_axis_object is degenerate')
        elif self.symmetry_axis_object is not None:
            raise PlacementContractError(
                'symmetry_axis_object is only valid for axial symmetry',
            )


@dataclass(frozen=True, eq=False)
class PlacementRegionRequest:
    """Versioned VLM placement region plus observed carried-object geometry."""

    goal_id: str
    stamp_ns: int
    image_frame: str
    request_id: str
    producer_epoch: str
    executor_epoch: str
    generation: int
    region: NormalizedPlacementRegion
    constraints: PlacementConstraints
    object_extent_m: np.ndarray
    tool_from_object: np.ndarray
    object_reference_identity: ObservedPerceptionIdentity
    object_reference_points_object: np.ndarray
    verification: PlacementOrientationVerification

    def __post_init__(self) -> None:
        strings = (
            self.goal_id,
            self.image_frame,
            self.request_id,
            self.producer_epoch,
            self.executor_epoch,
        )
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise PlacementContractError(
                'goal, frame, request, and producer identity must be non-empty strings',
            )
        if (
            len(self.goal_id) > 256
            or len(self.image_frame) > 256
            or len(self.request_id) > 256
            or len(self.producer_epoch) > 128
            or len(self.executor_epoch) > 256
        ):
            raise PlacementContractError('placement identity string is too long')
        _positive_stamp(self.stamp_ns, 'request')
        extent = _finite_vector(
            self.object_extent_m,
            3,
            'object_extent_m',
        ).copy()
        if np.any(extent <= 0.0):
            raise PlacementContractError('object_extent_m must be positive')
        tool_from_object = _finite_transform(
            self.tool_from_object,
            'tool_from_object',
        ).copy()
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 0 < self.generation < 2**63
        ):
            raise PlacementContractError('placement observation generation must be positive')
        if not isinstance(self.object_reference_identity, ObservedPerceptionIdentity):
            raise PlacementContractError('object reference identity is malformed')
        if self.object_reference_identity.stamp_ns >= self.stamp_ns:
            raise PlacementContractError(
                'object reference observation must predate placement observation',
            )
        reference = np.asarray(
            self.object_reference_points_object,
            dtype=float,
        ).copy()
        if (
            reference.ndim != 2
            or reference.shape[1] != 3
            or not MIN_OBJECT_REFERENCE_POINTS <= len(reference) <= MAX_OBJECT_REFERENCE_POINTS
            or not np.all(np.isfinite(reference))
        ):
            raise PlacementContractError(
                'object_reference_points_object must be a bounded finite Nx3 cloud',
            )
        if len(np.unique(reference, axis=0)) != len(reference):
            raise PlacementContractError('object reference points must be unique')
        measured_extent = np.ptp(reference, axis=0)
        allowed_extent_error = (
            OBJECT_REFERENCE_EXTENT_ABSOLUTE_TOLERANCE_M
            + OBJECT_REFERENCE_EXTENT_RELATIVE_TOLERANCE * extent
        )
        if (
            np.any(measured_extent <= 0.0)
            or np.any(np.abs(measured_extent - extent) > allowed_extent_error)
        ):
            raise PlacementContractError(
                'object extent does not match the frozen object reference cloud',
            )
        if not isinstance(self.verification, PlacementOrientationVerification):
            raise PlacementContractError('verification semantics are malformed')
        extent.setflags(write=False)
        tool_from_object.setflags(write=False)
        reference.setflags(write=False)
        object.__setattr__(self, 'object_extent_m', extent)
        object.__setattr__(self, 'tool_from_object', tool_from_object)
        object.__setattr__(self, 'object_reference_points_object', reference)

    @property
    def observation_identity(self) -> ObservedPerceptionIdentity:
        """Return the exact place-time target owner carried by schema v2."""
        return ObservedPerceptionIdentity(
            request_id=self.request_id,
            producer_epoch=self.producer_epoch,
            generation=self.generation,
            stamp_ns=self.stamp_ns,
            frame_id=self.image_frame,
        )


def parse_region_request(payload: str) -> PlacementRegionRequest:
    """Parse a strict schema-v2 JSON request without semantic class shortcuts."""
    try:
        data = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise PlacementContractError('placement request is not valid JSON') from error
    if not isinstance(data, Mapping):
        raise PlacementContractError('placement request must be a JSON object')
    required = {
        'schema_version', 'goal_id', 'stamp_ns', 'image_frame', 'request_id',
        'producer_epoch', 'executor_epoch', 'generation', 'region_xyxy',
        'object_extent_m',
        'tool_from_object', 'object_reference_identity',
        'object_reference_points_object', 'verification',
    }
    optional = {'avoid_xyxy', 'constraints'}
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing or unknown:
        raise PlacementContractError(
            f'placement request keys mismatch; missing={sorted(missing)}, '
            f'unknown={sorted(unknown)}',
        )
    if (
        isinstance(data['schema_version'], bool)
        or not isinstance(data['schema_version'], int)
        or data['schema_version'] != 2
    ):
        raise PlacementContractError('unsupported placement request schema_version')
    raw_constraints = data.get('constraints', {})
    if not isinstance(raw_constraints, Mapping):
        raise PlacementContractError('constraints must be a JSON object')
    allowed_constraints = {
        'min_clearance_m', 'max_surface_tilt_rad', 'preferred_yaw_rad',
        'yaw_tolerance_rad', 'min_support_fraction',
    }
    constraint_unknown = set(raw_constraints) - allowed_constraints
    if constraint_unknown:
        raise PlacementContractError(
            f'unknown placement constraints: {sorted(constraint_unknown)}',
        )
    numeric_contracts = (
        data['region_xyxy'],
        data.get('avoid_xyxy', ()),
        data['object_extent_m'],
        data['tool_from_object'],
        data['object_reference_points_object'],
    )
    if any(_contains_boolean(value) for value in numeric_contracts):
        raise PlacementContractError(
            'placement geometry arrays cannot contain JSON booleans',
        )
    for name, value in raw_constraints.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise PlacementContractError(
                f'placement constraint {name} must be numeric or null',
            )
    defaults = PlacementConstraints()
    try:
        for name in ('goal_id', 'image_frame', 'request_id', 'producer_epoch'):
            if not isinstance(data[name], str):
                raise PlacementContractError(f'{name} must be a JSON string')
        raw_generation = data['generation']
        if isinstance(raw_generation, bool) or not isinstance(raw_generation, int):
            raise PlacementContractError('generation must be a JSON integer')
        raw_reference_identity = data['object_reference_identity']
        if not isinstance(raw_reference_identity, Mapping):
            raise PlacementContractError('object_reference_identity must be an object')
        reference_identity_keys = {
            'request_id', 'producer_epoch', 'generation',
            'observation_stamp_ns', 'frame_id',
        }
        if set(raw_reference_identity) != reference_identity_keys:
            raise PlacementContractError(
                'object_reference_identity keys must be exactly '
                f'{sorted(reference_identity_keys)}',
            )
        for name in ('request_id', 'producer_epoch', 'frame_id'):
            if not isinstance(raw_reference_identity[name], str):
                raise PlacementContractError(
                    f'object_reference_identity.{name} must be a JSON string',
                )
        reference_generation = raw_reference_identity['generation']
        reference_stamp = raw_reference_identity['observation_stamp_ns']
        if (
            isinstance(reference_generation, bool)
            or not isinstance(reference_generation, int)
            or isinstance(reference_stamp, bool)
            or not isinstance(reference_stamp, int)
        ):
            raise PlacementContractError(
                'object reference generation and stamp must be JSON integers',
            )
        reference_identity = ObservedPerceptionIdentity(
            request_id=raw_reference_identity['request_id'],
            producer_epoch=raw_reference_identity['producer_epoch'],
            generation=reference_generation,
            stamp_ns=reference_stamp,
            frame_id=raw_reference_identity['frame_id'],
        )
        raw_verification = data['verification']
        if not isinstance(raw_verification, Mapping):
            raise PlacementContractError('verification must be a JSON object')
        verification_required = {
            'require_upright',
            'upright_axis_object',
            'orientation_symmetry',
        }
        verification_keys = verification_required | {
            'symmetry_axis_object',
        }
        verification_missing = verification_required - set(raw_verification)
        verification_unknown = set(raw_verification) - verification_keys
        if verification_missing or verification_unknown:
            raise PlacementContractError(
                'placement verification keys mismatch; '
                f'missing={sorted(verification_missing)}, '
                f'unknown={sorted(verification_unknown)}',
            )
        require_upright = raw_verification['require_upright']
        if not isinstance(require_upright, bool):
            raise PlacementContractError('require_upright must be boolean')
        if _contains_boolean(raw_verification['upright_axis_object']) or (
            'symmetry_axis_object' in raw_verification
            and _contains_boolean(raw_verification['symmetry_axis_object'])
        ):
            raise PlacementContractError(
                'placement verification axes cannot contain JSON booleans',
            )
        verification = PlacementOrientationVerification(
            require_upright=require_upright,
            upright_axis_object=tuple(
                float(value) for value in raw_verification['upright_axis_object']
            ),
            orientation_symmetry=str(raw_verification['orientation_symmetry']),
            symmetry_axis_object=(
                None
                if raw_verification.get('symmetry_axis_object') is None
                else tuple(float(value) for value in (
                    raw_verification['symmetry_axis_object']
                ))
            ),
        )
        constraints = PlacementConstraints(
            min_clearance_m=float(raw_constraints.get(
                'min_clearance_m', defaults.min_clearance_m,
            )),
            max_surface_tilt_rad=(
                None if raw_constraints.get('max_surface_tilt_rad') is None
                else float(raw_constraints['max_surface_tilt_rad'])
            ),
            preferred_yaw_rad=(
                None if raw_constraints.get('preferred_yaw_rad') is None
                else float(raw_constraints['preferred_yaw_rad'])
            ),
            yaw_tolerance_rad=float(raw_constraints.get(
                'yaw_tolerance_rad', defaults.yaw_tolerance_rad,
            )),
            min_support_fraction=float(raw_constraints.get(
                'min_support_fraction', defaults.min_support_fraction,
            )),
        )
        region = NormalizedPlacementRegion(
            tuple(float(value) for value in data['region_xyxy']),
            tuple(tuple(float(value) for value in box)
                  for box in data.get('avoid_xyxy', ())),
        )
        raw_stamp = data['stamp_ns']
        if isinstance(raw_stamp, bool) or not isinstance(raw_stamp, int):
            raise PlacementContractError('stamp_ns must be a JSON integer')
        request = PlacementRegionRequest(
            goal_id=data['goal_id'],
            stamp_ns=raw_stamp,
            image_frame=data['image_frame'],
            request_id=data['request_id'],
            producer_epoch=data['producer_epoch'],
            executor_epoch=data['executor_epoch'],
            generation=raw_generation,
            region=region,
            constraints=constraints,
            object_extent_m=np.asarray(data['object_extent_m'], dtype=float),
            tool_from_object=np.asarray(data['tool_from_object'], dtype=float),
            object_reference_identity=reference_identity,
            object_reference_points_object=np.asarray(
                data['object_reference_points_object'],
                dtype=float,
            ),
            verification=verification,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PlacementContractError(f'invalid placement request: {error}') from error
    return request


@dataclass(frozen=True, eq=False)
class PlacementPerceptionSnapshot:
    """One transformed snapshot bound to an exact source RGB-D frame."""

    rgb_stamp_ns: int
    depth_stamp_ns: int
    camera_info_stamp_ns: int
    scene_stamp_ns: int
    joint_stamp_ns: int
    image_frame: str
    frame_id: str
    organized_points: np.ndarray
    scene_points: np.ndarray
    gravity: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray


@dataclass(frozen=True)
class ObservedPerceptionIdentity:
    """Exact task-owned producer identity for one observed target frame."""

    request_id: str
    producer_epoch: str
    generation: int
    stamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        strings = (self.request_id, self.producer_epoch, self.frame_id)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise PlacementContractError('perception identity strings must be non-empty')
        if (
            len(self.request_id) > 256
            or len(self.producer_epoch) > 128
            or len(self.frame_id) > 256
        ):
            raise PlacementContractError('perception identity string is too long')
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 0 < self.generation < 2**63
        ):
            raise PlacementContractError('perception generation must be positive')
        _positive_stamp(self.stamp_ns, 'perception observation')


@dataclass(frozen=True, eq=False)
class ObservedPlacementRegionGeometry:
    """Planning-frame support geometry frozen from the selected RGB-D region."""

    frame_id: str
    plane_origin: np.ndarray
    plane_normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    support_coordinates: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise PlacementContractError('observed placement region frame is empty')
        normal = _finite_vector(self.plane_normal, 3, 'support plane normal')
        tangent_u = _finite_vector(self.tangent_u, 3, 'support tangent u')
        tangent_v = _finite_vector(self.tangent_v, 3, 'support tangent v')
        _finite_vector(self.plane_origin, 3, 'support plane origin')
        basis = np.stack((normal, tangent_u, tangent_v))
        if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5):
            raise PlacementContractError('support plane basis must be orthonormal')
        coordinates = np.asarray(self.support_coordinates, dtype=float)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or len(coordinates) < 3
            or not np.all(np.isfinite(coordinates))
        ):
            raise PlacementContractError(
                'observed placement support coordinates are malformed',
            )


@dataclass(frozen=True, eq=False)
class PlannedObjectGeometry:
    """Observed object model registered to its selected planned placement pose."""

    frame_id: str
    expected_pose: np.ndarray
    reference_points_object: np.ndarray
    support_normal: np.ndarray
    verification: PlacementOrientationVerification

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise PlacementContractError('planned object geometry frame is empty')
        expected_pose = _finite_transform(
            self.expected_pose,
            'planned object pose',
        ).copy()
        points = np.asarray(self.reference_points_object, dtype=float).copy()
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not MIN_OBJECT_REFERENCE_POINTS <= len(points) <= MAX_OBJECT_REFERENCE_POINTS
            or not np.all(np.isfinite(points))
        ):
            raise PlacementContractError('observed object reference model is malformed')
        if len(np.unique(points, axis=0)) != len(points):
            raise PlacementContractError('observed object reference points are not unique')
        normal = _finite_vector(self.support_normal, 3, 'object support normal')
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-5):
            raise PlacementContractError('object support normal must be unit length')
        if not isinstance(self.verification, PlacementOrientationVerification):
            raise PlacementContractError('object verification semantics are malformed')
        frozen_normal = normal.copy()
        expected_pose.setflags(write=False)
        points.setflags(write=False)
        frozen_normal.setflags(write=False)
        object.__setattr__(self, 'expected_pose', expected_pose)
        object.__setattr__(self, 'reference_points_object', points)
        object.__setattr__(self, 'support_normal', frozen_normal)


@dataclass(frozen=True, eq=False)
class _ObjectModelAnalysis:
    orientation_mode: str
    reference_center_object: np.ndarray
    reference_axes_object: np.ndarray
    reference_eigenvalues: np.ndarray
    symmetry_axis_object: np.ndarray | None
    symmetry_eigen_index: int | None
    upright_axis_object: np.ndarray
    upright_eigen_index: int | None
    signed_upright_profile: np.ndarray | None


@dataclass(frozen=True)
class PostReleaseVerificationConfig:
    """Generic fail-closed thresholds for observed post-release placement."""

    min_stable_duration_s: float = 0.50
    min_samples: int = 3
    min_target_points: int = MIN_OBJECT_REFERENCE_POINTS
    min_current_support_points: int = 24
    max_target_motion_m: float = 0.025
    min_region_support_fraction: float = 0.80
    min_gripper_clearance_m: float = 0.04
    min_gripper_aperture_m: float = 0.065
    max_sync_skew_s: float = 0.025
    max_observation_age_s: float = 0.35
    max_state_skew_s: float = 0.12
    observation_timeout_s: float = 1.0
    max_observation_gap_s: float = 0.30
    plane_distance_tolerance_m: float = 0.015
    region_neighbor_radius_m: float = 0.04
    support_neighbor_radius_m: float = 0.05
    bottom_band_m: float = 0.025
    max_bottom_height_m: float = 0.035
    max_plane_penetration_m: float = 0.012
    target_mask_dilation_px: int = 2
    max_geometry_samples: int = 4096
    gripper_probe_radius_m: float = 0.0
    target_depth_correspondence_tolerance_m: float = 0.012
    object_position_tolerance_m: float = 0.04
    object_orientation_tolerance_rad: float = 0.35
    upright_tolerance_rad: float = 0.26
    orientation_degeneracy_ratio: float = 1.20
    max_axial_transverse_ratio: float = 1.90
    max_symmetry_axis_alignment_error_rad: float = 0.12
    min_signed_upright_profile_asymmetry: float = 0.10
    min_signed_upright_profile_alignment: float = 0.60
    max_object_orientation_motion_rad: float = 0.10
    registration_distance_m: float = 0.035
    min_registration_inlier_fraction: float = 0.55
    max_registration_rms_m: float = 0.025
    min_object_reference_points: int = MIN_OBJECT_REFERENCE_POINTS
    fk_position_tolerance_m: float = 0.015
    fk_orientation_tolerance_rad: float = 0.10
    max_rejection_diagnostics: int = 8

    def __post_init__(self) -> None:
        positive = (
            self.min_stable_duration_s,
            self.max_target_motion_m,
            self.min_gripper_clearance_m,
            self.min_gripper_aperture_m,
            self.max_sync_skew_s,
            self.max_observation_age_s,
            self.max_state_skew_s,
            self.observation_timeout_s,
            self.max_observation_gap_s,
            self.plane_distance_tolerance_m,
            self.region_neighbor_radius_m,
            self.support_neighbor_radius_m,
            self.bottom_band_m,
            self.max_bottom_height_m,
            self.target_depth_correspondence_tolerance_m,
            self.object_position_tolerance_m,
            self.object_orientation_tolerance_rad,
            self.upright_tolerance_rad,
            self.max_symmetry_axis_alignment_error_rad,
            self.min_signed_upright_profile_asymmetry,
            self.max_object_orientation_motion_rad,
            self.registration_distance_m,
            self.max_registration_rms_m,
            self.fk_position_tolerance_m,
            self.fk_orientation_tolerance_rad,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('post-release metric and timing thresholds must be positive')
        if (
            not math.isfinite(self.min_region_support_fraction)
            or not 0.0 < self.min_region_support_fraction <= 1.0
        ):
            raise ValueError('post-release support fraction must be in (0, 1]')
        nonnegative = (
            self.max_plane_penetration_m,
            self.gripper_probe_radius_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError('post-release geometry margins cannot be negative')
        counts = (
            self.min_samples,
            self.min_target_points,
            self.min_current_support_points,
            self.max_geometry_samples,
            self.min_object_reference_points,
            self.max_rejection_diagnostics,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in counts
        ):
            raise ValueError('post-release sample counts must be positive')
        if self.max_geometry_samples < max(
            self.min_target_points,
            self.min_current_support_points,
        ):
            raise ValueError('max_geometry_samples is below a minimum point count')
        if (
            isinstance(self.target_mask_dilation_px, bool)
            or not isinstance(self.target_mask_dilation_px, int)
            or self.target_mask_dilation_px < 0
        ):
            raise ValueError('target mask dilation cannot be negative')
        if not math.isfinite(self.orientation_degeneracy_ratio) or (
            self.orientation_degeneracy_ratio <= 1.0
        ):
            raise ValueError('orientation degeneracy ratio must exceed one')
        if (
            not math.isfinite(self.max_axial_transverse_ratio)
            or self.max_axial_transverse_ratio < 1.0
        ):
            raise ValueError('axial transverse ratio cannot be below one')
        if (
            not math.isfinite(self.min_registration_inlier_fraction)
            or not 0.0 < self.min_registration_inlier_fraction <= 1.0
        ):
            raise ValueError('registration inlier fraction must be in (0, 1]')
        if (
            not math.isfinite(self.min_signed_upright_profile_alignment)
            or not 0.0 < self.min_signed_upright_profile_alignment <= 1.0
        ):
            raise ValueError('signed upright profile alignment must be in (0, 1]')


class PlaceExecutionCorrelation:
    """Correlate one executor epoch and one ordered placement contract."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = 'idle'
        self.goal_id = ''
        self.executor_epoch = ''
        self.trajectory_contract_id = ''
        self.trajectory_command_highwater = 0
        self.gripper_command_highwater = 0
        self.trajectory_source_highwater_ns = 0
        self.gripper_source_highwater_ns = 0
        self.approach_command_id: int | None = None
        self.approach_source_stamp_ns: int | None = None
        self.release_gripper_command_id: int | None = None
        self.release_source_stamp_ns: int | None = None
        self.retreat_command_id: int | None = None
        self.retreat_source_stamp_ns: int | None = None
        self.latest_gripper_command_id = 0
        self.latest_gripper_source_stamp_ns = 0
        self.pre_release_gripper_command_id: int | None = None
        self.pre_release_gripper_source_stamp_ns: int | None = None
        self.failure = ''

    @property
    def armed(self) -> bool:
        return self.state not in {'idle', 'invalid', 'complete'}

    @property
    def ready_for_release(self) -> bool:
        return self.state == 'await_release'

    @property
    def complete(self) -> bool:
        return self.state == 'complete'

    @staticmethod
    def _identity(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise PlacementContractError(f'{label} must be a bounded non-empty string')
        return value.strip()

    @staticmethod
    def _nonnegative_command_id(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PlacementContractError(f'{label} command_id must be nonnegative')
        return value

    @staticmethod
    def _nonnegative_stamp(value: object, label: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**63
        ):
            raise PlacementContractError(f'{label} source stamp must be nonnegative')
        return value

    def arm(
        self,
        *,
        goal_id: str,
        executor_epoch: str,
        trajectory_contract_id: str,
        trajectory_command_highwater: int,
        gripper_command_highwater: int,
        trajectory_source_highwater_ns: int,
        gripper_source_highwater_ns: int,
    ) -> None:
        self.reset()
        self.goal_id = self._identity(goal_id, 'place goal')
        self.executor_epoch = self._identity(executor_epoch, 'executor epoch')
        self.trajectory_contract_id = self._identity(
            trajectory_contract_id,
            'trajectory contract',
        )
        self.trajectory_command_highwater = self._nonnegative_command_id(
            trajectory_command_highwater,
            'trajectory high-water',
        )
        self.gripper_command_highwater = self._nonnegative_command_id(
            gripper_command_highwater,
            'gripper high-water',
        )
        self.trajectory_source_highwater_ns = self._nonnegative_stamp(
            trajectory_source_highwater_ns,
            'trajectory high-water',
        )
        self.gripper_source_highwater_ns = self._nonnegative_stamp(
            gripper_source_highwater_ns,
            'gripper high-water',
        )
        self.latest_gripper_command_id = self.gripper_command_highwater
        self.latest_gripper_source_stamp_ns = self.gripper_source_highwater_ns
        self.state = 'await_approach_active'

    def _invalidate(self, reason: str) -> None:
        self.failure = reason
        self.state = 'invalid'
        raise PlacementContractError(reason)

    @staticmethod
    def _command_id(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PlacementContractError(f'{label} command_id must be positive')
        return value

    def _validate_epoch(self, value: object) -> None:
        if self._identity(value, 'executor epoch') != self.executor_epoch:
            self._invalidate('executor epoch changed during placement execution')

    def _validate_trajectory_contract(self, value: object) -> None:
        if self._identity(value, 'trajectory contract') != self.trajectory_contract_id:
            self._invalidate('trajectory status belongs to a different contract')

    def observe_trajectory(
        self,
        *,
        status: str,
        segment: str,
        command_id: int,
        executor_epoch: str,
        trajectory_contract_id: str,
        source_stamp_ns: int,
    ) -> None:
        """Accept only active-to-succeeded pairs for the two place segments."""
        if self.state in {'idle', 'invalid'}:
            self._invalidate('place execution status arrived outside an active chain')
        if status not in {'active', 'succeeded'}:
            self._invalidate('place execution status is not active or succeeded')
        if segment not in {'place_approach', 'place_retreat'}:
            self._invalidate('place execution segment is not approach or retreat')
        try:
            command = self._command_id(command_id, segment)
            source_stamp = _positive_stamp(
                source_stamp_ns,
                f'{segment} source',
            )
            self._validate_epoch(executor_epoch)
            self._validate_trajectory_contract(trajectory_contract_id)
        except PlacementContractError as error:
            self._invalidate(str(error))
            return

        if self.state == 'await_approach_active':
            if segment != 'place_approach' or status != 'active':
                self._invalidate('place approach must start with active status')
            if (
                command <= self.trajectory_command_highwater
                or source_stamp <= self.trajectory_source_highwater_ns
            ):
                self._invalidate('place approach did not exceed execution high-water')
            self.approach_command_id = command
            self.approach_source_stamp_ns = source_stamp
            self.state = 'await_approach_success'
            return
        if self.state == 'await_approach_success':
            if segment != 'place_approach':
                self._invalidate('place retreat arrived before release')
            if command != self.approach_command_id:
                self._invalidate('place approach command_id changed before success')
            if source_stamp != self.approach_source_stamp_ns:
                self._invalidate('place approach source identity changed before success')
            if status == 'succeeded':
                self.pre_release_gripper_command_id = (
                    self.latest_gripper_command_id
                )
                self.pre_release_gripper_source_stamp_ns = (
                    self.latest_gripper_source_stamp_ns
                )
                self.state = 'await_release'
            return
        if self.state == 'await_release':
            if (
                segment == 'place_approach'
                and status == 'succeeded'
                and command == self.approach_command_id
                and source_stamp == self.approach_source_stamp_ns
            ):
                return
            self._invalidate('trajectory status arrived before measured release')
        if self.state == 'await_retreat_active':
            if (
                segment == 'place_approach'
                and status == 'succeeded'
                and command == self.approach_command_id
                and source_stamp == self.approach_source_stamp_ns
            ):
                return
            if segment != 'place_retreat' or status != 'active':
                self._invalidate('place retreat must start with active status')
            assert self.approach_command_id is not None
            assert self.release_source_stamp_ns is not None
            if command <= self.approach_command_id:
                self._invalidate('place retreat command_id did not follow approach')
            if source_stamp <= self.release_source_stamp_ns:
                self._invalidate('place retreat source did not follow release')
            self.retreat_command_id = command
            self.retreat_source_stamp_ns = source_stamp
            self.state = 'await_retreat_success'
            return
        if self.state == 'await_retreat_success':
            if segment != 'place_retreat':
                self._invalidate('place approach replayed after release')
            if command != self.retreat_command_id:
                self._invalidate('place retreat command_id changed before success')
            if source_stamp != self.retreat_source_stamp_ns:
                self._invalidate('place retreat source identity changed before success')
            if status == 'succeeded':
                self.state = 'complete'
            return
        if self.state == 'complete':
            if (
                segment == 'place_retreat'
                and status == 'succeeded'
                and command == self.retreat_command_id
                and source_stamp == self.retreat_source_stamp_ns
            ):
                return
            self._invalidate('trajectory status changed after completed retreat')
        self._invalidate('place execution chain is in an invalid state')

    def observe_gripper_command(
        self,
        gripper_command_id: int,
        *,
        executor_epoch: str,
        source_stamp_ns: int,
    ) -> None:
        """Track the executor's monotonic gripper identity during the chain."""
        try:
            command = self._nonnegative_command_id(
                gripper_command_id,
                'gripper',
            )
            source_stamp = self._nonnegative_stamp(
                source_stamp_ns,
                'gripper',
            )
            self._validate_epoch(executor_epoch)
        except PlacementContractError as error:
            self._invalidate(str(error))
            return
        if command < self.latest_gripper_command_id:
            self._invalidate('gripper command identity moved backwards')
        if source_stamp < self.latest_gripper_source_stamp_ns:
            self._invalidate('gripper command source clock moved backwards')
        if command == self.latest_gripper_command_id:
            if source_stamp != self.latest_gripper_source_stamp_ns:
                self._invalidate('gripper command source changed without a new command')
            return
        if source_stamp <= self.latest_gripper_source_stamp_ns:
            self._invalidate('new gripper command did not advance its source time')
        self.latest_gripper_command_id = command
        self.latest_gripper_source_stamp_ns = source_stamp

    def is_new_release(
        self,
        gripper_command_id: int,
        *,
        executor_epoch: str,
        source_stamp_ns: int,
    ) -> bool:
        """Return whether an accepted gripper command strictly follows approach."""
        if not isinstance(executor_epoch, str) or executor_epoch != self.executor_epoch:
            return False
        return bool(
            self.state == 'await_release'
            and self.approach_source_stamp_ns is not None
            and self.pre_release_gripper_command_id is not None
            and self.pre_release_gripper_source_stamp_ns is not None
            and gripper_command_id == self.latest_gripper_command_id
            and gripper_command_id > self.pre_release_gripper_command_id
            and source_stamp_ns == self.latest_gripper_source_stamp_ns
            and source_stamp_ns > self.pre_release_gripper_source_stamp_ns
            and source_stamp_ns > self.approach_source_stamp_ns
        )

    def observe_release(
        self,
        gripper_command_id: int,
        *,
        executor_epoch: str,
        source_stamp_ns: int,
    ) -> None:
        """Bind one nonzero measured release strictly between approach and retreat."""
        if self.state != 'await_release':
            self._invalidate('measured release arrived outside approach/retreat boundary')
        try:
            command = self._command_id(
                gripper_command_id,
                'release gripper',
            )
            source_stamp = _positive_stamp(
                source_stamp_ns,
                'release gripper source',
            )
            self._validate_epoch(executor_epoch)
        except PlacementContractError as error:
            self._invalidate(str(error))
            return
        if not self.is_new_release(
            command,
            executor_epoch=executor_epoch,
            source_stamp_ns=source_stamp,
        ):
            self._invalidate(
                'release gripper command does not strictly follow approach success',
            )
        self.release_gripper_command_id = command
        self.release_source_stamp_ns = source_stamp
        self.state = 'await_retreat_active'


@dataclass(frozen=True, eq=False)
class PostReleaseObservation:
    """One exact synchronized geometry and measured robot-state sample."""

    identity: ObservedPerceptionIdentity
    rgb_stamp_ns: int
    depth_stamp_ns: int
    target_stamp_ns: int
    joint_stamp_ns: int
    execution_status_received_ns: int
    now_ns: int
    geometry_frame_id: str
    organized_points: np.ndarray
    target_points: np.ndarray
    target_pixels_uv: np.ndarray
    gripper_probe_points: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    planning_from_tool_fk: np.ndarray
    planning_from_tool_tf: np.ndarray
    gripper_command_id: int
    gripper_source_stamp_ns: int
    gripper_aperture_m: float


@dataclass(frozen=True)
class PostReleaseVerificationResult:
    """Versioned terminal observed-placement evidence or failure."""

    state: str
    result: str
    failure: str
    goal_id: str
    identity: ObservedPerceptionIdentity
    release_gripper_command_id: int
    release_ack_stamp_ns: int
    observation_start_stamp_ns: int
    first_observation_stamp_ns: int
    last_observation_stamp_ns: int
    first_rgb_stamp_ns: int
    first_depth_stamp_ns: int
    first_target_stamp_ns: int
    last_rgb_stamp_ns: int
    last_depth_stamp_ns: int
    last_target_stamp_ns: int
    last_joint_stamp_ns: int
    last_execution_status_received_ns: int
    sample_count: int
    target_point_count: int
    stable_duration_s: float
    max_target_motion_m: float
    region_support_fraction: float
    target_gripper_clearance_m: float
    target_depth_correspondence_max_error_m: float
    object_position_error_m: float
    object_orientation_error_rad: float
    object_upright_error_rad: float
    object_registration_inlier_fraction: float
    object_registration_rms_m: float
    object_orientation_mode: str
    planned_object_pose: tuple[tuple[float, ...], ...]
    observed_object_center_m: tuple[float, float, float]
    rejected_sample_count: int
    rejected_sample_reasons: tuple[str, ...]
    geometry_frame_id: str

    def __post_init__(self) -> None:
        if self.state not in {'verified', 'failed'}:
            raise PlacementContractError('terminal verification state is invalid')
        expected_result = (
            POST_RELEASE_VERIFICATION_RESULT
            if self.state == 'verified'
            else 'post_release_verification_failed'
        )
        if self.result != expected_result:
            raise PlacementContractError('terminal verification result is inconsistent')
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.goal_id, self.geometry_frame_id)
        ):
            raise PlacementContractError('terminal verification identity is empty')
        if self.state == 'verified' and self.failure:
            raise PlacementContractError('verified terminal result cannot carry a failure')
        if self.state == 'failed' and not str(self.failure).strip():
            raise PlacementContractError('failed terminal result requires a failure')
        if not isinstance(self.identity, ObservedPerceptionIdentity):
            raise PlacementContractError('terminal perception identity is malformed')
        positive_integers = (
            self.release_gripper_command_id,
            self.release_ack_stamp_ns,
            self.observation_start_stamp_ns,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_integers
        ):
            raise PlacementContractError('terminal release identity is malformed')
        nonnegative_integers = (
            self.first_observation_stamp_ns,
            self.last_observation_stamp_ns,
            self.first_rgb_stamp_ns,
            self.first_depth_stamp_ns,
            self.first_target_stamp_ns,
            self.last_rgb_stamp_ns,
            self.last_depth_stamp_ns,
            self.last_target_stamp_ns,
            self.last_joint_stamp_ns,
            self.last_execution_status_received_ns,
            self.sample_count,
            self.target_point_count,
            self.rejected_sample_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in nonnegative_integers
        ):
            raise PlacementContractError('terminal counts or stamps are malformed')
        metrics = (
            self.stable_duration_s,
            self.max_target_motion_m,
            self.region_support_fraction,
            self.target_gripper_clearance_m,
            self.target_depth_correspondence_max_error_m,
            self.object_position_error_m,
            self.object_orientation_error_rad,
            self.object_upright_error_rad,
            self.object_registration_inlier_fraction,
            self.object_registration_rms_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in metrics):
            raise PlacementContractError('terminal metrics must be finite and nonnegative')
        if (
            self.region_support_fraction > 1.0
            or self.object_registration_inlier_fraction > 1.0
        ):
            raise PlacementContractError('terminal fractions cannot exceed one')
        if self.object_orientation_mode not in {'full', 'axial'}:
            raise PlacementContractError('terminal orientation mode is invalid')
        _finite_transform(self.planned_object_pose, 'terminal planned object pose')
        _finite_vector(
            self.observed_object_center_m,
            3,
            'terminal observed object center',
        )
        if any(not isinstance(reason, str) for reason in self.rejected_sample_reasons):
            raise PlacementContractError('terminal rejection diagnostics are malformed')

    def to_payload(self) -> dict[str, object]:
        """Serialize the stable public schema consumed by task acceptance."""
        return {
            'schema': POST_RELEASE_VERIFICATION_SCHEMA,
            'state': self.state,
            'result': self.result,
            'failure': self.failure,
            'observation_source': POST_RELEASE_OBSERVATION_SOURCE,
            'goal_id': self.goal_id,
            'place_goal_id': self.goal_id,
            'release_gripper_command_id': self.release_gripper_command_id,
            'request_id': self.identity.request_id,
            'producer_epoch': self.identity.producer_epoch,
            'generation': self.identity.generation,
            'frame_id': self.identity.frame_id,
            'geometry_frame_id': self.geometry_frame_id,
            'planning_observation_stamp_ns': self.identity.stamp_ns,
            'release_ack_stamp_ns': self.release_ack_stamp_ns,
            'observation_start_stamp_ns': self.observation_start_stamp_ns,
            'first_observation_stamp_ns': self.first_observation_stamp_ns,
            'last_observation_stamp_ns': self.last_observation_stamp_ns,
            'first_status_stamp_ns': self.first_observation_stamp_ns,
            'last_status_stamp_ns': self.last_observation_stamp_ns,
            'first_rgb_stamp_ns': self.first_rgb_stamp_ns,
            'first_depth_stamp_ns': self.first_depth_stamp_ns,
            'first_target_stamp_ns': self.first_target_stamp_ns,
            'last_rgb_stamp_ns': self.last_rgb_stamp_ns,
            'last_depth_stamp_ns': self.last_depth_stamp_ns,
            'last_target_stamp_ns': self.last_target_stamp_ns,
            'last_joint_stamp_ns': self.last_joint_stamp_ns,
            'last_execution_status_received_ns': (
                self.last_execution_status_received_ns
            ),
            'sample_count': self.sample_count,
            'target_point_count': self.target_point_count,
            'stable_duration_s': self.stable_duration_s,
            'max_target_motion_m': self.max_target_motion_m,
            'region_support_fraction': self.region_support_fraction,
            'target_gripper_clearance_m': self.target_gripper_clearance_m,
            'target_depth_correspondence_max_error_m': (
                self.target_depth_correspondence_max_error_m
            ),
            'object_position_error_m': self.object_position_error_m,
            'object_orientation_error_rad': (
                self.object_orientation_error_rad
            ),
            'object_upright_error_rad': self.object_upright_error_rad,
            'object_registration_inlier_fraction': (
                self.object_registration_inlier_fraction
            ),
            'object_registration_rms_m': self.object_registration_rms_m,
            'object_orientation_mode': self.object_orientation_mode,
            'planned_object_pose': self.planned_object_pose,
            'observed_object_center_m': self.observed_object_center_m,
            'rejected_sample_count': self.rejected_sample_count,
            'rejected_sample_reasons': self.rejected_sample_reasons,
        }


@dataclass(frozen=True)
class _PostReleaseSample:
    stamp_ns: int
    center: tuple[float, float, float]
    target_point_count: int
    support_fraction: float
    gripper_clearance_m: float
    rgb_stamp_ns: int
    depth_stamp_ns: int
    target_stamp_ns: int
    joint_stamp_ns: int
    execution_status_received_ns: int
    target_depth_correspondence_max_error_m: float
    object_position_error_m: float
    object_orientation_error_rad: float
    object_upright_error_rad: float
    object_registration_inlier_fraction: float
    object_registration_rms_m: float
    observed_object_rotation: tuple[tuple[float, float, float], ...] | None
    observed_upright_axis: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryWaypoint:
    """Canonical finite joint waypoint in one placement phase."""

    positions: tuple[float, ...]
    time_from_start_s: float
    phase: str


@dataclass(frozen=True, eq=False)
class RawTrajectorySegment:
    """One untrusted planner segment before canonical ordering and time merge."""

    phase: str
    joint_names: tuple[str, ...]
    positions: np.ndarray
    times_s: np.ndarray


@dataclass(frozen=True)
class PlacementTrajectoryContract:
    """Audited transit/approach/retreat trajectory ready for the executor."""

    goal_id: str
    frame_id: str
    joint_names: tuple[str, ...]
    points: tuple[TrajectoryWaypoint, ...]
    phase_start_indices: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.goal_id or not self.frame_id:
            raise PlacementContractError('trajectory goal and frame must be non-empty')
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise PlacementContractError('trajectory joint names must be unique')
        if len(self.points) < 2:
            raise PlacementContractError('placement trajectory needs at least two points')
        previous_time = -1.0
        valid_phases = {'transit', 'approach', 'retreat'}
        for point in self.points:
            positions = _finite_vector(
                point.positions, len(self.joint_names), 'trajectory positions',
            )
            if not np.all(np.isfinite(positions)):
                raise PlacementContractError('trajectory positions are non-finite')
            if (
                not math.isfinite(point.time_from_start_s)
                or point.time_from_start_s <= previous_time
            ):
                raise PlacementContractError('trajectory times must increase strictly')
            if point.phase not in valid_phases:
                raise PlacementContractError(f'unknown trajectory phase {point.phase!r}')
            previous_time = point.time_from_start_s
        starts = dict(self.phase_start_indices)
        if set(starts) != valid_phases:
            raise PlacementContractError('trajectory must declare all three phases')
        if not (0 == starts['transit'] <= starts['approach'] <= starts['retreat']):
            raise PlacementContractError('trajectory phase indices are out of order')
        if starts['retreat'] >= len(self.points):
            raise PlacementContractError('trajectory phase index exceeds point count')


def combine_trajectory_segments(
    *,
    goal_id: str,
    frame_id: str,
    expected_joint_names: Sequence[str],
    start_positions: object,
    segments: Sequence[RawTrajectorySegment],
    continuity_tolerance_rad: float,
) -> PlacementTrajectoryContract:
    """Audit and join transit/approach/retreat planner responses."""
    expected = tuple(expected_joint_names)
    if not expected or len(set(expected)) != len(expected):
        raise PlacementContractError('expected trajectory joints must be unique')
    if tuple(segment.phase for segment in segments) != (
        'transit', 'approach', 'retreat',
    ):
        raise PlacementContractError('planner segments must be transit/approach/retreat')
    if not math.isfinite(continuity_tolerance_rad) or continuity_tolerance_rad <= 0.0:
        raise PlacementContractError('trajectory continuity tolerance must be positive')
    previous = _finite_vector(start_positions, len(expected), 'trajectory start')
    points: list[TrajectoryWaypoint] = []
    phase_starts: list[tuple[str, int]] = []
    global_time = 0.0
    for segment in segments:
        incoming_names = tuple(segment.joint_names)
        if len(incoming_names) != len(set(incoming_names)) or set(incoming_names) != set(expected):
            raise PlacementContractError(
                f'{segment.phase} segment joints do not match the configured arm',
            )
        reorder = [incoming_names.index(name) for name in expected]
        positions = np.asarray(segment.positions, dtype=float)
        times = np.asarray(segment.times_s, dtype=float)
        if (
            positions.ndim != 2 or positions.shape[1] != len(incoming_names)
            or len(positions) < 2
        ):
            raise PlacementContractError(
                f'{segment.phase} segment needs at least two arm waypoints',
            )
        if times.shape != (len(positions),) or not np.all(np.isfinite(times)):
            raise PlacementContractError(f'{segment.phase} segment times are malformed')
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise PlacementContractError(
                f'{segment.phase} segment times must increase strictly',
            )
        positions = positions[:, reorder]
        if not np.all(np.isfinite(positions)):
            raise PlacementContractError(
                f'{segment.phase} segment positions contain non-finite values',
            )
        start_error = float(np.max(np.abs(positions[0] - previous)))
        if start_error > continuity_tolerance_rad:
            raise PlacementContractError(
                f'{segment.phase} starts {start_error:.4f} rad from the prior state',
            )
        phase_starts.append((segment.phase, len(points)))
        first_index = 0 if not points else 1
        for index in range(first_index, len(positions)):
            if not points:
                time_value = float(times[index] - times[0])
            else:
                time_value = global_time + float(times[index] - times[0])
            points.append(TrajectoryWaypoint(
                positions=tuple(float(value) for value in positions[index]),
                time_from_start_s=time_value,
                phase=segment.phase,
            ))
        if phase_starts[-1][1] >= len(points):
            raise PlacementContractError(f'{segment.phase} segment adds no motion')
        global_time = points[-1].time_from_start_s
        previous = positions[-1]
    return PlacementTrajectoryContract(
        goal_id=goal_id,
        frame_id=frame_id,
        joint_names=expected,
        points=tuple(points),
        phase_start_indices=tuple(phase_starts),
    )


@dataclass(frozen=True, eq=False)
class EvaluatedPlacementMotion:
    """Robot-specific IK/collision/path result consumed by the core planner."""

    score: float
    trajectory: PlacementTrajectoryContract

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise PlacementContractError('motion evaluation score must be finite')


@dataclass(frozen=True, eq=False)
class CandidateAudit:
    """Visualization-safe record of every candidate sent to motion planning."""

    candidate: PlacementCandidate
    feasible: bool
    motion_score: float | None
    reason: str


@dataclass(frozen=True, eq=False)
class PlacementOutput:
    """Selected placement, its executable contract, and candidate audit trail."""

    goal_id: str
    result: PlannedPlacement
    trajectory: PlacementTrajectoryContract
    candidates: tuple[CandidateAudit, ...]


def backproject_depth(
    depth_m: object,
    camera_matrix: object,
    transform_target_from_camera: object,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Back-project aligned measured depth into an organized target-frame cloud."""
    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    transform = _finite_transform(
        transform_target_from_camera, 'transform_target_from_camera',
    )
    if depth.ndim != 2:
        raise PlacementContractError('depth image must be two-dimensional')
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise PlacementContractError('camera matrix must be finite 3x3')
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise PlacementContractError('camera focal lengths must be positive')
    if not (
        math.isfinite(min_depth_m) and math.isfinite(max_depth_m)
        and 0.0 <= min_depth_m < max_depth_m
    ):
        raise PlacementContractError('depth interval must be finite and increasing')
    rows, columns = np.indices(depth.shape)
    valid = (
        np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    )
    cloud = np.full((*depth.shape, 3), np.nan, dtype=float)
    z = depth[valid]
    camera_points = np.column_stack((
        (columns[valid] - cx) * z / fx,
        (rows[valid] - cy) * z / fy,
        z,
    ))
    cloud[valid] = (
        camera_points @ transform[:3, :3].T + transform[:3, 3]
    )
    return cloud


def capture_observed_region_geometry(
    organized_points: object,
    region: NormalizedPlacementRegion,
    *,
    frame_id: str,
    plane_origin: object,
    plane_normal: object,
    tangent_u: object,
    tangent_v: object,
    plane_distance_tolerance_m: float,
    min_points: int,
    max_points: int,
) -> ObservedPlacementRegionGeometry:
    """Freeze only measured plane points inside a selected image region."""
    organized = np.asarray(organized_points, dtype=float)
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise PlacementContractError('organized region cloud must have shape (H, W, 3)')
    if (
        not math.isfinite(plane_distance_tolerance_m)
        or plane_distance_tolerance_m <= 0.0
    ):
        raise PlacementContractError('region plane tolerance must be positive')
    if min_points < 3 or max_points < min_points:
        raise PlacementContractError('region geometry point limits are invalid')
    origin = _finite_vector(plane_origin, 3, 'support plane origin')
    normal = _finite_vector(plane_normal, 3, 'support plane normal')
    basis_u = _finite_vector(tangent_u, 3, 'support tangent u')
    basis_v = _finite_vector(tangent_v, 3, 'support tangent v')
    height, width = organized.shape[:2]
    pixel_x = (np.arange(width, dtype=float) + 0.5) / width
    pixel_y = (np.arange(height, dtype=float) + 0.5) / height
    xx, yy = np.meshgrid(pixel_x, pixel_y)
    x1, y1, x2, y2 = region.xyxy
    selected = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    for ax1, ay1, ax2, ay2 in region.avoid_xyxy:
        selected &= ~(
            (xx >= ax1) & (xx <= ax2) & (yy >= ay1) & (yy <= ay2)
        )
    points = organized[selected]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_points:
        raise PlacementContractError('selected region has too few finite depth points')
    distances = np.abs((points - origin) @ normal)
    points = points[distances <= plane_distance_tolerance_m]
    if len(points) < min_points:
        raise PlacementContractError(
            'selected region has too few observed support-plane points',
        )
    coordinates = np.column_stack((
        (points - origin) @ basis_u,
        (points - origin) @ basis_v,
    ))
    coordinates = _deterministic_sample(coordinates, max_points)
    return ObservedPlacementRegionGeometry(
        frame_id=frame_id,
        plane_origin=origin,
        plane_normal=normal,
        tangent_u=basis_u,
        tangent_v=basis_v,
        support_coordinates=coordinates,
    )


def capture_planned_object_geometry(
    reference_points_object: object,
    *,
    frame_id: str,
    expected_object_pose: object,
    support_normal: object,
    verification: PlacementOrientationVerification,
    min_points: int,
    max_points: int,
) -> PlannedObjectGeometry:
    """Bind a frozen grasp-time object model to one expected placement pose."""
    points = np.asarray(reference_points_object, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
        or len(points) < min_points
    ):
        raise PlacementContractError(
            'frozen object reference has insufficient finite observed points',
        )
    if (
        isinstance(min_points, bool)
        or isinstance(max_points, bool)
        or not isinstance(min_points, int)
        or not isinstance(max_points, int)
        or min_points <= 0
        or max_points < min_points
    ):
        raise PlacementContractError('planned object point limits are invalid')
    sampled = _deterministic_sample(points, max_points)
    if len(np.unique(sampled, axis=0)) != len(sampled):
        raise PlacementContractError('frozen object reference points are not unique')
    return PlannedObjectGeometry(
        frame_id=frame_id,
        expected_pose=_finite_transform(
            expected_object_pose,
            'expected_object_pose',
        ),
        reference_points_object=sampled.copy(),
        support_normal=_unit_vector(support_normal, 'support normal'),
        verification=verification,
    )


class PostReleasePlacementVerifier:
    """Prove one release using only synchronized observations and feedback."""

    def __init__(
        self,
        config: PostReleaseVerificationConfig | None = None,
    ) -> None:
        self.config = config or PostReleaseVerificationConfig()
        self.reset()

    def _analyze_object_model(
        self,
        planned: PlannedObjectGeometry,
    ) -> _ObjectModelAnalysis:
        points = np.asarray(planned.reference_points_object, dtype=float)
        if len(points) < self.config.min_object_reference_points:
            raise PlacementContractError(
                'observed object model has insufficient reference points',
            )
        center = np.mean(points, axis=0)
        covariance = (points - center).T @ (points - center) / len(points)
        eigenvalues, axes = np.linalg.eigh(covariance)
        if (
            not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 1e-12
            or eigenvalues[-1] <= 1e-10
        ):
            raise PlacementContractError('observed object model is geometrically degenerate')
        ratio = self.config.orientation_degeneracy_ratio
        requested = planned.verification.orientation_symmetry
        symmetry_axis = None
        symmetry_index = None

        def axial_structure(index: int) -> bool:
            other = [item for item in range(3) if item != index]
            transverse_ratio = max(eigenvalues[other]) / min(eigenvalues[other])
            separated = all(
                max(eigenvalues[index], eigenvalues[item])
                / min(eigenvalues[index], eigenvalues[item]) >= ratio
                for item in other
            )
            return bool(
                transverse_ratio <= self.config.max_axial_transverse_ratio
                and separated
            )

        full_observable = bool(
            eigenvalues[1] / eigenvalues[0] >= ratio
            and eigenvalues[2] / eigenvalues[1] >= ratio
        )
        if requested == 'auto':
            axial_candidates = [
                index for index in (0, 2) if axial_structure(index)
            ]
            if len(axial_candidates) == 1:
                orientation_mode = 'axial'
                symmetry_index = axial_candidates[0]
                symmetry_axis = axes[:, symmetry_index]
            elif full_observable:
                orientation_mode = 'full'
            else:
                raise PlacementContractError(
                    'observed object orientation is fully degenerate or ambiguous',
                )
        elif requested == 'none':
            if not full_observable:
                raise PlacementContractError(
                    'full object orientation is not observable from degenerate geometry',
                )
            orientation_mode = 'full'
        else:
            orientation_mode = 'axial'
            assert planned.verification.symmetry_axis_object is not None
            symmetry_axis = _unit_vector(
                planned.verification.symmetry_axis_object,
                'symmetry_axis_object',
            )
            alignments = np.abs(axes.T @ symmetry_axis)
            symmetry_index = int(np.argmax(alignments))
            alignment_error = math.acos(float(np.clip(
                alignments[symmetry_index],
                0.0,
                1.0,
            )))
            if alignment_error > self.config.max_symmetry_axis_alignment_error_rad:
                raise PlacementContractError(
                    'declared symmetry axis is not observable in reference geometry',
                )
            if not axial_structure(symmetry_index):
                raise PlacementContractError(
                    'declared axial geometry lacks equal transverse variances '
                    'or a separated symmetry axis',
                )

        upright_axis = _unit_vector(
            planned.verification.upright_axis_object,
            'upright_axis_object',
        )
        upright_eigen_index = None
        if (
            orientation_mode == 'axial'
            and planned.verification.require_upright
            and symmetry_axis is not None
            and abs(float(np.dot(symmetry_axis, upright_axis)))
            < math.cos(self.config.max_symmetry_axis_alignment_error_rad)
        ):
            raise PlacementContractError(
                'upright axis is not observable under declared axial symmetry',
            )
        if orientation_mode == 'axial':
            upright_eigen_index = symmetry_index

        signed_upright_profile = None
        if planned.verification.require_upright:
            signed_upright_profile = _signed_axis_profile(points, upright_axis)
            if (
                np.linalg.norm(signed_upright_profile)
                < self.config.min_signed_upright_profile_asymmetry
            ):
                raise PlacementContractError(
                    'upright direction is not observable from frozen object geometry',
                )
        expected_upright = planned.expected_pose[:3, :3] @ upright_axis
        planned_upright_error = math.acos(float(np.clip(np.dot(
            expected_upright,
            planned.support_normal,
        ), -1.0, 1.0)))
        if (
            planned.verification.require_upright
            and planned_upright_error > self.config.upright_tolerance_rad
        ):
            raise PlacementContractError(
                'planned object pose violates the requested upright orientation',
            )
        return _ObjectModelAnalysis(
            orientation_mode=orientation_mode,
            reference_center_object=center,
            reference_axes_object=axes,
            reference_eigenvalues=eigenvalues,
            symmetry_axis_object=symmetry_axis,
            symmetry_eigen_index=symmetry_index,
            upright_axis_object=upright_axis,
            upright_eigen_index=upright_eigen_index,
            signed_upright_profile=signed_upright_profile,
        )

    def reset(self) -> None:
        """Clear all ownership and samples before a different placement goal."""
        self.state = 'idle'
        self.goal_id = ''
        self.identity: ObservedPerceptionIdentity | None = None
        self.region: ObservedPlacementRegionGeometry | None = None
        self.planned_object: PlannedObjectGeometry | None = None
        self.object_model: _ObjectModelAnalysis | None = None
        self.expected_joint_names: tuple[str, ...] = ()
        self.release_gripper_command_id: int | None = None
        self.release_gripper_source_stamp_ns: int | None = None
        self.release_ack_stamp_ns: int | None = None
        self.observation_start_stamp_ns: int | None = None
        self.samples: list[_PostReleaseSample] = []
        self.last_now_ns: int | None = None
        self.last_source_stamp_ns: int | None = None
        self.last_observation_received_ns: int | None = None
        self.rejected_sample_count = 0
        self.rejected_sample_reasons: list[str] = []
        self.terminal: PostReleaseVerificationResult | None = None

    @property
    def observing(self) -> bool:
        return self.state == 'observing'

    @property
    def armed(self) -> bool:
        return self.state in {'armed', 'observing'}

    def arm(
        self,
        *,
        goal_id: str,
        identity: ObservedPerceptionIdentity,
        region: ObservedPlacementRegionGeometry,
        planned_object: PlannedObjectGeometry,
        expected_joint_names: Sequence[str],
    ) -> None:
        """Bind one planned goal to its exact perception producer identity."""
        if not isinstance(goal_id, str) or not goal_id.strip():
            raise PlacementContractError('post-release goal_id is empty')
        if not isinstance(identity, ObservedPerceptionIdentity):
            raise PlacementContractError('post-release perception identity is invalid')
        if not isinstance(region, ObservedPlacementRegionGeometry):
            raise PlacementContractError('post-release observed region is invalid')
        if not isinstance(planned_object, PlannedObjectGeometry):
            raise PlacementContractError('post-release planned object is invalid')
        if planned_object.frame_id != region.frame_id:
            raise PlacementContractError(
                'planned object and observed region frames do not match',
            )
        joint_names = tuple(expected_joint_names)
        if (
            not joint_names
            or any(not isinstance(name, str) or not name.strip() for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise PlacementContractError(
                'post-release expected joint names must be non-empty and unique',
            )
        model = self._analyze_object_model(planned_object)
        self.reset()
        self.state = 'armed'
        self.goal_id = goal_id
        self.identity = identity
        self.region = region
        self.planned_object = planned_object
        self.object_model = model
        self.expected_joint_names = joint_names

    def begin_release(
        self,
        *,
        gripper_command_id: int,
        gripper_source_stamp_ns: int,
        acknowledgement_stamp_ns: int,
        observation_start_stamp_ns: int | None = None,
    ) -> None:
        """Open the observation window after measured release acknowledgement."""
        if self.state != 'armed' or self.identity is None:
            raise PlacementContractError('post-release verifier is not armed')
        if (
            isinstance(gripper_command_id, bool)
            or not isinstance(gripper_command_id, int)
            or gripper_command_id <= 0
        ):
            raise PlacementContractError('release gripper command ID is invalid')
        stamp = _positive_stamp(acknowledgement_stamp_ns, 'release acknowledgement')
        source_stamp = _positive_stamp(
            gripper_source_stamp_ns,
            'release gripper source',
        )
        if stamp <= self.identity.stamp_ns:
            raise PlacementContractError(
                'release acknowledgement predates its planning observation',
            )
        observation_start = (
            stamp
            if observation_start_stamp_ns is None
            else _positive_stamp(
                observation_start_stamp_ns,
                'post-release observation start',
            )
        )
        if observation_start < stamp:
            raise PlacementContractError(
                'post-release observation starts before release acknowledgement',
            )
        self.release_gripper_command_id = gripper_command_id
        self.release_gripper_source_stamp_ns = source_stamp
        self.release_ack_stamp_ns = stamp
        self.observation_start_stamp_ns = observation_start
        self.last_now_ns = observation_start
        self.last_observation_received_ns = observation_start
        self.state = 'observing'

    def _terminal_result(
        self,
        *,
        state: str,
        result: str,
        failure: str,
    ) -> PostReleaseVerificationResult:
        assert self.identity is not None
        assert self.region is not None
        assert self.planned_object is not None
        assert self.object_model is not None
        assert self.release_gripper_command_id is not None
        assert self.release_gripper_source_stamp_ns is not None
        assert self.release_ack_stamp_ns is not None
        assert self.observation_start_stamp_ns is not None
        first = self.samples[0] if self.samples else None
        last = self.samples[-1] if self.samples else None
        centers = np.asarray([sample.center for sample in self.samples], dtype=float)
        max_motion = 0.0
        if len(centers) > 1:
            differences = centers[:, None, :] - centers[None, :, :]
            max_motion = float(np.max(np.linalg.norm(differences, axis=2)))
        terminal = PostReleaseVerificationResult(
            state=state,
            result=result,
            failure=failure,
            goal_id=self.goal_id,
            identity=self.identity,
            release_gripper_command_id=self.release_gripper_command_id,
            release_ack_stamp_ns=self.release_ack_stamp_ns,
            observation_start_stamp_ns=self.observation_start_stamp_ns,
            first_observation_stamp_ns=0 if first is None else first.stamp_ns,
            last_observation_stamp_ns=0 if last is None else last.stamp_ns,
            first_rgb_stamp_ns=0 if first is None else first.rgb_stamp_ns,
            first_depth_stamp_ns=0 if first is None else first.depth_stamp_ns,
            first_target_stamp_ns=0 if first is None else first.target_stamp_ns,
            last_rgb_stamp_ns=0 if last is None else last.rgb_stamp_ns,
            last_depth_stamp_ns=0 if last is None else last.depth_stamp_ns,
            last_target_stamp_ns=0 if last is None else last.target_stamp_ns,
            last_joint_stamp_ns=0 if last is None else last.joint_stamp_ns,
            last_execution_status_received_ns=(
                0 if last is None else last.execution_status_received_ns
            ),
            sample_count=len(self.samples),
            target_point_count=(
                0 if not self.samples
                else min(sample.target_point_count for sample in self.samples)
            ),
            stable_duration_s=(
                0.0 if first is None or last is None
                else (last.stamp_ns - first.stamp_ns) * 1e-9
            ),
            max_target_motion_m=max_motion,
            region_support_fraction=(
                0.0 if not self.samples
                else min(sample.support_fraction for sample in self.samples)
            ),
            target_gripper_clearance_m=(
                0.0 if not self.samples
                else min(sample.gripper_clearance_m for sample in self.samples)
            ),
            target_depth_correspondence_max_error_m=(
                0.0 if not self.samples
                else max(
                    sample.target_depth_correspondence_max_error_m
                    for sample in self.samples
                )
            ),
            object_position_error_m=(
                0.0 if not self.samples
                else max(sample.object_position_error_m for sample in self.samples)
            ),
            object_orientation_error_rad=(
                0.0 if not self.samples
                else max(
                    sample.object_orientation_error_rad
                    for sample in self.samples
                )
            ),
            object_upright_error_rad=(
                0.0 if not self.samples
                else max(
                    sample.object_upright_error_rad
                    for sample in self.samples
                )
            ),
            object_registration_inlier_fraction=(
                0.0 if not self.samples
                else min(
                    sample.object_registration_inlier_fraction
                    for sample in self.samples
                )
            ),
            object_registration_rms_m=(
                0.0 if not self.samples
                else max(
                    sample.object_registration_rms_m
                    for sample in self.samples
                )
            ),
            object_orientation_mode=self.object_model.orientation_mode,
            planned_object_pose=tuple(
                tuple(float(value) for value in row)
                for row in self.planned_object.expected_pose
            ),
            observed_object_center_m=(
                (0.0, 0.0, 0.0) if last is None else last.center
            ),
            rejected_sample_count=self.rejected_sample_count,
            rejected_sample_reasons=tuple(self.rejected_sample_reasons),
            geometry_frame_id=self.region.frame_id,
        )
        self.state = state
        self.terminal = terminal
        return terminal

    def fail(self, reason: str) -> PostReleaseVerificationResult | None:
        """Fail the active release once; idle or already-terminal sessions ignore it."""
        if self.state != 'observing':
            return None
        detail = str(reason).strip() or 'unspecified_post_release_failure'
        return self._terminal_result(
            state='failed',
            result='post_release_verification_failed',
            failure=detail,
        )

    def reject_sample(
        self,
        reason: str,
        *,
        now_ns: int,
    ) -> PostReleaseVerificationResult | None:
        """Drop one recoverable observation without extending the valid timeout."""
        if self.state != 'observing':
            return None
        try:
            now = _positive_stamp(now_ns, 'post-release clock')
        except PlacementContractError as error:
            return self.fail(str(error))
        if self.last_now_ns is not None and now < self.last_now_ns:
            return self.fail('post-release clock moved backwards')
        self.last_now_ns = now
        self.samples.clear()
        self.rejected_sample_count += 1
        detail = str(reason).strip() or 'unspecified recoverable sample rejection'
        self.rejected_sample_reasons.append(detail[:256])
        if len(self.rejected_sample_reasons) > self.config.max_rejection_diagnostics:
            self.rejected_sample_reasons.pop(0)
        return None

    def tick(self, now_ns: int) -> PostReleaseVerificationResult | None:
        """Detect ROS clock rollback and target-observation timeout."""
        if self.state != 'observing':
            return None
        try:
            now = _positive_stamp(now_ns, 'post-release clock')
        except PlacementContractError as error:
            return self.fail(str(error))
        if self.last_now_ns is not None and now < self.last_now_ns:
            return self.fail('post-release clock moved backwards')
        self.last_now_ns = now
        assert self.last_observation_received_ns is not None
        timeout_ns = int(self.config.observation_timeout_s * 1e9)
        if now - self.last_observation_received_ns > timeout_ns:
            return self.fail('post-release target observation timed out or is occluded')
        return None

    def _validate_observation(self, observation: PostReleaseObservation) -> None:
        assert self.identity is not None
        assert self.region is not None
        assert self.release_gripper_command_id is not None
        assert self.release_ack_stamp_ns is not None
        assert self.observation_start_stamp_ns is not None
        if not isinstance(observation.identity, ObservedPerceptionIdentity):
            raise PlacementContractError('post-release observation identity is malformed')
        expected_owner = (
            self.identity.request_id,
            self.identity.producer_epoch,
            self.identity.generation,
            self.identity.frame_id,
        )
        observed_owner = (
            observation.identity.request_id,
            observation.identity.producer_epoch,
            observation.identity.generation,
            observation.identity.frame_id,
        )
        if observed_owner != expected_owner:
            raise PlacementContractError(
                'post-release observation mixed request/producer/generation/frame identity',
            )
        if observation.geometry_frame_id != self.region.frame_id:
            raise PlacementContractError('post-release geometry frame changed')
        source_stamps = (
            _positive_stamp(observation.rgb_stamp_ns, 'post-release RGB'),
            _positive_stamp(observation.depth_stamp_ns, 'post-release depth'),
            _positive_stamp(observation.target_stamp_ns, 'post-release target'),
            _positive_stamp(observation.identity.stamp_ns, 'post-release status'),
        )
        if min(source_stamps) <= self.observation_start_stamp_ns:
            raise PlacementContractError(
                'post-release RGB/depth/target/status must all be newer '
                'than observation start',
            )
        if observation.target_stamp_ns != observation.identity.stamp_ns:
            raise PlacementContractError(
                'post-release target/status source stamps do not match exactly',
            )
        if max(source_stamps) - min(source_stamps) > int(
            self.config.max_sync_skew_s * 1e9
        ):
            raise PlacementContractError(
                'post-release RGB/depth/target/status are not synchronized',
            )
        now = _positive_stamp(observation.now_ns, 'post-release clock')
        if self.last_now_ns is not None and now < self.last_now_ns:
            raise PlacementContractError('post-release clock moved backwards')
        if now < max(source_stamps):
            raise PlacementContractError('post-release source stamp is in the future')
        if now - max(source_stamps) > int(self.config.max_observation_age_s * 1e9):
            raise PlacementContractError('post-release observation is stale')
        if observation.identity.stamp_ns <= self.release_ack_stamp_ns:
            raise PlacementContractError('post-release observation predates release')
        joint_stamp = _positive_stamp(observation.joint_stamp_ns, 'arm state')
        if joint_stamp <= self.observation_start_stamp_ns:
            raise PlacementContractError(
                'post-release measured arm state predates observation start',
            )
        if abs(joint_stamp - observation.identity.stamp_ns) > int(
            self.config.max_state_skew_s * 1e9
        ):
            raise PlacementContractError('post-release arm state is not synchronized')
        execution_stamp = _positive_stamp(
            observation.execution_status_received_ns,
            'gripper status receipt',
        )
        if (
            execution_stamp < self.observation_start_stamp_ns
            or now < execution_stamp
            or now - execution_stamp > int(self.config.max_state_skew_s * 1e9)
        ):
            raise PlacementContractError('post-release gripper feedback is stale')
        names = tuple(observation.joint_names)
        positions = np.asarray(observation.joint_positions, dtype=float)
        if (
            names != self.expected_joint_names
            or len(set(names)) != len(names)
            or positions.shape != (len(names),)
            or not np.all(np.isfinite(positions))
        ):
            raise PlacementContractError('post-release measured arm state is malformed')
        measured_fk = _finite_transform(
            observation.planning_from_tool_fk,
            'post-release FK tool pose',
        )
        measured_tf = _finite_transform(
            observation.planning_from_tool_tf,
            'post-release TF tool pose',
        )
        fk_position_error = float(np.linalg.norm(
            measured_fk[:3, 3] - measured_tf[:3, 3],
        ))
        fk_orientation_error = _rotation_angle(
            measured_fk[:3, :3].T @ measured_tf[:3, :3],
        )
        if (
            fk_position_error > self.config.fk_position_tolerance_m
            or fk_orientation_error > self.config.fk_orientation_tolerance_rad
        ):
            raise PlacementContractError(
                'post-release measured joints disagree with stamped tool TF',
            )
        if observation.gripper_command_id != self.release_gripper_command_id:
            raise PlacementContractError('post-release gripper command ID changed')
        if (
            _positive_stamp(
                observation.gripper_source_stamp_ns,
                'post-release gripper source',
            )
            != self.release_gripper_source_stamp_ns
        ):
            raise PlacementContractError('post-release gripper source identity changed')
        aperture = float(observation.gripper_aperture_m)
        if not math.isfinite(aperture) or aperture < self.config.min_gripper_aperture_m:
            raise PlacementContractError('post-release gripper is not measured open')

    @staticmethod
    def _registration_metrics(
        observed: np.ndarray,
        reference: np.ndarray,
        threshold: float,
    ) -> tuple[float, float]:
        """Return conservative bidirectional registration metrics."""
        forward = _nearest_spatial_distances(observed, reference)
        reverse = _nearest_spatial_distances(reference, observed)
        forward_inliers = forward <= threshold
        reverse_inliers = reverse <= threshold
        fraction = min(
            float(np.mean(forward_inliers)),
            float(np.mean(reverse_inliers)),
        )
        directional_rms = []
        for distances, inliers in (
            (forward, forward_inliers),
            (reverse, reverse_inliers),
        ):
            directional_rms.append(
                float('inf')
                if not np.any(inliers)
                else float(np.sqrt(np.mean(distances[inliers] ** 2)))
            )
        return fraction, max(directional_rms)

    @staticmethod
    def _axial_coordinates(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
        axial = points @ axis
        radial = np.linalg.norm(points - axial[:, None] * axis, axis=1)
        return np.column_stack((axial, radial, np.zeros(len(points))))

    def _signed_upright_profile_alignment(
        self,
        model: _ObjectModelAnalysis,
        points: np.ndarray,
        candidate_axis: np.ndarray,
    ) -> float:
        reference_profile = model.signed_upright_profile
        if reference_profile is None:
            raise PlacementContractError(
                'frozen signed upright evidence is unavailable',
            )
        observed_profile = _signed_axis_profile(points, candidate_axis)
        reference_norm = float(np.linalg.norm(reference_profile))
        observed_norm = float(np.linalg.norm(observed_profile))
        if (
            reference_norm < self.config.min_signed_upright_profile_asymmetry
            or observed_norm < self.config.min_signed_upright_profile_asymmetry
        ):
            raise PlacementContractError(
                'observed upright direction is not observable from target geometry',
            )
        return float(np.clip(
            np.dot(reference_profile, observed_profile)
            / (reference_norm * observed_norm),
            -1.0,
            1.0,
        ))

    def _object_pose_metrics_for(
        self,
        planned: PlannedObjectGeometry,
        model: _ObjectModelAnalysis,
        targets: np.ndarray,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
    ]:
        target_points = np.asarray(targets, dtype=float)
        if (
            target_points.ndim != 2
            or target_points.shape[1] != 3
            or len(target_points) < self.config.min_object_reference_points
            or not np.all(np.isfinite(target_points))
        ):
            raise PlacementContractError(
                'observed object pose has insufficient finite target points',
            )
        expected_pose = np.asarray(planned.expected_pose, dtype=float)
        expected_rotation = expected_pose[:3, :3]
        reference = np.asarray(planned.reference_points_object, dtype=float)
        reference_center = np.asarray(model.reference_center_object, dtype=float)
        expected_center = (
            expected_rotation @ reference_center + expected_pose[:3, 3]
        )
        observed_center = np.mean(target_points, axis=0)
        position_error = float(np.linalg.norm(observed_center - expected_center))
        if position_error > self.config.object_position_tolerance_m:
            raise PlacementContractError(
                'observed object center does not match expected pose',
            )

        centered = target_points - observed_center
        covariance = centered.T @ centered / len(centered)
        observed_values, observed_axes = np.linalg.eigh(covariance)
        if (
            not np.all(np.isfinite(observed_values))
            or observed_values[0] <= 1e-12
            or observed_values[-1] <= 1e-10
        ):
            raise PlacementContractError(
                'observed object orientation is geometrically degenerate',
            )
        ratio = self.config.orientation_degeneracy_ratio
        upright_axis_object = _unit_vector(
            planned.verification.upright_axis_object,
            'upright_axis_object',
        )
        observed_rotation: np.ndarray | None
        if model.orientation_mode == 'full':
            if (
                observed_values[1] / observed_values[0] < ratio
                or observed_values[2] / observed_values[1] < ratio
            ):
                raise PlacementContractError(
                    'observed full orientation became degenerate',
                )
            candidates = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        rotation = (
                            observed_axes
                            @ np.diag((sx, sy, sz))
                            @ model.reference_axes_object.T
                        )
                        if np.linalg.det(rotation) <= 0.0:
                            continue
                        candidate_upright = _unit_vector(
                            rotation @ upright_axis_object,
                            'candidate observed upright axis',
                        )
                        if planned.verification.require_upright:
                            try:
                                signed_alignment = (
                                    self._signed_upright_profile_alignment(
                                        model,
                                        target_points,
                                        candidate_upright,
                                    )
                                )
                            except PlacementContractError:
                                continue
                            if (
                                signed_alignment
                                < self.config.min_signed_upright_profile_alignment
                            ):
                                continue
                        candidates.append((
                            _rotation_angle(expected_rotation.T @ rotation),
                            rotation,
                            candidate_upright,
                        ))
            if not candidates:
                if planned.verification.require_upright:
                    raise PlacementContractError(
                        'observed upright direction is not observable from '
                        'the frozen signed profile',
                    )
                raise PlacementContractError(
                    'observed full orientation has no proper registration',
                )
            orientation_error, observed_rotation, observed_upright = min(
                candidates,
                key=lambda item: item[0],
            )
            predicted = reference @ expected_rotation.T + expected_pose[:3, 3]
            inlier_fraction, registration_rms = self._registration_metrics(
                target_points,
                predicted,
                self.config.registration_distance_m,
            )
        else:
            assert model.symmetry_eigen_index is not None
            assert model.symmetry_axis_object is not None
            index = model.symmetry_eigen_index
            other = [item for item in range(3) if item != index]
            transverse_ratio = (
                max(observed_values[other]) / min(observed_values[other])
            )
            separated = all(
                max(observed_values[index], observed_values[item])
                / min(observed_values[index], observed_values[item]) >= ratio
                for item in other
            )
            if (
                transverse_ratio > self.config.max_axial_transverse_ratio
                or not separated
            ):
                raise PlacementContractError(
                    'observed axial geometry lost transverse equality or axis separation',
                )
            expected_axis = _unit_vector(
                expected_rotation @ model.symmetry_axis_object,
                'planned symmetry axis',
            )
            observed_axis = _unit_vector(
                observed_axes[:, index],
                'observed symmetry axis',
            )
            if planned.verification.require_upright:
                signed_alignment = self._signed_upright_profile_alignment(
                    model,
                    target_points,
                    observed_axis,
                )
                if (
                    abs(signed_alignment)
                    < self.config.min_signed_upright_profile_alignment
                ):
                    raise PlacementContractError(
                        'observed upright direction does not preserve the '
                        'frozen signed profile',
                    )
                observed_upright = (
                    observed_axis if signed_alignment >= 0.0 else -observed_axis
                )
                expected_upright = _unit_vector(
                    expected_rotation @ upright_axis_object,
                    'planned upright axis',
                )
                orientation_error = math.acos(float(np.clip(np.dot(
                    expected_upright,
                    observed_upright,
                ), -1.0, 1.0)))
            else:
                orientation_error = math.acos(float(np.clip(abs(np.dot(
                    expected_axis,
                    observed_axis,
                )), 0.0, 1.0)))
                observed_upright = observed_axis
            observed_rotation = None
            observed_object = (
                target_points - expected_pose[:3, 3]
            ) @ expected_rotation
            inlier_fraction, registration_rms = self._registration_metrics(
                self._axial_coordinates(
                    observed_object,
                    model.symmetry_axis_object,
                ),
                self._axial_coordinates(
                    reference,
                    model.symmetry_axis_object,
                ),
                self.config.registration_distance_m,
            )
        if orientation_error > self.config.object_orientation_tolerance_rad:
            raise PlacementContractError(
                'observed object orientation does not match expected pose',
            )
        upright_dot = float(np.dot(
            _unit_vector(observed_upright, 'observed upright axis'),
            planned.support_normal,
        ))
        if not planned.verification.require_upright:
            upright_dot = abs(upright_dot)
        upright_error = math.acos(float(np.clip(upright_dot, -1.0, 1.0)))
        if (
            planned.verification.require_upright
            and upright_error > self.config.upright_tolerance_rad
        ):
            raise PlacementContractError(
                'observed object violates requested upright orientation',
            )
        if (
            inlier_fraction < self.config.min_registration_inlier_fraction
            or not math.isfinite(registration_rms)
            or registration_rms > self.config.max_registration_rms_m
        ):
            raise PlacementContractError(
                'observed object-to-model registration is insufficient',
            )
        return (
            position_error,
            float(orientation_error),
            float(upright_error),
            inlier_fraction,
            registration_rms,
            observed_center,
            observed_rotation,
            _unit_vector(observed_upright, 'observed upright axis'),
        )

    def validate_observed_object_pose(
        self,
        planned: PlannedObjectGeometry,
        target_points: object,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
    ]:
        """Validate current target geometry against the frozen grasp model."""
        if not isinstance(planned, PlannedObjectGeometry):
            raise PlacementContractError('planned object geometry is malformed')
        return self._object_pose_metrics_for(
            planned,
            self._analyze_object_model(planned),
            np.asarray(target_points, dtype=float),
        )

    def _object_pose_metrics(
        self,
        targets: np.ndarray,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
    ]:
        assert self.planned_object is not None
        assert self.object_model is not None
        return self._object_pose_metrics_for(
            self.planned_object,
            self.object_model,
            targets,
        )

    def _geometry_metrics(
        self,
        observation: PostReleaseObservation,
    ) -> tuple[
        np.ndarray,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray | None,
        np.ndarray,
    ]:
        assert self.region is not None
        organized = np.asarray(observation.organized_points, dtype=float)
        targets_all = np.asarray(observation.target_points, dtype=float)
        pixels = np.asarray(observation.target_pixels_uv, dtype=float)
        probes = np.asarray(observation.gripper_probe_points, dtype=float)
        if organized.ndim != 3 or organized.shape[2] != 3:
            raise PlacementContractError('post-release RGB-D cloud is malformed')
        if (
            targets_all.ndim != 2
            or targets_all.shape[1] != 3
            or not np.all(np.isfinite(targets_all))
            or len(targets_all) < self.config.min_target_points
        ):
            raise PlacementContractError('post-release target has insufficient finite points')
        if (
            pixels.shape != (len(targets_all), 2)
            or not np.all(np.isfinite(pixels))
            or not np.allclose(pixels, np.rint(pixels), atol=1e-5)
        ):
            raise PlacementContractError('post-release target pixel correspondence is invalid')
        if probes.ndim != 2 or probes.shape[1] != 3 or not np.all(np.isfinite(probes)):
            raise PlacementContractError('post-release gripper probes are malformed')
        if len(probes) == 0:
            raise PlacementContractError('post-release gripper has no measured probes')
        height, width = organized.shape[:2]
        integer_pixels = np.rint(pixels).astype(int)
        if (
            np.any(integer_pixels[:, 0] < 0)
            or np.any(integer_pixels[:, 0] >= width)
            or np.any(integer_pixels[:, 1] < 0)
            or np.any(integer_pixels[:, 1] >= height)
        ):
            raise PlacementContractError('post-release target pixels exceed RGB-D image')
        if len(np.unique(integer_pixels, axis=0)) != len(integer_pixels):
            raise PlacementContractError('post-release target pixels are not unique')
        organized_targets = organized[
            integer_pixels[:, 1],
            integer_pixels[:, 0],
        ]
        if not np.all(np.isfinite(organized_targets)):
            raise PlacementContractError(
                'post-release target pixels lack measured organized depth',
            )
        correspondence_errors = np.linalg.norm(
            organized_targets - targets_all,
            axis=1,
        )
        correspondence_max_error = float(np.max(correspondence_errors))
        if (
            correspondence_max_error
            > self.config.target_depth_correspondence_tolerance_m
        ):
            raise PlacementContractError(
                'post-release target xyz does not match organized depth',
            )

        targets = _deterministic_sample(
            targets_all,
            self.config.max_geometry_samples,
        )
        origin = np.asarray(self.region.plane_origin, dtype=float)
        normal = np.asarray(self.region.plane_normal, dtype=float)
        tangent_u = np.asarray(self.region.tangent_u, dtype=float)
        tangent_v = np.asarray(self.region.tangent_v, dtype=float)
        support_reference = np.asarray(self.region.support_coordinates, dtype=float)
        target_offsets = targets - origin
        target_coordinates = np.column_stack((
            target_offsets @ tangent_u,
            target_offsets @ tangent_v,
        ))
        inside_region = _nearest_planar_distances(
            target_coordinates,
            support_reference,
        ) <= self.config.region_neighbor_radius_m
        region_fraction = float(np.mean(inside_region))

        occlusion_mask = np.zeros((height, width), dtype=bool)
        dilation = self.config.target_mask_dilation_px
        for du in range(-dilation, dilation + 1):
            for dv in range(-dilation, dilation + 1):
                uu = integer_pixels[:, 0] + du
                vv = integer_pixels[:, 1] + dv
                valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                occlusion_mask[vv[valid], uu[valid]] = True
        current = organized[~occlusion_mask]
        current = current[np.all(np.isfinite(current), axis=1)]
        if len(current) < self.config.min_current_support_points:
            raise PlacementContractError('post-release support is occluded')
        current_offsets = current - origin
        plane_distance = np.abs(current_offsets @ normal)
        current = current[plane_distance <= self.config.plane_distance_tolerance_m]
        current_offsets = current - origin
        current_coordinates = np.column_stack((
            current_offsets @ tangent_u,
            current_offsets @ tangent_v,
        ))
        lower = np.min(support_reference, axis=0) - self.config.region_neighbor_radius_m
        upper = np.max(support_reference, axis=0) + self.config.region_neighbor_radius_m
        bounded = np.all((current_coordinates >= lower) & (current_coordinates <= upper), axis=1)
        current_coordinates = current_coordinates[bounded]
        current_coordinates = _deterministic_sample(
            current_coordinates,
            self.config.max_geometry_samples,
        )
        current_coordinates = current_coordinates[
            _nearest_planar_distances(
                current_coordinates,
                support_reference,
            ) <= self.config.region_neighbor_radius_m
        ]
        if len(current_coordinates) < self.config.min_current_support_points:
            raise PlacementContractError(
                'post-release selected region has insufficient current support points',
            )

        target_heights = target_offsets @ normal
        bottom_height = float(np.percentile(target_heights, 2.5))
        if (
            bottom_height < -self.config.max_plane_penetration_m
            or bottom_height > self.config.max_bottom_height_m
        ):
            raise PlacementContractError('post-release target is not resting on support plane')
        bottom = target_heights <= bottom_height + self.config.bottom_band_m
        if np.count_nonzero(bottom) < 3:
            raise PlacementContractError('post-release target bottom is under-observed')
        bottom_supported = _nearest_planar_distances(
            target_coordinates[bottom],
            current_coordinates,
        ) <= self.config.support_neighbor_radius_m
        support_fraction = min(region_fraction, float(np.mean(bottom_supported)))
        if support_fraction < self.config.min_region_support_fraction:
            raise PlacementContractError(
                'post-release target support inside selected region is insufficient',
            )

        sampled_probes = _deterministic_sample(
            probes,
            self.config.max_geometry_samples,
        )
        clearance = float(np.min(np.linalg.norm(
            targets[:, None, :] - sampled_probes[None, :, :],
            axis=2,
        ))) - self.config.gripper_probe_radius_m
        if clearance < self.config.min_gripper_clearance_m:
            raise PlacementContractError('post-release gripper has not cleared target')
        (
            position_error,
            orientation_error,
            upright_error,
            registration_inlier_fraction,
            registration_rms,
            center,
            observed_rotation,
            observed_upright_axis,
        ) = self._object_pose_metrics(targets)
        return (
            center,
            len(targets_all),
            support_fraction,
            clearance,
            correspondence_max_error,
            position_error,
            orientation_error,
            upright_error,
            registration_inlier_fraction,
            registration_rms,
            observed_rotation,
            observed_upright_axis,
        )

    def observe(
        self,
        observation: PostReleaseObservation,
    ) -> PostReleaseVerificationResult | None:
        """Consume one fresh exact frame and verify only a continuous stable window."""
        if self.state != 'observing':
            return None
        try:
            self._validate_observation(observation)
            source_stamp = observation.identity.stamp_ns
            if (
                self.last_source_stamp_ns is not None
                and source_stamp < self.last_source_stamp_ns
            ):
                raise PlacementContractError(
                    'post-release observation source clock moved backwards',
                )
            if source_stamp == self.last_source_stamp_ns:
                self.last_now_ns = observation.now_ns
                return None
            self.last_source_stamp_ns = source_stamp
        except (PlacementContractError, TypeError, ValueError) as error:
            return self.fail(str(error))
        try:
            (
                center,
                point_count,
                support_fraction,
                clearance,
                correspondence_max_error,
                object_position_error,
                object_orientation_error,
                object_upright_error,
                object_registration_inlier_fraction,
                object_registration_rms,
                observed_rotation,
                observed_upright_axis,
            ) = (
                self._geometry_metrics(observation)
            )
        except (PlacementContractError, TypeError, ValueError) as error:
            return self.reject_sample(str(error), now_ns=observation.now_ns)

        sample = _PostReleaseSample(
            stamp_ns=source_stamp,
            center=tuple(float(value) for value in center),
            target_point_count=point_count,
            support_fraction=support_fraction,
            gripper_clearance_m=clearance,
            rgb_stamp_ns=observation.rgb_stamp_ns,
            depth_stamp_ns=observation.depth_stamp_ns,
            target_stamp_ns=observation.target_stamp_ns,
            joint_stamp_ns=observation.joint_stamp_ns,
            execution_status_received_ns=observation.execution_status_received_ns,
            target_depth_correspondence_max_error_m=(
                correspondence_max_error
            ),
            object_position_error_m=object_position_error,
            object_orientation_error_rad=object_orientation_error,
            object_upright_error_rad=object_upright_error,
            object_registration_inlier_fraction=(
                object_registration_inlier_fraction
            ),
            object_registration_rms_m=object_registration_rms,
            observed_object_rotation=(
                None
                if observed_rotation is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in observed_rotation
                )
            ),
            observed_upright_axis=tuple(
                float(value) for value in observed_upright_axis
            ),
        )
        reset_window = False
        if self.samples:
            gap_ns = source_stamp - self.samples[-1].stamp_ns
            reset_window = gap_ns > int(self.config.max_observation_gap_s * 1e9)
            prior_centers = np.asarray(
                [prior.center for prior in self.samples],
                dtype=float,
            )
            motion = np.linalg.norm(prior_centers - center, axis=1)
            reset_window = reset_window or bool(
                np.max(motion) > self.config.max_target_motion_m
            )
            assert self.object_model is not None
            assert self.planned_object is not None
            if self.object_model.orientation_mode == 'full':
                current_rotation = sample.observed_object_rotation
                if current_rotation is None:
                    reset_window = True
                else:
                    current_rotation_array = np.asarray(
                        current_rotation,
                        dtype=float,
                    )
                    angular_motion = [
                        _rotation_angle(
                            np.asarray(prior.observed_object_rotation, dtype=float).T
                            @ current_rotation_array
                        )
                        for prior in self.samples
                        if prior.observed_object_rotation is not None
                    ]
                    reset_window = reset_window or bool(
                        not angular_motion
                        or max(angular_motion)
                        > self.config.max_object_orientation_motion_rad
                    )
            else:
                current_axis = np.asarray(sample.observed_upright_axis, dtype=float)
                prior_axes = np.asarray(
                    [prior.observed_upright_axis for prior in self.samples],
                    dtype=float,
                )
                axis_dots = prior_axes @ current_axis
                if not self.planned_object.verification.require_upright:
                    axis_dots = np.abs(axis_dots)
                axis_motion = np.arccos(np.clip(axis_dots, -1.0, 1.0))
                reset_window = reset_window or bool(
                    np.max(axis_motion)
                    > self.config.max_object_orientation_motion_rad
                )
        if reset_window:
            self.samples.clear()
        self.samples.append(sample)
        self.last_now_ns = observation.now_ns
        self.last_observation_received_ns = observation.now_ns
        duration_s = (self.samples[-1].stamp_ns - self.samples[0].stamp_ns) * 1e-9
        if (
            len(self.samples) < self.config.min_samples
            or duration_s < self.config.min_stable_duration_s
        ):
            return None
        return self._terminal_result(
            state='verified',
            result=POST_RELEASE_VERIFICATION_RESULT,
            failure='',
        )


class PlacementCoordinator:
    """Gate synchronized observations and delegate robot motion evaluation."""

    def __init__(
        self,
        planner: ObservedPlacementPlanner,
        *,
        expected_joint_names: Sequence[str],
        max_sync_skew_s: float,
        max_snapshot_age_s: float,
    ) -> None:
        self.planner = planner
        self.expected_joint_names = tuple(expected_joint_names)
        if (
            not self.expected_joint_names
            or len(set(self.expected_joint_names)) != len(self.expected_joint_names)
        ):
            raise ValueError('expected joint names must be non-empty and unique')
        if max_sync_skew_s <= 0.0 or max_snapshot_age_s <= 0.0:
            raise ValueError('placement synchronization limits must be positive')
        self.max_sync_skew_ns = int(max_sync_skew_s * 1e9)
        self.max_snapshot_age_ns = int(max_snapshot_age_s * 1e9)

    def _validate_snapshot(
        self,
        request: PlacementRegionRequest,
        snapshot: PlacementPerceptionSnapshot,
        now_ns: int,
    ) -> np.ndarray:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise PlacementContractError('placement clock must be non-negative integer ns')
        if not snapshot.frame_id.strip():
            raise PlacementContractError('snapshot frame is empty')
        if snapshot.image_frame != request.image_frame:
            raise PlacementContractError(
                'placement request image frame does not match exact RGB-D source',
            )
        rgbd_stamps = (
            snapshot.rgb_stamp_ns,
            snapshot.depth_stamp_ns,
            snapshot.camera_info_stamp_ns,
        )
        if any(stamp != request.stamp_ns for stamp in rgbd_stamps):
            raise PlacementContractError(
                'placement request, RGB, depth, and camera-info stamps must match exactly',
            )
        stamps = np.asarray((
            request.stamp_ns,
            snapshot.scene_stamp_ns,
        ), dtype=np.int64)
        if np.max(stamps) - np.min(stamps) > self.max_sync_skew_ns:
            raise PlacementContractError('VLM/RGB/depth/scene observations are not synchronized')
        if now_ns < int(np.max(stamps)) or now_ns - int(np.max(stamps)) > self.max_snapshot_age_ns:
            raise PlacementContractError('placement perception snapshot is stale')
        if now_ns < snapshot.joint_stamp_ns or (
            now_ns - snapshot.joint_stamp_ns > self.max_snapshot_age_ns
        ):
            raise PlacementContractError('placement joint state is stale')
        incoming_names = tuple(snapshot.joint_names)
        incoming_positions = _finite_vector(
            snapshot.joint_positions, len(incoming_names), 'joint positions',
        )
        if len(set(incoming_names)) != len(incoming_names):
            raise PlacementContractError('joint state contains duplicate names')
        mapping = dict(zip(incoming_names, incoming_positions))
        missing = set(self.expected_joint_names) - set(mapping)
        if missing:
            raise PlacementContractError(
                f'joint state is missing configured joints: {sorted(missing)}',
            )
        return np.asarray([mapping[name] for name in self.expected_joint_names])

    def plan(
        self,
        request: PlacementRegionRequest,
        snapshot: PlacementPerceptionSnapshot,
        *,
        now_ns: int,
        evaluate: Callable[[PlacementCandidate, np.ndarray], EvaluatedPlacementMotion],
        control: PlanningControl | None = None,
    ) -> PlacementOutput:
        """Plan once from one immutable snapshot and audited motion callback."""
        checkpoint(control, 'placement coordinator snapshot validation')
        current_joints = self._validate_snapshot(request, snapshot, now_ns)
        audits: list[CandidateAudit] = []

        def audited_evaluate(
            candidate: PlacementCandidate,
            current: np.ndarray,
        ) -> EvaluatedPlacementMotion:
            try:
                checkpoint(control, 'placement coordinator motion evaluation')
                evaluation = evaluate(candidate, current)
                checkpoint(control, 'placement coordinator motion evaluation')
                if not isinstance(evaluation, EvaluatedPlacementMotion):
                    raise PlanningError('motion evaluator returned the wrong contract type')
                if evaluation.trajectory.goal_id != request.goal_id:
                    raise PlanningError('motion trajectory goal_id does not match request')
                if evaluation.trajectory.frame_id != snapshot.frame_id:
                    raise PlanningError('motion trajectory frame does not match perception')
                audits.append(CandidateAudit(
                    candidate, True, evaluation.score, '',
                ))
                return evaluation
            except PlanningAborted:
                raise
            except (PlanningError, PlacementContractError, ValueError) as error:
                audits.append(CandidateAudit(
                    candidate, False, None, f'{type(error).__name__}: {error}',
                ))
                raise PlanningError(str(error)) from error

        observed = ObservedPlacementInput(
            organized_points=snapshot.organized_points,
            scene_points=snapshot.scene_points,
            region=request.region,
            constraints=request.constraints,
            gravity=snapshot.gravity,
            object_extent_m=request.object_extent_m,
            tool_from_object=request.tool_from_object,
            organized_frame=snapshot.frame_id,
            scene_frame=snapshot.frame_id,
            organized_stamp_s=snapshot.depth_stamp_ns / 1e9,
            scene_stamp_s=snapshot.scene_stamp_ns / 1e9,
        )
        result = self.planner.plan(
            observed,
            current_joints=current_joints,
            evaluate=audited_evaluate,
            control=control,
        )
        checkpoint(control, 'placement coordinator output validation')
        evaluation = result.evaluation
        if not isinstance(evaluation, EvaluatedPlacementMotion):
            raise PlacementContractError('selected placement has no motion contract')
        return PlacementOutput(
            goal_id=request.goal_id,
            result=result,
            trajectory=evaluation.trajectory,
            candidates=tuple(audits),
        )


__all__ = [
    'CandidateAudit',
    'EvaluatedPlacementMotion',
    'ObservedPerceptionIdentity',
    'ObservedPlacementRegionGeometry',
    'POST_RELEASE_OBSERVATION_SOURCE',
    'POST_RELEASE_VERIFICATION_RESULT',
    'POST_RELEASE_VERIFICATION_SCHEMA',
    'PostReleaseObservation',
    'PostReleasePlacementVerifier',
    'PostReleaseVerificationConfig',
    'PostReleaseVerificationResult',
    'PlacementContractError',
    'PlacementCoordinator',
    'PlaceExecutionCorrelation',
    'PlacementOutput',
    'PlacementOrientationVerification',
    'PlacementPerceptionSnapshot',
    'PlacementRegionRequest',
    'PlacementTrajectoryContract',
    'PlannedObjectGeometry',
    'RawTrajectorySegment',
    'TrajectoryWaypoint',
    'backproject_depth',
    'capture_observed_region_geometry',
    'capture_planned_object_geometry',
    'combine_trajectory_segments',
    'parse_region_request',
]
