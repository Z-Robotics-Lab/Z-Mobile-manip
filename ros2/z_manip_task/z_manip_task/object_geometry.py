"""Observed carried-object frame and placement-orientation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


AXIS_SELECTORS = frozenset({
    'principal_long',
    'principal_middle',
    'principal_short',
})
ORIENTATION_SYMMETRIES = frozenset({'none', 'axial'})
MIN_REFERENCE_POINTS = 40
MAX_REFERENCE_POINTS = 512

_SELECTOR_INDEX = {
    'principal_long': 0,
    'principal_middle': 1,
    'principal_short': 2,
}


class ObjectGeometryError(ValueError):
    """Observed geometry cannot satisfy the requested semantic contract."""


@dataclass(frozen=True)
class PlacementVerificationSemantics:
    """VLM-provided semantic axes expressed as observable PCA selectors."""

    require_upright: bool
    upright_axis: str
    orientation_symmetry: str
    symmetry_axis: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.require_upright, bool):
            raise ObjectGeometryError('require_upright must be boolean')
        if self.upright_axis not in AXIS_SELECTORS:
            raise ObjectGeometryError('upright_axis is not a supported principal axis')
        if self.orientation_symmetry not in ORIENTATION_SYMMETRIES:
            raise ObjectGeometryError(
                'orientation_symmetry must be none or axial',
            )
        if self.orientation_symmetry == 'axial':
            if self.symmetry_axis not in AXIS_SELECTORS:
                raise ObjectGeometryError(
                    'axial symmetry requires an observable symmetry_axis',
                )
        elif self.symmetry_axis is not None:
            raise ObjectGeometryError(
                'symmetry_axis is only valid for axial symmetry',
            )


@dataclass(frozen=True)
class CarriedObjectObservationIdentity:
    """Perception owner of the immutable grasp-time reference cloud."""

    request_id: str
    producer_epoch: str
    generation: int
    observation_stamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        strings = (self.request_id, self.producer_epoch, self.frame_id)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ObjectGeometryError('carried-object identity strings must be non-empty')
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation <= 0
        ):
            raise ObjectGeometryError('carried-object generation must be positive')
        if (
            isinstance(self.observation_stamp_ns, bool)
            or not isinstance(self.observation_stamp_ns, int)
            or self.observation_stamp_ns <= 0
        ):
            raise ObjectGeometryError(
                'carried-object observation stamp must be positive',
            )

    def to_payload(self) -> dict[str, object]:
        return {
            'request_id': self.request_id,
            'producer_epoch': self.producer_epoch,
            'generation': self.generation,
            'observation_stamp_ns': self.observation_stamp_ns,
            'frame_id': self.frame_id,
        }


@dataclass(frozen=True)
class ObjectGeometryConfig:
    """Generic robust-PCA quality gates for one tracked object cloud."""

    min_points: int = MIN_REFERENCE_POINTS
    trim_mad_scale: float = 4.5
    extent_percentile: float = 2.0
    min_extent_m: float = 0.008
    min_axis_separation_ratio: float = 1.20
    max_axial_transverse_ratio: float = 1.90
    max_reference_points: int = MAX_REFERENCE_POINTS

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_points, bool)
            or not isinstance(self.min_points, int)
            or self.min_points < MIN_REFERENCE_POINTS
        ):
            raise ObjectGeometryError(
                f'min_points must be an integer of at least {MIN_REFERENCE_POINTS}',
            )
        if (
            isinstance(self.max_reference_points, bool)
            or not isinstance(self.max_reference_points, int)
            or self.max_reference_points < self.min_points
            or self.max_reference_points > MAX_REFERENCE_POINTS
        ):
            raise ObjectGeometryError(
                'max_reference_points must cover the minimum point count and '
                f'not exceed {MAX_REFERENCE_POINTS}',
            )
        positive = (
            self.trim_mad_scale,
            self.min_extent_m,
            self.min_axis_separation_ratio,
            self.max_axial_transverse_ratio,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ObjectGeometryError('object geometry thresholds must be positive')
        if self.min_axis_separation_ratio <= 1.0:
            raise ObjectGeometryError('min_axis_separation_ratio must exceed one')
        if self.max_axial_transverse_ratio < 1.0:
            raise ObjectGeometryError('max_axial_transverse_ratio cannot be below one')
        if (
            not math.isfinite(self.extent_percentile)
            or not 0.0 <= self.extent_percentile < 25.0
        ):
            raise ObjectGeometryError('extent_percentile must be in [0, 25)')


@dataclass(frozen=True, eq=False)
class CarriedObjectGeometry:
    """Object geometry frozen from the exact post-grasp tracked observation."""

    tool_from_object: np.ndarray
    object_extent_m: np.ndarray
    upright_axis_object: np.ndarray
    orientation_symmetry: str
    symmetry_axis_object: np.ndarray | None
    require_upright: bool
    retained_point_count: int
    principal_variances: np.ndarray
    reference_points_object: np.ndarray
    identity: CarriedObjectObservationIdentity

    def __post_init__(self) -> None:
        transform = np.asarray(self.tool_from_object, dtype=float)
        extent = np.asarray(self.object_extent_m, dtype=float)
        upright = np.asarray(self.upright_axis_object, dtype=float)
        variances = np.asarray(self.principal_variances, dtype=float)
        reference = np.asarray(self.reference_points_object, dtype=float)
        if not isinstance(self.identity, CarriedObjectObservationIdentity):
            raise ObjectGeometryError('carried-object observation identity is invalid')
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ObjectGeometryError('tool_from_object must be a finite 4x4 transform')
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
            raise ObjectGeometryError('tool_from_object has an invalid homogeneous row')
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ObjectGeometryError('tool_from_object rotation is not proper')
        if extent.shape != (3,) or not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            raise ObjectGeometryError('object_extent_m must be finite and positive')
        if (
            upright.shape != (3,)
            or not np.all(np.isfinite(upright))
            or not np.isclose(np.linalg.norm(upright), 1.0, atol=1e-6)
        ):
            raise ObjectGeometryError('upright_axis_object must be a unit vector')
        if self.orientation_symmetry not in ORIENTATION_SYMMETRIES:
            raise ObjectGeometryError('resolved orientation symmetry is invalid')
        if self.orientation_symmetry == 'axial':
            symmetry = np.asarray(self.symmetry_axis_object, dtype=float)
            if (
                symmetry.shape != (3,)
                or not np.all(np.isfinite(symmetry))
                or not np.isclose(np.linalg.norm(symmetry), 1.0, atol=1e-6)
            ):
                raise ObjectGeometryError('symmetry_axis_object must be a unit vector')
        elif self.symmetry_axis_object is not None:
            raise ObjectGeometryError('full orientation cannot retain a symmetry axis')
        if isinstance(self.retained_point_count, bool) or self.retained_point_count <= 0:
            raise ObjectGeometryError('retained_point_count must be positive')
        if variances.shape != (3,) or not np.all(np.isfinite(variances)):
            raise ObjectGeometryError('principal_variances must be finite')
        if (
            reference.ndim != 2
            or reference.shape[1] != 3
            or not MIN_REFERENCE_POINTS <= len(reference) <= MAX_REFERENCE_POINTS
            or not np.all(np.isfinite(reference))
        ):
            raise ObjectGeometryError(
                'reference_points_object must be a bounded finite object-frame cloud',
            )
        if len(np.unique(reference, axis=0)) != len(reference):
            raise ObjectGeometryError('reference_points_object must contain unique points')

    def verification_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            'require_upright': self.require_upright,
            'upright_axis_object': self.upright_axis_object.tolist(),
            'orientation_symmetry': self.orientation_symmetry,
        }
        if self.symmetry_axis_object is not None:
            payload['symmetry_axis_object'] = self.symmetry_axis_object.tolist()
        return payload


def parse_placement_verification(
    value: object,
) -> PlacementVerificationSemantics:
    """Parse a strict structured-VLM placement verification object."""
    if not isinstance(value, Mapping):
        raise ObjectGeometryError('placement_verification must be an object')
    required = {
        'require_upright',
        'upright_axis',
        'orientation_symmetry',
        'symmetry_axis',
    }
    if set(value) != required:
        raise ObjectGeometryError(
            'placement_verification keys must be exactly '
            f'{sorted(required)}',
        )
    symmetry_axis = value['symmetry_axis']
    if symmetry_axis is not None and not isinstance(symmetry_axis, str):
        raise ObjectGeometryError('symmetry_axis must be a string or null')
    return PlacementVerificationSemantics(
        require_upright=value['require_upright'],
        upright_axis=str(value['upright_axis']),
        orientation_symmetry=str(value['orientation_symmetry']),
        symmetry_axis=symmetry_axis,
    )


def _finite_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ObjectGeometryError(f'{label} must be a finite 4x4 transform')
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ObjectGeometryError(f'{label} has an invalid homogeneous row')
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ObjectGeometryError(f'{label} rotation is not proper')
    return transform


def _canonical_sign(axis: np.ndarray, references: tuple[np.ndarray, ...]) -> np.ndarray:
    """Resolve PCA sign deterministically against measured frame directions."""
    for reference in references:
        projection = float(np.dot(axis, reference))
        if abs(projection) > 1e-6:
            return axis if projection > 0.0 else -axis
    index = int(np.argmax(np.abs(axis)))
    return axis if axis[index] >= 0.0 else -axis


def _sample_reference_points(points: np.ndarray, limit: int) -> np.ndarray:
    """Bound a unique cloud deterministically while retaining every AABB face."""
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    if len(ordered) <= limit:
        return ordered

    selected = {
        int(function(ordered[:, axis]))
        for axis in range(3)
        for function in (np.argmin, np.argmax)
    }
    sample_count = limit - len(selected)
    selected.update(np.linspace(
        0,
        len(ordered) - 1,
        sample_count,
        dtype=int,
    ).tolist())
    if len(selected) < limit:
        for index in range(len(ordered)):
            selected.add(index)
            if len(selected) == limit:
                break
    return ordered[np.asarray(sorted(selected), dtype=int)]


def _robust_principal_axes(
    points: np.ndarray,
    config: ObjectGeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ObjectGeometryError('tracked object cloud must have shape (N, 3)')
    finite = np.unique(points[np.all(np.isfinite(points), axis=1)], axis=0)
    if len(finite) < config.min_points:
        raise ObjectGeometryError('tracked object cloud has insufficient finite points')
    median = np.median(finite, axis=0)
    radii = np.linalg.norm(finite - median, axis=1)
    radial_median = float(np.median(radii))
    radial_mad = float(np.median(np.abs(radii - radial_median)))
    robust_sigma = max(1.4826 * radial_mad, 1e-6)
    threshold = radial_median + config.trim_mad_scale * robust_sigma
    retained = finite[radii <= threshold]
    if len(retained) < config.min_points:
        raise ObjectGeometryError('robust object-cloud trimming retained too few points')
    center = np.median(retained, axis=0)
    centered = retained - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order]
    if (
        not np.all(np.isfinite(eigenvalues))
        or eigenvalues[0] <= 1e-10
        or eigenvalues[2] <= 1e-12
    ):
        raise ObjectGeometryError('tracked object geometry is rank-degenerate')
    return retained, eigenvalues, axes


def estimate_carried_object_geometry(
    points_base: object,
    base_from_tool: object,
    semantics: PlacementVerificationSemantics,
    identity: CarriedObjectObservationIdentity,
    *,
    config: ObjectGeometryConfig | None = None,
) -> CarriedObjectGeometry:
    """Estimate and freeze a scalable object frame from online tracked points."""
    if not isinstance(semantics, PlacementVerificationSemantics):
        raise ObjectGeometryError('placement verification semantics are invalid')
    if not isinstance(identity, CarriedObjectObservationIdentity):
        raise ObjectGeometryError('carried-object observation identity is invalid')
    settings = config or ObjectGeometryConfig()
    base_t_tool = _finite_transform(base_from_tool, 'base_from_tool')
    points = np.asarray(points_base, dtype=float)
    retained, variances, pca_axes = _robust_principal_axes(points, settings)

    adjacent_long = variances[0] / max(variances[1], 1e-12)
    adjacent_short = variances[1] / max(variances[2], 1e-12)
    if semantics.orientation_symmetry == 'none':
        if min(adjacent_long, adjacent_short) < settings.min_axis_separation_ratio:
            raise ObjectGeometryError(
                'full object orientation is not observable from ambiguous PCA axes',
            )
    else:
        assert semantics.symmetry_axis is not None
        symmetry_index = _SELECTOR_INDEX[semantics.symmetry_axis]
        other_indices = [index for index in range(3) if index != symmetry_index]
        distinct = all(
            max(variances[symmetry_index], variances[index])
            / max(min(variances[symmetry_index], variances[index]), 1e-12)
            >= settings.min_axis_separation_ratio
            for index in other_indices
        )
        transverse_ratio = (
            max(variances[index] for index in other_indices)
            / max(min(variances[index] for index in other_indices), 1e-12)
        )
        if not distinct or transverse_ratio > settings.max_axial_transverse_ratio:
            raise ObjectGeometryError(
                'declared axial symmetry is not observable in tracked geometry',
            )
        if (
            semantics.require_upright
            and semantics.upright_axis != semantics.symmetry_axis
        ):
            raise ObjectGeometryError(
                'requested upright axis is unobservable under axial symmetry',
            )

    upright_index = _SELECTOR_INDEX[semantics.upright_axis]
    if semantics.require_upright and semantics.orientation_symmetry == 'none':
        other_indices = [index for index in range(3) if index != upright_index]
        if any(
            max(variances[upright_index], variances[index])
            / max(min(variances[upright_index], variances[index]), 1e-12)
            < settings.min_axis_separation_ratio
            for index in other_indices
        ):
            raise ObjectGeometryError('requested upright axis is not observable')

    tool_rotation = base_t_tool[:3, :3]
    z_axis = _canonical_sign(
        pca_axes[:, upright_index],
        (np.array((0.0, 0.0, 1.0)), tool_rotation[:, 2], tool_rotation[:, 0]),
    )
    remaining = [index for index in range(3) if index != upright_index]
    x_index = min(remaining)
    x_seed = _canonical_sign(
        pca_axes[:, x_index],
        (tool_rotation[:, 0], np.array((1.0, 0.0, 0.0)), tool_rotation[:, 1]),
    )
    y_axis = np.cross(z_axis, x_seed)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-8:
        raise ObjectGeometryError('principal-axis frame construction is degenerate')
    y_axis /= y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    base_r_object = np.column_stack((x_axis, y_axis, z_axis))

    center = np.median(retained, axis=0)
    coordinates = (retained - center) @ base_r_object
    low = np.percentile(coordinates, settings.extent_percentile, axis=0)
    high = np.percentile(coordinates, 100.0 - settings.extent_percentile, axis=0)
    central_extent = high - low
    if np.any(central_extent < settings.min_extent_m):
        raise ObjectGeometryError('observed object extent is below the geometry floor')

    # The transmitted reference is also the collision geometry contract. Center
    # it on the complete robust-cloud AABB and preserve all six extrema while
    # bounding transport size, so its PTP exactly equals the reported extent.
    robust_low = np.min(coordinates, axis=0)
    robust_high = np.max(coordinates, axis=0)
    robust_midpoint = 0.5 * (robust_low + robust_high)
    object_center = center + base_r_object @ robust_midpoint
    reference_points = _sample_reference_points(
        coordinates - robust_midpoint,
        settings.max_reference_points,
    )
    extent = np.ptp(reference_points, axis=0)
    if np.any(extent < settings.min_extent_m):
        raise ObjectGeometryError('frozen object extent is below the geometry floor')
    base_t_object = np.eye(4)
    base_t_object[:3, :3] = base_r_object
    base_t_object[:3, 3] = object_center
    tool_t_object = np.linalg.inv(base_t_tool) @ base_t_object

    upright_axis_object = base_r_object.T @ pca_axes[:, upright_index]
    upright_axis_object /= np.linalg.norm(upright_axis_object)
    symmetry_axis_object = None
    if semantics.symmetry_axis is not None:
        symmetry_index = _SELECTOR_INDEX[semantics.symmetry_axis]
        symmetry_axis_object = base_r_object.T @ pca_axes[:, symmetry_index]
        symmetry_axis_object /= np.linalg.norm(symmetry_axis_object)

    return CarriedObjectGeometry(
        tool_from_object=tool_t_object,
        object_extent_m=extent,
        upright_axis_object=upright_axis_object,
        orientation_symmetry=semantics.orientation_symmetry,
        symmetry_axis_object=symmetry_axis_object,
        require_upright=semantics.require_upright,
        retained_point_count=len(retained),
        principal_variances=variances,
        reference_points_object=reference_points,
        identity=identity,
    )


__all__ = [
    'AXIS_SELECTORS',
    'CarriedObjectObservationIdentity',
    'CarriedObjectGeometry',
    'ObjectGeometryConfig',
    'ObjectGeometryError',
    'ORIENTATION_SYMMETRIES',
    'PlacementVerificationSemantics',
    'estimate_carried_object_geometry',
    'parse_placement_verification',
]
