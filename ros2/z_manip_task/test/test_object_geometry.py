"""Observed object-frame estimation and semantic ambiguity tests."""

import math

import numpy as np
import pytest

from z_manip_task.object_geometry import (
    CarriedObjectObservationIdentity,
    estimate_carried_object_geometry,
    ObjectGeometryConfig,
    ObjectGeometryError,
    parse_placement_verification,
    PlacementVerificationSemantics,
)


def _identity() -> CarriedObjectObservationIdentity:
    return CarriedObjectObservationIdentity(
        request_id='task-7',
        producer_epoch='bridge-a',
        generation=3,
        observation_stamp_ns=1_000_000_000,
        frame_id='wrist_camera_optical_frame',
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray((
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    ))


def _cylinder_points(
    *,
    radius: float = 0.025,
    length: float = 0.20,
    rotation: np.ndarray | None = None,
    center: tuple[float, float, float] = (0.45, 0.05, 0.72),
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    heights = np.linspace(-0.5 * length, 0.5 * length, 17)
    local = np.asarray([
        (radius * math.cos(angle), radius * math.sin(angle), height)
        for height in heights
        for angle in angles
    ])
    orientation = np.eye(3) if rotation is None else rotation
    return local @ orientation.T + np.asarray(center)


def _box_points(
    extent: tuple[float, float, float],
    rotation: np.ndarray,
) -> np.ndarray:
    axes = [np.linspace(-0.5 * value, 0.5 * value, 9) for value in extent]
    local = np.asarray([
        (x, y, z)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    ])
    return local @ rotation.T + np.asarray((0.40, -0.10, 0.60))


def test_side_grasp_upright_bottle_retains_nonidentity_tool_from_object() -> None:
    base_t_tool = np.eye(4)
    base_t_tool[:3, :3] = _rotation_y(math.pi / 2.0)
    base_t_tool[:3, 3] = (0.43, 0.05, 0.72)
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_long',
        orientation_symmetry='axial',
        symmetry_axis='principal_long',
    )

    geometry = estimate_carried_object_geometry(
        _cylinder_points(),
        base_t_tool,
        semantics,
        _identity(),
    )

    assert not np.allclose(geometry.tool_from_object, np.eye(4))
    assert geometry.object_extent_m[2] == pytest.approx(0.20, abs=0.01)
    assert np.abs(geometry.upright_axis_object) == pytest.approx((0.0, 0.0, 1.0))
    assert np.abs(geometry.symmetry_axis_object) == pytest.approx((0.0, 0.0, 1.0))
    assert geometry.verification_payload()['orientation_symmetry'] == 'axial'
    assert geometry.identity == _identity()
    assert len(geometry.reference_points_object) <= 512
    reference_extent = np.ptp(geometry.reference_points_object, axis=0)
    assert reference_extent == pytest.approx(geometry.object_extent_m, abs=0.01)


@pytest.mark.parametrize(
    ('visible_arc_degrees', 'accepted'),
    ((273.0, True), (260.0, False)),
)
def test_partial_view_axial_surface_uses_recorded_observability_boundary(
    visible_arc_degrees: float,
    accepted: bool,
) -> None:
    """Model the transverse anisotropy of a single-view cylindrical surface."""
    half_arc = math.radians(visible_arc_degrees) * 0.5
    angles = np.linspace(-half_arc, half_arc, 181)
    heights = np.linspace(-0.10, 0.10, 17)
    points = np.asarray([
        (0.025 * math.cos(angle), 0.025 * math.sin(angle), height)
        for height in heights
        for angle in angles
    ])
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_long',
        orientation_symmetry='axial',
        symmetry_axis='principal_long',
    )

    if accepted:
        geometry = estimate_carried_object_geometry(
            points,
            np.eye(4),
            semantics,
            _identity(),
        )
        transverse_ratio = (
            geometry.principal_variances[1]
            / geometry.principal_variances[2]
        )
        assert 1.35 < transverse_ratio < 1.90
    else:
        with pytest.raises(
            ObjectGeometryError,
            match='axial symmetry is not observable',
        ):
            estimate_carried_object_geometry(
                points,
                np.eye(4),
                semantics,
                _identity(),
            )


def test_robust_pca_trims_sparse_far_outliers_from_frozen_reference() -> None:
    points = _cylinder_points()
    outliers = np.asarray([
        (4.0 + index, -3.0, 2.0)
        for index in range(8)
    ])
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_long',
        orientation_symmetry='axial',
        symmetry_axis='principal_long',
    )

    geometry = estimate_carried_object_geometry(
        np.vstack((points, outliers)),
        np.eye(4),
        semantics,
        _identity(),
    )

    assert max(geometry.object_extent_m) < 0.25
    assert np.max(np.linalg.norm(geometry.reference_points_object, axis=1)) < 0.15


def test_duplicate_rgbd_points_are_deduplicated_before_freezing_reference() -> None:
    points = _cylinder_points()
    duplicated = np.vstack((points, points[::2], points[::3]))
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_long',
        orientation_symmetry='axial',
        symmetry_axis='principal_long',
    )

    geometry = estimate_carried_object_geometry(
        duplicated,
        np.eye(4),
        semantics,
        _identity(),
    )

    reference = geometry.reference_points_object
    assert len(reference) == len(np.unique(reference, axis=0))
    assert 40 <= len(reference) <= 512


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    (
        ({'min_points': 39}, 'at least 40'),
        ({'max_reference_points': 513}, 'not exceed 512'),
    ),
)
def test_reference_cloud_configuration_matches_transport_contract(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ObjectGeometryError, match=message):
        ObjectGeometryConfig(**kwargs)


def test_retained_tail_points_define_conservative_collision_extent() -> None:
    body = _box_points((0.12, 0.06, 0.025), np.eye(3))
    tail_x = np.linspace(0.46, 0.50, 10)
    tail_y = np.linspace(-0.11, -0.09, 5)
    tail_z = np.linspace(0.595, 0.605, 4)
    tail = np.asarray([
        (x, y, z)
        for x in tail_x
        for y in tail_y
        for z in tail_z
    ])
    semantics = PlacementVerificationSemantics(
        require_upright=False,
        upright_axis='principal_short',
        orientation_symmetry='none',
        symmetry_axis=None,
    )

    geometry = estimate_carried_object_geometry(
        np.vstack((body, tail)),
        np.eye(4),
        semantics,
        _identity(),
    )

    reference_extent = np.ptp(geometry.reference_points_object, axis=0)
    assert geometry.object_extent_m == pytest.approx(reference_extent, abs=1e-12)
    assert max(geometry.object_extent_m) > 0.13


def test_small_observable_object_keeps_exact_reference_extent() -> None:
    semantics = PlacementVerificationSemantics(
        require_upright=False,
        upright_axis='principal_short',
        orientation_symmetry='none',
        symmetry_axis=None,
    )

    geometry = estimate_carried_object_geometry(
        _box_points((0.024, 0.014, 0.0085), np.eye(3)),
        np.eye(4),
        semantics,
        _identity(),
    )

    assert geometry.object_extent_m == pytest.approx(
        np.ptp(geometry.reference_points_object, axis=0),
        abs=1e-12,
    )
    assert min(geometry.object_extent_m) >= 0.008


def test_axial_symmetry_requires_the_distinct_observed_axis() -> None:
    semantics = PlacementVerificationSemantics(
        require_upright=False,
        upright_axis='principal_long',
        orientation_symmetry='axial',
        symmetry_axis='principal_middle',
    )

    with pytest.raises(ObjectGeometryError, match='axial symmetry is not observable'):
        estimate_carried_object_geometry(
            _cylinder_points(), np.eye(4), semantics, _identity(),
        )


def test_asymmetric_box_supports_explicit_non_upright_full_orientation() -> None:
    rotation = _rotation_y(0.45)
    semantics = PlacementVerificationSemantics(
        require_upright=False,
        upright_axis='principal_short',
        orientation_symmetry='none',
        symmetry_axis=None,
    )

    geometry = estimate_carried_object_geometry(
        _box_points((0.24, 0.11, 0.045), rotation),
        np.eye(4),
        semantics,
        _identity(),
    )

    assert not geometry.require_upright
    assert geometry.orientation_symmetry == 'none'
    assert geometry.symmetry_axis_object is None
    assert sorted(geometry.object_extent_m) == pytest.approx(
        sorted((0.24, 0.11, 0.045)),
        abs=0.012,
    )


def test_fully_degenerate_shape_fails_closed_without_axis_hardcoding() -> None:
    points = _box_points((0.10, 0.10, 0.10), np.eye(3))
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_long',
        orientation_symmetry='none',
        symmetry_axis=None,
    )

    with pytest.raises(ObjectGeometryError, match='not observable'):
        estimate_carried_object_geometry(points, np.eye(4), semantics, _identity())


def test_upright_axis_unobservable_under_axial_symmetry_is_rejected() -> None:
    semantics = PlacementVerificationSemantics(
        require_upright=True,
        upright_axis='principal_short',
        orientation_symmetry='axial',
        symmetry_axis='principal_long',
    )

    with pytest.raises(ObjectGeometryError, match='upright axis is unobservable'):
        estimate_carried_object_geometry(
            _cylinder_points(), np.eye(4), semantics, _identity(),
        )


def test_structured_semantics_are_exact_and_do_not_default_missing_fields() -> None:
    value = {
        'require_upright': True,
        'upright_axis': 'principal_long',
        'orientation_symmetry': 'axial',
        'symmetry_axis': 'principal_long',
    }
    assert parse_placement_verification(value) == PlacementVerificationSemantics(
        True,
        'principal_long',
        'axial',
        'principal_long',
    )

    missing = dict(value)
    missing.pop('upright_axis')
    with pytest.raises(ObjectGeometryError, match='keys must be exactly'):
        parse_placement_verification(missing)


def test_symmetry_axis_is_forbidden_for_full_orientation() -> None:
    with pytest.raises(ObjectGeometryError, match='only valid for axial'):
        parse_placement_verification({
            'require_upright': False,
            'upright_axis': 'principal_short',
            'orientation_symmetry': 'none',
            'symmetry_axis': 'principal_long',
        })
