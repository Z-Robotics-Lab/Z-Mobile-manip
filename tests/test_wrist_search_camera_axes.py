"""Forward-kinematics regression guard for the bounded wrist-search axes.

The horizontal sweep used to be driven by PiPER J4, a rotation about the
forearm.  That axis is roughly 20 deg off the camera's own optical axis at the
search anchor, so sweeping it ROLLED the picture (1.11 deg of horizon tilt per
0.66 deg of azimuth) instead of panning it - the operator saw the view go
oblique rather than shake left and right.  The sweep now runs on J1, the base
yaw, whose axis is the world vertical, so the horizon tilt is an exact
invariant.

These tests re-derive that from the deployed URDF plus the measured hand-eye
calibration and fail if the pan axis is ever moved back to a joint that tilts
the camera horizon, or if the swept envelope leaves the firmware joint limits.
They are the durable part of the fix: the config alone cannot express "this
axis keeps the horizon level".
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from z_manip.control.wrist_search import BoundedWristSearch, WristSearchConfig
from z_manip.kinematics import KinematicChain


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_HOME = ROOT / "configs/piper_home.example.json"
# The robot assets live beside the stack checkout, so resolve them the way
# tests/test_robust_ik.py does: honour the deployed environment override, then
# the sibling checkout, then the known install path.  A git worktree sits two
# levels deeper, which is why the sibling lookup alone is not enough.
_URDF_CANDIDATES = (
    os.environ.get("Z_MANIP_ROBOT_URDF"),
    ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf",
    "/home/yusenzlabpc/Z-Robotics-Lab/go2W_Sim/assets/urdf/go2w_sensored.urdf",
)
_CALIBRATION_CANDIDATES = (
    os.environ.get("Z_MANIP_WRIST_CAMERA_CALIBRATION"),
    ROOT / "configs/piper_wrist_camera_calibration.json",
    "/home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/calibration/"
    "piper_wrist_camera_calibration.json",
)
# The measured Home is captured per robot and is deliberately not committed;
# fall back to the example anchor so the axis invariants are still exercised.
_HOME_CANDIDATES = (
    os.environ.get("Z_MANIP_PIPER_HOME"),
    ROOT / "configs/piper_home.json",
    "/home/yusenzlabpc/Z-Robotics-Lab/Z-Mobile-manip/configs/piper_home.json",
    EXAMPLE_HOME,
)


def _first_existing(candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None

# The pan axis must keep the horizon level.  J1 is exactly invariant (0.0 deg
# by construction); this bound only has to be small enough to reject a tilting
# axis, and J4 blows straight through it (16.5 deg at a single 15 deg step).
MAX_PAN_HORIZON_TILT_DEG = 0.05
# Over the whole grid the only tilt contribution is J5's, measured at most
# 0.37 deg over its +/-28 deg envelope.
MAX_GRID_HORIZON_TILT_DEG = 1.0
# A pan step must actually pan.  J1 delivers 1.000 deg of optical-axis azimuth
# per degree; J4 delivers 0.660 and is rejected here too.
MIN_AZIMUTH_PER_PAN_DEGREE = 0.80


def _wrist_executor_module():
    """Load the remote fixed-view executor, which owns the firmware limits."""

    spec = importlib.util.spec_from_file_location(
        "piper_wrist_search_executor",
        ROOT / "scripts/runtime/piper_wrist_search_executor.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def camera_pose():
    """Return ``joints -> base_from_camera``, composed exactly as runtime does.

    ``piper_planning_session_gate`` builds the camera pose as
    ``chain.forward(q) @ tip_from_camera``; reproducing that composition here
    keeps the test honest about the frame the operator actually looks through.
    """

    urdf = _first_existing(_URDF_CANDIDATES)
    calibration = _first_existing(_CALIBRATION_CANDIDATES)
    if urdf is None:
        pytest.skip("PiPER URDF unavailable")
    if calibration is None:
        pytest.skip("wrist-camera hand-eye calibration unavailable")
    document = json.loads(calibration.read_text(encoding="utf-8"))
    if document.get("calibrated") is not True:
        pytest.skip("hand-eye calibration is not marked calibrated")
    tip_from_camera = np.asarray(document["tip_from_camera"], dtype=float)
    chain = KinematicChain.from_urdf(urdf, "piper_base_link", "piper_gripper_base")

    def pose(joints):
        return chain.forward(np.asarray(joints, dtype=float)) @ tip_from_camera

    return pose


@pytest.fixture(scope="module")
def anchor():
    path = _first_existing(_HOME_CANDIDATES)
    assert path is not None, "the example Home anchor is committed and must exist"
    return np.asarray(
        json.loads(path.read_text(encoding="utf-8"))["joint_radians"], dtype=float
    )


def _horizon_tilt_deg(transform):
    """Image roll of the world vertical, in degrees.

    ``camera_color_optical_frame`` is REP-105 optical (+x image right, +y image
    down, +z along the optical axis).  Projecting world up into the image plane
    and measuring its angle from image-up gives the tilt an operator perceives
    as the picture being oblique.  0 deg means world-up points straight up.
    """

    rotation = np.asarray(transform)[:3, :3]
    up_in_camera = rotation.T @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(up_in_camera[0], -up_in_camera[1]))


def _azimuth_deg(transform):
    axis = np.asarray(transform)[:3, 2]
    return math.degrees(math.atan2(axis[1], axis[0]))


def _offset(anchor, index, radians):
    joints = np.asarray(anchor, dtype=float).copy()
    joints[index] += radians
    return joints


def test_pan_axis_pans_the_camera_without_tilting_the_horizon(camera_pose, anchor):
    config = WristSearchConfig()
    reference_tilt = _horizon_tilt_deg(camera_pose(anchor))
    reference_azimuth = _azimuth_deg(camera_pose(anchor))
    offsets = np.linspace(
        -config.max_pan_offset_rad, config.max_pan_offset_rad, 61
    )
    for offset in offsets:
        transform = camera_pose(_offset(anchor, config.pan_joint_index, offset))
        tilt = abs(_horizon_tilt_deg(transform) - reference_tilt)
        assert tilt <= MAX_PAN_HORIZON_TILT_DEG, (
            f"pan joint {config.pan_joint_index} tilts the horizon by {tilt:.3f} deg "
            f"at a {math.degrees(offset):+.1f} deg offset; the sweep axis must "
            "rotate about the world vertical"
        )
        if abs(offset) > 1e-6:
            azimuth = abs(_azimuth_deg(transform) - reference_azimuth)
            gain = azimuth / abs(math.degrees(offset))
            assert gain >= MIN_AZIMUTH_PER_PAN_DEGREE, (
                f"pan joint {config.pan_joint_index} only delivers {gain:.3f} deg of "
                "optical-axis azimuth per degree; it is not a pan axis"
            )


def test_forearm_roll_axis_would_fail_the_horizon_bound(camera_pose, anchor):
    """The bound above is discriminating, not vacuous.

    J4 (index 3) is the axis this defect shipped on.  A single default-sized
    step about it must violate the pan bound; if it ever stops doing so, the
    guard above has lost its teeth.
    """

    config = WristSearchConfig()
    reference = _horizon_tilt_deg(camera_pose(anchor))
    rolled = _horizon_tilt_deg(camera_pose(_offset(anchor, 3, config.pan_step_rad)))
    assert abs(rolled - reference) > 10.0 * MAX_PAN_HORIZON_TILT_DEG


def test_every_search_view_keeps_the_horizon_level(camera_pose, anchor):
    config = WristSearchConfig()
    search = BoundedWristSearch(config)
    reference = _horizon_tilt_deg(camera_pose(anchor))
    for view in search.views:
        joints = anchor.copy()
        joints[config.pan_joint_index] += view.pan_offset_rad
        joints[config.pitch_joint_index] += view.pitch_offset_rad
        tilt = abs(_horizon_tilt_deg(camera_pose(joints)) - reference)
        assert tilt <= MAX_GRID_HORIZON_TILT_DEG, (
            f"view {view.index} tilts the horizon by {tilt:.3f} deg"
        )


def test_pitch_axis_still_looks_up_and_down_cleanly(camera_pose, anchor):
    """The operator is explicitly happy with look-up/look-down; keep it."""

    config = WristSearchConfig()
    reference = _horizon_tilt_deg(camera_pose(anchor))

    def elevation(transform):
        return math.degrees(
            math.asin(float(np.clip(np.asarray(transform)[:3, 2][2], -1.0, 1.0)))
        )

    base_elevation = elevation(camera_pose(anchor))
    for offset in np.linspace(
        -config.max_pitch_offset_rad, config.max_pitch_offset_rad, 41
    ):
        transform = camera_pose(_offset(anchor, config.pitch_joint_index, offset))
        assert abs(_horizon_tilt_deg(transform) - reference) <= MAX_GRID_HORIZON_TILT_DEG
        if abs(offset) > 1e-6:
            gain = (elevation(transform) - base_elevation) / math.degrees(offset)
            assert 0.95 <= abs(gain) <= 1.05


def test_whole_arm_sweep_stays_inside_the_firmware_joint_limits(anchor):
    """J1 moves the WHOLE chain, so limits are checked on all six joints.

    The executor interpolates linearly between two views and only writes the
    pan and pitch indices, so every commanded configuration lies inside the
    (pan offset, pitch offset) rectangle.  A dense scan of that rectangle is
    therefore a complete proof rather than a sample.
    """

    limits = np.asarray(
        _wrist_executor_module().executor.JOINT_LIMITS_RAD, dtype=float
    )
    config = WristSearchConfig()
    worst = np.full(6, np.inf)
    pans = np.linspace(-config.max_pan_offset_rad, config.max_pan_offset_rad, 61)
    pitches = np.linspace(
        -config.max_pitch_offset_rad, config.max_pitch_offset_rad, 61
    )
    for pan in pans:
        for pitch in pitches:
            joints = anchor.copy()
            joints[config.pan_joint_index] += pan
            joints[config.pitch_joint_index] += pitch
            worst = np.minimum(
                worst, np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
            )
    assert np.all(worst > 0.0), (
        "the swept envelope leaves the firmware joint limits; worst per-joint "
        f"margin (deg) = {np.degrees(worst).round(3).tolist()}"
    )
    # The pan joint specifically must keep real headroom, not scrape a stop.
    assert worst[config.pan_joint_index] > math.radians(30.0)


def test_fixed_view_targets_reject_a_grid_that_leaves_the_limits():
    """The runtime gate that backs the test above must still fail closed."""

    module = _wrist_executor_module()
    config = WristSearchConfig()
    home = np.asarray(
        json.loads(EXAMPLE_HOME.read_text(encoding="utf-8"))["joint_radians"],
        dtype=float,
    )
    at_limit = home.copy()
    at_limit[config.pan_joint_index] = module.executor.JOINT_LIMITS_RAD[
        config.pan_joint_index, 1
    ]
    with pytest.raises(module.executor.SafetyError):
        module.fixed_view_targets(at_limit)
