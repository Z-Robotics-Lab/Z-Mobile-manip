"""The ROS2 contract under test — topic/frame names + M0 gate thresholds.

Single source for the strings and numbers the tests assert against. Every value
is the go2w source of truth (``~/Desktop/go2w/scripts/sim/{wrist_camera.py,
warehouse_nav.py}``), NOT a value from prose or memory. Where the runtime name
differs from what a spec might suggest, the runtime wins and it is noted here.

Nothing in this file imports Isaac (z-manip invariant): these are plain strings
and floats describing the ROS2 face both sim and real must present.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path

# ---------------------------------------------------------------- camera topics
# realsense2-aligned names (wrist_camera.TOPIC_*). Both sim and real publish
# these (sim==real contract, plan §9.3).
TOPIC_COLOR = "/camera/color/image_raw"                 # rgb8
TOPIC_COLOR_INFO = "/camera/color/camera_info"          # CameraInfo 848x480
TOPIC_DEPTH_ALIGNED = "/camera/aligned_depth_to_color/image_raw"  # 16UC1 mm

# ------------------------------------------------------------------ arm / pose
# named-pose switch channel (std_msgs/String ∈ {STOW,LOOKOUT,CARRY}).
TOPIC_NAMED_POSE = "/piper/named_pose"
# joint state / commanded target (both JointState, names j1..j6 + j7,j8).
# NOTE: runtime names are /piper/state and /piper/cmd (warehouse_nav.py:478-479),
# NOT a generic "/piper/joint_states". G-c compares these two per joint-name.
TOPIC_JOINT_STATE = "/piper/state"
TOPIC_JOINT_CMD = "/piper/cmd"
TOPIC_EE_POSE = "/piper/ee_pose"        # PoseStamped, EE GT (frame "world")

POSE_STOW = "STOW"
POSE_LOOKOUT = "LOOKOUT"
POSE_CARRY = "CARRY"
POSE_CYCLE = (POSE_STOW, POSE_LOOKOUT, POSE_CARRY)
POSE_DEFAULT = POSE_LOOKOUT             # wrist_camera.DEFAULT_POSE

# --------------------------------------------------------------- ground truth
TOPIC_CLOCK = "/clock"


def _load_office_scene() -> tuple[_Path, dict]:
    """Read the simulator's versioned scene description without importing Isaac."""
    default = _Path(__file__).resolve().parents[2] / "go2W_Sim/configs/manip_office_scene.json"
    path = _Path(_os.environ.get("Z_MANIP_SCENE_CONFIG", default)).expanduser()
    if not path.is_file():
        return path, {}
    with path.open(encoding="utf-8") as stream:
        data = _json.load(stream)
    if data.get("version") != 1:
        raise ValueError(f"unsupported manipulation scene version in {path}")
    return path, data


SCENE_CONFIG_PATH, _SCENE = _load_office_scene()
SCENE_OBJECTS = tuple(_SCENE.get("objects", ()))
PROPS = tuple(str(item["name"]) for item in SCENE_OBJECTS)
if len(PROPS) != len(set(PROPS)):
    raise ValueError(f"duplicate object name in {SCENE_CONFIG_PATH}")
PROP_XYZ = {item["name"]: tuple(float(v) for v in item["position"])
            for item in SCENE_OBJECTS}
PROP_ORIENTATION_WXYZ = {
    item["name"]: tuple(float(v) for v in item["orientation_wxyz"])
    for item in SCENE_OBJECTS
}
PROP_SHAPE_FAMILY = {item["name"]: str(item["shape_family"])
                     for item in SCENE_OBJECTS}
PROP_REQUIRES_UPRIGHT = {
    item["name"]: bool(item.get("requires_upright", False))
    for item in SCENE_OBJECTS
}


def _configured_up_axis(quaternion: tuple[float, ...]) -> tuple[float, float, float]:
    """Pick the signed local principal axis configured closest to world +Z."""
    w, x, y, z = quaternion
    world_z_row = (
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    index = max(range(3), key=lambda candidate: abs(world_z_row[candidate]))
    axis = [0.0, 0.0, 0.0]
    axis[index] = 1.0 if world_z_row[index] >= 0.0 else -1.0
    return tuple(axis)


PROP_UP_AXIS_LOCAL = {
    name: _configured_up_axis(quaternion)
    for name, quaternion in PROP_ORIENTATION_WXYZ.items()
}

_SHELF_PARTS = _SCENE.get("shelf", {}).get("parts", ())
_SUPPORT = next((item for item in _SHELF_PARTS
                 if item.get("name") == "shelf_surface"), None)
if _SUPPORT is None and _SCENE:
    raise ValueError(f"shelf_surface missing from {SCENE_CONFIG_PATH}")
SUPPORT_SURFACE_CENTER_XY = tuple(float(v) for v in (_SUPPORT or {}).get(
    "position", (0.0, 0.0, 0.0))[:2])
SUPPORT_SURFACE_HALF_XY = tuple(float(v) / 2.0 for v in (_SUPPORT or {}).get(
    "size", (0.0, 0.0, 0.0))[:2])
SUPPORT_SURFACE_TOP_Z = (
    float((_SUPPORT or {}).get("position", (0.0, 0.0, 0.0))[2])
    + float((_SUPPORT or {}).get("size", (0.0, 0.0, 0.0))[2]) / 2.0
)

# GT is an external acceptance oracle only. Runtime perception/planning/control
# are prohibited from consuming these topics or configured object poses.
PROP_POS_TOL_M = 0.10
Z_ON_SUPPORT_MIN = SUPPORT_SURFACE_TOP_Z - 0.005

# G-p7 立正 gate：YCB 规范系 Y 轴朝上（三件"高度"全在 bbox Y），生成时 Rx(+90°) 立正
# ⇒ 体 +Y 应对齐世界 +Z。由 GT odom 四元数算 (R·ŷ)·ẑ = 2(qy·qz+qw·qx)，沉降后
# tilt ≤ 15°——生成姿态错、沉降翻倒、后续被撞倒都由此拦住（G-p2 只看位置是盲区）。
PROP_TILT_MAX_DEG = 15.0
PROP_SETTLED_SPEED_MAX_MPS = 0.02
# reach gate（config 级，无需链）：缩放后台面 0.6066×0.4011（bbox 1.21323×1.00287 ×
# scale 0.5/0.4），物品到最近台缘水平距 ∈ [0.02, 0.15] m——上限=PiPER 可达窄带
# （626mm − 臂基距鼻端~0.3 − 站位余量），下限=不悬空出台。
PROP_EDGE_DIST_MAX_M = 0.15
PROP_EDGE_DIST_MIN_M = 0.02

def prop_odom_topic(name: str) -> str:
    """GT odom topic for a prop name (``/objects/<name>/odom``)."""
    return f"/objects/{name}/odom"

# ---------------------------------------------------------------------- frames
FRAME_OPTICAL = "camera_color_optical_frame"    # wrist_camera.CAM_OPTICAL_FRAME
# TF parent of the optical frame. Runtime default is "base"; the go2w side reads
# GO2W_CAM_TF_PARENT so a SLAM stack may relabel it (base_link/body). Overridable
# here too so sim==real without code edits.
FRAME_CAM_PARENT = _os.environ.get("GO2W_CAM_TF_PARENT", "base_link")

# ------------------------------------------------------------------ M0 gate values
# All from plan.md §5 M0 row + wrist_camera constants. Thresholds are the GATE,
# never softened to make a live run pass.
CAM_WIDTH = 848
CAM_HEIGHT = 480
# fx = focal/aperture*W = 1.93/2.65*848 = 617.60 px. Gate window [555,680] brackets
# the D435 color-stream family without pinning the exact sim value.
FX_MIN = 555.0
FX_MAX = 680.0
FX_NOMINAL = 617.60

ENC_COLOR = "rgb8"
ENC_DEPTH = "16UC1"

# G-a: color & depth hz floor, SIM-time domain (M0 gate N=10 frames per sim
# second = CAM_STRIDE 10 @ 100 Hz physics). Wall rate ≈ sim × RTF（本机 ~2.1 Hz）
# and is NOT the gate quantity; on the real robot stamps are wall ⇒ identical
# probe, identical gate（sim/real 通用）。
GA_HZ_MIN = 10.0
# Measurement tolerance, NOT a gate relaxation: stamp arithmetic yields e.g.
# 9.999999999999757 for an exactly-10.0 stream (IEEE dust). Nearest REAL
# failure modes are discrete strides (11 ⇒ 9.09 fps, 50 ⇒ 2.0 fps), so 0.05
# separates float noise from genuine deficits by two orders of margin.
GA_HZ_TOL = 0.05
# G-b: LOOKOUT optical-axis elevation vs horizontal ≤ 5° (|pitch|).
GB_PITCH_MAX_DEG = 5.0
# G-c: three-pose settle joint error < 0.05 rad.
GC_JOINT_ERR_MAX = 0.05
# G-d: RTF ≥ 0.15.
GD_RTF_MIN = 0.15
# G-e: nearest non-zero depth in a frame ≥ 0.28 m (near-clip; DEPTH_MIN_Z).
GE_MIN_DEPTH_M = 0.28

# GT rate: designed 5 Hz sim (warehouse_nav step % 20); gate ≥4 Hz sim-time.
GT_SIM_HZ = 5.0
GT_SIM_HZ_MIN = 4.0
GT_WALL_HZ_FLOOR = 0.5  # legacy wall-side floor（仅带宽观察用，非 gate）

# e2e "sees the scene": ≥5% of a depth frame in the [0.3, 3.0] m band.
E2E_INBAND_FRAC_MIN = 0.05

# Settle windows in SIM seconds (never wall — RTF 0.2 pitfall).
SETTLE_SIM_S = 3.0

# xfail marker for the two gates blocked by the in-flight arm-stiffness fix.
XFAIL_ARM_STIFFNESS = "arm stiffness fix in flight: task_6dd89fc1"
