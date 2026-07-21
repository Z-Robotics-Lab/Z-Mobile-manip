"""L0 adapter layer — the ONLY place platform differences live.

z-manip NEVER imports Isaac (invariant, ``docs/plan.md`` §0, §9). It consumes a
ROS2 contract; the Isaac-side publishers live in the go2w repo. Both adapters
here present the SAME topic names + encodings so that migrating sim→real is a
DDS-config change (which adapter + which DDS profile), not a code change:

    :mod:`z_manip.adapters.isaac_adapter` — consumes Isaac-published topics only
        (realsense2_camera-aligned camera face + /piper/* + TF).
    :mod:`z_manip.adapters.real_adapter`  — D435i (realsense2_camera Jazzy) +
        piper_sdk (CAN), same topic names + encodings.

The shared camera/topic contract (``docs/plan.md`` §9.3 — sim==real topic names
& encodings). This IS the M0 camera face the go2w side publishes:

    /camera/color/image_raw               Image rgb8
    /camera/color/camera_info             CameraInfo (K/D/W/H; 848×480; fx=fy≈616.92)
    /camera/aligned_depth_to_color/image_raw  Image 16UC1 (mm; near-clipped 0.28 m)
    TF  base → camera_color_optical_frame  (REP-105 optical: z fwd, x right, y down)

Simulation truth is intentionally absent from the runtime adapter. A separate
test harness may score outcomes after the task, but production code cannot read
those topics or methods. Skeleton only (M0). Each adapter's methods raises
``NotImplementedError``. The new
camera topic structure is a cross-package data-flow face landing in go2w (a CEO
gate flagged in the blueprint) — this module only DECLARES the contract z-manip
consumes; it does not publish it.
"""

from __future__ import annotations

# The sim==real topic contract (docs/plan.md §9.3). Single source of the names
# both adapters agree on; kept as constants so a rename is one edit, reviewed.
TOPIC_COLOR_IMAGE = "/camera/color/image_raw"                 # rgb8
TOPIC_COLOR_CAMERA_INFO = "/camera/color/camera_info"         # CameraInfo
TOPIC_ALIGNED_DEPTH = "/camera/aligned_depth_to_color/image_raw"  # 16UC1 mm
CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"           # REP-105 optical

# Camera intrinsics contract (848×480, D435 color HFOV ≈69°). The numeric values
# are DERIVED and OWNED on the Isaac side (go2w warehouse_nav.py); mirrored here
# only so the consumer can assert the CameraInfo it receives matches. Retype /
# fold into a frozen dataclass when the adapter is implemented (M1).
CAMERA_WIDTH = 848
CAMERA_HEIGHT = 480
CAMERA_FX = 616.92
CAMERA_FY = 616.92
CAMERA_CX = 424.0   # W/2 (OpenCV (W-1)/2 = 423.5 is the alternative; go2w pins one)
CAMERA_CY = 240.0   # H/2
# Depth near-clip (D435 min-Z): values below this are holed to 0 (docs/plan.md
# §M0 G-e, §9). Expressed in meters here; 16UC1 wire units are millimeters.
DEPTH_NEAR_CLIP_M = 0.28

__all__ = [
    "TOPIC_COLOR_IMAGE",
    "TOPIC_COLOR_CAMERA_INFO",
    "TOPIC_ALIGNED_DEPTH",
    "CAMERA_OPTICAL_FRAME",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CAMERA_FX",
    "CAMERA_FY",
    "CAMERA_CX",
    "CAMERA_CY",
    "DEPTH_NEAR_CLIP_M",
]
