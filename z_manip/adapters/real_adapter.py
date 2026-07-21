"""real_adapter (L0) — D435i (realsense2_camera) + piper_sdk (real robot side).

The real-hardware peer of :mod:`z_manip.adapters.isaac_adapter`, presenting the
SAME interface + the SAME topic names/encodings (:mod:`z_manip.adapters`, §9.3),
so migrating sim→real changes only which adapter + which DDS profile — never the
stack above. It consumes:

    /camera/color/image_raw, /camera/color/camera_info,
    /camera/aligned_depth_to_color/image_raw   — realsense2_camera (Jazzy) driver
    /piper/state, /piper/cmd                    — piper_sdk (CAN) bridge
    TF  base → camera_color_optical_frame        — realsense2_camera publishes this

There is NO ground-truth oracle on real hardware (the sim-only verify hook in
:class:`~z_manip.adapters.isaac_adapter.IsaacAdapter` has no counterpart here —
real verify uses aperture + wrist force + vision votes, ``docs/plan.md`` §3
VERIFY). Hardware interface (D435i / piper_sdk / calibration) freezes at M5 =
CEO gate (G8); this module only declares the consumer contract.

M0 skeleton: interface only; every method raises ``NotImplementedError``.
"""

from __future__ import annotations


class RealAdapter:
    """ROS2 consumer of the real-robot contract (D435i + piper_sdk)."""

    def get_color_image(self) -> object:
        """Latest ``/camera/color/image_raw`` frame (rgb8) from realsense2_camera.

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("RealAdapter is an M0 skeleton (M5, CEO-gated G8).")

    def get_aligned_depth(self) -> object:
        """Latest ``/camera/aligned_depth_to_color/image_raw`` (16UC1 mm).

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("RealAdapter is an M0 skeleton (M5, CEO-gated G8).")

    def get_camera_info(self) -> object:
        """Latest ``/camera/color/camera_info`` from realsense2_camera.

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("RealAdapter is an M0 skeleton (M5, CEO-gated G8).")
