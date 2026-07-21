"""isaac_adapter (L0) — consume Isaac-published ROS2 topics (sim side).

z-manip NEVER imports Isaac (invariant, ``docs/plan.md`` §0, §9). This adapter is
a pure ROS2 CONSUMER of the topics the go2w Isaac driver publishes, presented
behind the same interface as :mod:`z_manip.adapters.real_adapter` so the stack
above is platform-blind. What it consumes (all names from
:mod:`z_manip.adapters` — the sim==real contract §9.3):

    /camera/color/image_raw               (rgb8)
    /camera/color/camera_info             (CameraInfo, 848×480)
    /camera/aligned_depth_to_color/image_raw  (16UC1 mm, near-clipped 0.28 m)
    /piper/state, /piper/cmd              (JointState — arm state/target)
    TF  base → camera_color_optical_frame

Simulation truth is deliberately not exposed here. Acceptance tests may score
the completed task from outside the runtime process, but perception, planning,
execution, and verification use only camera, TF, and proprioception. M0
skeleton: interface only; every method raises ``NotImplementedError``. This
module does NOT publish the camera face — that Isaac-side publishing code lands
in go2w and its new-topic structure is a CEO gate flagged in the blueprint.
"""

from __future__ import annotations

class IsaacAdapter:
    """ROS2 consumer of the Isaac-published sim contract (no Isaac import)."""

    def get_color_image(self) -> object:
        """Latest ``/camera/color/image_raw`` frame (rgb8).

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("IsaacAdapter is an M0 skeleton.")

    def get_aligned_depth(self) -> object:
        """Latest ``/camera/aligned_depth_to_color/image_raw`` frame (16UC1 mm).

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("IsaacAdapter is an M0 skeleton.")

    def get_camera_info(self) -> object:
        """Latest ``/camera/color/camera_info`` (K/D/W/H; 848×480).

        Raises:
            NotImplementedError: in M0 — skeleton.
        """
        raise NotImplementedError("IsaacAdapter is an M0 skeleton.")
