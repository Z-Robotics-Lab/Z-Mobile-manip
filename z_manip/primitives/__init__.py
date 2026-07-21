"""L2 primitive layer — atomic ROS2-action primitives, no algorithms.

Primitives compose the L1 models into re-usable robot actions. Each is an atomic
ROS2 action (mirroring the reference stacks' primitive pattern:
``visual_servoing_base`` atomic primitives discoverable via an orchestrator's
``run_primitive``). A primitive owns the ROS boundary (parsing messages,
resolving TF, publishing feedback) and delegates all math to L1 models — it
contains no perception/grasp/plan algorithm itself.

Primitives (see each module):

- :mod:`z_manip.primitives.scan`       — in-place rotate + LOOKOUT arm sweep +
                                        per-frame detect.
- :mod:`z_manip.primitives.servo_base` — two-stage base servo; near stage
                                        directly drives ``/cmd_vel``
                                        (owner=manip_servo, ``docs/plan.md`` §1 G7).
- :mod:`z_manip.primitives.arm_goto`    — named pose / joint trajectory; publishes
                                        ``/piper/named_pose`` (String) or an arm
                                        action.
- :mod:`z_manip.primitives.track`       — EdgeTAM mask stream driver (track_3d).
- :mod:`z_manip.primitives.grasp_exec`  — pre-grasp → straight-line approach →
                                        close → lift.

Skeleton only (M0). Every primitive raises ``NotImplementedError``.
"""

from __future__ import annotations

__all__: list[str] = []
