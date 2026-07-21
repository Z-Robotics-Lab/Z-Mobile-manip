# Deployment configuration

`configs/go2w_piper.json` is a schema v2 deployment example. The loader is
strict: unknown sections, missing sections, unresolved environment variables,
invalid URDF/collision references, and inconsistent tool geometry stop startup.

## Schema v2 migration

Schema v2 adds two top-level sections:

- `tool_geometry`: measured parallel-gripper axes, contact interval, TCP, and
  collision-model aperture.
- `work_pose`: bounded, robot-specific SE(2) base-pose search parameters.

Schema v1 is intentionally not given implicit values for these fields. A wrong
TCP or closing axis can produce a valid-looking IK solution for the wrong
physical point, so the loader returns an actionable migration error instead.
To migrate, change `schema_version` to `2`, measure and add `tool_geometry`, and
add a tuned `work_pose` section. Do not copy the PiPER values to another robot.

All JSON arrays stored in frozen configuration objects are normalized to nested
tuples at the load boundary. This includes work-pose samples,
`grasp_plan.lift_direction_base`, and `grasp_plan.tool_from_tip`.

`work_pose.radial_distances_m` describes target positions in the future arm-base
frame. Calibrate these distances with margin for the complete pregrasp,
Cartesian contact approach, and the visual-servo exit envelope; the arm's bare
maximum reach is not a valid work distance.

Visual-servo convergence uses the configured depth and lateral tolerances as
entry gates. Once settling starts, `depth_exit_hysteresis_m` and
`lateral_exit_hysteresis_m` widen only the exit gates so bounded sensor noise
does not repeatedly clear an otherwise stationary convergence window.

## Tool transform contract

`grasp_plan.tool_from_tip` retains its historical field name. Its first and
third rotation columns are respectively the candidate tool X (closing) and Z
(approach) axes expressed in `robot.tip_link`; its translation is the candidate
contact TCP expressed in that frame. The loader verifies that:

- those axes match `tool_geometry` and form a right-handed rigid transform;
- the TCP lies inside the measured finger-contact interval;
- planned maximum grasp width does not exceed the collision aperture;
- the collision model parses, references real URDF links, identifies target
  contact capsules, and, for fixed tip-frame proxies, covers the configured
  contact interval and brackets the TCP along the closing axis.

Dynamic finger-link collision proxies remain supported. Their open-state
geometry is evaluated by runtime kinematics, so the loader performs structural
and URDF-frame validation without assuming a particular robot or joint layout.
