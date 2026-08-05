# Strict all-IK replay evidence (2026-07-23)

## Decision

Keep the normal planning path at four grasp symmetries and 64 hypotheses. The
bounded, translation-first Pinocchio search now resolves 11 of the 13 recorded
all-IK failures while preserving the 3 mm position tolerance, 0.03 rad
orientation tolerance, URDF limits, and collision checks. Increasing the search
to eight symmetries recovers one additional session, but its 11.85 s planning
time is incompatible with the 3 s close-range handoff target. The final session
cannot be made safe by IK tuning alone: an exactly reachable candidate fails the
attached-object lift collision check.

Do not globally increase symmetry count, loosen pose tolerances, or reduce the
2 mm attached-object collision threshold from this evidence. Persistent
failures should return a typed re-approach/re-perception disposition instead of
executing a marginal plan.

## Scope and isolation

The replay set contains the 13 immutable planning sessions that originally
ended with 64 IK rejections:

```text
20260720-033954  20260720-034126  20260720-034208
20260720-080241  20260722-100709  20260722-100915
20260722-101749  20260722-101946  20260722-102726
20260722-102927  20260722-103206  20260722-103409
20260722-103623
```

All counterfactuals used recorded RGB-D, target clouds, scene clouds, joint
states, calibration, and grasp candidates. Replays ran in Docker with
`--network none`, no device mounts, dropped Linux capabilities, and
`no-new-privileges`. No NUC, CAN, WebRTC, ROS Domain 20, base, or arm transport
was opened.

The production planner image was `z-manip-runtime:pinocchio` with Pinocchio
4.0.0. CasADi 3.7.2 exists only in the separate
`z-mobile-manip-whole-body:latest` image (also Pinocchio 4.0.0). Therefore these
primary matrix results are Pinocchio planner results. The separate CasADi
counterfactual below is explicitly identified and was not integrated into the
production planner.

## Before and after

The original reports produced zero valid plans. Their candidate search time was
5.438 s p50, 5.785 s p95, and 5.996 s maximum; total planning time was 6.010 s
p50, 6.378 s p95, and 6.594 s maximum.

Strict replays used the same 3 mm / 0.03 rad acceptance criteria:

| Search policy | Valid plans | Successful total p50 | Successful total p95 | Maximum successful total | Failed sessions |
| --- | ---: | ---: | ---: | ---: | --- |
| 4 symmetries / 64 hypotheses | 11/13 | 1.848 s | 2.844 s | 3.547 s | `080241`, `102927` |
| 8 symmetries / 128 hypotheses | 12/13 | 1.941 s | 6.573 s | 11.847 s | `080241` |
| 12 symmetries / 192 hypotheses | 12/13 | 2.070 s | 7.328 s | 13.317 s | `080241` |

Session `20260722-102927` is the only extra recovery from the larger search. It
selects candidate 44, symmetry 6 after 106 rejected hypotheses, with 11.301 s
candidate search and 11.847 s total time. This is useful diagnostic evidence,
but not an acceptable default or a 3 s handoff path. Twelve symmetries add no
success and worsen the tail.

The non-strict deployed tolerance replay can produce 13/13 plans in 1.808 s p50
and 2.435 s p95 with four symmetries. That result is intentionally excluded from
the strict acceptance decision because it uses the deployed 10 mm / 0.349 rad
tolerances rather than the requested 3 mm / 0.03 rad evidence boundary.

## CasADi strict counterfactual

A separate, non-production experiment rebuilt the calibrated six-joint FK in
CasADi 3.7.2 and solved each pregrasp/grasp/lift target with IPOPT. It did not
use `pinocchio.casadi` (that module is unavailable in the installed Pinocchio
build). Every returned state was projected inside the exact URDF bounds and
then independently rechecked with the production FK, 3 mm position tolerance,
0.03 rad orientation tolerance, URDF limits, fixture checks, scene collision,
attached-target collision, and trajectory validation. The replay remained
network-isolated and did not open hardware or ROS transports.

With four symmetries / 64 hypotheses, this counterfactual produced:

| Metric | Result |
| --- | ---: |
| Valid plans | 11/13 |
| Successful total p50 | 1.808 s |
| Successful total p95 | 1.943 s |
| Maximum successful total | 2.007 s |
| Maximum across every case | 8.333 s |
| All cases within the 3 s budget | No |

The valid sessions were `033954`, `034126`, `034208`, `100709`, `100915`,
`101749`, `101946`, `102726`, `103206`, `103409`, and `103623`. Session
`080241` remained invalid because the strict collision pipeline still rejects
the recorded scene/attached target. Session `102927` remained invalid and took
8.333 s total (7.554 s search), exposing an unbounded failure tail despite the
nominal 3 s outer search budget. In contrast, CasADi reduced `100915` from the
Pinocchio matrix's 3.547 s to 1.599 s.

Decision: do not integrate this CasADi fallback yet. Its 11 successful cases
are individually below 3 s, but the 8.333 s failure violates the whole-planning
budget, and CasADi is absent from the production planner image. The experiment
is positive feasibility evidence for a future bounded fallback only after the
solver call is made hard-cancellable and its dependency is added to the actual
planner image. It is not evidence for loosening IK or collision acceptance.

## Why `20260720-080241` is not an IK-only failure

The target centroid is approximately `[0.5834, -0.0896, 0.0194]` m in
`piper_base_link`. Under strict tolerances, candidate 46 / symmetry 0 reaches
the grasp and lift IK states, then fails the attached-target lift collision
check:

- four-symmetry replay: nearest payload-to-scene distance 1.555 mm against the
  2.000 mm threshold (margin -0.445 mm);
- twelve-symmetry replay: nearest distance 1.253 mm (margin -0.747 mm);
- collision kind: `attached_target`, at lift segment 1, sample 1;
- replay reason: the attached grasp target intersects the perceived scene.

The same session still has many other IK and approach-collision rejections, so
its final disposition is mixed IK/collision. More IK seeds cannot make the
reachable candidate collision-free, and a CasADi solution would still need to
pass the identical URDF and collision gates.

## Standoff and work-pose counterfactuals

Two additional strict sweeps tested whether a different base work pose alone
would repair `080241` while rigidly keeping candidate, target, and scene geometry
consistent:

- 35 base-frame translations over `dx = +/-0.12 m` and `dy = +/-0.14 m`:
  0/35 valid plans;
- 45 work poses over target `x = {0.50, 0.56, 0.62} m`,
  `y = {-0.14, 0, 0.14} m`, and scene yaw
  `{-20, -10, 0, 10, 20}` degrees: 0/45 valid plans.

This does not prove that every possible mobile-base pose is infeasible. It does
show that the tested local standoff/yaw optimizer cannot safely convert this
recorded grasp/scene bundle into a valid plan. The next corrective action is a
fresh close-range perception/grasp set or a typed base re-approach, not a wider
IK tolerance.

## Recommended runtime policy

1. Use four symmetries / 64 hypotheses for the normal translation-first path.
2. Preserve the bounded persistent Pinocchio warm-start policy from commit
   `e092bcd`; it improves search without stale-goal contamination.
3. Preserve exhaustive-failure caching and typed handoff disposition from
   commit `dcd6efe`; identical failed bundles must not pay the same full search
   repeatedly.
4. An eight-symmetry recovery may be exposed only as a separately budgeted
   diagnostic/recovery path. It is not the default because the recorded tail is
   11.85 s.
5. If strict search exhausts or an exact IK solution fails lift collision,
   request `NEED_BASE_APPROACH` / fresh close-range perception. Do not silently
   weaken the 3 mm / 0.03 rad or collision constraints.
6. Before introducing CasADi into grasp IK, add it to the actual planner image,
   bind it to the same calibrated PiPER model, joint limits, fixtures, and scene
   collision validation, and compare it on this exact replay set. Availability
   in the whole-body image is not integration into the planner.

## Evidence locations

The source planning reports remain under:

```text
artifacts/go2w_real/interactive_sessions/planning/<session>/artifacts/planning/planning_report.json
```

Ephemeral replay matrices produced during this audit were:

```text
/tmp/zmm-all-ik-matrix/summary.json
/tmp/zmm-all-ik-strict-matrix/summary.json
/tmp/zmm-standoff-counterfactual/summary.json
/tmp/zmm-work-pose-counterfactual/summary.json
```

They are intentionally not checked in because they contain generated replay
artifacts. The immutable session IDs, aggregate results, exact tolerances, and
decision boundary are recorded above.
