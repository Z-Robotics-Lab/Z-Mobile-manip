# Persistent Pinocchio IK warm-start benchmark

This benchmark isolates the persistent planning worker from all robot transports.
It ran in an unprivileged Docker container with `--network none`, read-only code,
recorded close-range artifacts, and no ROS, CAN, WebRTC, or hardware access.

## Change

The solver now prepends only the nearest valid historical IK solution before its
unchanged deterministic global seed set. Previously it prepended the three
nearest historical solutions. The extra historical seeds repeated local search
in the same basin and increased tail latency without changing the selected plan.

No acceptance or safety criterion changed:

- exact IK tolerances remain 3 mm translation and 0.03 rad orientation;
- all deterministic global restarts remain available;
- URDF joint limits remain mandatory;
- approach, self, fixed-fixture, and point-cloud collision checks are unchanged.

## Recorded replay

Five close-range sessions were replayed through one persistent worker. The table
uses the last two rounds after process warm-up (10 plans per variant).

| Variant | Success | p50 wall | p95/max wall | Worst candidate rejections |
| --- | ---: | ---: | ---: | ---: |
| Three historical seeds | 10/10 | 1.246 s | 1.864 s | 20 |
| Nearest historical seed | 10/10 | 1.161 s | 1.752 s | 20 |

This reduces steady-state p50 by 6.9% and p95 by 6.0%. Candidate selection,
symmetry selection, and rejection counts were unchanged in every paired replay.
The optimized cold first round also remained 5/5 successful; its p50 was 1.173 s
and its maximum was 2.093 s.

## Perturbation replay

The five sessions were replayed five times each with deterministic 3 mm Gaussian
target perturbations, 1 degree joint-seed perturbations, and shuffled candidate
order. All 25/25 trials produced a valid plan. Planner-search latency was
1.304 s p50 and 1.737 s p95 (1.777 s maximum). Container wall time, including
fresh process/import/setup cost on every trial, was 2.427 s p50 and 2.899 s p95.

The relevant unit and integration suite completed with 52 passing tests.

## All-IK-failure handoff evidence

The recorded `20260722-103623` failure rejected all 64 hypotheses in IK after
5.640 s of search. Its URDF-derived conservative maximum tip radius is 0.750742 m.
Candidate grasp radii were 0.4645--0.4803 m; even the conservative
`grasp radius + 0.12 m pregrasp` bound was at most 0.6003 m. Therefore none of
the 64 targets was rejected by the proof-safe maximum-radius gate. None reached
within 20 mm translation while merely missing orientation; all 64 retained
66--634 mm best translation residuals. This is evidence for requesting another
base approach / fresh close-range observation, rather than relaxing IK, joint,
or collision limits or repeatedly spending the complete 64-hypothesis budget.
