# All-IK retry disposition benchmark (2026-07-22)

This change was evaluated offline against every `planning_report.json` under
`artifacts/`; no ROS, CAN, NUC, WebRTC, or robot transport was opened.

## Recorded evidence

All 207 reports were valid JSON:

| outcome | reports | timed reports | search p50 / p95 | total p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| valid plan | 137 | 103 | 0.366 / 3.307 s | 0.917 / 3.886 s |
| exhaustive all-IK failure | 13 | 13 | 5.438 / 5.785 s | 6.010 / 6.378 s |
| other fail-closed outcome | 57 | 24 | 4.370 / 6.332 s | 5.004 / 8.209 s |

The valid and all-IK target-radius intervals overlap (`0.353--0.652 m` versus
`0.361--0.674 m`).  A scalar distance gate would therefore reject recorded
successful plans and was not added.

## Policy

The first immutable source/joint snapshot still runs the complete planner.
Only a complete, non-truncated report in which every evaluated hypothesis was
rejected at the IK stage is classified as `NEED_BASE_APPROACH`.  The resident
worker caches that evidence using the artifact identity, measured/planning
joints, backend, and planner arguments.  A repeat of the exact request writes
the typed report without another IK search.  Changed camera artifacts, joints,
backend, or planner settings miss the cache and run normally.

Writing a cached typed report for the 13 recorded all-IK failures took 0.083 ms
p50 and 0.090 ms p95 over 260 offline writes (maximum 0.241 ms), replacing a
repeated 5.438 s p50 / 5.785 s p95 search.

URDF joint limits, exact IK tolerances, deterministic seeds, aperture checks,
and collision checks are unchanged.  Collision or mixed-stage failures remain
ordinary fail-closed planner errors.
