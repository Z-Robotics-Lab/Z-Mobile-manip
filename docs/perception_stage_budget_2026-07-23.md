# Perception two-second stage budget — 2026-07-23

This report is offline-only. It reads immutable interactive reports and the
recorded rosbag; it does not initialize ROS, join Domain 20, or open CAN,
WebRTC, camera, or robot transports.

## Measured bottleneck

For the 39 classified fresh-grounding sessions at or after
`20260722-090000`:

| Stage | p50 | p95 | Two-second budget status |
|---|---:|---:|---|
| Passive capture window | 0.253 s | 0.255 s | pass |
| Fresh perception core | 4.730 s | 7.367 s | fail |
| Interactive wrapper overhead | 1.291 s | 1.589 s | fail |
| Fresh UI end-to-end | 6.024 s | 8.462 s | fail |

Successful grasp generation retained a median of 64 candidates over 38
reports. The offline rosbag CPU replay separately measures target filtering,
scene exclusion, and 64-candidate generation at about 0.182 s p50 with
byte-identical output. Therefore neither the 0.25 s passive capture nor current
CPU post-processing is the remaining dominant cost.

The rosbag request-to-first-exact-six-artifact-bundle measurement is 3.997 s
p50 and 6.227 s p95. Together with the interactive measurements above, this
localizes the principal fresh-path bottleneck to grounding/tracker production
of the first complete bundle. The additional stable 1.29–1.59 s belongs to the
interactive wrapper path. Historical reports do not contain the newer internal
stage fields, so this evidence does not split model inference from EdgeTAM
initialization; claiming that either one alone is the bottleneck would be a
guess.

## Executable budget

`scripts/benchmark_perception_latency.py` now fails closed when measurements
are missing and checks both latency and candidate-count quality:

```bash
python3 scripts/benchmark_perception_latency.py \
  artifacts/go2w_real/interactive_sessions/perception \
  --not-before-session 20260722-090000 \
  --minimum-samples 5 \
  --check \
  --output /tmp/perception-latency.json
```

The current target is:

- passive capture p95 at most 0.30 s;
- fresh core p50/p95 at most 1.50/1.70 s;
- wrapper overhead p50/p95 at most 0.20/0.30 s;
- fresh UI total p50/p95 at most 1.80/2.00 s;
- median successful grasp-candidate count at least 32.

The latest historical set exits `2`, as intended: capture and candidate
quality pass, while fresh core, wrapper overhead, and total latency fail. A new
post-change live artifact set must pass this command before perception is
reported as meeting the two-second target.

## Resident wrapper lifecycle

The deployed perception runner now keeps one Python worker resident and imports
ROS, OpenCV, and the grasp stack only once. Every UI request still executes the
same dry-run validation, exact target-identity/0.5 s age gate, six-artifact
bundle checks, passive joint capture, and candidate generation. The one-shot
container remains the compatibility fallback for isolated output paths.
The resident context also keeps its subscription endpoints discovered between
requests. It never spins callbacks, caches perception evidence, or publishes;
each request still owns an isolated node and a fresh exact-bundle transaction.
