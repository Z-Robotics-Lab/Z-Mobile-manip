# Mobile handoff benchmark — 2026-07-22

This note records the offline evidence used to tune the mobile handoff pipeline.
No replay in this report opened ROS, CAN, WebRTC, or a PiPER transport.

## Dataset

- Rosbag: `mobile-tuning-20260722-182620-mobile-handoff-tuning`
- Duration: 956.0 s
- Size: 17.5 GiB, four valid Zstd MCAP splits
- Planning captures in the bag time window: 11
- Depth-servo replay: **68/270** trace records are inside the strict rosbag
  `[start, end)` window; **202** older records are excluded. The bounded report
  is complete with no integrity issues.

## Planning result

The target-cloud median is transformed into `piper_base_link` before planning.
The production gate now classifies the result as:

- `< 0.60 m`: `NEAR_FIELD_IK`
- `0.60–0.70 m`: `PRECISION_IK`
- `> 0.70 m`: `NEED_BASE_APPROACH`; do not launch the expensive IK container

This distinction is essential: six captures were 0.875–1.560 m from the arm
base and were physically outside the handoff workspace. They are not IK
failures. They now return `NEED_BASE_APPROACH` in about 0.16 s.

For the five genuine handoff captures (0.371–0.507 m), the original
one-container-per-click replay produced:

- success: **5/5**
- planner wall time: p50 **2.42 s**, p95/max **3.12 s**
- search time: p50 **1.23 s**, p95/max **1.88 s**
- four of five planning calls completed below 3.0 s

The production warm planner runner removes only Docker/Python cold-start
jitter. It keeps the artifact tree read-only, writes through a unique scratch
mount, has no network or devices, and atomically promotes the completed report.
On the same five captures:

- success: **5/5**
- planner wall times: **2.25, 2.43, 1.75, 1.81, 1.93 s**
- p95/max: **2.43 s**
- candidate count, symmetry coverage, selected candidate, and rejection count
  are unchanged from the cold-run baseline

The replay report is `/tmp/z-mobile-planning-replay-2f8c3d5.json` on the tuning
machine. Reproduce it with `scripts/offline/planning_replay_benchmark.py` as
documented in the README.

## Perception result

Historical captures in this bag predate the warm-runner change and therefore
remain slow (fresh perception p50 about 6.03 s). The current implementation:

- reuses EdgeTAM only when the normalized target instruction hash matches;
- keeps the perception runner warm instead of paying a Docker/import cold start
  for every UI click;
- records explicit per-stage timing for future regression reports.

Existing valid tracked-target samples show a core p50 of **0.84 s**. The
expected steady UI path is about **1.68 s**, but the `< 2 s` claim must be
validated with a new post-change real capture. Fresh YOLOE acquisition and
same-target tracked refresh must be reported separately.

An additional 4090-only benchmark isolated dynamic YOLOE prompt setup. The
detector, 640-pixel input, confidence threshold, and box selection remained
unchanged. Persisting MobileCLIP and caching only the exact normalized phrase
reduced a new prompt request from **0.251–0.328 s** of prompt setup plus
inference to **0.010–0.015 s** end to end after service warmup. Embeddings for
`white charger`, `red bottle`, and `black box` were elementwise identical to
the former implementation (`max_abs_diff = 0`). Historical fresh perception
still contains about 1.46 s of capture/wrapper overhead, so this optimization
does not by itself establish the fresh `< 2 s` target.

## Safety and regression checks

- Final IK position/orientation acceptance and collision checks were retained.
- Impossible targets are rejected from a URDF-derived reach bound before
  multi-seed IK.
- Candidate scoring now tests useful symmetries earlier without pruning the
  candidate or symmetry set.
- A verified `NEED_BASE_APPROACH` result is propagated as a recoverable mobile
  disposition; incomplete gate evidence remains fail-closed.
- The mobile-handoff lifecycle remains owned until its passive watcher records
  success, typed recovery, failure, or timeout; a second approach cannot race
  the fresh planning/execution worker.
- Warm-planner reports are bounded, non-symlink JSON files and are promoted
  atomically. Missing output or runner failure cannot masquerade as success.
- Full offline suite: **991 passed, 50 skipped, 0 failed**.
- The skipped tests require unavailable live milestone state or optional host
  Pinocchio/CasADi packages; Docker replay covers the planner used here.

## Next live validation

After the operator returns, record a new short bag and check:

1. same-target tracked perception UI time is below 2 s;
2. every capture inside 0.70 m reaches planning rather than base approach;
3. close-range warm planner calls remain below 3 s;
4. no motion command is issued when the disposition is `NEED_BASE_APPROACH`.
