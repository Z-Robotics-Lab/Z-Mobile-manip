# Mobile handoff benchmark — 2026-07-22

This note records the offline evidence used to tune the mobile handoff pipeline.
No replay in this report opened ROS, CAN, WebRTC, or a PiPER transport.

Two evidence generations are intentionally kept separate below. The rosbag
records the older, live-observed pipeline and is the authority for its actual
latency and failure modes. The newer warm-worker, Monte Carlo, and overlap
results are **offline optimized benchmarks** derived from those captures. They
do not establish a successful physical grasp or replace the next live run.

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

### Current persistent-planner benchmark

Keeping the planner process and robot model resident removes the remaining
per-call setup cost. On the same five genuine close-range captures, the current
persistent worker produced:

- success: **5/5**;
- planner wall time: p50 **1.266 s**, max **1.814 s**;
- worker setup: about **3 ms**.

The success criterion remains the production candidate, IK, and collision
gate; this is not a relaxed feasibility check. It is nevertheless an offline
benchmark: no arm motion or grasp execution occurred.

Target and joint-state perturbation replay adds two robustness checks around
the recorded close-range states:

- realistic noise, 3 mm target sigma and 0.5 deg joint sigma: **25/25**;
- stress noise, 5 mm target sigma and 1.0 deg joint sigma: **15/15**.

These trials demonstrate repeatable offline IK/planning feasibility near the
recorded states. They do not measure calibration error, actuator tracking, or
physical grasp success.

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

The historical artifact set further bounds what can be claimed: fresh
grounding core p50 is **4.604 s**, its wrapper total p50 is **5.596 s**, and
wrapper overhead p50 is **1.314 s**. All **375** reports predate the new stage
timing fields; consequently the tracked-path wrapper total and overhead have
zero instrumented samples. The implementation now emits those fields, but a
new live artifact set is required before reporting fresh or tracked end-to-end
perception below 2 s.

An additional 4090-only benchmark isolated dynamic YOLOE prompt setup. The
detector, 640-pixel input, confidence threshold, and box selection remained
unchanged. Persisting MobileCLIP and caching only the exact normalized phrase
reduced a new prompt request from **0.251–0.328 s** of prompt setup plus
inference to **0.010–0.015 s** end to end after service warmup. Embeddings for
`white charger`, `red bottle`, and `black box` were elementwise identical to
the former implementation (`max_abs_diff = 0`). Historical fresh perception
still contains about 1.46 s of capture/wrapper overhead, so this optimization
does not by itself establish the fresh `< 2 s` target.

## Recorded handoff lifecycle (historical implementation)

The benchmark now pairs depth-servo `stopped` records, interactive perception
attempts, linked planning attempts, and the passive-joint source timestamp in
the strict rosbag `[start, end)` window. These timings describe the
**implementation recorded in the bag**, before the warm perception/planning
runner changes above. They are not measurements of the current warm runners.

Five handoff transactions have complete stop, fresh-perception, and planning
timestamps:

- base stop -> fresh perception start: p50 **1.846 s**, p95/max **2.221 s**;
- fresh perception: p50 **5.905 s**, p95/max **7.176 s**;
- perception finish -> planner start: p50 **0.010 s**, p95 **0.014 s**;
- planning: p50 **7.177 s**, p95/max **7.603 s**;
- base stop -> plan finish: p50 **14.671 s**, p95/max **16.525 s**.

The latest overwritten handoff log contains one explicit passive-joint source
stamp. It places the first accepted post-stop sample **1.841 s** after stop and
only **0.025 s** before fresh perception starts. Older logs do not retain this
evidence, so the benchmark reports one joint-source sample rather than
extrapolating it to all five transactions. All five linked plans were blocked;
no executor/grasp-start timestamp exists in these artifacts. Worker launch is
therefore not counted as grasp execution.

The depth-servo transition itself is small but intentional:

- handoff settle -> probe: p50 **0.300 s**, p95 **0.354 s**;
- probe -> stopped: p50 **0.150 s**, p95 **0.200 s**.

### Strict post-stop passive-joint latency

The bag contains **17,045** `/piper/state` messages. All 17,045 have the exact
six PiPER joint names and six finite positions. Source-stamp inter-arrival is
50.06 ms at p50, 54.77 ms at p95, and 54.80 ms at p99. Long ownership/offline
gaps are kept separate from normal observer cadence rather than hidden by a
larger acceptance window.

Across the six recorded `stopped` lifecycle boundaries, the first complete
joint message whose **source stamp is strictly newer than stop** was actually
delivered after 190.3–219.8 ms (p50 about 201.4 ms). The source stamp itself
crossed the boundary within 0.8–32.5 ms, but the bag shows a median
record-minus-source delay of 191.8 ms. Accepting the first message delivered
after stop would therefore be unsafe: its source can still predate stop.

The readiness watcher now polls its unchanged strict gate every **10 ms**
instead of 50 ms. This removes up to 40 ms of avoidable detection delay after
the valid sample arrives. It does not change the 2 s fail-closed timeout or
relax the post-stop source epoch, complete-six-joint, read-only, freshness,
zero-command, URDF-limit, or stationary-range requirements.

The direct receive-only CAN evidence window remains **0.25 s**. Recent reports
contain 150–151 filtered frames (50–51 per joint feedback CAN ID), with a
4.84–4.91 ms joint snapshot span and zero measured joint range. Shortening that
window would weaken slow-motion detection, and it already runs in parallel
with perception, so it is not a successful-handoff critical-path saving.

The 10–14 ms perception/planner orchestration gap is negligible. The actionable
critical path is the post-stop joint-readiness wait, fresh perception, and IK
search. The safe latency change is to start the read-only fresh capture and
the post-stop passive-joint watcher concurrently, then join both before any
transform, planning, or motion gate. The joint sample must still be measured
after the base-stop timestamp; this does not relax readiness or collision
acceptance. A continuous passive cache should expose the accepted source stamp
instead of adding a fixed sleep.

The current safe parallel handoff implementation starts the read-only capture
and post-stop joint watcher together and joins both before planning. Based on
the recorded lifecycle timestamps, its expected critical-path recovery is
about **0.904 s**. This is a projection from overlapping two already-required
operations, not a live latency measurement and not permission to bypass the
post-stop joint epoch check.

The current offline projection combines the unverified tracked UI estimate of
about **1.68 s** with the persistent-planner p50 of **1.266 s**, giving
**2.946 s** before grasp-start overhead. The planner maximum would instead give
**3.494 s**. Therefore neither the perception `< 2 s` goal nor the complete
replan-to-grasp `< 3 s` goal is claimed yet; both require a post-change live
capture that includes the executor start timestamp.

Reproduce the bounded lifecycle report with:

```bash
python3 scripts/offline/mobile_handoff_benchmark.py \
  --bag /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/rosbags/mobile-tuning-20260722-182620-mobile-handoff-tuning \
  --sessions-root /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/interactive_sessions \
  --trace-jsonl /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/latest/depth-servo.trace.jsonl \
  --grasp-log /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/planning_sessions/piper-grasp.log \
  --output /tmp/z-mobile-handoff-lifecycle.json
```

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

1. fresh and same-target tracked wrapper totals are emitted and separately
   remain below 2 s;
2. every capture inside 0.70 m reaches planning rather than base approach;
3. persistent close-range planner p50/max agree with the 1.266/1.814 s offline
   benchmark;
4. base stop -> executor/grasp start is measured, including the projected
   0.904 s safe-overlap recovery, and is below 3 s;
5. no motion command is issued when the disposition is `NEED_BASE_APPROACH`.
