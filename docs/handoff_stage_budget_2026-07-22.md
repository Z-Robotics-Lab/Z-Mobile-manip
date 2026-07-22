# Handoff stage budget — 2026-07-22

This is a filesystem-only latency projection. It does not open ROS, CAN,
WebRTC, a camera, or a robot driver. A projection is not accepted as evidence
that the grasp executor started.

## Three-second critical path

Fresh RGB-D acquisition and the post-stop passive-joint watcher may run in
parallel. Planning may start only after both finish:

```text
base stopped
  ├─ fresh perception ───────┐
  └─ post-stop passive joint ├─ planning ─ executor start
                             ┘
```

The current offline p50 inputs are:

- tracked/fresh perception estimate: **1.680 s**;
- perception-to-planner dispatch: **0.010228 s**;
- persistent planner p50: **1.266 s**, max **1.814 s**;
- the only recorded post-stop passive sample: **1.840508 s**;
- executor-start dispatch: **not recorded**.

The strict projections are:

| Scenario | Plan finish after stop | Budget remaining for executor start |
|---|---:|---:|
| p50, recorded passive readiness | 3.116736 s | -0.116736 s |
| p50, projected 50 ms passive cache | 2.956228 s | 0.043772 s |
| planner max, projected 50 ms passive cache | 3.504228 s | -0.504228 s |

Even the optimistic p50 projection leaves only **43.772 ms** for dispatching
and proving executor start. The `< 3 s` target is therefore not established.
The planner-max scenario misses it by **504.228 ms before executor start**.

The recorded perception-to-planner gap is only 10–14 ms. Removing that gap
would not make the worst case pass and would risk conflating two immutable
artifacts. No validation stage was removed. The useful work is to measure and
reduce fresh perception, maintain a continuously updated passive cache while
still requiring a sample newer than base stop, reduce planner tail latency,
and record an explicit executor-start receipt.

## Strict evidence rule

A live `< 3 s` result is accepted only when the same transaction contains:

1. base-stop timestamp;
2. fresh RGB-D source epoch at or after base stop;
3. passive-joint source epoch at or after base stop;
4. a successful plan-finish timestamp after both inputs;
5. a grasp/executor-start timestamp after plan finish.

The current bag has five paired stop/perception/planning transactions but
**0/5 complete evidence chains**. All recorded plans in those transactions
were blocked and there is no grasp-start timestamp.

## Reproduce

First create the bounded lifecycle report as documented in
`performance_benchmark_2026-07-22.md`, then run:

```bash
python3 scripts/offline/handoff_stage_budget.py \
  --profile scripts/offline/profiles/handoff_optimized_20260722.json \
  --handoff-report /tmp/z-mobile-handoff-lifecycle.json \
  --output /tmp/z-mobile-handoff-budget.json
```

The simulator is fail-closed: missing executor latency remains `null`, and a
projection can never satisfy the recorded-evidence gate.
