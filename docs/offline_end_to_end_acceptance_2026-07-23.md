# Offline end-to-end acceptance (2026-07-23)

This is the fail-closed acceptance result for the stopped tuning bag and its
interactive-session artifacts.  The evaluator is transport-free: it imports
no ROS or robot SDK modules and sends no commands.

## Reproduce

```bash
python3 scripts/offline/end_to_end_acceptance_summary.py \
  --bag-replay-report /tmp/z-mobile-offline-acceptance-20260722/mobile-pipeline-replay.json \
  --perception-report /tmp/z-mobile-antipodal-optimized2/report.json \
  --planning-replay-report /tmp/z-mobile-offline-acceptance-20260722/planning-replay.json \
  --interactive-root /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/interactive_sessions \
  --receipts-root /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/planning_sessions/execution-receipts \
  --json-output /tmp/z-mobile-offline-end-to-end-acceptance.json \
  --markdown-output /tmp/z-mobile-offline-end-to-end-acceptance.md
```

Add `--require-acceptance` in CI when a non-zero exit status is desired for a
rejected or evidence-incomplete run.

## Result

Overall verdict: **INCOMPLETE_EVIDENCE**.

| Stage | Evidence | Result | Observation |
|---|---|---:|---|
| Bag integrity | measured | PASS | 37 topics, 177,414 messages, valid MCAP framing, no transport or motion commands |
| Fresh perception | measured | FAIL | 12/12 exact bundles; p50 3.997 s, p95 6.227 s against the 2.0 s goal |
| Tracked perception | counterfactual | UNMEASURED | 7/9 exact identities existed; 6/9 were at most 0.5 s old |
| Close planning | measured offline replay | PASS | 5/5 near-field trials; planner wall p95 2.868 s against the 3.0 s goal |
| All-IK disposition | measured | PASS | 5 complete, non-truncated, all-IK reports mapped to `NEED_BASE_APPROACH`; 0 ambiguous |
| Executor start | unmeasured | UNMEASURED | no valid hash-linked pregrasp receipt falls inside this bag window |

The repository does contain a valid older pregrasp/approach/lift receipt chain,
but its executor start (`1784714704795191901`) predates this bag's start
(`1784715981466293780`).  It is therefore excluded rather than reused as proof
for this recording.

## Evidence contract

- Fresh perception is measured from request to the first exact six-artifact
  bundle in the stopped bag.
- Tracked reuse is deliberately labelled `COUNTERFACTUAL`: cache availability
  in recorded data is not proof that the production runtime used that cache.
- An exhaustive all-IK disposition requires a complete, non-truncated report
  in which every recorded rejection has stage `ik`. Mixed or incomplete
  failures are never promoted.
- Executor start requires a `pregrasp-receipt.json` whose
  `planning_report_sha256` matches an immutable interactive planning report,
  whose `planned_grasp_sha256` matches that report, whose source stamp matches
  the linked perception report, and whose timestamp follows successful plan
  completion inside the bag window.
- A process launch, log line, or planned trajectory is not executor-start
evidence. Missing evidence remains `UNMEASURED`.
