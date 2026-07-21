# Staged pick-and-place contract

The dashboard exposes three explicit physical actions. None of these actions
may silently run a later action:

1. `pick_hold` moves from measured Home through the selected, checked plan,
   closes the gripper, lifts the object, and stops while holding it.
2. `return_home_holding` follows the exact checked outbound joint paths in
   reverse while keeping the gripper closed, then stops at measured Home.
3. `place_back` is also the recovery action. From Home it follows the checked
   paths forward to the original grasp pose. From `holding_at_lift` it directly
   reverses the checked lift. In both cases it opens only at the original grasp
   pose, then reverses approach and transit to Home.

The durable workflow phases are:

```text
ready_at_home
    -- pick_hold --> holding_at_lift
    -- return_home_holding --> holding_at_home
holding_at_lift
    -- place_back --> placed_back_at_home
holding_at_home
    -- place_back --> placed_back_at_home
```

## Transition invariants

- Every transition is bound to one immutable planning artifact and one
  planning session. A receipt from another artifact or session is never
  accepted.
- Every successful receipt contains the previous receipt's digest. This forms
  an ordered chain and prevents a stale receipt from authorizing a later
  action.
- `return_home_holding` executes the exact reverse of the recorded lift,
  approach, and transit joint paths. It does not interpolate a new direct Home
  path.
- The gripper stays closed from verified contact until the original grasp pose
  is reached during `place_back`. It is never opened by `pick_hold` or
  `return_home_holding`.
- `place_back` releases at the original grasp pose and only then performs the
  reverse approach and transit.
- Home verification is required at the beginning of `pick_hold`, at the end of
  `return_home_holding`, and at the end of `place_back`.
- A failed or interrupted action does not advance the durable phase and does
  not manufacture a success receipt.
- Returning Home without the held-object workflow explicitly clears the
  workflow. Historical receipts may remain for diagnosis but cannot authorize
  a new transition.
- Starting a new perception/planning task invalidates an unfinished workflow;
  receipts and paths from the prior task cannot cross-contaminate it.

## HTTP and dashboard contract

The loopback workbench uses these explicit actions:

| Button | POST endpoint | Action header | Valid starting phase |
| --- | --- | --- | --- |
| Pick & Hold | `/api/grasp/pick-hold` | `grasp-pick-hold` | `ready_at_home` |
| Return Home Holding | `/api/grasp/return-home-holding` | `grasp-return-home-holding` | `holding_at_lift` |
| Place Back | `/api/grasp/place-back` | `grasp-place-back` | `holding_at_lift` or `holding_at_home` |

`GET /api/grasp/status` exposes `workflow.phase`, `artifact_id`,
`planning_session_id`, `holding_object`, and `at_home`. Buttons are enabled
only for the valid next transition. The browser must not infer success from a
request being accepted; it waits for the corresponding terminal workflow
phase.

All POSTs retain the existing same-origin, exact action-header, bounded JSON
body, single-active-action, and bounded-speed checks. No action accepts a path,
joint target, artifact identifier, session identifier, or gripper command from
the browser.
