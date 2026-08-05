# Wrist-frame VLM blind test (2026-07-10)

Input: the same 848x480 Isaac Office wrist frame, captured from
`/camera/color/image_raw` after the shelf scene was loaded. The frame contains a
cracker box, soup can, mustard bottle, mug, banana, and power drill. No scene
configuration or object state was sent to the models.

| Model | Mustard | Power drill | Mug | Latency |
|---|---|---|---|---|
| `qwen/qwen3.5-35b-a3b` | Correct full bottle/body; cap+shelf avoid; camera `+z` approach | Correct drill/handle; chuck avoid; `+z` | Correct mug/body; opening avoid; `+z` | 9.26-10.33 s |
| `qwen/qwen3-vl-32b-instruct` | Correct box but wrong camera `+y` approach; no avoid region | Correct drill/body; bit avoid; `+z` | Target only; no part, avoid region, or approach | 3.31-5.32 s |
| `qwen/qwen3-vl-30b-a3b-instruct` | Rejected: returned coordinates outside the required normalized contract | Not advanced after schema failure | Not advanced after schema failure | n/a |

The original decision used Qwen3.5-35B-A3B as the semantic-affordance primary
and the 32B model as a latency fallback. A 2026-07-14 live rerun exposed a
quality-routing failure on the shelf task: Qwen3.5 returned the mustard bottle
correctly in 28.0 s but exceeded the former shared 25 s timeout; 32B returned a
structurally valid box on empty shelf background in 6.4 s. The
`qwen/qwen3-vl-235b-a22b-instruct` model returned the correct bottle in 25.5 s.
Both geometry-qualified models remain in the bounded route. Live Office runs on
2026-07-15 measured the 235B VL model at 16.6-18.0 s while Qwen3.5 repeatedly
reached its 40 s provider bound, so runtime order is now 235B VL primary and
Qwen3.5 fallback. Typed transient curl failures get one fresh-process retry;
timeouts and rejected geometry do not. The 32B model is not accepted as the
production fallback for this geometry-sensitive task.

VLM output remains advisory: EdgeTAM must maintain the instance, RGB-D supplies
geometry, and collision/IK/planning remain the authority on executable motion.

## EdgeTAM handoff

The Qwen3.5 mustard box `(466,302)-(500,420)` was passed unchanged to
`yonigozlan/EdgeTAM-hf` through Transformers 4.57.6 on the local RTX 4090 D.
EdgeTAM refined it to `(470,346)-(492,423)` with 1,476 mask pixels. A subsequent
frame propagation retained `track_id=1`, produced `(469,346)-(492,423)`, and
1,517 mask pixels. This confirms the runtime division: VLM seeds once; EdgeTAM,
not repeated VLM calls, owns the persistent target used by servo and RGB-D.
