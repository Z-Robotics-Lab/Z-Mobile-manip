# Closed-bag grounding benchmark (2026-07-23)

This benchmark replays immutable grounding seeds from the closed mobile-handoff
bag. It never initializes ROS publishers or opens robot transports. Both replay
containers run with `--network none`.

## Dataset

- Bag: `mobile-tuning-20260722-182620-mobile-handoff-tuning`
- 20 exact seed images
- 12 unique grounding requests
- 16 recorded `init_bbox` outcomes
- 12 exact fresh six-artifact perception bundles
- Scenes include visible and absent targets, motion blur, and retry offers

Recorded `init_bbox` is operational evidence from the previous pipeline, not
human-labelled ground truth. Threshold comparisons below therefore measure
agreement/retention, not detector precision or recall.

## Reproduce

Extract immutable request/seed/offer/bbox/frame identities in the ROS runtime:

```bash
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp \
  -v "$PWD:/workspace:ro" \
  -v "/home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/rosbags:/bags:ro" \
  -v "/tmp/z-mobile-grounding-seed-bench:/out" \
  z-manip-runtime:jazzy \
  python3 /workspace/scripts/offline/grounding_seed_benchmark.py extract \
  --bag /bags/mobile-tuning-20260722-182620-mobile-handoff-tuning \
  --output /out
```

Replay the local YOLOE-11S model without network access:

```bash
docker run --rm --gpus all --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp \
  -v "$PWD:/workspace:ro" \
  -v "/tmp/z-mobile-grounding-seed-bench:/bench" \
  -v "/home/yusenzlabpc/Z-Robotics-Lab/models:/models:ro" \
  z-mobile-manip-yoloe:latest \
  python3 /workspace/scripts/offline/grounding_seed_benchmark.py replay-yoloe \
  --manifest /bench/seed_manifest.json \
  --grounding-service /workspace/scripts/runtime/go2w_local_grounding_service.py \
  --model /models/yoloe-11s-seg.pt \
  --output /bench/yoloe-report.json
```

## Results

The first seed offer for each unique request took 0.489 s median and 0.656 s
p95. Retry offers are excluded because their intentional delay would otherwise
inflate the request latency. EdgeTAM took 0.422 s median and about 0.667 s p95
from accepted `init_bbox` to the first exact tracker frame.

On 19 seeds with a matching instruction, resident YOLOE-11S took 6.7 ms median
total after load (inference median 6.4 ms). Total p95 was 123 ms, including
prompt/class setup; the one-time cold path was 1.13 s. Current offline
six-artifact decode, filtering, antipodal geometry, and artifact writes took
0.181 s median and 0.182 s p95.

The measured component budget is therefore approximately:

```text
first offer       0.489 s median   0.656 s p95
YOLOE             0.007 s median   0.123 s p95
EdgeTAM acquire   0.422 s median   0.667 s p95
post-processing   0.181 s median   0.182 s p95
------------------------------------------------
component sum     1.10 s median    1.63 s p95
```

This is a component budget, not a certified live end-to-end measurement. It
does show that the current resident-local path has room below two seconds. The
historical fresh full-bundle path was 3.997 s median / 6.227 s p95 because the
old offer-to-`init_bbox` stage alone cost 3.181 s median / 4.828 s p95.

## Confidence-gate conclusion

Production geometry-qualified candidates retained by the local detector:

- detector contract `>= 0.20`: 10 seeds;
- old wrist-search gate `>= 0.55`: 6 seeds;
- three recorded no-`init_bbox` samples had no geometry-qualified candidate.

The wrist gate was therefore a second, conflicting threshold after the
detector had already applied confidence, finite-coordinate, border, and area
checks. Wrist search now uses the detector contract of `0.20` and still requires
two consecutive observations.

The detector threshold itself is not lowered. The four visible charger views
scored about 0.335, 0.055, 0.195, and 0.042; dropping the detector to 0.05 would
still miss one view while the small negative set is insufficient to establish
false-positive safety. Prompt replay also showed that grouped aliases preserve
the best single-prompt score; alias dilution was not the cause. For black boxes,
the `small black box` alias was materially stronger than `black box` and should
remain.
