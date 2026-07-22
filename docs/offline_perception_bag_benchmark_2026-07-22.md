# Offline perception bag benchmark

This benchmark reads the stopped MCAP bag directly. It never calls `ros2 bag
play`, initializes `rclpy`, creates a publisher, imports robot drivers, or
opens a network/CAN/WebRTC transport.

## Reproduce

```bash
docker run --rm --network none \
  -e ROS_DOMAIN_ID=184 -e ROS_LOCALHOST_ONLY=1 -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace:ro" \
  -v /home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/rosbags/mobile-tuning-20260722-182620-mobile-handoff-tuning:/bag:ro \
  -v /tmp/z-mobile-perception-bag-benchmark:/out \
  z-manip-runtime:jazzy bash -lc \
  'source /opt/ros/jazzy/setup.bash && python3 /workspace/scripts/offline/perception_bag_benchmark.py \
    --bag /bag --output /out/report.json --maximum-cpu-bundles 3 --cpu-repeats 3'
```

## Recorded evidence

- 12/12 grounding requests produced an exact six-artifact bundle.
- Recorded fresh request-to-bundle latency: min 0.844 s, p50 3.997 s, p95
  6.227 s, max 6.959 s. The current bag therefore does not establish the
  requested sub-2-second fresh path.
- Seven of nine repeated same-instruction requests had an exact cached tracker
  identity available at request time. Six were at most 0.5 s old; one was
  7.788 s old. Exact identity alone therefore needs a separate age limit before
  it is treated as a safe fast path.
- 14,272 RGB, aligned-depth, and camera-info frames had the exact same source
  timestamp. RGB-to-depth timestamp delta p50 and p95 were both 0 ms.
- On three exact recorded bundles, each replayed three times, decode, filtering,
  antipodal proposal, and representative artifact writes took about 0.692 s
  p50. Antipodal proposal itself consumed about 0.67-0.69 s and dominates this
  CPU-only portion.

The CPU measurement excludes container startup, API handling, YOLOE, and
EdgeTAM inference. The fresh measurement includes all recorded perception and
transport delay but not caller-side process startup before the request was
recorded.
