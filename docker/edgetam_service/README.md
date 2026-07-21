# EdgeTAM persistent-mask service

This directory packages a platform-neutral HTTP service around the official
Hugging Face Transformers `EdgeTamVideoModel` and `Sam2VideoProcessor` streaming
API. It accepts RGB JPEG frames and pixel boxes only. It does not import ROS,
Isaac Sim, or robot code, and it has no object-pose or simulator ground-truth
input.

The model is loaded on the first `init` request, so `/health` remains fast and
does not allocate GPU memory. See the upstream
[EdgeTAM Video documentation](https://huggingface.co/docs/transformers/en/model_doc/edgetam_video)
for the underlying streaming inference API.

## Build and run

The image expects an NVIDIA driver compatible with CUDA 12.8 and the NVIDIA
Container Toolkit:

```bash
docker build -t z-manip-edgetam docker/edgetam_service
docker run --rm --gpus all \
  --name z-manip-edgetam \
  -p 127.0.0.1:8092:8092 \
  -v edgetam-hf-cache:/models/huggingface \
  -e EDGETAM_DEVICE=cuda \
  z-manip-edgetam
curl --fail http://127.0.0.1:8092/health
```

The first tracking request downloads `yonigozlan/EdgeTAM-hf` into the
mounted Hugging Face cache. Pre-populate that volume and set
`HF_HUB_OFFLINE=1` for an offline robot. No API key is used by this service.

Useful runtime settings:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `EDGETAM_MODEL_ID` | `yonigozlan/EdgeTAM-hf` | Hub model ID or mounted local path |
| `EDGETAM_DEVICE` | `cuda` | Torch device; `cpu` is supported for diagnostics |
| `EDGETAM_SESSION_TIMEOUT_S` | `30` | Idle time before a session is destroyed |
| `EDGETAM_MAX_SESSIONS` | `4` | Upper bound on resident inference sessions |
| `EDGETAM_MAX_FRAMES_PER_SESSION` | `1800` | Forces periodic re-detection before sequence wrap or drift |
| `EDGETAM_MIN_MASK_PIXELS` | `16` | Empty/thin mask rejection threshold |
| `EDGETAM_MIN_SCORE` | `0.35` | Foreground-probability rejection threshold |
| `EDGETAM_VISION_CACHE_FRAMES` | `8` | Per-session vision feature cache bound |
| `EDGETAM_STREAM_HISTORY_FRAMES` | `32` | Rolling frame/output history retained for streaming |

Use one service process per GPU for predictable latency. Requests for distinct
sessions may arrive concurrently, but model inference is serialized because the
shared CUDA model and streaming processor are not treated as re-entrant.

## Protocol

All JSON documents carry `"protocol":"z-manip.edgetam/v1"`. Images are base64
encoded JPEG bytes. Boxes use half-open pixel coordinates `[x1,y1,x2,y2]`.

Endpoints:

* `GET /health`
* `POST /v1/sessions/init` with `session_id`, `frame_seq: 0`,
  `image_jpeg_b64`, and `bbox_xyxy`
* `POST /v1/sessions/update` with `session_id`, the next consecutive
  `frame_seq`, and `image_jpeg_b64`
* `POST /v1/sessions/reset` with `session_id`

A successful init or update returns this shape:

```json
{
  "protocol": "z-manip.edgetam/v1",
  "status": "tracking",
  "session_id": "pick-17",
  "track_id": "stable-opaque-id",
  "frame_seq": 4,
  "image_size": [640, 480],
  "bbox_xyxy": [182, 91, 311, 366],
  "score": 0.93,
  "mask_rle": {
    "encoding": "coco_rle",
    "size": [480, 640],
    "counts": [43862, 15, 462, 18]
  }
}
```

`mask_rle` is uncompressed COCO RLE: counts alternate background/foreground
runs over the column-major flattened boolean mask. The first count may be zero.
The client verifies that runs cover the full image and that `bbox_xyxy` exactly
bounds the decoded non-empty mask.

Session identity, track identity, dimensions, and sequence are immutable.
Timeouts, duplicate or skipped frames, dimension changes, low-confidence/empty
masks, and malformed outputs destroy the tracking lock. Motion must stop and a
new detection/init cycle must begin after any such error.
Streaming sessions retain only a bounded recent history (32 frames by default,
never less than the model's mask-memory/object-pointer requirement). This keeps
GPU memory independent of session duration while preserving conditioning-frame
state. Configure the conservative rolling window with
`EDGETAM_STREAM_HISTORY_FRAMES`.
