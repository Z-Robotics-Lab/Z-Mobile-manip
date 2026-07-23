"""Strict, platform-neutral client for the external EdgeTAM mask service.

The wire protocol intentionally carries only JPEG observations and image-space
results.  It has no ROS, simulator, depth, or object-database dependency.  A
caller may inject a transport for tests or use :class:`UrllibJsonTransport` for
the standalone HTTP service in ``docker/edgetam_service``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import re
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np


PROTOCOL_VERSION = "z-manip.edgetam/v1"
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TERMINAL_SESSION_ERROR_CODES = frozenset(
    {
        "image_size_changed",
        "out_of_order",
        "session_closed",
        "session_expired",
        "session_frame_limit",
        "tracking_lost",
        "unknown_session",
    },
)


class EdgeTamServiceError(RuntimeError):
    """Base error for transport and protocol failures."""


class EdgeTamTransportError(EdgeTamServiceError):
    """The service could not be reached or returned an HTTP-level error."""


class EdgeTamProtocolError(EdgeTamServiceError):
    """A request or response violated the versioned tracking protocol."""


class EdgeTamTrackingLost(EdgeTamServiceError):
    """The active identity is no longer safe for control to consume."""

    def __init__(self, message: str, *, reason_code: str = "tracking_lost") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@runtime_checkable
class JsonTransport(Protocol):
    """Injectable JSON request boundary used by :class:`EdgeTamServiceClient`."""

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Perform one request and return a decoded JSON object."""


@dataclass(frozen=True, eq=False)
class EdgeTamTrack:
    """One validated persistent-mask observation."""

    session_id: str
    track_id: str
    frame_seq: int
    image_size: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask: np.ndarray

    # Discriminates from :class:`EdgeTamCoast` without an isinstance import.
    coasting: bool = False


@dataclass(frozen=True, eq=False)
class EdgeTamCoast:
    """A session-preserving keep-alive frame that carries no mask.

    The service produced no trustworthy target mask this frame (brief
    occlusion / motion blur) but kept the identity alive.  Identity fields stay
    in strict lockstep with the tracking contract.  Callers must never treat a
    coast as an observation: it has no mask, bbox, or score and must not be
    published or verified.
    """

    session_id: str
    track_id: str
    frame_seq: int
    image_size: tuple[int, int]

    coasting: bool = True


def _json_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EdgeTamProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _image_size(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EdgeTamProtocolError("image_size must be [width, height]")
    width = _json_int(value[0], name="image_size width", minimum=1)
    height = _json_int(value[1], name="image_size height", minimum=1)
    return width, height


def _bbox(value: object, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EdgeTamProtocolError("bbox_xyxy must contain four integers")
    x1, y1, x2, y2 = (
        _json_int(item, name=f"bbox_xyxy[{index}]")
        for index, item in enumerate(value)
    )
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise EdgeTamProtocolError("bbox_xyxy is empty or outside the image")
    return x1, y1, x2, y2


def encode_coco_rle(mask: object) -> dict[str, object]:
    """Encode a 2-D bool mask as uncompressed COCO column-major RLE."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("mask must be a non-empty 2-D array")
    flat = array.reshape(-1, order="F")
    counts: list[int] = []
    current = False
    run = 0
    for pixel in flat:
        value = bool(pixel)
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    return {
        "encoding": "coco_rle",
        "size": [int(array.shape[0]), int(array.shape[1])],
        "counts": counts,
    }


def decode_coco_rle(value: object, *, max_pixels: int = 16_777_216) -> np.ndarray:
    """Decode and strictly validate an uncompressed COCO bool-mask RLE."""

    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if not isinstance(value, Mapping):
        raise EdgeTamProtocolError("mask_rle must be an object")
    if value.get("encoding") != "coco_rle":
        raise EdgeTamProtocolError("mask_rle encoding must be coco_rle")
    size = value.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise EdgeTamProtocolError("mask_rle size must be [height, width]")
    height = _json_int(size[0], name="mask_rle height", minimum=1)
    width = _json_int(size[1], name="mask_rle width", minimum=1)
    if height * width > max_pixels:
        raise EdgeTamProtocolError("mask_rle exceeds the configured pixel limit")
    counts = value.get("counts")
    if not isinstance(counts, (list, tuple)) or not counts:
        raise EdgeTamProtocolError("mask_rle counts must be a non-empty list")

    parsed: list[int] = []
    for index, count in enumerate(counts):
        count = _json_int(count, name=f"mask_rle counts[{index}]")
        if index > 0 and count == 0:
            raise EdgeTamProtocolError("only the first COCO RLE run may be zero")
        parsed.append(count)
    pixel_count = height * width
    if sum(parsed) != pixel_count:
        raise EdgeTamProtocolError("mask_rle runs do not cover the declared image")

    flat = np.empty(pixel_count, dtype=bool)
    offset = 0
    foreground = False
    for count in parsed:
        flat[offset : offset + count] = foreground
        offset += count
        foreground = not foreground
    return flat.reshape((height, width), order="F")


class UrllibJsonTransport:
    """Small stdlib HTTP transport with bounded response decoding."""

    def __init__(self, base_url: str, *, max_response_bytes: int = 8_000_000):
        parsed = urlparse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("EdgeTAM service URL must be absolute HTTP(S)")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials must not be embedded in the service URL")
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes is too small")
        self._base_url = base_url.rstrip("/")
        self._max_response_bytes = int(max_response_bytes)

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("service paths must be absolute and host-relative")
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urlrequest.Request(
            self._base_url + path,
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urlrequest.urlopen(request, timeout=timeout_s) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise EdgeTamTransportError(
                        f"service returned unexpected content type {content_type!r}",
                    )
                raw = response.read(self._max_response_bytes + 1)
        except urlerror.HTTPError as exc:
            detail = ""
            error_code = ""
            try:
                payload_raw = exc.read(32_768)
                error_doc = json.loads(payload_raw.decode("utf-8"))
                error = error_doc.get("error", {})
                if isinstance(error, Mapping):
                    code = error.get("code")
                    message = error.get("message")
                    if isinstance(code, str):
                        error_code = code.strip()[:128]
                    if isinstance(message, str):
                        message = message.replace("\n", " ").strip()[:256]
                    else:
                        message = ""
                    detail = error_code or message
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            suffix = f" ({detail})" if detail else ""
            message = f"EdgeTAM HTTP {exc.code}{suffix}"
            if exc.code == 410 or error_code in _TERMINAL_SESSION_ERROR_CODES:
                raise EdgeTamTrackingLost(
                    message,
                    reason_code=error_code or "remote_session_gone",
                ) from exc
            raise EdgeTamTransportError(message) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise EdgeTamTransportError(f"EdgeTAM request failed: {exc}") from exc
        if len(raw) > self._max_response_bytes:
            raise EdgeTamTransportError("EdgeTAM response exceeds configured limit")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EdgeTamTransportError("EdgeTAM response is not valid UTF-8 JSON") from exc
        if not isinstance(decoded, Mapping):
            raise EdgeTamTransportError("EdgeTAM response root must be an object")
        return decoded


class EdgeTamServiceClient:
    """Stateful client that fails closed on every identity or mask anomaly."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8092",
        *,
        transport: JsonTransport | None = None,
        request_timeout_s: float = 5.0,
        session_idle_timeout_s: float = 30.0,
        min_mask_pixels: int = 16,
        min_score: float = 0.35,
        max_jpeg_bytes: int = 8_000_000,
        max_image_pixels: int = 4_194_304,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if request_timeout_s <= 0.0 or session_idle_timeout_s <= 0.0:
            raise ValueError("request and session timeouts must be positive")
        if min_mask_pixels < 1:
            raise ValueError("min_mask_pixels must be positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if max_jpeg_bytes < 1024:
            raise ValueError("max_jpeg_bytes is too small")
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be positive")
        self._transport = transport or UrllibJsonTransport(base_url)
        self._request_timeout_s = float(request_timeout_s)
        self._session_idle_timeout_s = float(session_idle_timeout_s)
        self._min_mask_pixels = int(min_mask_pixels)
        self._min_score = float(min_score)
        self._max_jpeg_bytes = int(max_jpeg_bytes)
        self._max_image_pixels = int(max_image_pixels)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._clear()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._session_id is not None

    def health(self) -> Mapping[str, Any]:
        response = self._request("GET", "/health", None)
        if response.get("protocol") != PROTOCOL_VERSION or response.get("status") != "ok":
            raise EdgeTamProtocolError("incompatible or unhealthy EdgeTAM service")
        return response

    def init(
        self,
        image_jpeg: bytes,
        bbox_xyxy: Sequence[int],
        *,
        session_id: str,
        frame_seq: int = 0,
    ) -> EdgeTamTrack:
        with self._lock:
            if self._session_id is not None:
                raise EdgeTamProtocolError("reset the active session before reinitializing")
            session_id = self._validate_session_id(session_id)
            if frame_seq != 0:
                raise EdgeTamProtocolError("a new tracking session must start at frame_seq 0")
            encoded = self._encode_jpeg(image_jpeg)
            raw_bbox = self._request_bbox(bbox_xyxy)
            payload = {
                "protocol": PROTOCOL_VERSION,
                "session_id": session_id,
                "frame_seq": 0,
                "image_jpeg_b64": encoded,
                "bbox_xyxy": list(raw_bbox),
            }
            try:
                response = self._request("POST", "/v1/sessions/init", payload)
                track = self._parse_track(
                    response,
                    expected_session=session_id,
                    expected_seq=0,
                    expected_track=None,
                    expected_size=None,
                )
            except Exception:
                self._clear()
                raise
            self._session_id = session_id
            self._track_id = track.track_id
            self._image_size = track.image_size
            self._last_frame_seq = 0
            self._last_activity_s = self._monotonic()
            return track

    def update(
        self,
        image_jpeg: bytes,
        *,
        frame_seq: int | None = None,
    ) -> EdgeTamTrack | EdgeTamCoast:
        with self._lock:
            self._require_active()
            now = self._monotonic()
            if now - float(self._last_activity_s) > self._session_idle_timeout_s:
                self._clear()
                raise EdgeTamTrackingLost("EdgeTAM session exceeded its idle timeout")
            expected_seq = int(self._last_frame_seq) + 1
            if frame_seq is None:
                frame_seq = expected_seq
            if isinstance(frame_seq, bool) or frame_seq != expected_seq:
                self._clear()
                raise EdgeTamTrackingLost(
                    f"out-of-order frame: expected {expected_seq}, got {frame_seq}",
                )
            payload = {
                "protocol": PROTOCOL_VERSION,
                "session_id": self._session_id,
                "frame_seq": frame_seq,
                "image_jpeg_b64": self._encode_jpeg(image_jpeg),
            }
            try:
                response = self._request("POST", "/v1/sessions/update", payload)
                if response.get("status") == "coasting":
                    result: EdgeTamTrack | EdgeTamCoast = self._parse_coast(
                        response,
                        expected_session=str(self._session_id),
                        expected_seq=frame_seq,
                        expected_track=str(self._track_id),
                        expected_size=self._image_size,
                    )
                else:
                    result = self._parse_track(
                        response,
                        expected_session=str(self._session_id),
                        expected_seq=frame_seq,
                        expected_track=str(self._track_id),
                        expected_size=self._image_size,
                    )
            except Exception:
                self._clear()
                raise
            self._last_frame_seq = frame_seq
            self._last_activity_s = self._monotonic()
            return result

    def reset(self) -> None:
        with self._lock:
            session_id = self._session_id
            self._clear()
            if session_id is None:
                return
            payload = {"protocol": PROTOCOL_VERSION, "session_id": session_id}
            response = self._request("POST", "/v1/sessions/reset", payload)
            if (
                response.get("protocol") != PROTOCOL_VERSION
                or response.get("status") != "reset"
                or response.get("session_id") != session_id
            ):
                raise EdgeTamProtocolError("invalid EdgeTAM reset acknowledgement")

    def _clear(self) -> None:
        self._session_id: str | None = None
        self._track_id: str | None = None
        self._image_size: tuple[int, int] | None = None
        self._last_frame_seq: int | None = None
        self._last_activity_s: float | None = None

    def _require_active(self) -> None:
        if self._session_id is None:
            raise EdgeTamTrackingLost("no active EdgeTAM tracking session")

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        try:
            response = self._transport.request(
                method,
                path,
                payload,
                self._request_timeout_s,
            )
        except EdgeTamServiceError:
            raise
        except Exception as exc:
            raise EdgeTamTransportError(f"EdgeTAM transport failed: {exc}") from exc
        if not isinstance(response, Mapping):
            raise EdgeTamProtocolError("EdgeTAM transport returned a non-object response")
        return response

    def _parse_track(
        self,
        response: Mapping[str, Any],
        *,
        expected_session: str,
        expected_seq: int,
        expected_track: str | None,
        expected_size: tuple[int, int] | None,
    ) -> EdgeTamTrack:
        if response.get("protocol") != PROTOCOL_VERSION:
            raise EdgeTamProtocolError("EdgeTAM protocol version mismatch")
        if response.get("status") != "tracking":
            raise EdgeTamTrackingLost("EdgeTAM service did not report a tracking state")
        if response.get("session_id") != expected_session:
            raise EdgeTamProtocolError("EdgeTAM session identity changed")
        frame_seq = _json_int(response.get("frame_seq"), name="frame_seq")
        if frame_seq != expected_seq:
            raise EdgeTamTrackingLost(
                f"EdgeTAM response sequence changed: expected {expected_seq}, got {frame_seq}",
            )
        track_id = response.get("track_id")
        if not isinstance(track_id, str) or not track_id or len(track_id) > 256:
            raise EdgeTamProtocolError("track_id must be a non-empty bounded string")
        if expected_track is not None and track_id != expected_track:
            raise EdgeTamTrackingLost("EdgeTAM track identity changed")

        image_size = _image_size(response.get("image_size"))
        if image_size[0] * image_size[1] > self._max_image_pixels:
            raise EdgeTamProtocolError("image_size exceeds the configured pixel limit")
        if expected_size is not None and image_size != expected_size:
            raise EdgeTamTrackingLost("EdgeTAM image dimensions changed within a session")
        bbox = _bbox(response.get("bbox_xyxy"), image_size)
        score_value = response.get("score")
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise EdgeTamProtocolError("score must be numeric")
        score = float(score_value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise EdgeTamProtocolError("score must be finite and in [0, 1]")
        if score < self._min_score:
            raise EdgeTamTrackingLost("EdgeTAM confidence fell below the configured threshold")

        mask = decode_coco_rle(
            response.get("mask_rle"),
            max_pixels=self._max_image_pixels,
        )
        width, height = image_size
        if mask.shape != (height, width):
            raise EdgeTamProtocolError("mask dimensions do not match image_size")
        ys, xs = np.nonzero(mask)
        if len(xs) < self._min_mask_pixels:
            raise EdgeTamTrackingLost("EdgeTAM returned an empty or too-small target mask")
        mask_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        if bbox != mask_bbox:
            raise EdgeTamProtocolError("bbox_xyxy does not exactly bound mask_rle")
        mask.setflags(write=False)
        return EdgeTamTrack(
            session_id=expected_session,
            track_id=track_id,
            frame_seq=frame_seq,
            image_size=image_size,
            bbox_xyxy=bbox,
            score=score,
            mask=mask,
        )

    def _parse_coast(
        self,
        response: Mapping[str, Any],
        *,
        expected_session: str,
        expected_seq: int,
        expected_track: str | None,
        expected_size: tuple[int, int] | None,
    ) -> EdgeTamCoast:
        """Validate a session-preserving coast frame.

        The identity contract is enforced exactly as for a tracking frame
        (protocol, session_id, track_id, frame_seq, image_size); only the
        mask/bbox/score are absent because there is no observation to publish.
        """
        if response.get("protocol") != PROTOCOL_VERSION:
            raise EdgeTamProtocolError("EdgeTAM protocol version mismatch")
        if response.get("session_id") != expected_session:
            raise EdgeTamProtocolError("EdgeTAM session identity changed")
        frame_seq = _json_int(response.get("frame_seq"), name="frame_seq")
        if frame_seq != expected_seq:
            raise EdgeTamTrackingLost(
                f"EdgeTAM coast sequence changed: expected {expected_seq}, got {frame_seq}",
            )
        track_id = response.get("track_id")
        if not isinstance(track_id, str) or not track_id or len(track_id) > 256:
            raise EdgeTamProtocolError("track_id must be a non-empty bounded string")
        if expected_track is not None and track_id != expected_track:
            raise EdgeTamTrackingLost("EdgeTAM track identity changed")
        image_size = _image_size(response.get("image_size"))
        if image_size[0] * image_size[1] > self._max_image_pixels:
            raise EdgeTamProtocolError("image_size exceeds the configured pixel limit")
        if expected_size is not None and image_size != expected_size:
            raise EdgeTamTrackingLost("EdgeTAM image dimensions changed within a session")
        return EdgeTamCoast(
            session_id=expected_session,
            track_id=track_id,
            frame_seq=frame_seq,
            image_size=image_size,
        )

    def _encode_jpeg(self, image_jpeg: bytes) -> str:
        if not isinstance(image_jpeg, bytes):
            raise EdgeTamProtocolError("image_jpeg must be bytes")
        if not 4 <= len(image_jpeg) <= self._max_jpeg_bytes:
            raise EdgeTamProtocolError("JPEG payload is empty or exceeds the configured limit")
        if not image_jpeg.startswith(b"\xff\xd8") or not image_jpeg.endswith(b"\xff\xd9"):
            raise EdgeTamProtocolError("image payload is not a complete JPEG byte stream")
        return base64.b64encode(image_jpeg).decode("ascii")

    @staticmethod
    def _validate_session_id(session_id: object) -> str:
        if not isinstance(session_id, str) or _SESSION_RE.fullmatch(session_id) is None:
            raise EdgeTamProtocolError("session_id has an invalid format")
        return session_id

    @staticmethod
    def _request_bbox(value: Sequence[int]) -> tuple[int, int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise EdgeTamProtocolError("bbox_xyxy must contain four integer pixels")
        parsed = tuple(
            _json_int(item, name=f"bbox_xyxy[{index}]")
            for index, item in enumerate(value)
        )
        if parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
            raise EdgeTamProtocolError("bbox_xyxy must be non-empty")
        return parsed
