#!/usr/bin/env python3
"""Display-only OpenCV tracking for the dashboard perception panes."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any

MAX_RENDER_IMAGE_BYTES = 2 * 1024 * 1024


class LivePerceptionError(ValueError):
    """A live display frame cannot be rendered without misrepresenting state."""


class LivePerceptionRenderer:
    """Track one fresh immutable perception result on the live RGB stream.

    This component is display-only.  It reads the observer JPEG and verified
    perception artifacts, performs bounded OpenCV tracking, and exposes no ROS,
    CAN, planning, or actuator surface.
    """

    def __init__(
        self,
        camera: Any,
        artifacts: Any | None,
        *,
        clock_ns: Any = time.time_ns,
        reference_max_age_s: float = 15.0,
    ) -> None:
        self.camera = camera
        self.artifacts = artifacts
        self._clock_ns = clock_ns
        self.reference_max_age_ns = round(reference_max_age_s * 1_000_000_000)
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._camera_etag: str | None = None
        self._reference_bbox: tuple[int, int, int, int] | None = None
        self._previous_gray: Any = None
        self._tracking_points: Any = None
        self._anchor_transform: Any = None
        self._mask: Any = None
        self._annotation: Any = None
        self._rendered: dict[str, bytes] = {}
        self._etags: dict[str, str] = {}
        self._state = "offline"
        self._detail = "no live perception anchor is available"
        self._reference_age_s: float | None = None

    @staticmethod
    def _decode(payload: bytes, flags: int) -> Any:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
        if image is None:
            raise LivePerceptionError("OpenCV could not decode a live-render input")
        return image

    @staticmethod
    def _encode(extension: str, image: Any) -> bytes:
        import cv2

        options = (
            [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            if extension == ".jpg"
            else [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        )
        ok, encoded = cv2.imencode(extension, image, options)
        if not ok or not 1 <= len(encoded) <= MAX_RENDER_IMAGE_BYTES:
            raise LivePerceptionError("OpenCV live-render output is invalid or oversized")
        return encoded.tobytes()

    def _reset(self, state: str, detail: str) -> None:
        self._previous_gray = None
        self._tracking_points = None
        self._anchor_transform = None
        self._camera_etag = None
        self._rendered = {}
        self._etags = {}
        self._state = state
        self._detail = detail[:512]

    def _initialize(self, session_id: str, payloads: dict[str, bytes], frame: Any) -> None:
        import cv2
        import numpy as np

        mask = self._decode(payloads["mask"], cv2.IMREAD_GRAYSCALE)
        overlay = self._decode(payloads["overlay"], cv2.IMREAD_COLOR)
        candidates = self._decode(payloads["candidates"], cv2.IMREAD_COLOR)
        height, width = frame.shape[:2]
        if mask.shape[:2] != (height, width) or overlay.shape[:2] != (height, width) or candidates.shape[:2] != (height, width):
            raise LivePerceptionError("live RGB and perception anchor dimensions differ")
        binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            raise LivePerceptionError("perception anchor mask is empty")
        x, y, w, h = cv2.boundingRect(points)
        if w < 4 or h < 4 or w * h > width * height * 0.5:
            raise LivePerceptionError("perception anchor mask has an invalid target extent")
        annotation_delta = cv2.absdiff(candidates, overlay)
        annotation_strength = np.max(annotation_delta, axis=2)
        annotation = np.where(annotation_strength >= 24, 255, 0).astype(np.uint8)
        annotation = cv2.dilate(annotation, np.ones((3, 3), dtype=np.uint8), iterations=1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_region = cv2.dilate(
            binary,
            np.ones((11, 11), dtype=np.uint8),
            iterations=1,
        )
        tracking_points = cv2.goodFeaturesToTrack(
            gray,
            mask=feature_region,
            maxCorners=100,
            qualityLevel=0.008,
            minDistance=3.0,
            blockSize=5,
        )
        if tracking_points is None or len(tracking_points) < 4:
            raise LivePerceptionError(
                "target has too few visual features for stable affine tracking",
            )
        self._session_id = session_id
        self._reference_bbox = (x, y, w, h)
        self._previous_gray = gray
        self._tracking_points = tracking_points.astype(np.float32)
        self._anchor_transform = np.eye(3, dtype=np.float64)
        self._mask = binary
        self._annotation = (annotation, candidates)

    def _track_affine(self, frame: Any) -> tuple[Any, tuple[int, int, int, int]]:
        """Advance a quality-gated local affine track by one RGB frame."""
        import cv2
        import numpy as np

        if (
            self._previous_gray is None
            or self._tracking_points is None
            or self._anchor_transform is None
            or self._mask is None
        ):
            raise LivePerceptionError("OpenCV affine tracking anchor is unavailable")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        forward, status_forward, error_forward = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._tracking_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if forward is None or status_forward is None:
            raise LivePerceptionError("OpenCV lost the target optical flow")
        backward, status_backward, _error_backward = cv2.calcOpticalFlowPyrLK(
            gray,
            self._previous_gray,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or status_backward is None:
            raise LivePerceptionError("OpenCV could not verify target optical flow")

        previous = self._tracking_points.reshape(-1, 2)
        current = forward.reshape(-1, 2)
        reverse = backward.reshape(-1, 2)
        valid = (
            status_forward.reshape(-1).astype(bool)
            & status_backward.reshape(-1).astype(bool)
            & np.isfinite(current).all(axis=1)
            & np.isfinite(reverse).all(axis=1)
            & (np.linalg.norm(reverse - previous, axis=1) <= 1.5)
        )
        if error_forward is not None:
            valid &= error_forward.reshape(-1) <= 30.0
        previous = previous[valid]
        current = current[valid]
        if len(previous) < 4:
            raise LivePerceptionError("target optical flow has fewer than four verified points")

        step, inlier_mask = cv2.estimateAffinePartial2D(
            previous,
            current,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=500,
            confidence=0.995,
            refineIters=10,
        )
        if step is None or inlier_mask is None or not np.all(np.isfinite(step)):
            raise LivePerceptionError("target affine motion could not be estimated")
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(inliers)
        predicted = cv2.transform(previous[None, :, :], step)[0]
        reprojection = np.linalg.norm(predicted - current, axis=1)
        median_reprojection = float(np.median(reprojection[inliers]))
        linear = step[:, :2]
        determinant = float(np.linalg.det(linear))
        scale = math.sqrt(max(determinant, 0.0))
        translation = float(np.linalg.norm(step[:, 2]))
        height, width = frame.shape[:2]
        if (
            inlier_count < 4
            or inlier_ratio < 0.60
            or median_reprojection > 1.25
            or not 0.80 <= scale <= 1.25
            or translation > 0.25 * math.hypot(width, height)
        ):
            raise LivePerceptionError(
                "target affine track failed its quality gate "
                f"(points={inlier_count}, ratio={inlier_ratio:.2f}, "
                f"reprojection={median_reprojection:.2f}px)",
            )

        step_homogeneous = np.eye(3, dtype=np.float64)
        step_homogeneous[:2] = step
        anchor_transform = step_homogeneous @ self._anchor_transform
        live_mask = cv2.warpAffine(
            self._mask,
            anchor_transform[:2],
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask_points = cv2.findNonZero(live_mask)
        if mask_points is None:
            raise LivePerceptionError("tracked target moved outside the live frame")
        x, y, w, h = cv2.boundingRect(mask_points)
        anchor_area = max(1, int(np.count_nonzero(self._mask)))
        live_area = int(np.count_nonzero(live_mask))
        if (
            w < 4
            or h < 4
            or not 0.45 <= live_area / anchor_area <= 2.20
            or x < 0
            or y < 0
            or x + w > width
            or y + h > height
        ):
            raise LivePerceptionError("tracked target extent is inconsistent with its anchor")

        feature_region = cv2.dilate(
            live_mask,
            np.ones((11, 11), dtype=np.uint8),
            iterations=1,
        )
        refreshed = cv2.goodFeaturesToTrack(
            gray,
            mask=feature_region,
            maxCorners=100,
            qualityLevel=0.008,
            minDistance=3.0,
            blockSize=5,
        )
        if refreshed is None or len(refreshed) < 4:
            raise LivePerceptionError("target no longer has enough verified visual features")
        self._previous_gray = gray
        self._tracking_points = refreshed.astype(np.float32)
        self._anchor_transform = anchor_transform
        return anchor_transform[:2].copy(), (x, y, w, h)

    def _render(
        self,
        frame: Any,
        transform: Any,
        bbox: tuple[int, int, int, int],
    ) -> None:
        import cv2
        import numpy as np

        if self._reference_bbox is None or self._mask is None or self._annotation is None:
            raise LivePerceptionError("live perception tracker has no initialized anchor")
        x, y, w, h = bbox
        transform = np.asarray(transform, dtype=np.float32)
        height, width = frame.shape[:2]
        live_mask = cv2.warpAffine(
            self._mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        overlay = frame.copy()
        active = live_mask >= 128
        if np.any(active):
            tint = np.asarray([70, 220, 115], dtype=np.float32)
            overlay[active] = np.clip(
                overlay[active].astype(np.float32) * 0.52 + tint * 0.48,
                0,
                255,
            ).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(live_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (85, 255, 145), 2, cv2.LINE_AA)
        candidate_view = overlay.copy()
        annotation_mask, candidate_anchor = self._annotation
        warped_annotation = cv2.warpAffine(
            annotation_mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_candidates = cv2.warpAffine(
            candidate_anchor,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        candidate_view[warped_annotation >= 128] = warped_candidates[warped_annotation >= 128]
        cv2.putText(overlay, "OpenCV affine tracked mask", (max(4, x), max(16, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (85, 255, 145), 1, cv2.LINE_AA)
        cv2.putText(candidate_view, "OpenCV affine tracked candidates", (max(4, x), max(16, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 225, 255), 1, cv2.LINE_AA)
        self._rendered = {
            "mask": self._encode(".png", live_mask),
            "overlay": self._encode(".jpg", overlay),
            "candidates": self._encode(".jpg", candidate_view),
        }
        self._etags = {
            key: '"live-perception-' + hashlib.sha256(payload).hexdigest()[:24] + '"'
            for key, payload in self._rendered.items()
        }

    def snapshot(self, kind: str) -> tuple[str, bytes | None, str | None, float | None, float | None, str]:
        if kind not in {"mask", "overlay", "candidates"}:
            raise ValueError("unsupported live perception image")
        with self._lock:
            camera_state, camera_payload, camera_etag, camera_age_s, camera_detail = self.camera.snapshot()
            if camera_state != "live" or camera_payload is None or camera_etag is None:
                self._reset(camera_state, camera_detail)
                return self._state, None, None, camera_age_s, self._reference_age_s, self._detail
            if self.artifacts is None:
                self._reset("offline", "interactive perception artifacts are unavailable")
                return self._state, None, None, camera_age_s, None, self._detail
            try:
                session_id, anchors, anchor_mtime_ns = self.artifacts.perception_snapshot()
                reference_age_ns = max(0, int(self._clock_ns()) - anchor_mtime_ns)
                self._reference_age_s = reference_age_ns / 1_000_000_000.0
                frame = self._decode(camera_payload, 1)
                if session_id != self._session_id:
                    if reference_age_ns > self.reference_max_age_ns:
                        self._session_id = session_id
                        self._reset("stale", "run perception to establish a fresh OpenCV tracking anchor")
                        return self._state, None, None, camera_age_s, self._reference_age_s, self._detail
                    self._initialize(session_id, anchors, frame)
                    bbox = self._reference_bbox
                    transform = self._anchor_transform[:2].copy()
                    tracking_state = "fresh"
                elif camera_etag != self._camera_etag:
                    if self._previous_gray is None:
                        self._reset("stale", "OpenCV tracking anchor is no longer valid; rerun perception")
                        return self._state, None, None, camera_age_s, self._reference_age_s, self._detail
                    transform, bbox = self._track_affine(frame)
                    tracking_state = "tracked"
                else:
                    return self._state, self._rendered.get(kind), self._etags.get(kind), camera_age_s, self._reference_age_s, self._detail
                assert bbox is not None
                self._render(frame, transform, bbox)
                self._camera_etag = camera_etag
                self._state = tracking_state
                self._detail = f"quality-gated OpenCV affine tracking session {session_id}"
            except LivePerceptionError as error:
                self._reset("stale", str(error))
                return self._state, None, None, camera_age_s, self._reference_age_s, self._detail
            except Exception as error:
                self._reset("invalid", f"OpenCV live rendering failed: {type(error).__name__}: {error}")
                return self._state, None, None, camera_age_s, self._reference_age_s, self._detail
            return self._state, self._rendered.get(kind), self._etags.get(kind), camera_age_s, self._reference_age_s, self._detail
