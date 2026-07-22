"""Fail-closed identity checks for reusing persistent tracked perception."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class TrackingReuseContract:
    """Exact active tracker identity observed before a same-target fast path."""

    request_id: str
    instruction_sha256: str
    producer_epoch: str
    generation: int
    track_id: str
    observation_stamp_ns: int
    observation_frame_id: str

    def same_identity(self, other: "TrackingReuseContract") -> bool:
        """Compare stable producer/request/track identity, not observation age."""

        return bool(
            self.request_id == other.request_id
            and self.instruction_sha256 == other.instruction_sha256
            and self.producer_epoch == other.producer_epoch
            and self.generation == other.generation
            and self.track_id == other.track_id
            and self.observation_frame_id == other.observation_frame_id
        )

    def accepts_bundle(
        self,
        manifest: Mapping[str, object],
        *,
        stamp_ns: int,
        frame_id: str,
    ) -> bool:
        """Return true only for a newer bundle from this exact tracked object.

        RGB-D artifacts are volatile and arrive after this contract is observed.
        The manifest therefore may have a newer result stamp, but it must retain
        the same tracker id and camera frame and may never move backwards in
        time relative to the validated status observation.
        """

        try:
            manifest_stamp_ns = int(manifest["result_stamp_ns"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return bool(
            manifest.get("schema") == "z_manip.tracker_frame.v1"
            and manifest_stamp_ns == stamp_ns
            and stamp_ns >= self.observation_stamp_ns
            and str(manifest.get("frame_id", "")) == frame_id
            and frame_id == self.observation_frame_id
            and str(manifest.get("track_id", "")) == self.track_id
            and str(manifest.get("session_id", ""))
            and str(manifest.get("seed_id", ""))
        )

    def accepts_fresh_bundle(
        self,
        manifest: Mapping[str, object],
        *,
        stamp_ns: int,
        frame_id: str,
        latest_observation_stamp_ns: int,
        max_age_s: float,
    ) -> bool:
        """Accept this identity only while its RGB-D evidence is recent."""

        if (
            not math.isfinite(max_age_s)
            or max_age_s <= 0.0
            or latest_observation_stamp_ns <= 0
            or not self.accepts_bundle(
                manifest,
                stamp_ns=stamp_ns,
                frame_id=frame_id,
            )
        ):
            return False
        age_ns = max(0, int(latest_observation_stamp_ns) - int(stamp_ns))
        return age_ns <= int(max_age_s * 1_000_000_000)


def parse_tracking_reuse_contract(
    values: Mapping[str, str],
    *,
    expected_instruction_sha256: str,
) -> TrackingReuseContract | None:
    """Parse a reusable status or reject it without partial trust."""

    if (
        values.get("schema") != "z_manip.perception_status.v1"
        or values.get("valid") != "true"
        or values.get("instruction_sha256") != expected_instruction_sha256
    ):
        return None
    try:
        generation = int(values["generation"])
        observation_stamp_ns = int(values["observation_stamp_ns"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    request_id = values.get("request_id", "").strip()
    producer_epoch = values.get("producer_epoch", "").strip()
    track_id = values.get("track_id", "").strip()
    observation_frame_id = values.get("observation_frame_id", "").strip()
    if (
        generation < 0
        or observation_stamp_ns <= 0
        or not request_id
        or not producer_epoch
        or not track_id
        or not observation_frame_id
    ):
        return None
    return TrackingReuseContract(
        request_id=request_id,
        instruction_sha256=expected_instruction_sha256,
        producer_epoch=producer_epoch,
        generation=generation,
        track_id=track_id,
        observation_stamp_ns=observation_stamp_ns,
        observation_frame_id=observation_frame_id,
    )
