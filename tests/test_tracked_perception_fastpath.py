from __future__ import annotations

from z_manip.perception.tracked_reuse import (
    parse_tracking_reuse_contract,
)


SHA = "a" * 64


def _status(**overrides: str) -> dict[str, str]:
    values = {
        "schema": "z_manip.perception_status.v1",
        "valid": "true",
        "instruction_sha256": SHA,
        "request_id": "request-1",
        "producer_epoch": "epoch-1",
        "generation": "7",
        "track_id": "track-4",
        "observation_stamp_ns": "1000000000",
        "observation_frame_id": "camera_color_optical_frame",
    }
    values.update(overrides)
    return values


def _manifest(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema": "z_manip.tracker_frame.v1",
        "seed_id": "seed-1",
        "session_id": "session-1",
        "track_id": "track-4",
        "result_stamp_ns": 1_100_000_000,
        "frame_id": "camera_color_optical_frame",
    }
    values.update(overrides)
    return values


def test_exact_same_target_status_accepts_only_new_same_track_bundle():
    contract = parse_tracking_reuse_contract(
        _status(),
        expected_instruction_sha256=SHA,
    )
    assert contract is not None
    assert contract.accepts_bundle(
        _manifest(),
        stamp_ns=1_100_000_000,
        frame_id="camera_color_optical_frame",
    )
    next_observation = parse_tracking_reuse_contract(
        _status(observation_stamp_ns="1200000000"),
        expected_instruction_sha256=SHA,
    )
    assert next_observation is not None
    assert contract.same_identity(next_observation)


def test_reuse_identity_changes_on_producer_generation_or_track_change():
    contract = parse_tracking_reuse_contract(
        _status(),
        expected_instruction_sha256=SHA,
    )
    assert contract is not None
    for changed in (
        _status(producer_epoch="epoch-2"),
        _status(generation="8"),
        _status(track_id="track-5"),
    ):
        candidate = parse_tracking_reuse_contract(
            changed,
            expected_instruction_sha256=SHA,
        )
        assert candidate is not None
        assert not contract.same_identity(candidate)


def test_reuse_rejects_instruction_identity_or_invalid_status():
    assert (
        parse_tracking_reuse_contract(
            _status(instruction_sha256="b" * 64),
            expected_instruction_sha256=SHA,
        )
        is None
    )
    assert (
        parse_tracking_reuse_contract(
            _status(valid="false"),
            expected_instruction_sha256=SHA,
        )
        is None
    )


def test_reuse_rejects_old_cross_track_or_cross_frame_bundle():
    contract = parse_tracking_reuse_contract(
        _status(),
        expected_instruction_sha256=SHA,
    )
    assert contract is not None
    cases = (
        (_manifest(track_id="track-9"), 1_100_000_000, "camera_color_optical_frame"),
        (_manifest(result_stamp_ns=900_000_000), 900_000_000, "camera_color_optical_frame"),
        (_manifest(frame_id="other_camera"), 1_100_000_000, "other_camera"),
        (_manifest(), 1_200_000_000, "camera_color_optical_frame"),
    )
    for manifest, stamp, frame in cases:
        assert not contract.accepts_bundle(
            manifest,
            stamp_ns=stamp,
            frame_id=frame,
        )


def test_reuse_status_requires_complete_observation_identity():
    for key in (
        "request_id",
        "producer_epoch",
        "track_id",
        "observation_frame_id",
    ):
        assert (
            parse_tracking_reuse_contract(
                _status(**{key: ""}),
                expected_instruction_sha256=SHA,
            )
            is None
        )
    assert (
        parse_tracking_reuse_contract(
            _status(observation_stamp_ns="0"),
            expected_instruction_sha256=SHA,
        )
        is None
    )
