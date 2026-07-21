from __future__ import annotations

import pytest

from z_manip.control.go2w_posture import (
    CommandOwner,
    Go2WPostureAdapter,
    PostureFeedback,
    PostureLimits,
    PosturePhase,
    PostureTarget,
    SportCommand,
    SportCommandArbiter,
    SportRequest,
    feedback_from_mapping,
    feedback_from_sources,
    get_body_height_from_response,
)


def _feedback(
    stamp_s: float,
    *,
    height: float = 0.0,
    pitch: float = 0.0,
    linear: float = 0.0,
    yaw_rate: float = 0.0,
) -> PostureFeedback:
    return PostureFeedback(
        stamp_s=stamp_s,
        body_height_m=height,
        roll_rad=0.0,
        pitch_rad=pitch,
        yaw_rad=0.0,
        base_linear_x_mps=linear,
        base_yaw_rate_rps=yaw_rate,
    )


class FakeTransport:
    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.requests: list[SportRequest] = []

    def send(self, request: SportRequest):
        self.requests.append(request)
        return {"data": {"header": {"status": {"code": self.code}}}}


def test_shadow_adapter_calculates_bounded_height_without_transport():
    adapter = Go2WPostureAdapter(mode="shadow")
    adapter.set_target(PostureTarget(body_height_m=-0.08, pitch_rad=-0.10))
    adapter.observe(_feedback(1.0))

    output = adapter.tick(now_s=1.0)

    assert output.phase == PosturePhase.COMMANDING
    assert output.command.name == "BodyHeight"
    assert output.command.parameter["data"] == pytest.approx(-0.01)
    assert output.command.would_send is True
    assert output.command.sent is False
    assert output.command.reason == "shadow: command not transmitted"


def test_live_mode_cannot_exist_without_explicit_transport():
    with pytest.raises(ValueError, match="injected transport"):
        Go2WPostureAdapter(mode="live")


def test_live_adapter_surfaces_robot_refusal():
    transport = FakeTransport(code=-1)
    adapter = Go2WPostureAdapter(mode="live", transport=transport)
    adapter.set_target(PostureTarget(body_height_m=-0.04))
    adapter.observe(_feedback(2.0))

    output = adapter.tick(now_s=2.0)

    assert output.phase == PosturePhase.FAULT
    assert output.command.sent is True
    assert output.command.accepted is False
    assert transport.requests[0].command == SportCommand.BODY_HEIGHT


def test_posture_waits_for_base_quiet_by_default():
    adapter = Go2WPostureAdapter(mode="shadow")
    adapter.set_target(PostureTarget(body_height_m=-0.04))
    adapter.observe(_feedback(3.0, linear=0.10))

    output = adapter.tick(now_s=3.0)

    assert output.phase == PosturePhase.WAITING_BASE_QUIET
    assert output.command.would_send is False
    assert "base velocity" in output.command.reason


def test_reactive_deployment_can_explicitly_allow_posture_while_moving():
    adapter = Go2WPostureAdapter(
        mode="shadow",
        limits=PostureLimits(allow_posture_while_moving=True),
    )
    adapter.set_target(PostureTarget(body_height_m=-0.04))
    adapter.observe(_feedback(4.0, linear=0.10))

    output = adapter.tick(now_s=4.0)

    assert output.phase == PosturePhase.COMMANDING
    assert output.command.name == "BodyHeight"


def test_stale_or_missing_feedback_never_generates_posture_command():
    adapter = Go2WPostureAdapter(mode="shadow")
    adapter.set_target(PostureTarget(body_height_m=-0.04))
    missing = adapter.tick(now_s=5.0)
    adapter.observe(_feedback(5.0))
    stale = adapter.tick(now_s=5.51)

    assert missing.phase == stale.phase == PosturePhase.BLOCKED
    assert missing.command.would_send is stale.command.would_send is False


def test_height_and_euler_are_serialized_without_queue_growth():
    arbiter = SportCommandArbiter()
    adapter = Go2WPostureAdapter(mode="shadow", arbiter=arbiter)
    adapter.set_target(PostureTarget(body_height_m=-0.08, pitch_rad=-0.10))
    adapter.observe(_feedback(6.0))

    first = adapter.tick(now_s=6.0)
    assert first.command.name == "BodyHeight"
    assert arbiter.pending == 1

    # At the next command slot, consume the queued Euler command instead of
    # appending another stale height+Euler pair.
    second = adapter.tick(now_s=6.21)
    assert second.command.name == "Euler"
    assert second.command.parameter["y"] == pytest.approx(-0.034906585, abs=1e-8)
    assert arbiter.pending == 0


def test_full_stop_flushes_move_and_posture_and_dispatches_first():
    arbiter = SportCommandArbiter()
    arbiter.submit(SportRequest(SportCommand.MOVE, {"x": 0.1, "y": 0.0, "z": 0.0}))
    arbiter.submit(SportRequest(SportCommand.EULER, {"x": 0.0, "y": -0.1, "z": 0.0}))
    transport = FakeTransport()
    adapter = Go2WPostureAdapter(mode="live", transport=transport, arbiter=arbiter)

    evidence = adapter.dispatch_full_stop()

    assert evidence.name == "StopMove"
    assert transport.requests == [SportRequest(SportCommand.STOP_MOVE, {})]
    assert arbiter.pending == 0
    assert arbiter.owner == CommandOwner.FULL_STOP


def test_arbiter_coalesces_move_and_prioritizes_posture_over_move():
    arbiter = SportCommandArbiter()
    arbiter.submit(SportRequest(SportCommand.MOVE, {"x": 0.1}))
    arbiter.submit(SportRequest(SportCommand.MOVE, {"x": 0.2}))
    arbiter.submit(SportRequest(SportCommand.EULER, {"x": 0.0, "y": 0.1, "z": 0.0}))

    assert arbiter.pop_next().command == SportCommand.EULER
    move = arbiter.pop_next()
    assert move is not None
    assert move.command == SportCommand.MOVE
    assert move.parameter["x"] == 0.2


def test_get_body_height_queries_are_coalesced_and_serialized():
    arbiter = SportCommandArbiter()
    arbiter.submit(SportRequest(SportCommand.MOVE, {"x": 0.1}))
    arbiter.submit(SportRequest(SportCommand.GET_BODY_HEIGHT, {}))
    arbiter.submit(SportRequest(SportCommand.GET_BODY_HEIGHT, {}))

    query = arbiter.pop_next()
    assert query is not None
    assert query.command == SportCommand.GET_BODY_HEIGHT
    assert query.api_id == 1024
    assert arbiter.owner == CommandOwner.FEEDBACK
    assert arbiter.pending == 1


@pytest.mark.parametrize(
    "response,expected,path",
    [
        (
            {"data": {"header": {"status": {"code": 0}}, "data": -0.04}},
            -0.04,
            "data.data",
        ),
        (
            {"data": {"header": {"status": {"code": 0}}, "data": "-0.03"}},
            -0.03,
            "data.data<json>",
        ),
        (
            {
                "data": {
                    "header": {"status": {"code": 0}},
                    "parameter": '{"body_height":-0.02}',
                }
            },
            -0.02,
            "data.parameter<json>.body_height",
        ),
        ({"height": 0.01}, 0.01, "height"),
    ],
)
def test_get_body_height_response_envelopes_are_parsed_strictly(
    response, expected, path,
):
    height, evidence_path = get_body_height_from_response(response)
    assert height == pytest.approx(expected)
    assert evidence_path == path


def test_get_body_height_never_substitutes_status_or_ambiguous_values():
    with pytest.raises(ValueError, match="no supported height"):
        get_body_height_from_response(
            {"data": {"header": {"status": {"code": 0}}}},
        )
    with pytest.raises(ValueError, match="refused"):
        get_body_height_from_response(
            {"data": {"header": {"status": {"code": -1}}, "data": -0.04}},
        )
    with pytest.raises(ValueError, match="conflicting"):
        get_body_height_from_response(
            {"data": {"data": -0.04, "height": -0.02}},
        )


def test_shadow_replay_fuses_query_offset_without_hand_written_nominal():
    feedback = feedback_from_sources(
        {
            "data": {
                "body_height": 0.312,
                "imu_state": {"rpy": [0.01, -0.08, 0.02]},
                "velocity": [0.0, 0.0, 0.0],
            }
        },
        {
            "data": {
                "header": {"status": {"code": 0}},
                "data": '{"height":-0.055}',
            }
        },
        stamp_s=12.0,
    )

    assert feedback.body_height_m == pytest.approx(-0.055)
    assert feedback.pitch_rad == pytest.approx(-0.08)
    assert "GetBodyHeight:data.data<json>.height" in feedback.source


def test_feedback_normalizer_requires_measured_height_and_attitude():
    feedback = feedback_from_mapping(
        {
            "body_height": 0.29,
            "imu_state": {"rpy": [0.01, -0.02, 0.03]},
            "velocity": [0.1, -0.02, 0.04],
        },
        stamp_s=7.0,
    )
    assert feedback.body_height_m == 0.29
    assert feedback.pitch_rad == -0.02
    assert feedback.planar_speed_mps == pytest.approx(0.10198039)

    with pytest.raises(ValueError, match="body_height"):
        feedback_from_mapping(
            {"imu_state": {"rpy": [0.0, 0.0, 0.0]}},
            stamp_s=7.0,
        )


def test_status_document_is_stable_for_ui_consumers():
    adapter = Go2WPostureAdapter(mode="shadow")
    adapter.set_target(PostureTarget(body_height_m=-0.04, pitch_rad=-0.08))
    adapter.observe(_feedback(8.0))
    document = adapter.tick(now_s=8.0).status_document(mode="shadow")

    assert document["schema"] == "z_manip.go2w_posture_status.v1"
    assert document["mode"] == "shadow"
    assert document["phase"] == "commanding"
    assert document["command_owner"] == "posture"
    assert document["body_height"]["current_m"] == 0.0
    assert document["body_height"]["target_m"] == -0.04
    assert document["attitude"]["target_pitch_rad"] == -0.08
    assert document["base"]["quiet"] is True
    assert document["command"]["would_send"] is True


def test_reached_requires_continuous_measured_settle_window():
    adapter = Go2WPostureAdapter(mode="shadow")
    adapter.set_target(PostureTarget(body_height_m=-0.04, pitch_rad=-0.08))
    adapter.observe(_feedback(9.0, height=-0.04, pitch=-0.08))
    settling = adapter.tick(now_s=9.0)
    adapter.observe(_feedback(9.36, height=-0.04, pitch=-0.08))
    reached = adapter.tick(now_s=9.36)

    assert settling.phase == PosturePhase.SETTLING
    assert reached.phase == PosturePhase.REACHED


@pytest.mark.parametrize(
    "target,match",
    [
        (PostureTarget(body_height_m=-0.13), "body-height"),
        (PostureTarget(body_height_m=0.0, pitch_rad=0.3), "pitch"),
    ],
)
def test_targets_outside_initial_manipulation_envelope_are_rejected(target, match):
    adapter = Go2WPostureAdapter(mode="shadow")
    with pytest.raises(ValueError, match=match):
        adapter.set_target(target)
