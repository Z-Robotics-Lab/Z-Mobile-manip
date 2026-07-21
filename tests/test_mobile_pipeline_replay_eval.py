from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "mobile_pipeline_replay_eval.py"
SPEC = importlib.util.spec_from_file_location("mobile_pipeline_replay_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


def test_loader_accepts_multiline_concatenated_api_objects(tmp_path):
    path = tmp_path / "api.jsonl"
    path.write_text(
        '{"sample_unix_ns":100,\n"approach":{}}\n'
        '{"sample_unix_ns":200,"approach":{}}\n',
        encoding="utf-8",
    )

    records, integrity = EVAL.load_json_stream(path)

    assert len(records) == 2
    assert integrity["decode_errors"] == []
    assert integrity["monotonic_violations"] == 0
    assert integrity["first_unix_ns"] == 100
    assert integrity["last_unix_ns"] == 200


def test_rosbag_inspection_checks_topics_and_mcap_framing(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    topic_lines = []
    for topic in EVAL.REQUIRED_TOPICS:
        topic_lines.extend((
            f"        name: {topic}",
            "      message_count: 3",
        ))
    (bag / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  storage_identifier: mcap\n"
        "  message_count: 27\n"
        "  topics_with_message_count:\n"
        + "\n".join(topic_lines)
        + "\n",
        encoding="utf-8",
    )
    (bag / "run_0.mcap").write_bytes(EVAL.MCAP_MAGIC + b"payload" + EVAL.MCAP_MAGIC)

    report = EVAL.inspect_rosbag(bag)

    assert report["storage_identifier"] == "mcap"
    assert report["framing_valid"] is True
    assert report["required_topics_missing"] == []
    assert report["required_topics_empty"] == []


def test_evaluation_quantifies_all_control_stages():
    trace = [{
        "phase": "handoff_ready",
        "tracking": True,
        "output": {
            "phase": "handoff_ready",
            "published_linear_x": 0.12,
            "published_angular_z": 0.03,
        },
        "reactive": {"handoff_ready": True},
        "posture_status": {
            "age_s": 0.02,
            "document": {
                "phase": "tracking",
                "command": {"codes": {"Euler": 0}},
            },
        },
        "arm_view_status": {
            "age_s": 0.02,
            "document": {"accepted_seq": 7},
        },
        "whole_body": {
            "command": {
                "intent": {
                    "body_roll_rps": 0.01,
                    "body_pitch_rps": -0.02,
                    "piper_joint1_rps": 0.03,
                    "piper_joint2_rps": 0.0,
                    "piper_joint3_rps": 0.0,
                    "piper_joint4_rps": 0.0,
                    "piper_joint5_rps": 0.0,
                    "piper_joint6_rps": 0.0,
                },
            },
        },
    }]
    bag = {
        "topics": {topic: 5 for topic in EVAL.REQUIRED_TOPICS},
        "required_topics_missing": [],
        "required_topics_empty": [],
        "framing_valid": True,
    }

    report = EVAL.evaluate(api_records=[], trace_records=trace, bag=bag)

    assert report["complete"] is True
    assert report["stages"]["track"]["tracking_ratio"] == 1.0
    assert report["stages"]["track"]["state"] == "tracked"
    assert report["stages"]["base"]["active_command_samples"] == 1
    assert report["stages"]["base"]["state"] == "active"
    assert report["stages"]["posture"]["nonzero_roll_pitch_samples"] == 1
    assert report["stages"]["posture"]["state"] == "acknowledged"
    assert report["stages"]["arm"]["nonzero_joint_intent_samples"] == 1
    assert report["stages"]["arm"]["state"] == "active_with_fresh_ack"
    assert report["stages"]["arm"]["accepted_seq_max"] == 7
    assert report["stages"]["handoff"]["observed"] is True
    assert report["stages"]["handoff"]["state"] == "observed"
    assert report["transport_opened"] is False
    assert report["motion_commands_sent"] == 0


def test_evaluation_flags_missing_ack_and_posture_fault():
    trace = [{
        "phase": "whole_body_approach",
        "tracking": True,
        "output": {"published_linear_x": 0.1, "published_angular_z": 0.0},
        "posture_status": {
            "age_s": 0.02,
            "document": {"phase": "fault", "detail": "Euler refused"},
        },
        "whole_body": {
            "command": {
                "intent": {
                    "body_roll_rps": 0.01,
                    "body_pitch_rps": 0.0,
                    **{f"piper_joint{joint}_rps": (0.1 if joint == 1 else 0.0) for joint in range(1, 7)},
                },
            },
        },
    }]

    report = EVAL.evaluate(api_records=[], trace_records=trace, bag=None)

    assert report["complete"] is False
    assert "arm intents have no fresh executor ACK evidence" in report["issues"]
    assert "posture command faults were recorded" in report["issues"]
    assert report["stages"]["posture"]["state"] == "fault"
    assert report["stages"]["arm"]["state"] == "active_without_fresh_ack"


def test_evaluation_flags_bag_arm_intents_that_executor_never_accepted():
    trace = [{
        "phase": "whole_body_approach",
        "tracking": True,
        "output": {"published_linear_x": 0.1, "published_angular_z": 0.0},
        "arm_view_status": {
            "age_s": 0.02,
            "document": {"accepted_seq": -1},
        },
        "whole_body": {
            "command": {
                "intent": {
                    "body_roll_rps": 0.0,
                    "body_pitch_rps": 0.0,
                    **{f"piper_joint{joint}_rps": 0.0 for joint in range(1, 7)},
                },
            },
        },
    }]
    bag = {
        "topics": {
            **{topic: 5 for topic in EVAL.REQUIRED_TOPICS},
            "/z_manip/reactive/arm_view_intent": 2138,
        },
        "required_topics_missing": [],
        "required_topics_empty": [],
        "framing_valid": True,
    }

    report = EVAL.evaluate(api_records=[], trace_records=trace, bag=bag)

    assert report["stages"]["arm"]["state"] == "intent_unacknowledged"
    assert report["stages"]["arm"]["accepted_sequence_observed"] is False
    assert (
        "PiPER executor never accepted a recorded arm intent sequence"
        in report["issues"]
    )
