#!/usr/bin/env python3
"""Evaluate a recorded mobile-manipulation run without ROS or robot drivers.

The evaluator reads the API status JSON stream, the depth-servo trace, and
rosbag2 metadata/MCAP framing.  It never imports ROS, WebRTC, CAN, or PiPER
drivers and cannot publish a command.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


REPORT_SCHEMA = "z_mobile_manip.offline_pipeline_evaluation.v1"
MCAP_MAGIC = b"\x89MCAP0\r\n"
REQUIRED_TOPICS = (
    "/z_manip/grounding/request",
    "/track_3d/is_tracking",
    "/track_3d/selected_target_pointcloud",
    "/z_manip/reactive/posture_intent",
    "/go2w/posture_state",
    "/z_manip/reactive/arm_view_intent",
    "/z_manip/reactive/arm_view_status",
    "/piper/state",
)


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_json_stream(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode concatenated or line-delimited JSON objects with diagnostics."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as error:
            errors.append({
                "line": error.lineno,
                "column": error.colno,
                "byte_offset": error.pos,
                "message": error.msg,
            })
            break
        if not isinstance(value, dict):
            errors.append({
                "byte_offset": index,
                "message": f"record is {type(value).__name__}, expected object",
            })
        else:
            records.append(value)
        index = end

    stamps: list[int] = []
    for record in records:
        raw = record.get("sample_unix_ns", record.get("updated_unix_ns"))
        try:
            stamps.append(int(raw))
        except (TypeError, ValueError):
            pass
    violations = sum(b < a for a, b in zip(stamps, stamps[1:]))
    gaps_s = [(b - a) / 1e9 for a, b in zip(stamps, stamps[1:]) if b >= a]
    return records, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "records": len(records),
        "decode_errors": errors,
        "timestamped_records": len(stamps),
        "monotonic_violations": violations,
        "first_unix_ns": stamps[0] if stamps else None,
        "last_unix_ns": stamps[-1] if stamps else None,
        "duration_s": ((stamps[-1] - stamps[0]) / 1e9 if len(stamps) >= 2 else 0.0),
        "max_gap_s": max(gaps_s, default=0.0),
    }


def _yaml_scalar(text: str, name: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*([^\n]+?)\s*$", text)
    return match.group(1).strip('"\'') if match else None


def inspect_rosbag(path: Path) -> dict[str, Any]:
    bag_dir = path if path.is_dir() else path.parent
    metadata = bag_dir / "metadata.yaml"
    text = metadata.read_text(encoding="utf-8") if metadata.is_file() else ""
    topics: dict[str, int] = {}
    current_topic: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            current_topic = stripped.split(":", 1)[1].strip().strip('"\'')
        elif current_topic is not None and stripped.startswith("message_count:"):
            try:
                topics[current_topic] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                topics[current_topic] = -1
            current_topic = None

    mcap_files = [path] if path.is_file() else sorted(path.glob("*.mcap"))
    framing = []
    for mcap in mcap_files:
        size = mcap.stat().st_size
        with mcap.open("rb") as stream:
            head = stream.read(len(MCAP_MAGIC))
            if size >= len(MCAP_MAGIC):
                stream.seek(-len(MCAP_MAGIC), 2)
                tail = stream.read(len(MCAP_MAGIC))
            else:
                tail = b""
        framing.append({
            "path": str(mcap.resolve()),
            "bytes": size,
            "header_magic_valid": head == MCAP_MAGIC,
            "footer_magic_valid": tail == MCAP_MAGIC,
        })

    missing = [topic for topic in REQUIRED_TOPICS if topic not in topics]
    empty = [topic for topic in REQUIRED_TOPICS if topics.get(topic) == 0]
    return {
        "path": str(bag_dir.resolve()),
        "metadata_present": metadata.is_file(),
        "storage_identifier": _yaml_scalar(text, "storage_identifier"),
        "declared_message_count": int(_yaml_scalar(text, "message_count") or 0),
        "topic_count": len(topics),
        "topics": topics,
        "required_topics_missing": missing,
        "required_topics_empty": empty,
        "mcap_files": framing,
        "framing_valid": bool(framing) and all(
            item["header_magic_valid"] and item["footer_magic_valid"]
            for item in framing
        ),
    }


def _runtime_records(api: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in api:
        approach = record.get("approach")
        runtime = approach.get("runtime") if isinstance(approach, dict) else None
        if isinstance(runtime, dict) and runtime:
            result.append(runtime)
    return result


def _phase_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("phase", "unknown")) for item in records).items()))


def _transitions(records: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    previous = None
    for item in records:
        phase = str(item.get("phase", "unknown"))
        if phase != previous:
            result.append(phase)
            previous = phase
    return result


def evaluate(
    *,
    api_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    bag: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = trace_records or _runtime_records(api_records)
    tracking = [item for item in runtime if item.get("tracking") is True]
    outputs = [item.get("output") for item in runtime if isinstance(item.get("output"), dict)]
    whole_body = []
    posture_status = []
    arm_status = []
    for item in runtime:
        whole = item.get("whole_body")
        command = whole.get("command") if isinstance(whole, dict) else None
        if isinstance(command, dict):
            whole_body.append(command)
        posture = item.get("posture_status")
        document = posture.get("document") if isinstance(posture, dict) else None
        if isinstance(document, dict):
            posture_status.append((posture, document))
        arm = item.get("arm_view_status")
        document = arm.get("document") if isinstance(arm, dict) else None
        if isinstance(document, dict):
            arm_status.append((arm, document))

    base_active = [
        item for item in outputs
        if abs(_finite(item.get("published_linear_x")) or 0.0) > 1e-4
        or abs(_finite(item.get("published_angular_z")) or 0.0) > 1e-4
    ]
    posture_intents = [
        item.get("intent", {}) for item in whole_body
        if isinstance(item.get("intent"), dict)
    ]
    posture_nonzero = [
        intent for intent in posture_intents
        if abs(_finite(intent.get("body_roll_rps")) or 0.0) > 1e-5
        or abs(_finite(intent.get("body_pitch_rps")) or 0.0) > 1e-5
    ]
    arm_nonzero = [
        intent for intent in posture_intents
        if any(abs(_finite(intent.get(f"piper_joint{joint}_rps")) or 0.0) > 1e-5 for joint in range(1, 7))
    ]
    posture_codes = Counter()
    posture_faults = Counter()
    for _wrapper, document in posture_status:
        codes = document.get("command", {}).get("codes", {})
        if isinstance(codes, dict):
            for name, code in codes.items():
                posture_codes[f"{name}:{code}"] += 1
        if document.get("phase") == "fault":
            posture_faults[str(document.get("detail", "fault"))] += 1

    arm_ack = []
    for _wrapper, document in arm_status:
        try:
            arm_ack.append(int(document.get("accepted_seq", -1)))
        except (TypeError, ValueError):
            arm_ack.append(-1)
    fresh_arm = [
        wrapper for wrapper, _doc in arm_status
        if (_finite(wrapper.get("age_s")) is not None and float(wrapper["age_s"]) <= 0.30)
    ]
    handoff_phases = {
        "reached", "handoff_probe", "handoff_ready", "handoff", "grasp",
    }
    handoff_samples = [
        item for item in runtime
        if str(item.get("phase")) in handoff_phases
        or str(item.get("output", {}).get("phase")) in handoff_phases
        or item.get("reactive", {}).get("handoff_ready") is True
    ]

    topic_counts = bag.get("topics", {}) if bag else {}
    arm_intent_message_count = int(
        topic_counts.get("/z_manip/reactive/arm_view_intent", 0) or 0
    )
    arm_ack_observed = bool(arm_ack and max(arm_ack) >= 0)
    api_workflows = [
        record.get("approach", {}).get("workflow", {}) for record in api_records
        if isinstance(record.get("approach"), dict)
    ]
    workflow_phases = Counter(
        str(item.get("phase", "unknown")) for item in api_workflows if isinstance(item, dict)
    )

    stages = {
        "detect": {
            "state": "observed" if (
                topic_counts.get("/z_manip/grounding/request", 0) > 0
                or workflow_phases.get("detecting", 0) > 0
            ) else "not_observed",
            "grounding_requests": topic_counts.get("/z_manip/grounding/request"),
            "tracker_initializations": topic_counts.get("/track_3d/init_bbox"),
            "api_detecting_samples": workflow_phases.get("detecting", 0),
        },
        "track": {
            "state": "tracked" if tracking else "not_tracked",
            "runtime_samples": len(runtime),
            "tracking_samples": len(tracking),
            "tracking_ratio": len(tracking) / len(runtime) if runtime else 0.0,
            "target_cloud_messages": topic_counts.get("/track_3d/selected_target_pointcloud"),
            "tracking_flag_messages": topic_counts.get("/track_3d/is_tracking"),
        },
        "base": {
            "state": "active" if base_active else "inactive",
            "output_samples": len(outputs),
            "active_command_samples": len(base_active),
            "max_abs_linear_mps": max((abs(_finite(x.get("published_linear_x")) or 0.0) for x in outputs), default=0.0),
            "max_abs_yaw_rps": max((abs(_finite(x.get("published_angular_z")) or 0.0) for x in outputs), default=0.0),
            "cmd_vel_messages": topic_counts.get("/cmd_vel"),
            "cmd_vel_safe_messages": topic_counts.get("/cmd_vel_safe"),
        },
        "posture": {
            "state": (
                "fault"
                if posture_faults
                else "acknowledged"
                if posture_status
                else "intent_without_status"
                if posture_nonzero
                else "not_exercised"
            ),
            "optimizer_intent_samples": len(posture_intents),
            "nonzero_roll_pitch_samples": len(posture_nonzero),
            "status_samples": len(posture_status),
            "command_code_counts": dict(sorted(posture_codes.items())),
            "fault_counts": dict(posture_faults.most_common()),
            "intent_messages": topic_counts.get("/z_manip/reactive/posture_intent"),
            "state_messages": topic_counts.get("/go2w/posture_state"),
        },
        "arm": {
            "state": (
                "active_with_fresh_ack"
                if arm_nonzero and fresh_arm and arm_ack_observed
                else "active_without_fresh_ack"
                if arm_nonzero
                else "intent_unacknowledged"
                if arm_intent_message_count > 0 and not arm_ack_observed
                else "not_exercised"
            ),
            "optimizer_intent_samples": len(posture_intents),
            "nonzero_joint_intent_samples": len(arm_nonzero),
            "status_samples": len(arm_status),
            "fresh_status_samples": len(fresh_arm),
            "accepted_seq_min": min(arm_ack, default=None),
            "accepted_seq_max": max(arm_ack, default=None),
            "accepted_sequence_observed": arm_ack_observed,
            "intent_messages": arm_intent_message_count,
            "status_messages": topic_counts.get("/z_manip/reactive/arm_view_status"),
            "joint_state_messages": topic_counts.get("/piper/state"),
        },
        "handoff": {
            "state": "observed" if handoff_samples else "not_reached",
            "samples": len(handoff_samples),
            "observed": bool(handoff_samples),
            "workflow_grasp_samples": workflow_phases.get("grasp", 0),
        },
    }

    issues = []
    if not runtime:
        issues.append("no runtime records")
    if bag and (bag["required_topics_missing"] or not bag["framing_valid"]):
        issues.append("rosbag is missing required topics or valid MCAP framing")
    if runtime and not tracking:
        issues.append("no tracked target samples")
    if posture_nonzero and not posture_status:
        issues.append("posture intents have no status/ACK evidence")
    if arm_nonzero and not fresh_arm:
        issues.append("arm intents have no fresh executor ACK evidence")
    if arm_intent_message_count > 0 and not arm_ack_observed:
        issues.append("PiPER executor never accepted a recorded arm intent sequence")
    if posture_faults:
        issues.append("posture command faults were recorded")

    return {
        "schema": REPORT_SCHEMA,
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "runtime_source": "depth_servo_trace" if trace_records else "api_status_runtime",
        "phase_counts": _phase_counts(runtime),
        "phase_transitions": _transitions(runtime),
        "stages": stages,
        "issues": issues,
        "complete": not issues,
    }


def _newest(root: Path, pattern: str) -> Path:
    matches = [item for item in root.glob(pattern) if item.is_file()]
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} below {root}")
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--api-jsonl", type=Path)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="fail when the report has integrity issues")
    args = parser.parse_args()
    if args.artifacts_root:
        if args.api_jsonl is None:
            try:
                args.api_jsonl = _newest(args.artifacts_root, "rosbags/*-api-status.jsonl")
            except FileNotFoundError:
                # Current recorders intentionally keep the command-free API
                # snapshot optional. The depth-servo trace remains the
                # authoritative WebRTC base-command evidence.
                pass
        args.trace_jsonl = args.trace_jsonl or args.artifacts_root / "latest/depth-servo.trace.jsonl"
        if args.bag is None:
            newest_mcap = _newest(args.artifacts_root, "rosbags/**/*.mcap")
            args.bag = newest_mcap.parent
    if args.trace_jsonl is None:
        parser.error("provide --artifacts-root or --trace-jsonl")
    return args


def main() -> int:
    args = parse_args()
    if args.api_jsonl is None:
        api = []
        api_integrity = {
            "path": None,
            "bytes": 0,
            "records": 0,
            "decode_errors": [],
            "timestamped_records": 0,
            "monotonic_violations": 0,
            "first_unix_ns": None,
            "last_unix_ns": None,
            "duration_s": 0.0,
            "max_gap_s": 0.0,
            "optional_absent": True,
        }
    else:
        api, api_integrity = load_json_stream(args.api_jsonl)
    trace, trace_integrity = load_json_stream(args.trace_jsonl)
    bag = inspect_rosbag(args.bag) if args.bag else None
    report = evaluate(api_records=api, trace_records=trace, bag=bag)
    report["integrity"] = {
        "api_status": api_integrity,
        "depth_servo_trace": trace_integrity,
        "rosbag": bag,
    }
    decode_failed = bool(api_integrity["decode_errors"] or trace_integrity["decode_errors"])
    if decode_failed:
        report["issues"].append("one or more JSON streams are truncated or malformed")
        report["complete"] = False
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if args.strict and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
