#!/usr/bin/env python3
"""Fail-closed audit for an offline Go2W manipulation debug bundle.

This program reads JSON and referenced artifact files only.  It deliberately
has no ROS, actuator, SocketCAN, network, or subprocess integration.  A failed
audit still produces a machine-readable report so the dashboard can explain
which safety claim was rejected without trying to recover by commanding the
robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any


BUNDLE_SCHEMA = "z_manip.debug_bundle.v1"
JOINT_REPORT_SCHEMA = "z_manip.piper_passive_joint_report.v1"
CALIBRATION_SCHEMA = "z_manip.piper_camera_calibration.v1"
AUDIT_SCHEMA = "z_manip.debug_safety_audit.v1"
MAX_JSON_BYTES = 16 * 1024 * 1024

_SSH_KEY_NAMES = {
    "authorized_keys",
    "identity",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}


def _failure(code: str, message: str, **details: object) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _read_json(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if not path.is_file():
            return None, f"{label} is not a regular file"
        size = path.stat().st_size
        if size > MAX_JSON_BYTES:
            return None, f"{label} exceeds the {MAX_JSON_BYTES}-byte limit"
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"cannot read {label}: {error}"
    if not isinstance(value, dict):
        return None, f"{label} must contain a JSON object"
    return value, None


def _is_sensitive_path(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    name = path.name.lower()
    if any(part == ".ssh" or "license" in part for part in lowered):
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    if name in _SSH_KEY_NAMES or any(name.startswith(f"{item}.") for item in _SSH_KEY_NAMES):
        return True
    return path.suffix.lower() in {".key", ".pem", ".ppk"}


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_file(
    raw_path: str | Path,
    *,
    base: Path,
    root: Path,
    label: str,
) -> tuple[Path | None, str | None]:
    try:
        source = Path(raw_path).expanduser()
        candidate = source if source.is_absolute() else base / source
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        return None, f"cannot resolve {label}: {error}"
    if not _within(resolved, root):
        return None, f"{label} escapes artifact root"
    if _is_sensitive_path(resolved):
        return None, f"{label} refers to a sensitive env/license/SSH-key path"
    if not resolved.is_file():
        return None, f"{label} is not a regular file"
    return resolved, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_zero(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _six_finite(values: object, *, nonnegative: bool = False) -> bool:
    if not isinstance(values, list) or len(values) != 6:
        return False
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        return False
    return all(
        math.isfinite(value) and (not nonnegative or value >= 0.0)
        for value in numeric
    )


def _joint_report_valid(document: dict[str, Any]) -> bool:
    return bool(
        document.get("schema") == JOINT_REPORT_SCHEMA
        and document.get("read_only") is True
        and document.get("complete_joint_feedback") is True
        and document.get("zero_transmit_verified") is True
        and _exact_zero(document.get("interface_tx_packet_delta"))
        and _six_finite(document.get("joint_positions_rad"))
        and _six_finite(document.get("joint_ranges_rad"), nonnegative=True)
    )


def _quality_passes(document: dict[str, Any]) -> bool:
    quality = document.get("quality")
    limits = document.get("quality_limits")
    if not isinstance(quality, dict) or not isinstance(limits, dict):
        return False
    try:
        sample_count = int(document["sample_count"])
        min_samples = int(limits["min_samples"])
        axis_rank = int(quality["rotation_axis_rank"])
        min_axis_rank = int(limits["min_rotation_axis_rank"])
        rotation_span = float(quality["max_pair_rotation_rad"])
        min_rotation_span = float(limits["min_rotation_span_rad"])
        translation_rmse = float(quality["translation_rmse_m"])
        max_translation_rmse = float(limits["max_translation_rmse_m"])
        rotation_rmse = float(quality["rotation_rmse_rad"])
        max_rotation_rmse = float(limits["max_rotation_rmse_rad"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    floats = (
        rotation_span,
        min_rotation_span,
        translation_rmse,
        max_translation_rmse,
        rotation_rmse,
        max_rotation_rmse,
    )
    return bool(
        all(math.isfinite(value) for value in floats)
        and sample_count >= min_samples >= 1
        and axis_rank >= min_axis_rank >= 1
        and rotation_span >= min_rotation_span >= 0.0
        and 0.0 <= translation_rmse <= max_translation_rmse
        and 0.0 <= rotation_rmse <= max_rotation_rmse
    )


def _calibration_valid(document: dict[str, Any]) -> bool:
    return bool(
        document.get("schema") == CALIBRATION_SCHEMA
        and document.get("calibrated") is True
        and document.get("synthetic") is False
        and str(document.get("calibration_id", "")).strip()
        and _quality_passes(document)
    )


def _artifact_references(
    bundle: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        return None, "bundle artifacts must be an object"
    normalized: dict[str, dict[str, Any]] = {}
    for name, reference in artifacts.items():
        if not isinstance(name, str) or not name:
            return None, "artifact names must be non-empty strings"
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            return None, f"artifact {name!r} must contain a string path"
        normalized[name] = reference
    return normalized, None


def audit_bundle(
    bundle_path: Path,
    artifact_root: Path,
    *,
    joint_report: Path | None = None,
) -> dict[str, object]:
    """Return a complete audit report; validation failures never raise."""

    errors: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, code: str, message: str, **details: object) -> None:
        checks.append({"name": name, "passed": passed, "details": details})
        if not passed:
            errors.append(_failure(code, message, **details))

    try:
        root = artifact_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        root = artifact_root.expanduser().absolute()
        check(
            "artifact_root",
            False,
            "INVALID_ARTIFACT_ROOT",
            "artifact root cannot be resolved",
            error=str(error),
        )
    else:
        check(
            "artifact_root",
            root.is_dir() and not _is_sensitive_path(root),
            "INVALID_ARTIFACT_ROOT",
            "artifact root must be a non-sensitive directory",
            root=str(root),
        )

    resolved_bundle, bundle_path_error = _resolve_file(
        bundle_path,
        base=Path.cwd(),
        root=root,
        label="bundle",
    )
    check(
        "bundle_path",
        resolved_bundle is not None,
        "UNSAFE_BUNDLE_PATH",
        bundle_path_error or "bundle path rejected",
    )
    bundle: dict[str, Any] = {}
    if resolved_bundle is not None:
        loaded, load_error = _read_json(resolved_bundle, "bundle")
        check(
            "bundle_json",
            loaded is not None,
            "INVALID_BUNDLE_JSON",
            load_error or "bundle JSON rejected",
        )
        bundle = loaded or {}
    else:
        checks.append({"name": "bundle_json", "passed": False, "details": {}})

    check(
        "bundle_schema",
        bundle.get("schema") == BUNDLE_SCHEMA,
        "UNSUPPORTED_BUNDLE_SCHEMA",
        f"bundle schema must be {BUNDLE_SCHEMA}",
        observed=bundle.get("schema"),
    )
    mode = bundle.get("mode")
    mode_safe = bool(
        isinstance(mode, dict)
        and mode.get("read_only") is True
        and mode.get("planning_only") is True
        and mode.get("offline_artifact_reader") is True
    )
    check(
        "offline_read_only_mode",
        mode_safe,
        "UNSAFE_BUNDLE_MODE",
        "bundle must explicitly declare read-only planning-only offline mode",
    )
    safety = bundle.get("safety")
    motion_safe = bool(
        isinstance(safety, dict)
        and _exact_zero(safety.get("motion_commands_published"))
        and safety.get("transport_opened") is False
        and safety.get("ros_imported") is False
        and safety.get("can_opened") is False
        and (
            safety.get("upstream_reported_motion_commands_published") is None
            or _exact_zero(safety.get("upstream_reported_motion_commands_published"))
        )
    )
    check(
        "zero_motion_and_transport",
        motion_safe,
        "MOTION_OR_TRANSPORT_REPORTED",
        "bundle must prove zero motion publication and no ROS/CAN transport",
    )

    references, references_error = _artifact_references(bundle)
    check(
        "artifact_manifest",
        references is not None,
        "INVALID_ARTIFACT_MANIFEST",
        references_error or "artifact manifest rejected",
    )
    resolved_artifacts: dict[str, Path] = {}
    raw_paths: set[str] = set()
    if references is not None and resolved_bundle is not None:
        for name, reference in references.items():
            raw_path = str(reference["path"])
            resolved, path_error = _resolve_file(
                raw_path,
                base=resolved_bundle.parent,
                root=root,
                label=f"artifact {name}",
            )
            path_ok = resolved is not None
            if path_ok and resolved is not None:
                expected_size = reference.get("size_bytes")
                if expected_size is not None:
                    path_ok = (
                        isinstance(expected_size, int)
                        and not isinstance(expected_size, bool)
                        and expected_size == resolved.stat().st_size
                    )
                expected_digest = reference.get("sha256")
                if path_ok and expected_digest is not None:
                    path_ok = bool(
                        isinstance(expected_digest, str)
                        and re.fullmatch(r"[0-9a-f]{64}", expected_digest)
                        and _sha256(resolved) == expected_digest
                    )
            check(
                f"artifact:{name}",
                path_ok,
                "UNSAFE_OR_TAMPERED_ARTIFACT",
                path_error or f"artifact {name!r} failed path or integrity validation",
                artifact=name,
            )
            if path_ok and resolved is not None:
                resolved_artifacts[name] = resolved
                raw_paths.add(raw_path)

    visualization = bundle.get("visualization")
    overlay_allowed = (
        visualization.get("robot_overlay_allowed")
        if isinstance(visualization, dict)
        else None
    )
    images = visualization.get("images") if isinstance(visualization, dict) else None
    image_refs_safe = isinstance(images, dict)
    if isinstance(images, dict) and resolved_bundle is not None:
        for raw in images.values():
            if not isinstance(raw, str):
                image_refs_safe = False
                continue
            resolved, _error_text = _resolve_file(
                raw,
                base=resolved_bundle.parent,
                root=root,
                label="visualization image",
            )
            if resolved is None or raw not in raw_paths:
                image_refs_safe = False
    check(
        "visualization_image_references",
        image_refs_safe,
        "UNMANIFESTED_VISUALIZATION_IMAGE",
        "visualization images must refer to validated manifest artifacts",
    )

    calibration_path = resolved_artifacts.get("camera_calibration")
    calibration_document = None
    if calibration_path is not None:
        calibration_document, _calibration_error = _read_json(
            calibration_path,
            "camera calibration",
        )
    calibration_ok = bool(
        calibration_document is not None
        and _calibration_valid(calibration_document)
    )
    if not calibration_ok:
        check(
            "uncalibrated_robot_overlay_disabled",
            overlay_allowed is False,
            "ROBOT_OVERLAY_WITHOUT_VALID_CALIBRATION",
            "robot overlay must be explicitly false without valid real calibration",
            robot_overlay_allowed=overlay_allowed,
        )
    else:
        checks.append({
            "name": "uncalibrated_robot_overlay_disabled",
            "passed": True,
            "details": {"valid_calibration": True},
        })

    selected_joint_path: Path | None = None
    joint_path_error = None
    if joint_report is not None:
        selected_joint_path, joint_path_error = _resolve_file(
            joint_report,
            base=Path.cwd(),
            root=root,
            label="joint report",
        )
    elif "joint_report" in resolved_artifacts:
        selected_joint_path = resolved_artifacts["joint_report"]

    joint_document = None
    joint_error = joint_path_error
    if selected_joint_path is not None:
        joint_document, joint_error = _read_json(selected_joint_path, "joint report")
    joint_evidence_supplied = joint_report is not None or "joint_report" in resolved_artifacts
    if joint_evidence_supplied:
        joint_ok = bool(
            joint_document is not None
            and _joint_report_valid(joint_document)
        )
        check(
            "passive_joint_zero_tx_evidence",
            joint_ok,
            "INVALID_OR_NONZERO_TX_JOINT_REPORT",
            joint_error or "joint report lacks complete passive zero-TX evidence",
        )
    else:
        joint_ok = False
        checks.append({
            "name": "passive_joint_zero_tx_evidence",
            "passed": True,
            "details": {"status": "not_supplied", "required_for_robot_overlay": True},
        })
    if not joint_ok:
        check(
            "jointless_robot_overlay_disabled",
            overlay_allowed is False,
            "ROBOT_OVERLAY_WITHOUT_PASSIVE_JOINT_EVIDENCE",
            "robot overlay must be explicitly false without passive zero-TX joint evidence",
            robot_overlay_allowed=overlay_allowed,
        )
    else:
        checks.append({
            "name": "jointless_robot_overlay_disabled",
            "passed": True,
            "details": {"passive_zero_tx_joint_evidence": True},
        })

    passed = not errors
    return {
        "schema": AUDIT_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "passed": passed,
        "mode": {
            "read_only": True,
            "planning_only": True,
            "offline_artifact_reader": True,
        },
        "motion_commands_published": 0,
        "artifact_root": str(root),
        "bundle": None if resolved_bundle is None else str(resolved_bundle),
        "calibration_valid": calibration_ok,
        "can_zero_transmit_verified": joint_ok,
        "robot_overlay_allowed": overlay_allowed is True and calibration_ok and joint_ok,
        "checks": checks,
        "errors": errors,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    audit = audit_bundle(
        args.bundle,
        args.artifact_root,
        joint_report=args.joint_report,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "passed": audit["passed"],
        "error_count": len(audit["errors"]),
    }, indent=2))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
