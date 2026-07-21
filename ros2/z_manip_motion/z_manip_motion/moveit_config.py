"""Fail-closed normalization for external MoveIt configuration inputs."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree


class MoveItConfigError(ValueError):
    """Raised when an external MoveIt input cannot be used safely."""


_SENSOR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_POINT_CLOUD_UPDATER = "occupancy_map_monitor/PointCloudOctomapUpdater"
_DEPTH_IMAGE_UPDATER = "occupancy_map_monitor/DepthImageOctomapUpdater"


def _require_file(path: Path, *, reference: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MoveItConfigError(
            f"mesh resource does not exist or is not a file: {reference!r} -> {resolved}"
        )
    return resolved


def _default_package_share(package_name: str) -> str:
    from ament_index_python.packages import get_package_share_directory

    return get_package_share_directory(package_name)


def _resolve_mesh_uri(
    reference: str,
    *,
    urdf_directory: Path,
    package_share_lookup: Callable[[str], str],
) -> Path:
    parsed = urlparse(reference)
    if parsed.query or parsed.fragment:
        raise MoveItConfigError(f"mesh URI cannot contain query or fragment: {reference!r}")

    if not parsed.scheme:
        candidate = Path(unquote(reference))
        if not candidate.is_absolute():
            candidate = urdf_directory / candidate
        return _require_file(candidate, reference=reference)

    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise MoveItConfigError(f"non-local file mesh URI is unsupported: {reference!r}")
        return _require_file(Path(unquote(parsed.path)), reference=reference)

    if parsed.scheme == "package":
        package_name = parsed.netloc
        relative = PurePosixPath(unquote(parsed.path).lstrip("/"))
        if not package_name or not relative.parts or ".." in relative.parts:
            raise MoveItConfigError(f"invalid package mesh URI: {reference!r}")
        try:
            package_share = Path(package_share_lookup(package_name)).resolve()
        except Exception as error:
            raise MoveItConfigError(
                f"cannot resolve mesh package {package_name!r}: {reference!r}"
            ) from error
        candidate = (package_share / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(package_share)
        except ValueError as error:
            raise MoveItConfigError(f"mesh escapes package share: {reference!r}") from error
        return _require_file(candidate, reference=reference)

    raise MoveItConfigError(f"unsupported mesh URI scheme in {reference!r}")


def normalized_robot_description(
    urdf_path: str | Path,
    *,
    package_share_lookup: Callable[[str], str] | None = None,
) -> str:
    """Load a URDF and canonicalize every mesh reference to a checked file URI."""

    candidate = Path(urdf_path).expanduser().resolve()
    if not candidate.is_file():
        raise MoveItConfigError(f"robot description file does not exist: {candidate}")
    try:
        root = ElementTree.fromstring(candidate.read_text())
    except (OSError, ElementTree.ParseError) as error:
        raise MoveItConfigError(f"invalid robot description XML {candidate}: {error}") from error
    if root.tag != "robot":
        raise MoveItConfigError(f"robot description root must be <robot>: {candidate}")

    lookup = package_share_lookup or _default_package_share
    for mesh in root.iter("mesh"):
        reference = (mesh.get("filename") or "").strip()
        if not reference:
            raise MoveItConfigError("every URDF <mesh> must have a non-empty filename")
        resolved = _resolve_mesh_uri(
            reference,
            urdf_directory=candidate.parent,
            package_share_lookup=lookup,
        )
        mesh.set("filename", resolved.as_uri())

    return ElementTree.tostring(root, encoding="unicode")


def _sensor_entries(profile: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if set(profile) != {"sensors"}:
        unexpected = sorted(set(profile) - {"sensors"})
        raise MoveItConfigError(
            f"3-D sensor profile has unsupported top-level keys: {unexpected}"
        )
    sensors = profile.get("sensors")
    if isinstance(sensors, Mapping):
        entries = list(sensors.items())
    elif isinstance(sensors, Sequence) and not isinstance(sensors, (str, bytes)):
        entries = []
        for index, item in enumerate(sensors):
            if not isinstance(item, Mapping):
                raise MoveItConfigError(f"3-D sensor entry {index} must be a mapping")
            item = dict(item)
            name = item.pop("name", None)
            entries.append((name, item))
    else:
        raise MoveItConfigError("3-D sensor profile 'sensors' must be a mapping or list")

    if not entries:
        raise MoveItConfigError("3-D sensor profile must define at least one sensor")
    normalized: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for name, parameters in entries:
        if not isinstance(name, str) or not _SENSOR_NAME.fullmatch(name):
            raise MoveItConfigError(f"invalid 3-D sensor name: {name!r}")
        if name in seen:
            raise MoveItConfigError(f"duplicate 3-D sensor name: {name!r}")
        if not isinstance(parameters, Mapping):
            raise MoveItConfigError(f"3-D sensor {name!r} parameters must be a mapping")
        seen.add(name)
        normalized.append((name, parameters))
    return normalized


def _validate_parameter_value(name: str, value: Any) -> None:
    scalar = isinstance(value, (bool, int, float, str))
    sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if not scalar and not sequence:
        raise MoveItConfigError(f"3-D sensor parameter {name!r} must be scalar or a list")
    values = value if sequence else (value,)
    if any(
        isinstance(item, (Mapping, Sequence))
        and not isinstance(item, (str, bytes))
        for item in values
    ):
        raise MoveItConfigError(f"3-D sensor parameter {name!r} cannot be nested")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values):
        raise MoveItConfigError(f"3-D sensor parameter {name!r} must be finite")


def moveit_sensor_parameters(
    profile: Mapping[str, Any],
    *,
    point_cloud_topic: str,
    depth_image_topic: str,
    filtered_cloud_topic: str,
    max_range: float,
) -> dict[str, Any]:
    """Convert a named sensor profile to MoveIt 2's flat ROS parameter form."""

    topics = {
        "point_cloud_topic": point_cloud_topic,
        "image_topic": depth_image_topic,
        "filtered_cloud_topic": filtered_cloud_topic,
    }
    if any(not isinstance(value, str) or not value.strip() for value in topics.values()):
        raise MoveItConfigError("3-D sensor topic overrides must be non-empty strings")
    if not math.isfinite(max_range) or max_range <= 0.0:
        raise MoveItConfigError("3-D sensor max_range override must be finite and positive")

    entries = _sensor_entries(profile)
    result: dict[str, Any] = {"sensors": [name for name, _ in entries]}
    for name, raw_parameters in entries:
        parameters = dict(raw_parameters)
        plugin = parameters.get("sensor_plugin")
        if not isinstance(plugin, str) or not plugin.strip():
            raise MoveItConfigError(f"3-D sensor {name!r} requires sensor_plugin")

        if plugin == _POINT_CLOUD_UPDATER:
            parameters["point_cloud_topic"] = point_cloud_topic
            parameters["max_range"] = max_range
            parameters["filtered_cloud_topic"] = filtered_cloud_topic
        elif plugin == _DEPTH_IMAGE_UPDATER:
            parameters["image_topic"] = depth_image_topic
            parameters["far_clipping_plane_distance"] = max_range
            parameters["filtered_cloud_topic"] = filtered_cloud_topic
        else:
            for key, value in topics.items():
                if key in parameters:
                    parameters[key] = value
            if "max_range" in parameters:
                parameters["max_range"] = max_range
            if "far_clipping_plane_distance" in parameters:
                parameters["far_clipping_plane_distance"] = max_range

        for key, value in parameters.items():
            if not isinstance(key, str) or not key or "." in key:
                raise MoveItConfigError(
                    f"invalid parameter name for 3-D sensor {name!r}: {key!r}"
                )
            _validate_parameter_value(f"{name}.{key}", value)
            result[f"{name}.{key}"] = value
    return result
