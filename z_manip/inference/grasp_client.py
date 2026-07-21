"""Model-independent client for learned 6-DoF grasp inference.

The wire contract is msgpack-friendly, but the core client accepts an injected
transport so protocol and safety validation require neither ZMQ nor a running
model server.  The optional production transport imports ``zmq`` and
``msgpack`` only when its first request is made.

Only observed object points, aligned colors, and scene bounds cross this
boundary.  There is intentionally no object-pose input: returned poses must be
expressed directly in the request's observation frame.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from z_manip.models.grasp_source import GraspGenerationError


PROTOCOL_VERSION = "z-manip.grasp.v1"
GRASP_CONVENTION = "x_closing_y_binormal_z_approach"
_OPERATIONS = frozenset(("health", "metadata", "infer"))
_FORBIDDEN_FIELDS = frozenset(("object_pose", "gt_pose", "world_object_pose"))


class GraspInferenceError(GraspGenerationError):
    """Recoverable learned-backend failure.

    This derives from ``GraspGenerationError`` so the existing grasp-source
    cascade can continue to its geometric backend without special coupling.
    """


class GraspInferenceUnavailable(GraspInferenceError):
    """The provider, endpoint, or optional transport dependency is unavailable."""


class GraspInferenceTimeout(GraspInferenceUnavailable):
    """The model server did not answer within the configured deadline."""


class GraspInferenceProtocolError(GraspInferenceError):
    """The server response violated the versioned inference contract."""


@runtime_checkable
class InferenceTransport(Protocol):
    """Synchronous request/reply seam implemented by real and test transports."""

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Send one operation and return its decoded mapping."""
        ...


@dataclass(frozen=True)
class GraspInferenceConfig:
    provider: str
    endpoint: str
    timeout_s: float = 1.5
    max_grasps: int = 128
    convention: str = GRASP_CONVENTION

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("grasp inference provider must be non-empty")
        if not self.endpoint.strip():
            raise ValueError("grasp inference endpoint must be non-empty")
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("grasp inference timeout_s must be finite and positive")
        if self.max_grasps < 1:
            raise ValueError("grasp inference max_grasps must be positive")
        if self.convention != GRASP_CONVENTION:
            raise ValueError(f"unsupported grasp convention {self.convention!r}")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        prefix: str = "Z_MANIP_GRASP_",
    ) -> "GraspInferenceConfig":
        values = os.environ if environ is None else environ
        try:
            endpoint = values[f"{prefix}ENDPOINT"]
        except KeyError as error:
            raise ValueError(f"missing {prefix}ENDPOINT") from error
        provider = values.get(f"{prefix}PROVIDER", "learned")
        try:
            timeout_s = float(values.get(f"{prefix}TIMEOUT_S", "1.5"))
            max_grasps = int(values.get(f"{prefix}MAX_GRASPS", "128"))
        except ValueError as error:
            raise ValueError(f"invalid numeric {prefix} environment setting") from error
        return cls(
            provider=provider,
            endpoint=endpoint,
            timeout_s=timeout_s,
            max_grasps=max_grasps,
        )


@dataclass(frozen=True)
class HealthStatus:
    provider: str
    model: str
    model_version: str
    ready: bool


@dataclass(frozen=True)
class BackendMetadata:
    provider: str
    model: str
    model_version: str
    convention: str
    operations: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class GraspInferenceResult:
    grasps: np.ndarray
    scores: np.ndarray
    widths: np.ndarray
    frame: str
    convention: str
    provider: str
    model: str
    model_version: str


def _wire_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def _array_from_wire(value: object, label: str) -> np.ndarray:
    """Decode a compact array envelope, with lists allowed for simple servers."""

    if isinstance(value, Mapping):
        required = frozenset(("dtype", "shape", "data"))
        if set(value) != required:
            raise GraspInferenceProtocolError(
                f"{label} array envelope must contain exactly {sorted(required)}",
            )
        try:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or not (
                np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
            ):
                raise TypeError(f"unsupported non-numeric dtype {dtype}")
            shape = tuple(int(dimension) for dimension in value["shape"])
            if any(dimension < 0 for dimension in shape):
                raise ValueError("negative array dimension")
            data = value["data"]
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError("array data is not bytes")
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(data) != expected_bytes:
                raise ValueError(
                    f"array payload has {len(data)} bytes; expected {expected_bytes}",
                )
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        except (TypeError, ValueError, OverflowError) as error:
            raise GraspInferenceProtocolError(f"invalid {label} array envelope: {error}") from error
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise GraspInferenceProtocolError(f"invalid {label} array: {error}") from error


class ZmqMsgpackTransport:
    """Lazy optional REQ transport; one socket per request avoids stale REQ state."""

    def __init__(
        self,
        endpoint: str,
        *,
        importer: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        if not endpoint:
            raise ValueError("ZMQ endpoint must be non-empty")
        self.endpoint = endpoint
        self._importer = importer

    def _dependencies(self) -> tuple[Any, Any]:
        try:
            return self._importer("zmq"), self._importer("msgpack")
        except (ImportError, ModuleNotFoundError) as error:
            raise GraspInferenceUnavailable(
                "msgpack-over-ZMQ transport requires optional packages "
                "'pyzmq' and 'msgpack'",
            ) from error

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        if operation not in _OPERATIONS:
            raise GraspInferenceProtocolError(f"unsupported inference operation {operation!r}")
        zmq, msgpack = self._dependencies()
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.endpoint)
            request = {
                "protocol": PROTOCOL_VERSION,
                "operation": operation,
                "payload": dict(payload),
            }
            socket.send(msgpack.packb(request, use_bin_type=True))
            if not socket.poll(max(1, int(np.ceil(timeout_s * 1000.0))), zmq.POLLIN):
                raise GraspInferenceTimeout(
                    f"grasp inference {operation} timed out after {timeout_s:.3f} s",
                )
            response = msgpack.unpackb(socket.recv(), raw=False, strict_map_key=False)
        except GraspInferenceError:
            raise
        except Exception as error:
            raise GraspInferenceUnavailable(
                f"grasp inference transport failed: {type(error).__name__}: {error}",
            ) from error
        finally:
            socket.close(linger=0)
        if not isinstance(response, Mapping):
            raise GraspInferenceProtocolError("model server response must be a mapping")
        return response


class GraspInferenceClient:
    """Validate observation-only requests and fail-closed 6-DoF responses."""

    def __init__(
        self,
        config: GraspInferenceConfig,
        *,
        transport: InferenceTransport | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.transport = transport or ZmqMsgpackTransport(config.endpoint)
        self._monotonic_fn = monotonic_fn

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        transport: InferenceTransport | None = None,
    ) -> "GraspInferenceClient":
        return cls(GraspInferenceConfig.from_env(environ), transport=transport)

    def _request(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            start = float(self._monotonic_fn())
        except Exception as error:
            raise GraspInferenceUnavailable(f"inference deadline clock failed: {error}") from error
        if not np.isfinite(start):
            raise GraspInferenceUnavailable("inference deadline clock is non-finite")
        try:
            response = self.transport.request(
                operation,
                payload,
                timeout_s=self.config.timeout_s,
            )
        except GraspInferenceError:
            raise
        except TimeoutError as error:
            raise GraspInferenceTimeout(
                f"grasp inference {operation} timed out after {self.config.timeout_s:.3f} s",
            ) from error
        except Exception as error:
            raise GraspInferenceUnavailable(
                f"grasp inference {operation} failed: {type(error).__name__}: {error}",
            ) from error
        try:
            elapsed = float(self._monotonic_fn()) - start
        except Exception as error:
            raise GraspInferenceUnavailable(f"inference deadline clock failed: {error}") from error
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise GraspInferenceUnavailable("inference deadline clock moved backwards or is non-finite")
        if elapsed > self.config.timeout_s:
            raise GraspInferenceTimeout(
                f"grasp inference {operation} exceeded {self.config.timeout_s:.3f} s deadline",
            )
        if not isinstance(response, Mapping):
            raise GraspInferenceProtocolError("model server response must be a mapping")
        forbidden = self._forbidden_fields(response)
        if forbidden:
            raise GraspInferenceProtocolError(
                f"model server returned forbidden pose fields: {sorted(forbidden)}",
            )
        if response.get("protocol") != PROTOCOL_VERSION:
            raise GraspInferenceProtocolError(
                f"expected protocol {PROTOCOL_VERSION!r}, got {response.get('protocol')!r}",
            )
        if response.get("status") != "ok":
            detail = response.get("error", "provider reported failure")
            raise GraspInferenceUnavailable(f"{self.config.provider} {operation} failed: {detail}")
        return response

    @staticmethod
    def _forbidden_fields(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            found.update(_FORBIDDEN_FIELDS.intersection(value))
            for nested in value.values():
                found.update(GraspInferenceClient._forbidden_fields(nested))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found.update(GraspInferenceClient._forbidden_fields(nested))
        return found

    def _model_fields(self, response: Mapping[str, Any]) -> tuple[str, str, str]:
        values = tuple(response.get(field) for field in ("provider", "model", "model_version"))
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise GraspInferenceProtocolError(
                "response provider, model, and model_version must be non-empty strings",
            )
        provider, model, model_version = values
        if provider != self.config.provider:
            raise GraspInferenceProtocolError(
                f"configured provider {self.config.provider!r} answered as {provider!r}",
            )
        return provider, model, model_version

    def health(self) -> HealthStatus:
        response = self._request("health", {})
        provider, model, model_version = self._model_fields(response)
        if response.get("ready") is not True:
            raise GraspInferenceUnavailable(f"{provider}/{model} is not ready")
        return HealthStatus(provider, model, model_version, True)

    def metadata(self) -> BackendMetadata:
        response = self._request("metadata", {})
        provider, model, model_version = self._model_fields(response)
        convention = response.get("convention")
        if convention != self.config.convention:
            raise GraspInferenceProtocolError(
                f"provider convention {convention!r} does not match {self.config.convention!r}",
            )
        operations = response.get("operations")
        if not isinstance(operations, (list, tuple)) or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise GraspInferenceProtocolError("metadata operations must be a string list")
        operation_set = set(operations)
        if not _OPERATIONS.issubset(operation_set):
            raise GraspInferenceProtocolError(
                f"provider metadata is missing operations {sorted(_OPERATIONS - operation_set)}",
            )
        return BackendMetadata(
            provider,
            model,
            model_version,
            convention,
            tuple(operations),
        )

    @staticmethod
    def _observation_arrays(
        object_points: object,
        colors: object | None,
        scene_bounds: object,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        try:
            points = np.asarray(object_points, dtype=np.float32)
            bounds = np.asarray(scene_bounds, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise GraspInferenceProtocolError(f"invalid inference observation: {error}") from error
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 1:
            raise GraspInferenceProtocolError("object_points must have shape (N, 3), N >= 1")
        if not np.all(np.isfinite(points)):
            raise GraspInferenceProtocolError("object_points contain non-finite values")
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise GraspInferenceProtocolError("scene_bounds must be a finite (2, 3) array")
        if np.any(bounds[0] >= bounds[1]):
            raise GraspInferenceProtocolError("scene_bounds minima must be below maxima")
        tolerance = 1e-5
        if np.any(points < bounds[0] - tolerance) or np.any(points > bounds[1] + tolerance):
            raise GraspInferenceProtocolError("object_points lie outside scene_bounds")

        packed_colors = None
        if colors is not None:
            try:
                raw_colors = np.asarray(colors)
            except (TypeError, ValueError) as error:
                raise GraspInferenceProtocolError(f"invalid colors array: {error}") from error
            if raw_colors.shape != points.shape:
                raise GraspInferenceProtocolError("colors must align with object_points as (N, 3)")
            if raw_colors.dtype == np.bool_ or not np.issubdtype(raw_colors.dtype, np.number):
                raise GraspInferenceProtocolError("colors must be numeric RGB values")
            if not np.all(np.isfinite(raw_colors)):
                raise GraspInferenceProtocolError("colors contain non-finite values")
            if np.issubdtype(raw_colors.dtype, np.integer):
                if np.any(raw_colors < 0) or np.any(raw_colors > 255):
                    raise GraspInferenceProtocolError("integer colors must be in [0, 255]")
                packed_colors = raw_colors.astype(np.uint8)
            else:
                if np.any(raw_colors < 0.0) or np.any(raw_colors > 1.0):
                    raise GraspInferenceProtocolError("floating colors must be in [0, 1]")
                packed_colors = raw_colors.astype(np.float32)
        return points, packed_colors, bounds

    def infer(
        self,
        *,
        object_points: object,
        scene_bounds: object,
        frame: str,
        colors: object | None = None,
    ) -> GraspInferenceResult:
        """Return strictly validated poses from observation geometry only."""

        if not isinstance(frame, str) or not frame.strip():
            raise GraspInferenceProtocolError("inference frame must be a non-empty string")
        points, packed_colors, bounds = self._observation_arrays(
            object_points,
            colors,
            scene_bounds,
        )
        payload = {
            "object_points": _wire_array(points),
            "colors": None if packed_colors is None else _wire_array(packed_colors),
            "scene_bounds": _wire_array(bounds),
            "frame": frame,
            "convention": self.config.convention,
            "max_grasps": self.config.max_grasps,
        }
        response = self._request("infer", payload)
        provider, model, model_version = self._model_fields(response)
        response_frame = response.get("frame")
        if response_frame != frame:
            raise GraspInferenceProtocolError(
                f"provider returned frame {response_frame!r}; expected observation frame {frame!r}",
            )
        convention = response.get("convention")
        if convention != self.config.convention:
            raise GraspInferenceProtocolError(
                f"provider returned convention {convention!r}; expected {self.config.convention!r}",
            )
        try:
            grasps = np.asarray(
                _array_from_wire(response.get("grasps"), "grasps"),
                dtype=np.float64,
            )
            scores = np.asarray(
                _array_from_wire(response.get("scores"), "scores"),
                dtype=np.float64,
            )
            widths = np.asarray(
                _array_from_wire(response.get("widths"), "widths"),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise GraspInferenceProtocolError(f"response arrays must be numeric: {error}") from error
        self._validate_candidates(grasps, scores, widths)
        return GraspInferenceResult(
            grasps=grasps,
            scores=scores,
            widths=widths,
            frame=response_frame,
            convention=convention,
            provider=provider,
            model=model,
            model_version=model_version,
        )

    def _validate_candidates(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        widths: np.ndarray,
    ) -> None:
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            raise GraspInferenceProtocolError("grasps must have shape (N, 4, 4)")
        count = len(grasps)
        if count < 1:
            raise GraspInferenceProtocolError("provider returned no grasp candidates")
        if count > self.config.max_grasps:
            raise GraspInferenceProtocolError(
                f"provider returned {count} grasps; requested at most {self.config.max_grasps}",
            )
        if scores.shape != (count,) or widths.shape != (count,):
            raise GraspInferenceProtocolError("scores and widths must align one-to-one with grasps")
        if not all(np.all(np.isfinite(array)) for array in (grasps, scores, widths)):
            raise GraspInferenceProtocolError("grasp response contains non-finite values")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise GraspInferenceProtocolError("grasp scores must be normalized to [0, 1]")
        if np.any(widths <= 0.0):
            raise GraspInferenceProtocolError("grasp widths must be positive meters")
        expected_bottom = np.broadcast_to((0.0, 0.0, 0.0, 1.0), (count, 4))
        if not np.allclose(grasps[:, 3, :], expected_bottom, atol=1e-6):
            raise GraspInferenceProtocolError("grasp matrices have invalid homogeneous bottom rows")
        rotations = grasps[:, :3, :3]
        orthogonality = np.swapaxes(rotations, 1, 2) @ rotations
        if not np.allclose(orthogonality, np.eye(3), atol=1e-4):
            raise GraspInferenceProtocolError("grasp rotations are not orthonormal")
        determinants = np.linalg.det(rotations)
        if np.any(determinants <= 0.0):
            raise GraspInferenceProtocolError("grasp rotations are left-handed reflections")
        if not np.allclose(determinants, 1.0, atol=1e-4):
            raise GraspInferenceProtocolError("grasp rotations do not have determinant +1")
