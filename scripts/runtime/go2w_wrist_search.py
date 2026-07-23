#!/usr/bin/env python3
"""Bounded wrist-camera search coordinator used by the loopback workbench.

The coordinator is transport-agnostic.  In shadow mode it never invokes the
motion adapter.  Live mode is available only when the server was started with
a fixed executable and ``Z_MANIP_ENABLE_WRIST_SEARCH=1``; the browser cannot
provide a joint target or command.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from z_manip.control.wrist_search import (
    BoundedWristSearch,
    WristSearchConfig,
    WristSearchDecision,
    WristSearchPhase,
)


class DetectorProbe:
    """Call the existing loopback-only open-vocabulary detector."""

    def __init__(
        self,
        camera_image: Path,
        *,
        endpoint: str = "http://127.0.0.1:8771/ground",
        timeout_s: float = 2.0,
    ) -> None:
        self.camera_image = camera_image.expanduser().resolve()
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)

    def __call__(self, target: str) -> tuple[bool, float | None, str]:
        try:
            image = self.camera_image.read_bytes()
            if not 1 <= len(image) <= 512 * 1024:
                return False, None, "camera image is unavailable or oversized"
            payload = json.dumps({
                "schema": "z_manip.local_grounding_request.v1",
                "instruction": target,
                "image_base64": base64.b64encode(image).decode("ascii"),
            }).encode("utf-8")
            request = Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.timeout_s) as response:
                document = json.loads(response.read())
            detection = document.get("target") if isinstance(document, dict) else None
            confidence = detection.get("confidence") if isinstance(detection, dict) else None
            value = float(confidence)
            if not np.isfinite(value):
                raise ValueError("non-finite detector confidence")
            return True, value, str(detection.get("label", "target"))
        except HTTPError as error:
            if error.code == 422:
                return False, None, "target not detected"
            return False, None, f"detector HTTP {error.code}"
        except (OSError, ValueError, TypeError, json.JSONDecodeError, URLError) as error:
            return False, None, f"detector unavailable: {type(error).__name__}"


class FixedWristMotion:
    """Invoke one fixed script by view index; never accepts joint targets."""

    def __init__(self, script: Path, log_path: Path) -> None:
        self.script = script.expanduser().resolve()
        self.log_path = log_path.expanduser().resolve()
        if not self.script.is_file() or not self.script.stat().st_mode & 0o111:
            raise FileNotFoundError(f"fixed wrist-search script is unavailable: {self.script}")
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def stop(self) -> None:
        """Interrupt a currently running fixed-view command, if any."""
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)

    def __call__(self, view_index: int, speed_percent: int) -> np.ndarray:
        process = subprocess.Popen(
            [str(self.script), str(int(view_index)), str(int(speed_percent))],
            cwd=self.script.parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._lock:
            self._process = process
        try:
            stdout, _stderr = process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            self.stop()
            stdout, _stderr = process.communicate()
            raise RuntimeError("fixed wrist view exceeded its 30 second bound")
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_bytes(stdout[-64 * 1024 :])
        if process.returncode != 0:
            raise RuntimeError(f"fixed wrist view failed with code {process.returncode}")
        document = None
        for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("phase") == "complete":
                document = candidate
                break
        if document is None:
            raise RuntimeError("fixed wrist view omitted its completion receipt")
        joints = np.asarray(document.get("final_joints_rad"), dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("fixed wrist view returned invalid measured joints")
        return joints


class WristSearchCoordinator:
    """Run a finite search and publish inspectable state to the UI owner."""

    def __init__(
        self,
        home_joints_rad: np.ndarray,
        detector: Callable[[str], tuple[bool, float | None, str]],
        *,
        motion: Callable[[int, int], np.ndarray] | None = None,
        config: WristSearchConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        home = np.asarray(home_joints_rad, dtype=float)
        if home.shape != (6,) or not np.isfinite(home).all():
            raise ValueError("wrist search requires a finite six-joint Home anchor")
        self.home = home.copy()
        self.detector = detector
        self.motion = motion
        self.config = config or WristSearchConfig()
        self.sleep = sleep
        self.clock = clock
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._revision = 0
        self._status: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "mode": None,
            "target": None,
            "view_index": None,
            "view_count": len(BoundedWristSearch(self.config).views),
            "confidence": None,
            "confirmations": 0,
            "operator_authorized": False,
            "message": "Wrist search is idle.",
            "failure": None,
        }

    @property
    def live_enabled(self) -> bool:
        with self._lock:
            one_shot = self._status.get("operator_authorized") is True
        return self.motion is not None and (
            os.environ.get("Z_MANIP_ENABLE_WRIST_SEARCH") == "1" or one_shot
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
            result.update(
                schema="z_manip.wrist_search_status.v1",
                revision=self._revision,
                live_enabled=self.live_enabled,
            )
            return result

    def stop(self) -> None:
        self._cancel.set()
        stop_motion = getattr(self.motion, "stop", None)
        if callable(stop_motion):
            stop_motion()
        self._update(active=False, phase="stopped", message="Wrist search cancelled.")

    def _update(self, **values: Any) -> None:
        with self._lock:
            if all(self._status.get(key) == value for key, value in values.items()):
                return
            self._status.update(values)
            self._revision += 1

    def run(
        self,
        target: str,
        *,
        mode: str,
        speed_percent: int,
        cancel: threading.Event | None = None,
        operator_present: bool = False,
    ) -> bool:
        if mode not in {"shadow", "live"}:
            raise ValueError("wrist search mode must be shadow or live")
        one_shot_authorized = bool(operator_present) and self.motion is not None
        if mode == "live" and not (self.live_enabled or one_shot_authorized):
            self._update(
                active=False,
                phase="locked",
                failure="live wrist search requires operator-present enablement",
                message="Live wrist search is locked while the operator is absent.",
            )
            return False
        self._update(operator_authorized=one_shot_authorized)
        try:
            return self._run_authorized(
                target,
                mode=mode,
                speed_percent=speed_percent,
                cancel=cancel,
            )
        finally:
            # Browser confirmation authorizes only this one finite scan.  A
            # later task must be confirmed again unless the operator explicitly
            # enabled the service environment for a supervised session.
            self._update(operator_authorized=False)

    def _run_authorized(
        self,
        target: str,
        *,
        mode: str,
        speed_percent: int,
        cancel: threading.Event | None,
    ) -> bool:
        if not 1 <= int(speed_percent) <= 20:
            raise ValueError("wrist search speed must be within 1..20 percent")
        self._cancel = threading.Event()
        external_cancel = cancel
        search = BoundedWristSearch(self.config)
        now = self.clock()
        decision = search.start(self.home, now_s=now)
        self._update(
            active=True,
            phase=decision.phase.value,
            mode=mode,
            target=target,
            view_index=0,
            confidence=None,
            confirmations=0,
            failure=None,
            message="Searching the current camera view first.",
        )
        confirmed = False
        cancelled = False
        last_live_view = 0
        try:
            while decision.phase not in {
                WristSearchPhase.FOUND,
                WristSearchPhase.EXHAUSTED,
                WristSearchPhase.STOPPED,
            }:
                if self._cancel.is_set() or (external_cancel is not None and external_cancel.is_set()):
                    cancelled = True
                    search.stop()
                    self._update(active=False, phase="stopped", message="Wrist search cancelled.")
                    return False
                if decision.phase is WristSearchPhase.MOVE:
                    view = decision.view
                    assert view is not None
                    self._update(
                        phase="move" if mode == "live" else "shadow_view",
                        view_index=view.index,
                        message=(
                            f"Moving to bounded wrist view {view.index + 1}/{len(search.views)}."
                            if mode == "live"
                            else f"Shadowing wrist view {view.index + 1}/{len(search.views)}."
                        ),
                    )
                    measured = (
                        self.motion(view.index, min(int(speed_percent), 12))
                        if mode == "live" and self.motion is not None
                        else np.asarray(decision.target_joints_rad, dtype=float)
                    )
                    if mode == "live" and self.motion is not None:
                        last_live_view = view.index
                    decision = search.update_motion(measured, now_s=self.clock())
                    if decision.phase is WristSearchPhase.SETTLE:
                        self.sleep(self.config.settle_s)
                        decision = search.update_motion(measured, now_s=self.clock())
                    continue
                if decision.phase is WristSearchPhase.OBSERVE:
                    visible, confidence, detail = self.detector(target)
                    decision = search.observe(
                        visible=visible,
                        confidence=confidence,
                        now_s=self.clock(),
                    )
                    self._update(
                        phase=decision.phase.value,
                        confidence=confidence,
                        confirmations=decision.confirmations,
                        message=detail if not visible else f"Detector confidence {confidence:.3f}.",
                    )
                    if decision.phase is WristSearchPhase.OBSERVE:
                        self.sleep(self.config.observation_period_s)
                    continue
                raise RuntimeError(f"unexpected wrist search phase: {decision.phase}")
            found = decision.phase is WristSearchPhase.FOUND
            confirmed = found
            self._update(
                active=False,
                phase=decision.phase.value,
                confidence=decision.confidence,
                confirmations=decision.confirmations,
                message=(
                    "Target confirmed; handing the stable view to EdgeTAM and depth servo."
                    if found
                    else "Finite wrist search exhausted without a confirmed target."
                ),
                failure=None if found else "target not found in bounded wrist search",
            )
            return found
        finally:
            if (
                not confirmed
                and not cancelled
                and last_live_view != 0
                and mode == "live"
                and self.motion is not None
            ):
                # A failed search must not strand the wrist off-anchor: the
                # next Find would probe the current view from an arbitrary
                # pose and miss a target that is plainly visible from Home.
                try:
                    self.motion(0, min(int(speed_percent), 12))
                    self._update(
                        message=(
                            "Search ended without a target; wrist returned "
                            "to the Home anchor view."
                        ),
                    )
                except Exception as error:
                    self._update(
                        message=(
                            "Wrist anchor restore failed; run Reset + Home "
                            f"before the next Find: {error}"
                        ),
                    )
