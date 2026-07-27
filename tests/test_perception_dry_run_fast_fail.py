"""WS-B item #4: fail-fast no-seed in the perception dry-run bundle wait.

The dry-run imports rclpy/cv_bridge/ROS message packages at module scope, which
are unavailable on the PC.  Stub exactly those receive-only ROS surfaces so the
pure decision helpers, the argument gate, and the report classification can be
exercised without a live ROS graph.  The fast-fail decision itself is factored
into pure module-level helpers precisely so it is testable here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRY_RUN = ROOT / "scripts" / "runtime" / "go2w_perception_dry_run.py"
INTEGRATION = ROOT / "scripts" / "runtime" / "go2w_interactive_sessions.py"


def _install_ros_stubs() -> list[str]:
    """Register minimal stand-ins for the dry-run's ROS-only imports.

    Returns the names added so the caller can remove them again after the
    module is loaded -- the imported module keeps its own references to the
    stub classes, so nothing leaks into the rest of the pytest session.
    """

    added: list[str] = []

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        added.append(name)
        return mod

    cv_bridge = module("cv_bridge")
    cv_bridge.CvBridge = object

    diagnostic_msgs = module("diagnostic_msgs")
    diagnostic_msgs_msg = module("diagnostic_msgs.msg")
    diagnostic_msgs.msg = diagnostic_msgs_msg

    class _DiagnosticStatus:
        OK = 0
        WARN = 1
        ERROR = 2

    diagnostic_msgs_msg.DiagnosticArray = object
    diagnostic_msgs_msg.DiagnosticStatus = _DiagnosticStatus

    rclpy = module("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy_node = module("rclpy.node")
    rclpy_node.Node = object
    rclpy.node = rclpy_node
    rclpy_qos = module("rclpy.qos")
    rclpy_qos.DurabilityPolicy = type("DurabilityPolicy", (), {"TRANSIENT_LOCAL": 1})
    rclpy_qos.QoSProfile = object
    rclpy_qos.ReliabilityPolicy = type("ReliabilityPolicy", (), {"RELIABLE": 1})
    rclpy_qos.qos_profile_sensor_data = object
    rclpy.qos = rclpy_qos

    sensor_msgs = module("sensor_msgs")
    sensor_msgs_msg = module("sensor_msgs.msg")
    sensor_msgs.msg = sensor_msgs_msg
    sensor_msgs_msg.CameraInfo = object
    sensor_msgs_msg.Image = object
    sensor_msgs_msg.PointCloud2 = object
    sensor_msgs_py = module("sensor_msgs_py")
    point_cloud2 = module("sensor_msgs_py.point_cloud2")
    sensor_msgs_py.point_cloud2 = point_cloud2

    std_msgs = module("std_msgs")
    std_msgs_msg = module("std_msgs.msg")
    std_msgs.msg = std_msgs_msg
    std_msgs_msg.Bool = object
    std_msgs_msg.String = object

    return added


def _load_dry_run():
    added = _install_ros_stubs()
    sys.path.insert(0, str(DRY_RUN.parent))
    spec = importlib.util.spec_from_file_location("go2w_perception_dry_run", DRY_RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["go2w_perception_dry_run"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # The module now holds its own references to the stub classes; drop the
        # stub modules so no other test in this session sees a fake ROS graph.
        for name in added:
            sys.modules.pop(name, None)
    return module


def _load_integration():
    sys.path.insert(0, str(INTEGRATION.parent))
    spec = importlib.util.spec_from_file_location(
        "go2w_interactive_sessions", INTEGRATION,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["go2w_interactive_sessions"] = module
    spec.loader.exec_module(module)
    return module


DRY = _load_dry_run()


# ---------------------------------------------------------------------------
# The seed-acceptance predicate: a seed-bearing phase, a public-valid bundle,
# or a live track id each prove grounding produced a seed.
# ---------------------------------------------------------------------------

def test_seed_predicate_true_for_tracking_phase():
    assert DRY._status_indicates_seed(
        phase="tracking", valid_field="false", track_id="",
    ) is True


def test_seed_predicate_true_for_waiting_tracker_phase():
    assert DRY._status_indicates_seed(
        phase="WAITING_TRACKER", valid_field="", track_id="",
    ) is True


def test_seed_predicate_true_for_public_valid_or_live_track():
    assert DRY._status_indicates_seed(
        phase="grounding", valid_field="true", track_id="",
    ) is True
    assert DRY._status_indicates_seed(
        phase="grounding", valid_field="false", track_id="edge-7",
    ) is True


def test_seed_predicate_false_while_only_grounding():
    # No seed yet: grounding is still running, no valid bundle, no track.
    assert DRY._status_indicates_seed(
        phase="grounding", valid_field="false", track_id="",
    ) is False
    assert DRY._status_indicates_seed(
        phase="waiting_frame", valid_field="", track_id="",
    ) is False


# ---------------------------------------------------------------------------
# The fast-fail decision: abort only when the stack is alive but no seed has
# been accepted within the bounded budget.  Keep waiting once a seed exists.
# ---------------------------------------------------------------------------

def test_fastfail_keeps_waiting_once_seed_accepted():
    # This is the EdgeTAM depth-dropout coast: a seed exists, depth is still
    # being acquired, so the full wait is preserved even far past the budget.
    assert DRY._no_seed_fastfail_reason(
        seed_accepted=True, status_observed=True, elapsed_s=99.0, budget_s=8.0,
    ) is None


def test_fastfail_waits_before_the_budget_elapses():
    assert DRY._no_seed_fastfail_reason(
        seed_accepted=False, status_observed=True, elapsed_s=5.0, budget_s=8.0,
    ) is None


def test_fastfail_needs_a_live_perception_stack():
    # No status observed at all -> the classic timeout path handles it, not the
    # legible target-not-found abort.
    assert DRY._no_seed_fastfail_reason(
        seed_accepted=False, status_observed=False, elapsed_s=99.0, budget_s=8.0,
    ) is None


def test_fastfail_budget_zero_disables_the_abort():
    assert DRY._no_seed_fastfail_reason(
        seed_accepted=False, status_observed=True, elapsed_s=99.0, budget_s=0.0,
    ) is None


def test_fastfail_aborts_no_seed_with_grounding_failed_reason():
    reason = DRY._no_seed_fastfail_reason(
        seed_accepted=False, status_observed=True, elapsed_s=8.0, budget_s=8.0,
    )
    assert reason is not None
    # The "grounding_failed" prefix is load-bearing: the session layer maps it
    # to PERCEPTION_TARGET_NOT_FOUND (asserted below).
    assert reason.startswith("grounding_failed")
    assert "no segmentation seed" in reason


# ---------------------------------------------------------------------------
# The new --no-seed-timeout gate.
# ---------------------------------------------------------------------------

def _args(*extra):
    return DRY._arguments(
        ["--instruction", "white charger", "--output", "/tmp/zmb-out", *extra],
    )


def test_no_seed_timeout_defaults_to_the_module_constant():
    args = _args()
    assert args.no_seed_timeout == DRY.DEFAULT_NO_SEED_TIMEOUT_S


def test_no_seed_timeout_accepts_zero_and_positive():
    assert _args("--no-seed-timeout", "0").no_seed_timeout == 0.0
    assert _args("--no-seed-timeout", "6.5").no_seed_timeout == 6.5


@pytest.mark.parametrize("bad", ["-1", "nan", "-0.001"])
def test_no_seed_timeout_rejects_negative_or_nonfinite(bad):
    with pytest.raises(SystemExit):
        _args("--no-seed-timeout", bad)


# ---------------------------------------------------------------------------
# End-to-end legibility + fail-closed: the report the fast-fail writes maps to
# PERCEPTION_TARGET_NOT_FOUND and is never retried as a fresh-seed recovery.
# ---------------------------------------------------------------------------

def test_no_seed_report_maps_to_target_not_found_and_is_not_retried(tmp_path):
    integration = _load_integration()
    reason = DRY._no_seed_fastfail_reason(
        seed_accepted=False, status_observed=True, elapsed_s=8.0, budget_s=8.0,
    )
    # A faithful copy of the report the dry-run writes on a no-seed fast-fail.
    report = {
        "read_only": True,
        "perception_bundle_valid": False,
        "perception_failure": reason,
        "seed_accepted": False,
        "no_seed_fast_fail": True,
        "no_seed_timeout_s": 8.0,
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    result = integration.FixedReadOnlyBackend._perception_failure_result(tmp_path, 5)
    assert result.error_code == "PERCEPTION_TARGET_NOT_FOUND"
    # Fail closed: a genuinely absent target is deterministic, so a fresh-seed
    # retry is never issued for this disposition.
    assert integration.FixedReadOnlyBackend._perception_retryable(tmp_path, 5) is False


def test_fast_fail_is_wired_into_the_bundle_wait_loop():
    # Guard the loop integration and report surface so the pure decision cannot
    # be silently disconnected from main().
    source = DRY_RUN.read_text(encoding="utf-8")
    assert "no_seed_reason = _no_seed_fastfail_reason(" in source
    assert "no_seed_fast_fail = True" in source
    assert "\"no_seed_fast_fail\": no_seed_fast_fail," in source
    assert "no_seed_budget = min(args.no_seed_timeout, args.timeout)" in source
