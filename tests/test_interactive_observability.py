"""WS-B item #15: cold-path observability in the interactive session backend.

  (a) a bounded worker stdout/stderr tail is folded into a failed attempt when
      the perception process died without a structured report;
  (b) the rc=70 fingerprint self-heal count and cumulative cost are persisted;
  (c) the VLM reuse hit-ratio (how often a fresh YOLOE+VLM grounding was
      avoided by tracking reuse) is persisted;
  (d) the passive-window supervision granularity, the selected-evidence gate
      that decides whether another remote observation period is opened, and the
      fail-closed shape of the predicate that may cut one short.

(a)-(c) run only on cold paths (a completed/failed attempt or a self-heal), so
they add no I/O or subprocess cost to the hot success path.

(d) guards a safety property at the source level: the zero-TX probe attests
that a nominally read-only request never saw the arm transmit, so no latency
optimisation may be able to destroy a window's verdict.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "scripts" / "runtime" / "go2w_interactive_sessions.py"
# The integration module imports a sibling runtime helper by bare name; make
# scripts/runtime importable so this suite also runs in isolation.
sys.path.insert(0, str(INTEGRATION.parent))


def _integration_module():
    spec = importlib.util.spec_from_file_location(
        "go2w_interactive_sessions_obs", INTEGRATION,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["go2w_interactive_sessions_obs"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _integration_module()


def _backend():
    return MODULE.FixedReadOnlyBackend(
        MODULE.ServerRuntimeConfig.from_server_environment({}),
    )


def _timing_records(log_path: Path, stage: str):
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("stage") == stage:
            records.append(payload)
    return records


# ---------------------------------------------------------------------------
# (a) worker stderr tail
# ---------------------------------------------------------------------------

def test_worker_stderr_tail_none_and_missing_return_empty(tmp_path):
    assert MODULE.FixedReadOnlyBackend._worker_stderr_tail(None) == ""
    assert MODULE.FixedReadOnlyBackend._worker_stderr_tail(tmp_path / "absent.log") == ""


def test_worker_stderr_tail_keeps_diagnostics_and_drops_timing_json(tmp_path):
    log = tmp_path / "perception.log"
    log.write_text(
        "importing ROS + grasp stack\n"
        + json.dumps({"schema": "z_manip.interactive_timing.v1", "stage": "x"})
        + "\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: RGB-D subscription never matched a publisher\n",
        encoding="utf-8",
    )
    tail = MODULE.FixedReadOnlyBackend._worker_stderr_tail(log)
    assert "RGB-D subscription never matched a publisher" in tail
    assert "importing ROS" in tail
    # Structured timing markers are not human diagnostics; they are dropped.
    assert "interactive_timing" not in tail
    assert " | " in tail  # lines joined into one bounded record


def test_worker_stderr_tail_is_bounded(tmp_path):
    log = tmp_path / "perception.log"
    log.write_text("\n".join(f"diagnostic line {i}" for i in range(500)), encoding="utf-8")
    tail = MODULE.FixedReadOnlyBackend._worker_stderr_tail(log)
    assert len(tail) <= MODULE.MAX_PERCEPTION_ERROR_CHARS
    # Only the most recent lines survive the tail.
    assert "diagnostic line 499" in tail
    assert "diagnostic line 0 " not in tail


def test_worker_stderr_tail_rejects_symlink(tmp_path):
    real = tmp_path / "real.log"
    real.write_text("boom\n", encoding="utf-8")
    link = tmp_path / "link.log"
    link.symlink_to(real)
    assert MODULE.FixedReadOnlyBackend._worker_stderr_tail(link) == ""


def test_process_failure_without_report_folds_in_the_stderr_tail(tmp_path):
    log = tmp_path / "perception.log"
    log.write_text(
        "RuntimeError: grounding bridge exited before publishing status\n",
        encoding="utf-8",
    )
    # Empty output dir -> no report.json -> the no-report PROCESS_FAILED branch.
    result = MODULE.FixedReadOnlyBackend._perception_failure_result(
        tmp_path, 1, log_path=log,
    )
    assert result.error_code == "PERCEPTION_PROCESS_FAILED"
    assert "worker stderr tail:" in result.message
    assert "grounding bridge exited before publishing status" in result.message


def test_process_failure_without_report_or_log_is_unchanged(tmp_path):
    result = MODULE.FixedReadOnlyBackend._perception_failure_result(tmp_path, 1)
    assert result.error_code == "PERCEPTION_PROCESS_FAILED"
    assert result.message == "read-only perception process failed without a valid report"


# ---------------------------------------------------------------------------
# (b) rc=70 fingerprint self-heal cost
# ---------------------------------------------------------------------------

def test_selfheal_cost_is_counted_and_accumulated(tmp_path):
    backend = _backend()
    log = tmp_path / "perception.log"

    backend._persist_fingerprint_selfheal_cost(
        log, cost_s=0.5, healed=True, healed_return_code=0,
    )
    backend._persist_fingerprint_selfheal_cost(
        log, cost_s=0.25, healed=False, healed_return_code=None,
    )

    records = _timing_records(log, "perception_fingerprint_selfheal_cost")
    assert len(records) == 2
    assert records[0]["selfheal_count"] == 1
    assert records[0]["selfheal_cost_total_s"] == 0.5
    assert records[0]["healed"] is True
    assert records[1]["selfheal_count"] == 2
    assert records[1]["selfheal_cost_total_s"] == 0.75
    assert records[1]["healed"] is False
    assert records[1]["trigger"] == "resident_worker_fingerprint_mismatch"
    assert backend._fingerprint_selfheal_count == 2
    assert backend._fingerprint_selfheal_cost_s == 0.75


# ---------------------------------------------------------------------------
# (c) VLM reuse hit-ratio
# ---------------------------------------------------------------------------

def test_reuse_hit_ratio_scores_reused_as_a_vlm_avoided_hit():
    backend = _backend()

    fresh = backend._grounding_observability(reused=False)
    assert fresh == {
        "vlm_avoided_reuse_hits": 0,
        "grounding_requests_scored": 1,
        "vlm_reuse_hit_ratio": 0.0,
    }

    reused = backend._grounding_observability(reused=True)
    assert reused["vlm_avoided_reuse_hits"] == 1
    assert reused["grounding_requests_scored"] == 2
    assert reused["vlm_reuse_hit_ratio"] == 0.5

    backend._grounding_observability(reused=True)
    third = backend._grounding_observability(reused=False)
    # 2 reuse hits over 4 scored requests.
    assert third["vlm_avoided_reuse_hits"] == 2
    assert third["grounding_requests_scored"] == 4
    assert third["vlm_reuse_hit_ratio"] == 0.5


# ---------------------------------------------------------------------------
# (d) passive-window supervision granularity
# ---------------------------------------------------------------------------

def test_supervision_granularity_is_far_below_the_observation_period():
    # The request must return on the perception worker's clock, not on the
    # boundary of a fixed remote observation period it no longer needs.
    assert MODULE.PASSIVE_CAPTURE_POLL_SECONDS > 0.0
    assert MODULE.PASSIVE_CAPTURE_POLL_SECONDS < float(
        MODULE.PASSIVE_CAPTURE_SECONDS,
    ) / 10.0


def test_selected_evidence_gate_admits_only_a_complete_zero_tx_report(tmp_path):
    backend = MODULE.FixedReadOnlyBackend
    output = tmp_path / "selected"
    output.mkdir()
    selected = output / "selected_passive_joint_report.json"

    assert backend._selected_passive_report_valid(output) is False

    selected.write_text(json.dumps({
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": False,
        "interface_tx_packet_delta": 3,
    }), encoding="utf-8")
    assert backend._selected_passive_report_valid(output) is False

    selected.write_text(json.dumps({
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
    }), encoding="utf-8")
    assert backend._selected_passive_report_valid(output) is True


def test_selected_evidence_gate_rejects_a_symlinked_report(tmp_path):
    output = tmp_path / "symlinked"
    output.mkdir()
    real = tmp_path / "real_report.json"
    real.write_text(json.dumps({
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
    }), encoding="utf-8")
    (output / "selected_passive_joint_report.json").symlink_to(real)

    assert MODULE.FixedReadOnlyBackend._selected_passive_report_valid(
        output,
    ) is False


def test_the_stop_predicate_never_keys_on_worker_liveness_alone():
    """Source-level guard on the fail-closed shape of the abandon predicate.

    The supervision loop is ``while perception_worker_running()``, so its only
    exit is the worker going away.  If that is also sufficient for stop(), then
    the probe in flight at that moment always satisfies stop() and the final
    passive window of every attempt is abandoned by construction -- taking its
    zero-TX verdict with it.  Abandoning must require a positive proof of
    redundancy established BEFORE the window was opened.
    """

    source = Path(MODULE.__file__).read_text(encoding="utf-8")
    predicate = source.split("def passive_window_unneeded()")[1]
    predicate = predicate.split("return")[1].split("\n\n")[0]
    assert "window_redundant_at_open" in predicate
    # Never "the worker is gone, therefore throw the observation away".
    assert "not perception_worker_running() or" not in predicate


def test_structural_completeness_is_a_distinct_predicate_from_the_zero_tx_gate():
    # Two different questions, two different functions.  Conflating them is
    # what let a completed report of an ACTUAL transmission be deleted and the
    # request reported as a success.
    backend = MODULE.FixedReadOnlyBackend
    assert backend._passive_report_valid is not (
        backend._passive_report_structurally_complete
    )
    source = Path(MODULE.__file__).read_text(encoding="utf-8")
    # The discard path must consult completeness, never the verdict.
    discard = source.split("if abandoned and not ")[1].split(":")[0]
    assert discard == "structurally_complete"


def test_reuse_hit_ratio_survives_object_new_backend_without_init():
    # A backend built without __init__ (as some harnesses do) still records the
    # hit-ratio via getattr defaults rather than raising AttributeError.
    backend = object.__new__(MODULE.FixedReadOnlyBackend)
    fields = backend._grounding_observability(reused=True)
    assert fields["vlm_reuse_hit_ratio"] == 1.0
    assert fields["grounding_requests_scored"] == 1
