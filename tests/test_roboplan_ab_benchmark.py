"""The A/B benchmark is only useful if it cannot quietly report a fake win.

Every test here targets one way this script could print a number that is not the
number it claims:

* a row where one side measured nothing still naming a winner,
* a throughput that is really the sample size,
* a peak-acceleration estimator that spaces its second difference wrongly and
  under-reports both sides by the same plausible-looking factor,
* a "planning" row run over two different worlds because the parked scene cloud
  drifted into the arm's reach,
* the benchmark's own ``enabled=True`` leaking into the deployed stack config.

Almost all of it runs without roboplan: the arithmetic is pure numpy and the
degradation contract is exactly what a host without the library exercises.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from z_manip.planning.time_parameterization import TimedJointTrajectory

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "roboplan_ab_benchmark.py"
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
CONFIG = ROOT / "configs/go2w_piper.json"

SPEC = importlib.util.spec_from_file_location("roboplan_ab_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
# Registered before execution: ``@dataclass`` resolves string annotations
# through ``sys.modules[cls.__module__]`` and raises on a module that is not
# there yet.  The other scripts/offline tests get away without this only because
# their scripts carry no dataclass.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")


# --------------------------------------------------------------- pure arithmetic


def test_a_side_that_measured_nothing_never_declares_a_winner() -> None:
    """A NaN column must read "-", not hand the win to the other side.

    ``_stats`` returns NaN whenever a backend produced no samples -- every path
    rejected, every solve failed.  Every NaN comparison is False, so a naive
    ``new > old`` silently declares the side that measured NOTHING the loser,
    which is the most flattering possible way to report a total failure.
    """

    assert MODULE.Metric("x", float("nan"), 1.0, better="low").winner() == "-"
    assert MODULE.Metric("x", 1.0, float("nan"), better="high").winner() == "-"
    assert MODULE.Metric("x", 1.0, float("inf"), better="low").winner() == "-"


@pytest.mark.parametrize(
    ("old", "new", "better", "expected"),
    [
        (10.0, 5.0, "low", "new"),
        (10.0, 5.0, "high", "old"),
        (5.0, 5.0, "low", "tie"),
        (5.0, 5.0, None, "-"),
    ],
)
def test_winner_follows_the_stated_direction(old, new, better, expected) -> None:
    assert MODULE.Metric("x", old, new, better=better).winner() == expected


def test_throughput_reports_running_out_of_candidates_not_budget() -> None:
    """A backend that clears the whole list has NOT had its throughput measured.

    Returning only the count makes "107 candidates in 0.6 s" indistinguishable
    from "the benchmark only had 107 candidates", and the second is a statement
    about ``--ik-targets``, not about the solver.
    """

    resolved, usable, spent = MODULE._throughput([0.01] * 4, [True] * 4, 0.6)
    assert (resolved, usable, spent) == (4, 4, False)

    resolved, usable, spent = MODULE._throughput([0.25] * 4, [True] * 4, 0.6)
    assert (resolved, usable, spent) == (2, 2, True)


def test_throughput_counts_only_candidates_that_finished_inside_the_budget() -> None:
    # 0.5 + 0.2 overruns 0.6: the second candidate is not resolved inside it.
    resolved, usable, spent = MODULE._throughput([0.5, 0.2, 0.01], [False, True, True], 0.6)
    assert (resolved, usable, spent) == (1, 0, True)


def test_peak_fractions_space_acceleration_on_the_sample_midpoint() -> None:
    """Non-uniform sample spacing must use the midpoint interval, not dt[1:].

    TOPPRA's emitted times are not uniformly spaced.  Dividing the velocity
    difference by the *following* interval instead of the midpoint between the
    two is a plausible-looking off-by-one that scales the reported peak
    acceleration by an arbitrary factor -- here 1.6x -- and it would do so for
    both retimers, so no cross-column check would catch it.
    """

    trajectory = TimedJointTrajectory(
        positions=np.asarray([[0.0], [0.1], [0.1], [0.0]]),
        times_s=np.asarray([0.0, 0.1, 0.5, 0.6]),
    )
    velocity_cap = np.asarray([2.0])
    acceleration_cap = np.asarray([1.0])

    peak_v, peak_a = MODULE._peak_fractions(trajectory, velocity_cap, acceleration_cap)

    # velocities: [1.0, 0.0, -1.0]; midpoint spacings: [0.25, 0.25]
    assert peak_v == pytest.approx(0.5)
    assert peak_a == pytest.approx(4.0)


class _FakePlanner:
    """A ``segment_valid`` predicate driven by an explicit verdict list."""

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def segment_valid(self, first, second, *, control=None) -> bool:
        verdict = self._verdicts[self.calls]
        self.calls += 1
        return verdict


def test_a_query_only_one_backend_must_search_is_dropped() -> None:
    """The rigged-row guard, and the one this benchmark nearly shipped.

    The deployment clearance makes the numpy predicate the stricter of the two
    on this model, so filtering queries on it alone hands roboplan a pile of
    pairs its own straight-line early return answers in microseconds.  The
    latency column would then report a three-order-of-magnitude "search win"
    over queries roboplan never searched.
    """

    states = np.arange(8.0).reshape(8, 1)
    # pair 0: blocked for both  -> kept
    # pair 1: blocked for old only -> dropped, counted
    # pair 2: blocked for new only -> dropped, counted
    # pair 3: free for both        -> dropped, counted
    old = _FakePlanner([False, False, True, True])
    new = _FakePlanner([False, True, False, True])

    pairs, counts = MODULE._searchable_pairs(states, old, new, queries=4)

    assert len(pairs) == 1
    assert np.array_equal(pairs[0][0], states[0])
    assert counts == {"examined": 4, "both_free": 1, "old_only": 1, "new_only": 1}


def test_the_pair_filter_stops_once_it_has_enough_queries() -> None:
    states = np.arange(12.0).reshape(12, 1)
    old = _FakePlanner([False] * 6)
    new = _FakePlanner([False] * 6)

    pairs, counts = MODULE._searchable_pairs(states, old, new, queries=2)

    assert len(pairs) == 2
    assert counts["examined"] == 2
    assert old.calls == 2  # no wasted collision checks past the quota


# ------------------------------------------------------------- benchmark honesty


def test_a_reachable_scene_cloud_fails_the_benchmark_closed(monkeypatch) -> None:
    """The planning row is an A/B only while the parked cloud blocks nothing.

    Both planners are compared over "one world".  If the cloud drifts into the
    arm's reach the numpy planner alone is solving a harder problem, and the row
    reports that as a roboplan search win.  ``build_fixtures`` therefore proves
    the cloud inert instead of trusting the constant.
    """

    with pytest.raises(RuntimeError, match="inside the arm's reach"):
        _fixtures(monkeypatch, plane_z_m=0.15)


def test_the_parked_cloud_is_inert_where_the_benchmark_puts_it(monkeypatch) -> None:
    """The inverse of the test above: the shipped constant must actually pass."""

    assert _fixtures(monkeypatch).chain.dof == 6


def test_the_benchmark_never_enables_roboplan_in_the_deployed_config(monkeypatch) -> None:
    """``enabled=True`` lives in this script's own object and nowhere else.

    This checkout is bind-mounted read-only into the running robot.  A benchmark
    that opted the deployment in -- by mutating the parsed section, or by reusing
    it -- would switch a live planner.
    """

    fixtures = _fixtures(monkeypatch)
    assert fixtures.roboplan_config.enabled is True
    assert fixtures.stack.roboplan is not None
    assert fixtures.stack.roboplan.runtime.enabled is False
    assert fixtures.roboplan_config is not fixtures.stack.roboplan.runtime
    # And the file on disk is untouched by having run any of this.
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["roboplan"]["enabled"] is False


def test_every_row_either_reports_metrics_or_names_why_it_did_not(
    tmp_path, monkeypatch,
) -> None:
    """The degradation contract, in whichever environment this runs.

    Off the 4090 this is the whole benchmark: every row must skip with a stated
    reason rather than crash.  In the container the same assertion pins that a
    row never emits both an empty metric list and no explanation, which is how a
    silently-broken row would look identical to a fast one.
    """

    output = tmp_path / "report.json"
    monkeypatch.setenv("Z_MANIP_ROBOT_URDF", str(URDF))
    monkeypatch.setattr(sys, "argv", [
        "roboplan_ab_benchmark.py",
        "--config", str(CONFIG),
        "--output", str(output),
        "--ik-targets", "2",
        "--plan-queries", "1",
        "--plan-pair-pool", "2",
        "--timing-paths", "1",
        "--timing-vertices", "2",
        "--collision-states", "8",
    ])

    assert MODULE.main() == 0

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == MODULE.SCHEMA
    assert document["transport_opened"] is False
    assert document["motion_commands_sent"] == 0
    assert len(document["comparisons"]) == 5
    for comparison in document["comparisons"]:
        if "skipped" in comparison:
            assert comparison["skipped"].strip(), comparison["name"]
            assert "metrics" not in comparison, comparison["name"]
        else:
            assert comparison["metrics"], comparison["name"]


def test_render_prints_the_reason_instead_of_an_empty_table() -> None:
    text = MODULE.render(
        [MODULE.Comparison("row", "old", "new", ("context",), skipped="no roboplan")],
        {"urdf": "u", "config": "c", "seed": 1, "roboplan_version": "absent"},
    )
    assert "SKIPPED: no roboplan" in text
    assert "winner" not in text


def _fixtures(monkeypatch, *, plane_z_m: float | None = None):
    """Build the shared fixtures, optionally moving the parked scene plane."""

    monkeypatch.setenv("Z_MANIP_ROBOT_URDF", str(URDF))
    if plane_z_m is not None:
        monkeypatch.setattr(MODULE, "_SCENE_PLANE_Z_M", plane_z_m)
    return MODULE.build_fixtures(CONFIG, 20260803, sanity_states=64)
