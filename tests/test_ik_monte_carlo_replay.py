from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/offline/ik_monte_carlo_replay.py"
SPEC = importlib.util.spec_from_file_location("ik_monte_carlo_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bounded_planning_seed_never_exceeds_urdf_limits() -> None:
    lower = np.array([-1.0, 0.0, -2.0])
    upper = np.array([1.0, 2.0, 0.0])
    start = np.array([0.99, 0.01, -0.01])

    perturbed, applied = MODULE._bounded_planning_seed(
        start,
        rng=np.random.default_rng(7),
        sigma_rad=10.0,
        lower=lower,
        upper=upper,
    )

    assert np.all(perturbed >= lower)
    assert np.all(perturbed <= upper)
    np.testing.assert_allclose(applied, perturbed - start)


def test_planner_command_keeps_measured_and_planning_joints_distinct(
    tmp_path: Path,
) -> None:
    measured = np.arange(6, dtype=float)
    planning = measured + 0.25

    command = MODULE._planner_command(
        tmp_path / "perception",
        tmp_path / "planning",
        measured,
        planning,
        image="local:test",
    )

    assert f"--joints={MODULE._csv(measured)}" in command
    assert f"--planning-joints={MODULE._csv(planning)}" in command


def test_artifact_perturbation_is_deterministic_and_preserves_candidate_set(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    grasps = np.repeat(np.eye(4)[None, :, :], 4, axis=0)
    grasps[:, 0, 3] = np.arange(4)
    np.savez(
        source / "grasp_candidates.npz",
        grasps=grasps,
        scores=np.arange(4, dtype=float),
        widths=np.arange(4, dtype=float) + 0.01,
        centroid=np.array([0.1, 0.2, 0.3]),
    )
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.save(source / "target_points.npy", points)

    first_offset, first_order = MODULE._perturb_artifacts(
        source,
        tmp_path / "first",
        rng=np.random.default_rng(19),
        target_sigma_m=0.003,
    )
    second_offset, second_order = MODULE._perturb_artifacts(
        source,
        tmp_path / "second",
        rng=np.random.default_rng(19),
        target_sigma_m=0.003,
    )

    np.testing.assert_allclose(first_offset, second_offset)
    assert first_order == second_order
    assert sorted(first_order) == [0, 1, 2, 3]
    np.testing.assert_allclose(
        np.load(tmp_path / "first/target_points.npy"),
        points + first_offset,
    )


def test_distribution_records_expected_quantiles() -> None:
    result = MODULE._distribution([1.0, 2.0, 3.0, 4.0])

    assert result["min"] == 1.0
    assert result["p50"] == 2.5
    assert result["p95"] == 3.8499999999999996
    assert result["max"] == 4.0
