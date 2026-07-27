"""Bit-equality tests for the Pinocchio IK inner-loop fast paths.

The damped-least-squares loop evaluates several vector norms per iteration and
per line-search step.  ``_euclidean_norm`` replaces ``numpy.linalg.norm`` there
purely to drop dispatch overhead, so the returned double must be identical for
every input the solver can produce -- an off-by-one-ulp norm would move an
acceptance decision at the 6 mm / 0.06 rad gates.
"""

from __future__ import annotations

import numpy as np

from z_manip.kinematics.pinocchio_ik import PinocchioIKSolver, _euclidean_norm


def _vectors() -> list[np.ndarray]:
    generator = np.random.default_rng(20260727)
    samples = [
        np.zeros(3),
        np.asarray([0.006, 0.0, 0.0]),
        np.asarray([1e-18, -1e-18, 1e-18]),
        np.asarray([1e6, -1e6, 1e6]),
    ]
    for size in (3, 6):
        samples.extend(generator.normal(0.0, 1.0, size=size) for _ in range(2000))
        samples.extend(generator.normal(0.0, 1e-8, size=size) for _ in range(500))
    return samples


def test_euclidean_norm_is_bit_identical_to_numpy() -> None:
    for vector in _vectors():
        assert _euclidean_norm(vector) == float(np.linalg.norm(vector))


def test_euclidean_norm_matches_on_non_contiguous_views() -> None:
    generator = np.random.default_rng(11)
    block = generator.normal(0.0, 1.0, size=(64, 12))
    for row in block:
        strided = row[::2]
        assert _euclidean_norm(strided) == float(np.linalg.norm(strided))


def test_seed_deduplication_matches_the_pairwise_allclose_scan() -> None:
    generator = np.random.default_rng(5)
    lower = np.full(6, -2.6)
    upper = np.full(6, 2.6)

    def reference(seeds: list[np.ndarray]) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for seed in seeds:
            bounded = np.minimum(
                np.maximum(np.asarray(seed, dtype=float), lower + 1e-10),
                upper - 1e-10,
            )
            if not any(np.allclose(bounded, other, atol=1e-10) for other in unique):
                unique.append(bounded)
        return unique

    for _ in range(300):
        seeds = [generator.uniform(-3.0, 3.0, size=6) for _ in range(8)]
        # Duplicates, clipped duplicates and near-tolerance neighbours all have
        # to be classified the same way as the pairwise scan.
        seeds.append(seeds[0].copy())
        seeds.append(seeds[1] + 1e-12)
        seeds.append(seeds[2] + 1e-4)
        seeds.append(np.full(6, 9.0))
        seeds.append(np.full(6, 11.0))
        expected = reference(seeds)
        actual = PinocchioIKSolver._deduplicated_seeds(seeds, lower, upper)
        assert len(actual) == len(expected)
        for produced, wanted in zip(actual, expected):
            assert np.array_equal(produced, wanted)


def test_weighted_cost_is_bit_identical_to_the_norm_form() -> None:
    generator = np.random.default_rng(3)
    for _ in range(2000):
        twist = generator.normal(0.0, 1.0, size=6)
        weights = np.abs(generator.normal(1.0, 20.0, size=6))
        expected = float(np.linalg.norm(weights * np.asarray(twist, dtype=float)))
        assert PinocchioIKSolver._weighted_cost(twist, weights) == expected
