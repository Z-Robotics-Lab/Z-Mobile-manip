"""Offline regression tests for the final PiPER fixed-fixture guard."""

from __future__ import annotations

from pathlib import Path

from z_manip.fixed_self_collision import FixedSelfCollisionGuard, _segment_distance


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MODEL = ROOT / "configs/piper_collision_capsules.json"

# Recorded poses from the live Mid-360 strike, all on one joint-space approach
# ray.  2026-07-28: re-recorded against the corrected head geometry.  The pose
# this file used to call `prior` (RECORDED_STRIKE, the ray at t=1.0) was never
# clear: the superseded 32.5 mm capsule at the wrong pose reported +25.5 mm of
# margin there while the gripper was 5.4 mm inside the sensor's metal.  The
# last waypoint on the same recorded approach that is genuinely outside the
# corrected envelope is the ray at t=0.35 -- CLEAR_APPROACH.  The
# hardware-referenced proof lives in tests/test_lidar_keepout.py.
CLEAR_APPROACH = [-0.01715, 0.0658, -0.00315, -0.01575, 0.11585, 0.0]
RECORDED_STRIKE = [-0.049, 0.188, -0.009, -0.045, 0.331, 0.0]
DEEPER_STRIKE = [-0.139, 0.313, -0.009, -0.121, 0.358, 0.0]
RECORDED_CONTACT = [-0.176, 0.775, 0.003, -0.196, 0.367, -0.096]


def guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=MODEL)


def pair_margin(checker: FixedSelfCollisionGuard, joints, name: str) -> float:
    """Smallest margin over the configured pairs that involve ``name``.

    ``check_state`` reports only the global argmin, which at a deep contact
    pose is a neighbouring head fixture rather than the Mid-360 itself.
    """

    world = checker._world_capsules(joints)
    margins = [
        _segment_distance(*world[first][:2], *world[second][:2])
        - (world[first][2] + world[second][2] + checker.clearance_m)
        for first, second in checker.pairs
        if name in (first, second)
    ]
    assert margins
    return min(margins)


def test_recorded_mid360_contact_is_rejected_but_prior_pose_is_clear() -> None:
    checker = guard()

    assert checker.check_state(CLEAR_APPROACH).valid
    witness = checker.check_state(RECORDED_CONTACT)
    assert not witness.valid
    assert witness.minimum_margin_m < -0.04
    # The global argmin here is the platform head the sensor is bolted to, so
    # assert the Mid-360 keep-out itself is breached instead of depending on
    # which neighbouring fixture happens to win the minimum.
    assert pair_margin(checker, RECORDED_CONTACT, "mid360") < 0.0
    assert pair_margin(checker, RECORDED_CONTACT, "mid360_bracket") < 0.0

    # The pose the superseded model certified as clear is now rejected: this is
    # the fail-open that dented the sensor.
    strike = checker.check_state(RECORDED_STRIKE)
    assert not strike.valid
    assert "mid360" in strike.witness.pair


def test_continuous_step_blocks_entry_before_recorded_contact() -> None:
    checker = guard()

    decision = checker.check_step(CLEAR_APPROACH, DEEPER_STRIKE)
    assert not decision.allowed
    assert not decision.escaping
    assert "enters" in decision.reason
    assert "mid360" in decision.witness.pair


def test_guard_permits_only_monotonic_escape_from_conservative_envelope() -> None:
    checker = guard()
    # Both recorded poses sit inside the corrected envelope.  The guard must not
    # trap the arm there, but must admit only a strictly improving step.
    just_inside = DEEPER_STRIKE
    shallower = RECORDED_STRIKE
    assert not checker.check_state(just_inside).valid
    assert not checker.check_state(shallower).valid

    escape = checker.check_step(just_inside, shallower)
    assert escape.allowed
    assert escape.escaping
    assert escape.target_margin_m > escape.current_margin_m

    hold = checker.check_step(just_inside, just_inside)
    assert not hold.allowed
    assert not hold.escaping


def test_rear_nuc_capsule_and_arm_pairs_are_loaded() -> None:
    checker = guard()
    names = {capsule.name for capsule in checker.capsules}
    assert "nuc" in names
    nuc = next(capsule for capsule in checker.capsules if capsule.name == "nuc")
    # The rear NUC enclosure is a fixed fixture behind the arm base, so it must
    # participate in supplemental self-collision like the mid360/platform head.
    assert nuc.supplemental
    nuc_pairs = {pair for pair in checker.pairs if "nuc" in pair}
    # Mirrors the mid360 fixture: upper_arm + every distal arm link (14 pairs).
    assert len(nuc_pairs) == 14
    assert {"nuc", "upper_arm"} == set(next(p for p in nuc_pairs if "upper_arm" in p))


def test_rearward_shoulder_lean_into_the_nuc_is_rejected() -> None:
    checker = guard()
    # A nominal forward viewing pose is clear, outside every fixture envelope
    # (the binding one is the rear NUC).
    clear = [0.0, 0.9, -1.0, 0.0, 0.6, 0.0]
    assert checker.check_state(clear).valid

    # Leaning the shoulder pitch (J2) rearward past its stop swings the upper
    # arm behind the base into the rear NUC keep-out -- the unguarded motion
    # from the incident.  The new capsule makes this a hard rejection.
    breach = [0.0, -0.6, -2.0, 0.0, -0.4, 0.0]
    witness = checker.check_state(breach)
    assert not witness.valid
    assert "nuc" in witness.witness.pair
    assert witness.minimum_margin_m < -0.05

    prior = [0.0, 0.3, -2.0, 0.0, -0.4, 0.0]
    assert checker.check_state(prior).valid
    decision = checker.check_step(prior, breach)
    assert not decision.allowed
    assert "enters" in decision.reason
    assert "nuc" in decision.witness.pair

