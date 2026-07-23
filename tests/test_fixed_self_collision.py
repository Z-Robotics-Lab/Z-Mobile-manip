"""Offline regression tests for the final PiPER fixed-fixture guard."""

from __future__ import annotations

from pathlib import Path

from z_manip.fixed_self_collision import FixedSelfCollisionGuard


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MODEL = ROOT / "configs/piper_collision_capsules.json"


def guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=MODEL)


def test_recorded_mid360_contact_is_rejected_but_prior_pose_is_clear() -> None:
    checker = guard()
    prior = [-0.049, 0.188, -0.009, -0.045, 0.331, 0.0]
    contact = [-0.176, 0.775, 0.003, -0.196, 0.367, -0.096]

    assert checker.check_state(prior).valid
    witness = checker.check_state(contact)
    assert not witness.valid
    assert "mid360" in witness.witness.pair
    assert witness.minimum_margin_m < -0.04


def test_continuous_step_blocks_entry_before_recorded_contact() -> None:
    checker = guard()
    prior = [-0.049, 0.188, -0.009, -0.045, 0.331, 0.0]
    first_penetration = [-0.139, 0.313, -0.009, -0.121, 0.358, 0.0]

    decision = checker.check_step(prior, first_penetration)
    assert not decision.allowed
    assert not decision.escaping
    assert "enters" in decision.reason
    assert "mid360" in decision.witness.pair


def test_guard_permits_only_monotonic_escape_from_conservative_envelope() -> None:
    checker = guard()
    just_inside = [-0.139, 0.313, -0.009, -0.121, 0.358, 0.0]
    clear = [-0.049, 0.188, -0.009, -0.045, 0.331, 0.0]

    escape = checker.check_step(just_inside, clear)
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
    # A nominal forward viewing pose is clear; the nearest fixture is the front
    # Mid-360, well outside its envelope.
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

