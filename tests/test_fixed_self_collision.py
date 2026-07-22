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

