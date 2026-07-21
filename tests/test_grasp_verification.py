import numpy as np

from z_manip.verification.grasp import (
    GraspVerificationConfig,
    GraspVerifier,
    VerificationSample,
    VerificationState,
)


def _sample(stamp, z, target_z, aperture=0.04, locked=True):
    return VerificationSample(
        stamp_s=stamp,
        gripper_aperture_m=aperture,
        ee_position_base=(0.4, 0.0, z),
        target_centroid_base=None if target_z is None else (0.4, 0.0, target_z),
        tracker_locked=locked,
    )


def test_grasp_verification_requires_aperture_lift_and_visual_following():
    verifier = GraspVerifier(GraspVerificationConfig(
        min_lift_m=0.06,
        hold_time_s=0.15,
        track_loss_timeout_s=0.3,
    ))

    assert verifier.update(_sample(0.0, 0.20, 0.18)).state == VerificationState.PENDING
    assert verifier.update(_sample(0.1, 0.24, 0.22)).state == VerificationState.PENDING
    lifted = verifier.update(_sample(0.2, 0.28, 0.26))
    assert lifted.state == VerificationState.PENDING
    success = verifier.update(_sample(0.36, 0.28, 0.26))

    assert success.state == VerificationState.SUCCESS
    assert success.lift_m >= 0.06
    assert success.relative_target_drift_m < 0.01


def test_fully_closed_gripper_fails_as_empty_grasp():
    verifier = GraspVerifier()
    result = verifier.update(_sample(0.0, 0.2, 0.18, aperture=0.001))
    assert result.state == VerificationState.FAILED
    assert "empty" in result.reason


def test_target_loss_and_slippage_fail_closed():
    verifier = GraspVerifier(GraspVerificationConfig(track_loss_timeout_s=0.2))
    verifier.update(_sample(0.0, 0.2, 0.18))
    assert verifier.update(_sample(0.1, 0.24, None, locked=False)).state == VerificationState.PENDING
    lost = verifier.update(_sample(0.31, 0.26, None, locked=False))
    assert lost.state == VerificationState.FAILED
    assert "tracking" in lost.reason

    verifier.reset()
    verifier.update(_sample(1.0, 0.2, 0.18))
    slipped = verifier.update(_sample(1.2, 0.28, 0.18))
    assert slipped.state == VerificationState.FAILED
    assert "drift" in slipped.reason


def test_verifier_rejects_time_reversal_and_nonfinite_measurements():
    verifier = GraspVerifier()
    verifier.update(_sample(2.0, 0.2, 0.18))
    assert verifier.update(_sample(1.9, 0.2, 0.18)).state == VerificationState.FAILED

    verifier.reset()
    bad = VerificationSample(0.0, np.nan, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), True)
    assert verifier.update(bad).state == VerificationState.FAILED
