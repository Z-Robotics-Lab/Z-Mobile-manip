"""Tests for the task runtime's pre-lift verification ordering."""

import pytest

from z_manip.verification.grasp import (
    GraspVerificationConfig,
    GraspVerifier,
    VerificationResult,
    VerificationSample,
    VerificationState,
)
from z_manip_task.grasp_verification import establish_baseline_before_lift


def test_pending_baseline_is_sampled_before_lift_publish():
    events = []

    def sample():
        events.append('baseline')
        return VerificationResult(VerificationState.PENDING)

    result = establish_baseline_before_lift(
        sample,
        lambda: events.append('publish_lift'),
    )

    assert result.state is VerificationState.PENDING
    assert events == ['baseline', 'publish_lift']


def test_pre_lift_baseline_allows_post_lift_hold_verification():
    verifier = GraspVerifier(GraspVerificationConfig(
        min_lift_m=0.06,
        hold_time_s=0.25,
    ))
    events = []

    def sample(stamp, ee_z, target_z):
        return verifier.update(VerificationSample(
            stamp_s=stamp,
            gripper_aperture_m=0.03,
            ee_position_base=(0.0, 0.0, ee_z),
            target_centroid_base=(0.0, 0.0, target_z),
            tracker_locked=True,
        ))

    establish_baseline_before_lift(
        lambda: sample(0.0, 0.20, 0.18),
        lambda: events.append('publish_lift'),
    )
    assert sample(0.10, 0.28, 0.26).state is VerificationState.PENDING
    assert sample(0.36, 0.28, 0.26).state is VerificationState.SUCCESS
    assert events == ['publish_lift']


@pytest.mark.parametrize(
    ('result', 'state'),
    (
        (
            VerificationResult(VerificationState.FAILED, 'empty grasp'),
            'failed',
        ),
        (VerificationResult(VerificationState.SUCCESS), 'success'),
    ),
)
def test_nonpending_baseline_never_publishes_lift(result, state):
    published = []

    with pytest.raises(ValueError, match=f'baseline is {state}'):
        establish_baseline_before_lift(
            lambda: result,
            lambda: published.append(True),
        )

    assert not published


def test_missing_baseline_data_never_publishes_lift():
    published = []

    def missing_sample():
        raise ValueError('fresh synchronized target lock is unavailable')

    with pytest.raises(ValueError, match='target lock'):
        establish_baseline_before_lift(
            missing_sample,
            lambda: published.append(True),
        )

    assert not published
