"""Fail-closed ordering for the pre-lift grasp-verification baseline."""

from __future__ import annotations

from collections.abc import Callable

from z_manip.verification.grasp import VerificationResult, VerificationState


def establish_baseline_before_lift(
    sample_verifier: Callable[[], VerificationResult],
    publish_lift: Callable[[], None],
) -> VerificationResult:
    """Publish lift only after the verifier accepts a pending baseline sample."""
    result = sample_verifier()
    if result.state is not VerificationState.PENDING:
        reason = f': {result.reason}' if result.reason else ''
        raise ValueError(
            f'pre-lift grasp baseline is {result.state.value}{reason}',
        )
    publish_lift()
    return result
