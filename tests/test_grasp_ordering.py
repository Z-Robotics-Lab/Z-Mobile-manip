"""The overhead penalty must price BOTH vertical approach directions.

``lateral_approach_scores`` penalised only ``max(0, -approach.up)``, so an
approach pointing straight UP scored exactly ``overhead_penalty_weight``
(0.4 as deployed, configs/go2w_piper.json:183) above its own mirror image.
Measured consequence on an open cup seen at 70 degrees elevation: the
top-ranked candidate had ``approach.up = +1.000`` with its TCP 2.0 mm above
the bench and its pregrasp 98 mm below it -- reaching the cup from inside the
flight case it stands on.
"""

import numpy as np
import pytest

from z_manip.models.grasp_ordering import lateral_approach_scores


def _approaches(*axes):
    poses = np.repeat(np.eye(4)[None, :, :], len(axes), axis=0)
    for index, axis in enumerate(axes):
        poses[index, :3, 2] = axis
    return poses


def test_reaching_up_from_under_the_support_is_penalised_like_reaching_down():
    poses = _approaches(
        (0.0, 0.0, 1.0),   # from below, up through the bench
        (0.0, 0.0, -1.0),  # overhead, down onto the object
        (0.0, 1.0, 0.0),   # side entry
    )
    ranked, bonuses, penalties = lateral_approach_scores(
        poses,
        np.ones(3),
        lateral_weight=0.5,
        overhead_penalty_weight=0.4,
    )

    assert penalties[0] == pytest.approx(penalties[1]) == pytest.approx(0.4)
    assert penalties[2] == pytest.approx(0.0)
    assert bonuses[0] == pytest.approx(bonuses[1]) == pytest.approx(0.0)
    assert ranked[2] > ranked[0]
    assert ranked[2] > ranked[1]


def test_an_oblique_approach_is_priced_between_side_and_vertical():
    root = np.sqrt(0.5)
    poses = _approaches((0.0, root, root), (0.0, root, -root))
    _ranked, _bonuses, penalties = lateral_approach_scores(
        poses,
        np.ones(2),
        lateral_weight=0.5,
        overhead_penalty_weight=0.4,
    )

    np.testing.assert_allclose(penalties, (0.4 * root, 0.4 * root))
