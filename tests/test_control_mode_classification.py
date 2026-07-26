"""One shared rule for classifying an injected callback's control transport.

``work_pose`` and ``standoff`` used to carry byte-identical copies of this
classification (differing only in arity and error text). They now delegate to
``z_manip.concurrency.control.classify_control_mode``; these tests pin the rule
itself and the fact that both call sites agree on it.

``grasp_pipeline._path_validator_control_mode`` deliberately does NOT delegate:
it tries positional binding first and raises on a non-inspectable callable.
That divergence is observable, so it is pinned here too rather than assumed.
"""

import pytest

from z_manip.concurrency.control import classify_control_mode
from z_manip.planning.grasp_pipeline import _path_validator_control_mode
from z_manip.planning.standoff import _evaluator_control_mode as _standoff_mode
from z_manip.planning.work_pose import _evaluator_control_mode as _work_pose_mode


def _opaque(callback):
    """Make ``callback`` un-inspectable the way a C builtin or proxy is."""

    callback.__signature__ = "not a signature"
    return callback


def test_keyword_only_control_is_classified_as_keyword():
    def evaluate(candidate, *, control=None):
        return control

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "keyword"


def test_trailing_positional_control_is_classified_as_positional():
    # Positional transport is the fallback for a callback that takes the
    # control object in a trailing slot it does not name ``control``; a
    # parameter actually named ``control`` binds as a keyword and wins above.
    def evaluate(candidate, budget):
        return budget

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "positional"

    def positional_only(candidate, control, /):
        return control

    assert classify_control_mode(
        positional_only,
        arity=1,
        requirement="unused",
    ) == "positional"


def test_a_parameter_named_control_is_bound_by_keyword_even_when_positional():
    def evaluate(candidate, control):
        return control

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "keyword"


def test_callback_without_a_control_slot_stays_legacy():
    def evaluate(candidate):
        return candidate

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "legacy"


def test_permissive_wrapper_prefers_keyword_transport():
    # An opaque *args/**kwargs wrapper binds every shape. Keyword transport
    # keeps ``control`` out of the wrapped callable's positional slots.
    def evaluate(*args, **kwargs):
        return kwargs

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "keyword"


def test_non_inspectable_callable_keeps_the_legacy_invocation():
    @_opaque
    def evaluate(candidate, *, control=None):
        return control

    assert classify_control_mode(
        evaluate,
        arity=1,
        requirement="unused",
    ) == "legacy"


def test_a_callback_matching_no_supported_shape_raises_the_requirement():
    def evaluate(first, second, third, fourth):
        return first

    with pytest.raises(TypeError, match="must accept a candidate"):
        classify_control_mode(
            evaluate,
            arity=1,
            requirement="evaluator must accept a candidate and optional control",
        )


@pytest.mark.parametrize(
    ("mode", "arity"),
    [(_work_pose_mode, 1), (_standoff_mode, 3)],
)
def test_work_pose_and_standoff_share_one_classification(mode, arity):
    names = tuple(f"observation_{index}" for index in range(arity))
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - build signatures with the module's real arity
        "\n".join(
            (
                f"def keyword({', '.join(names)}, *, control=None): pass",
                f"def positional({', '.join(names)}, budget): pass",
                f"def legacy({', '.join(names)}): pass",
                f"def wrong({', '.join(names)}, a, b, c, d): pass",
            )
        ),
        namespace,
    )

    assert mode(namespace["keyword"]) == "keyword"
    assert mode(namespace["positional"]) == "positional"
    assert mode(namespace["legacy"]) == "legacy"
    assert mode(_opaque(namespace["legacy"])) == "legacy"
    with pytest.raises(TypeError, match="optional control"):
        mode(namespace["wrong"])


def test_work_pose_and_standoff_keep_their_own_requirement_messages():
    def wrong(a, b, c, d, e):
        return a

    with pytest.raises(TypeError, match="work-pose evaluator"):
        _work_pose_mode(wrong)
    with pytest.raises(TypeError, match="standoff evaluator"):
        _standoff_mode(wrong)


def test_path_validator_precedence_deliberately_differs_from_the_shared_rule():
    def permissive(*args, **kwargs):
        return kwargs

    assert _path_validator_control_mode(permissive) == "positional"
    assert classify_control_mode(
        permissive,
        arity=1,
        requirement="unused",
    ) == "keyword"


def test_path_validator_rejects_a_non_inspectable_callable():
    @_opaque
    def validate(path, *, control=None):
        return control

    with pytest.raises(TypeError, match="inspectable signature"):
        _path_validator_control_mode(validate)
    assert classify_control_mode(
        validate,
        arity=1,
        requirement="unused",
    ) == "legacy"
