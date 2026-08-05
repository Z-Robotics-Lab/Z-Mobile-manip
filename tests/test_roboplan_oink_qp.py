"""Why there is no ``z_manip/control/roboplan_oink_qp.py``, and the guard that outlives it.

An OInK-backed :class:`WholeBodyQPSolver` was specified as a drop-in third
backend beside ``ScipyReferenceBoxQP`` and ``CasadiBoxQP``.  It is not
buildable against roboplan 0.6.0, for two independent reasons that these tests
pin so the next person reads a failing assertion instead of re-deriving them:

*   **ProxQP is not reachable as a bare QP.**  ``roboplan.optimal_ik`` exposes
    classes only; the sole solve entry point is ``Oink.solveIk``, which takes a
    ``Scene`` and a list of ``Task`` objects, never a Hessian and a gradient.
    No roboplan submodule has a QP surface, ``proxsuite`` ships in the image as
    C++ headers with no Python module, and ``Task`` cannot be subclassed from
    Python -- so a custom Jacobian/residual cannot be supplied either.  The only
    settable-objective task, ``ConfigurationTask``, has an identity Jacobian and
    per-joint weights, so its Hessian contribution is strictly DIAGONAL; the
    whole-body Hessian ``J^T W J + (control + smooth) I`` is dense and couples
    the base columns to the arm columns.  ``Task.weight`` is exposed as a full
    ``n x n`` matrix, which looks like the escape hatch and is not one: it is
    read-only, and measured against the real model it comes back as exactly
    ``diag(sqrt(joint_weights))`` with off-diagonal entries identically 0.0.
    That is the single fact route (b) dies on, so it is pinned by test rather
    than asserted here.

*   **No reachable joint group has ``CONTROL_DOF`` variables.**  ``Oink`` is
    sized by a ``Scene`` joint group: 6 for ``piper_arm``, 24 for the full
    robot.  The control vector is 10, and four of its entries -- base forward,
    base yaw, body roll, body pitch -- correspond to no joint in the model at
    all.  The Go2-W base is four continuous *wheel* joints, and roll/pitch are
    SPORT body-attitude commands.  This one is structural: a roboplan release
    cannot fix it.

Reformulating the caller's problem into OInK tasks was rejected rather than
attempted: it would silently change the optimization the caller posed, which
``whole_body_optimizer.py`` deliberately does not allow (the SciPy backend is
the "deterministic offline reference" every other backend must agree with).

There was also nothing to win on latency.  Measured on 64 real linearized
problems from ``WholeBodyShadowOptimizer`` (32 free + the same 32 with roll and
pitch locked, as ``whole_body_runtime.py:67-68`` locks them), independently
re-measured twice: ``casadi-qrqp`` 0.080 ms mean / 0.082 ms p95,
``scipy-lbfgsb-reference`` 0.162-0.166 ms mean, against ``horizon_dt_s`` of
200 ms.  The QP is already 0.04% of the control period.

There IS a robustness gap, but it does not belong to OInK and a third backend
would not close it: on that same sweep ``casadi-qrqp`` solved 64/64 while
``scipy-lbfgsb-reference`` raised ``RuntimeError`` on 1/64 -- a well-conditioned
(cond(H) = 3.45) box whose optimum sits in a corner, where L-BFGS-B reports
ABNORMAL_TERMINATION_IN_LNSRCH against the ``ftol=1e-12`` in
``whole_body_optimizer.py:288``.  That is a defect in the reference backend
itself, filed separately; it is recorded here because "the deterministic
reference every other backend must agree with" is the premise this whole
workstream was specified against, and it does not hold unconditionally.

The first four tests need no roboplan and guard the live NUC thin deploy: they
are the part of this file worth keeping whatever happens to the verdict above.
The last three need roboplan and pin the surface that produced it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from z_manip.control.whole_body_model import ARM_DOF, CONTROL_DOF

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "z_manip/control"
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"

# What a module under ``z_manip/control/`` may not pull in at import time.
# ``roboplan``/``pinocchio``/``casadi`` are not installed on the NUC at all.
# ``z_manip.planning`` is subtler and is the one an OInK adapter would actually
# reach for: ``install_go2w_reactive_runtime.sh`` copies no part of it, and it
# is where ``roboplan_runtime.scene_handle`` -- the sanctioned Scene accessor --
# lives, so importing it at module scope is an ImportError on the robot by a
# second route.
# # ponytail: prefix list, not "everything the install script omits".  Widen
# only when a real module needs it; a blanket rule would flag
# ``z_manip.kinematics.chain``, which IS copied.
_NUC_FORBIDDEN = ("roboplan", "pinocchio", "casadi", "z_manip.planning")


def _module_scope_imports(source: str, package: str) -> set[str]:
    """Dotted modules imported when this source is *executed*, relative resolved.

    Imports inside a function body are excluded -- that is the sanctioned way
    for ``z_manip/control/**`` to touch an optional heavy dependency.  Imports
    under a module-level ``try``/``if`` are NOT excluded.  A bare one is an
    ImportError on the robot; a ``try``/``except ImportError`` one is worse than
    that, because it does not fail at all -- the module loads with the
    dependency silently set to ``None`` and the first caller finds out.  Both
    are rejected.  Relative imports are resolved against ``package`` because
    ``from ..planning.roboplan_runtime import scene_handle`` is the same defect
    as the absolute spelling.

    # ponytail: this also flags ``if TYPE_CHECKING: import roboplan``, which is
    # in fact runtime-safe.  Conservative on purpose -- no module here needs it
    # today.  If one ever does, exclude that single ``if`` test by name rather
    # than teaching this function to evaluate module-level conditions.
    """

    tree = ast.parse(source)
    deferred = {
        id(node)
        for scope in ast.walk(tree)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(scope)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in deferred:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")[: -(node.level - 1) or None]
                names.add(".".join((*parts, *filter(None, (node.module,)))))
            elif node.module:
                names.add(node.module)
    return names


def _nuc_offenders(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        for banned in _NUC_FORBIDDEN
        if name == banned or name.startswith(f"{banned}.")
    }


def _package_of(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts[:-1])


# (source, what the NUC guard must report).  A scanner stubbed to return
# nothing passes the package scan below on today's clean tree; these cases are
# what make that scan mean something.
_SCANNER_CASES = (
    ("import roboplan", {"roboplan"}),
    ("import roboplan.optimal_ik as oik", {"roboplan.optimal_ik"}),
    ("from roboplan.core import Scene", {"roboplan.core"}),
    ("try:\n    import casadi\nexcept ImportError:\n    casadi = None", {"casadi"}),
    ("if True:\n    import pinocchio", {"pinocchio"}),
    ("def f():\n    import roboplan", set()),
    ("class C:\n    def m(self):\n        import pinocchio", set()),
    (
        "from z_manip.planning.roboplan_runtime import scene_handle",
        {"z_manip.planning.roboplan_runtime"},
    ),
    (
        "from ..planning.roboplan_runtime import scene_handle",
        {"z_manip.planning.roboplan_runtime"},
    ),
    ("from .whole_body_model import CONTROL_DOF", set()),
    ("from z_manip.kinematics.chain import rotation_log", set()),
)


def test_import_scanner_reports_the_patterns_that_would_break_the_nuc():
    """The self-check for the guard below, which is otherwise unfalsifiable.

    Both directions matter: a scanner that misses the deferred-import pattern
    would ban the only sanctioned way to use an optional dependency here, and a
    scanner that misses the relative or ``z_manip.planning`` spellings would
    wave through the exact import an OInK adapter is most likely to write.
    """

    for source, expected in _SCANNER_CASES:
        found = _nuc_offenders(_module_scope_imports(source, "z_manip.control"))
        assert found == expected, f"{source!r} -> {found}, expected {expected}"


def test_control_package_never_imports_at_module_scope_what_the_nuc_lacks():
    """The NUC thin deploy has no roboplan, no pinocchio, and no z_manip.planning.

    Read the install list before widening or narrowing this.  As of today
    ``scripts/runtime/install_go2w_reactive_runtime.sh`` copies exactly TWO
    things out of this package -- ``go2w_posture.py`` and
    ``configs/nuc-control-init.py`` renamed to ``__init__.py`` (a docstring and
    nothing else) -- so on today's install list a new module here would not
    reach the robot at all, and that is why module-scope ``scipy`` in
    ``whole_body_optimizer.py`` is not a finding.

    The guard is deliberately package-wide anyway.  The install list is a shell
    script that grows a filename whenever the NUC needs one more behaviour, and
    the narrower reality above is not something the author of that one-line
    change will re-derive.  Scanning everything costs nothing and is the guard
    for whoever does land an OInK adapter under ``z_manip/control/``.
    """

    modules = sorted(CONTROL.glob("**/*.py"))
    # A moved package would make the scan below vacuously green.
    assert len(modules) >= 11, f"expected the control package, found {modules}"

    offenders = {
        path.name: sorted(
            _nuc_offenders(
                _module_scope_imports(
                    path.read_text(encoding="utf-8"),
                    _package_of(path),
                ),
            ),
        )
        for path in modules
    }
    offenders = {name: found for name, found in offenders.items() if found}
    assert offenders == {}, f"module-scope heavy imports reach the NUC: {offenders}"


def test_control_package_imports_with_the_heavy_dependencies_absent():
    """The invariant itself, not a proxy for it -- and it runs everywhere.

    A subprocess because ``z_manip.control`` is already in ``sys.modules`` by
    the time any test in this file runs (this module imports from it), so an
    in-process ``import z_manip.control`` is a cache hit that asserts nothing.
    A meta-path blocker rather than ``skipif(roboplan installed)`` because the
    machine that HAS roboplan is exactly the machine where an accidental
    dependency would go unnoticed.

    The ``sys.modules`` half stands on its own: it holds whether or not the
    blocker fires, so a blocker that silently went inert cannot make this pass
    while a real eager import is present.
    """

    probe = """
import sys

BLOCKED = ("roboplan", "pinocchio", "casadi")


class _Absent:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not installed on the NUC")
        return None


sys.meta_path.insert(0, _Absent())
import z_manip.control  # noqa: F401
loaded = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED)
assert not loaded, f"importing z_manip.control loaded {loaded}"
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"z_manip.control is not NUC-importable:\n{result.stderr}"


def test_control_package_exports_nothing_defined_outside_it():
    """Structural, because a naming convention is not a guard.

    An adapter that imports roboplan lazily still imports clean above; only a
    re-export from ``__init__`` reveals that ``z_manip.control`` has grown a
    dependency the robot cannot satisfy.  Checked by defining module rather than
    by looking for "oink"/"roboplan" in the name, so a re-exported
    ``roboplan.core.Scene`` is caught whatever it is called.
    """

    import z_manip.control as control

    foreign = {}
    for name in control.__all__:
        value = getattr(control, name)
        if isinstance(value, ModuleType):
            # A module object carries ``__name__``, never ``__module__``, so it
            # took the getattr default below and passed.  Mutation-tested: a
            # re-exported foreign MODULE was a silent green before this branch,
            # while a re-exported foreign CLASS was already caught.
            origin = value.__name__
        else:
            # Plain data (PHASE_POLICY is a dict) has no ``__module__``;
            # anything that does -- class, function, or instance via its type
            # -- must be ours.
            origin = getattr(value, "__module__", "z_manip.control")
        if not origin.startswith("z_manip.control"):
            foreign[name] = origin
    assert foreign == {}, f"z_manip.control re-exports foreign symbols: {foreign}"


def test_roboplan_still_exposes_no_bare_qp():
    """If this fails, roboplan changed and the OInK box-QP backend is worth revisiting.

    Pinned as an exact surface rather than a spot check: a *new* method on
    ``Oink`` is precisely the event that would reopen this workstream, and it
    would otherwise go unnoticed on a version bump.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import importlib.util
    import pkgutil

    import roboplan
    import roboplan.optimal_ik as oik

    assert {n for n in dir(oik.Oink) if not n.startswith("_")} == {
        "enforceBarriers", "num_variables", "q_indices", "settings", "solveIk",
        "v_indices",
    }, (
        "Oink's surface changed.  Do NOT widen this set to make the suite green: "
        "a new method is the trigger to re-read this module's docstring and "
        "re-check whether a bare (H, g, lower, upper) QP is now reachable."
    )
    # No roboplan submodule offers a free function taking (H, g, lower, upper),
    # and none has a QP surface at all -- checked across the package, because
    # the next release could put one anywhere.
    for info in pkgutil.iter_modules(roboplan.__path__):
        module = importlib.import_module(f"roboplan.{info.name}")
        assert not [
            name for name in dir(module)
            if any(key in name.lower() for key in ("qp", "prox", "quadratic"))
        ], f"roboplan.{info.name} grew a QP surface"
    assert not [
        name for name, value in vars(oik).items()
        if not name.startswith("_") and callable(value) and not isinstance(value, type)
    ]
    # ProxQP is vendored as C++ headers only; not even a bypass is available.
    assert importlib.util.find_spec("proxsuite") is None

    # The objective cannot be supplied: Task has no Python constructor, and the
    # only concrete tasks are a diagonal one and a frame one whose Jacobian
    # comes from the model.
    with pytest.raises(TypeError):
        type("CustomTask", (oik.Task,), {})()
    concrete = {
        name for name, value in vars(oik).items()
        if isinstance(value, type) and issubclass(value, oik.Task) and value is not oik.Task
    }
    assert concrete == {"ConfigurationTask", "FrameTask"}


def _handle():
    """The real-model Scene shared by the two tests below.

    ``roboplan_runtime.scene_handle`` caches per config, so the ~0.65 s build
    is paid once however many tests here call this.
    """

    from z_manip.planning.roboplan_runtime import RoboplanConfig, scene_handle

    class _RobotConfig:
        urdf_path = URDF
        base_link = "piper_base_link"
        tip_link = "piper_gripper_base"
        acceleration_limits = (2.0,) * ARM_DOF

    return scene_handle(RoboplanConfig(enabled=True), _RobotConfig())


@pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")
def test_the_only_settable_objective_is_still_strictly_diagonal():
    """Route (b) -- reformulate the caller's QP as OInK tasks -- dies here.

    ``Task.weight`` is an ``n x n`` matrix and reads like the place a dense
    Hessian could go.  It is not: it is read-only, and the one task whose
    objective the caller supplies fills it from a 1-D ``joint_weights`` as
    ``diag(sqrt(joint_weights))``.  A sum of diagonals stays diagonal, so no
    set of ``ConfigurationTask`` s reproduces the whole-body
    ``J^T W J + (control + smooth) I``, which couples base columns to arm
    columns.  ``FrameTask`` cannot help either -- its Jacobian comes from the
    model and only two scalar costs are settable.

    Pinned because it is the load-bearing half of the verdict that a version
    bump could quietly reverse.  If ``weight`` grows a setter, or the
    constructor starts accepting a matrix, this fails -- and that is the
    trigger to re-read the module docstring, not to relax the assertion.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import roboplan.optimal_ik as oik

    handle = _handle()
    with handle.exclusive() as scene:
        oink = oik.Oink(scene, handle.group)
        size = oink.num_variables
        # Distinct per-joint weights: an implementation that ignored them and
        # handed back the identity would survive a bare "is it diagonal" check.
        joint_weights = np.arange(1.0, size + 1.0)
        task = oik.ConfigurationTask(
            oink,
            handle.extract(handle.seed),
            joint_weights,
        )
        weight = np.asarray(task.weight)

        assert weight.shape == (size, size)
        assert np.array_equal(weight, np.diag(np.diag(weight))), (
            f"ConfigurationTask.weight grew off-diagonal terms:\n{weight}"
        )
        assert np.allclose(np.diag(weight) ** 2, joint_weights)

        with pytest.raises(AttributeError):
            task.weight = np.eye(size) + 0.3
        with pytest.raises(TypeError):
            oik.ConfigurationTask(oink, handle.extract(handle.seed), np.eye(size) + 0.3)


@pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")
def test_no_oink_instance_can_hold_the_whole_body_control_vector():
    """The structural half of the finding, measured against the real model.

    ``Oink`` is sized by a Scene joint group.  ``CONTROL_DOF`` falls strictly
    between the two reachable sizes and no group can ever land on it: base
    forward/yaw and body roll/pitch are SPORT commands with no joint behind
    them, while the joints that do move the base are four continuous wheels.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import roboplan.optimal_ik as oik

    handle = _handle()
    with handle.exclusive() as scene:
        sizes = {
            "piper_arm": oik.Oink(scene, handle.group).num_variables,
            "full robot": oik.Oink(scene).num_variables,
        }
    assert sizes["piper_arm"] == ARM_DOF
    assert sizes["piper_arm"] < CONTROL_DOF < sizes["full robot"], (
        f"a joint group now brackets or matches CONTROL_DOF={CONTROL_DOF}: {sizes}"
    )
