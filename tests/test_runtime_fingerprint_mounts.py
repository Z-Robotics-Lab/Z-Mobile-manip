"""R9 REPAIR -- the launcher and the servo must hash the SAME FILES.

THE DEFECT THIS FILE EXISTS FOR, MEASURED ON THIS MACHINE.
``z_manip_runtime_fingerprint.runtime_inputs()`` returns 99 paths against the
live checkout.  Exactly one of them is outside ``$STACK_ROOT``::

    /home/yusenzlabpc/Z-Robotics-Lab/go2W_Sim/assets/urdf/go2w_sensored.urdf

``go2w_depth_servo.sh`` measured the fingerprint on the HOST and exported it as
``Z_MANIP_RUNTIME_FINGERPRINT``; ``go2w_depth_servo.py`` built
``RuntimeFingerprintWatch(launch=<that value>)`` and re-measured ``live`` INSIDE
the container.  The container's mount list was ``$STACK_ROOT:$STACK_ROOT:ro``
plus that URDF at the UNRELATED path ``/robot/go2w_sensored.urdf`` -- verified
read-only against the running runtime image, which has no ``go2W_Sim`` at all
and whose Dockerfile contains no ``COPY``.  ``runtime_inputs``'s
``if path.is_file()`` filter dropped the URDF silently, so ``live`` was a digest
over 98 files and ``runtime_fingerprint_state`` asserted ``mutated`` on every
single servo launch of a completely frozen tree::

    launcher (host, 99 files) : 834e35138ae0...
    servo    (container, 98)  : 87e820e6b6c2...
    EQUAL? False

Consequence: ``depth-servo.json.runtime_fingerprint.mutated: true`` forever,
``runtime_fingerprint_mutated: true`` on EVERY trace row -- the primary forensic
record -- and ``RUNTIME_TREE_MUTATED_DURING_RUN`` standing permanently on the
operator's only interface, which makes the one real mid-run edit R9 exists to
catch indistinguishable from background.

WHY THE ORIGINAL SUITE STAYED GREEN (ground rule 5 exactly).  Every servo
fingerprint test fed ``launch`` and ``live`` from the SAME synthetic probe
(``probe=lambda: ("a" * 64, None)``), and the launcher test only asserted that
the measurement precedes ``docker run``.  Nothing crossed the container
boundary.  The tests below do, without docker: the digest's per-file name key is
``path.relative_to(WORKSPACE_ROOT)``, so it depends only on the tree SHAPE, and
a synthetic host tree and a synthetic container tree built from the launcher's
own mount list can be compared byte-for-byte on this host.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_MODULE = ROOT / "scripts" / "runtime" / "z_manip_runtime_fingerprint.py"
SERVO_LAUNCHER = ROOT / "scripts" / "runtime" / "go2w_depth_servo.sh"

#: The launcher's loop variable.  Named here so a rename in the shell script
#: makes the mount invisible to ``_self_mounted_sources`` below and FAILS these
#: tests rather than silently reinstating the defect.
LAUNCHER_MANIFEST_VAR = "fingerprint_input"

#: Fixed sibling scripts ``runtime_inputs`` names explicitly.
_FIXED_SIBLINGS = (
    "go2w_interactive_sessions.py",
    "go2w_perception_dry_run.py",
    "go2w_perception_worker.py",
    "piper_planning_dry_run.py",
    "piper_planning_worker.py",
)


def _load_fingerprint_module(stack_root: Path):
    """Import the real module FROM a synthetic tree.

    ``STACK_ROOT``/``WORKSPACE_ROOT`` are derived from ``__file__``, so loading
    the same source out of a temporary tree is what lets a test stand in for
    "the host" and "the container" without either.
    """

    path = stack_root / "scripts" / "runtime" / "z_manip_runtime_fingerprint.py"
    spec = importlib.util.spec_from_file_location(
        f"z_manip_runtime_fingerprint_{abs(hash(str(path)))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_workspace(root: Path, *, with_external_urdf: bool) -> Path:
    """A minimal tree with the same SHAPE ``runtime_inputs`` walks."""

    stack = root / "Z-Mobile-manip"
    runtime = stack / "scripts" / "runtime"
    runtime.mkdir(parents=True)
    shutil.copy(FINGERPRINT_MODULE, runtime / FINGERPRINT_MODULE.name)
    for name in _FIXED_SIBLINGS:
        (runtime / name).write_text(f"# {name}\n", encoding="utf-8")
    (stack / "z_manip" / "verification").mkdir(parents=True)
    (stack / "z_manip" / "__init__.py").write_text("", encoding="utf-8")
    (stack / "z_manip" / "verification" / "passive_capture.py").write_text(
        "PASSIVE = 1\n", encoding="utf-8"
    )
    (stack / "configs").mkdir()
    (stack / "configs" / "go2w_piper.json").write_text("{}\n", encoding="utf-8")
    (stack / "configs" / "rosbag_sensor_qos.yaml").write_text("{}\n", encoding="utf-8")
    if with_external_urdf:
        urdf_dir = root / "go2W_Sim" / "assets" / "urdf"
        urdf_dir.mkdir(parents=True)
        (urdf_dir / "go2w_sensored.urdf").write_text("<robot/>\n", encoding="utf-8")
    return stack


#: The array the launcher accumulates self-mounts into.  Building it and then
#: not expanding it into ``docker run`` would look exactly like a fix, so the
#: expansion is asserted separately.
LAUNCHER_MOUNT_ARRAY = "FINGERPRINT_INPUT_MOUNTS"


def _docker_run_block(text: str) -> str:
    start = text.index("docker run --rm")
    return text[start:]


def _self_mounted_sources(*, stack_root: Path, external: list[Path]) -> list[Path]:
    """Host paths the launcher mounts AT THEIR OWN ABSOLUTE PATH.

    Only ``-v "$X:$X:mode"`` counts.  ``-v "$WHOLE_BODY_URDF:/robot/...:ro"``
    deliberately does not: the in-container ``runtime_inputs`` looks the file up
    by its host path, and the digest keys each file by that path, so a remapped
    copy satisfies neither.  That distinction IS the shipped defect.

    Scanned over the WHOLE script, not just the ``docker run`` block, because
    the per-input mounts are accumulated into an array by a loop above it.  The
    caller must therefore also prove that array reaches ``docker run``.
    """

    text = SERVO_LAUNCHER.read_text(encoding="utf-8")
    expansions = {
        "$STACK_ROOT": [stack_root],
        f"${LAUNCHER_MANIFEST_VAR}": list(external),
    }
    sources: list[Path] = []
    for spec in re.findall(r'-v "([^"]+)"', text):
        parts = spec.split(":")
        if len(parts) != 3:
            continue
        source, destination, _mode = parts
        if source != destination:
            continue
        sources.extend(expansions.get(source, []))
    return sources


def _external_inputs(module, stack: Path) -> list[Path]:
    """Hashed inputs outside the checkout, derived WITHOUT the repair's helper.

    Computed here rather than through ``module.external_runtime_inputs`` on
    purpose: the two tests below must fail against the PRE-REPAIR module on the
    defect itself (two digests that cannot match), not on an AttributeError for
    a function the repair introduced.  ``external_runtime_inputs`` is pinned
    against this same derivation by its own test.
    """

    return [path for path in module.runtime_inputs() if not path.is_relative_to(stack)]


@pytest.fixture()
def synthetic_host(tmp_path):
    workspace = tmp_path / "host"
    workspace.mkdir()
    stack = _build_workspace(workspace, with_external_urdf=True)
    return workspace, stack, _load_fingerprint_module(stack)


def test_the_hashed_set_reaches_outside_the_checkout_at_all(synthetic_host):
    """Guard the premise: if nothing were external, the rest would be vacuous."""

    _workspace, stack, module = synthetic_host
    external = _external_inputs(module, stack)
    assert external, "runtime_inputs no longer reaches outside STACK_ROOT"
    assert [path.name for path in external] == ["go2w_sensored.urdf"]


def test_external_runtime_inputs_is_exactly_that_derivation(synthetic_host):
    """The helper the launcher shells out to must agree with the definition."""

    _workspace, stack, module = synthetic_host
    assert list(module.external_runtime_inputs()) == _external_inputs(module, stack)


def test_every_hashed_input_is_mounted_at_its_own_absolute_path(synthetic_host):
    """THE CONTRACT THAT WAS BROKEN.

    Fails against the shipped launcher: the URDF is reachable only at
    ``/robot/go2w_sensored.urdf``, so it is unmounted at the path the container
    hashes it by.
    """

    _workspace, stack, module = synthetic_host
    sources = _self_mounted_sources(
        stack_root=stack, external=_external_inputs(module, stack)
    )
    missing = tuple(
        path
        for path in module.runtime_inputs()
        if not any(
            path == source or path.is_relative_to(source) for source in sources
        )
    )
    assert missing == (), (
        "go2w_depth_servo.sh does not bind-mount these hashed inputs at their "
        f"own absolute path, so the servo cannot reproduce the launcher's "
        f"digest: {[str(path) for path in missing]}"
    )


def test_the_container_view_hashes_to_the_launcher_value(tmp_path):
    """The decisive one: build the container's view from the mount list.

    No docker.  ``runtime_fingerprint`` keys each file by its path relative to
    ``WORKSPACE_ROOT``, so two trees with the same shape hash identically and a
    tree MISSING one hashed file cannot.
    """

    host_ws = tmp_path / "host"
    host_ws.mkdir()
    host_stack = _build_workspace(host_ws, with_external_urdf=True)
    host_module = _load_fingerprint_module(host_stack)
    host_fingerprint = host_module.runtime_fingerprint()

    sources = _self_mounted_sources(
        stack_root=host_stack, external=_external_inputs(host_module, host_stack)
    )

    # Reproduce ONLY what the mount list makes visible, at the same relative
    # position under WORKSPACE_ROOT.
    container_ws = tmp_path / "container"
    container_ws.mkdir()
    for source in sources:
        relative = source.relative_to(host_ws)
        destination = container_ws / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            # A repeated `-v` of the same source is what docker does anyway.
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy(source, destination)

    container_stack = container_ws / host_stack.relative_to(host_ws)
    container_module = _load_fingerprint_module(container_stack)
    container_fingerprint = container_module.runtime_fingerprint()

    assert container_fingerprint == host_fingerprint, (
        "the servo re-measures inside the container and can only ever produce "
        "this digest, so runtime_fingerprint_state would report mutated=True on "
        "every launch of a frozen tree "
        f"(launcher hashed {len(host_module.runtime_inputs())} inputs, "
        f"container {len(container_module.runtime_inputs())})"
    )


def test_the_per_input_mounts_actually_reach_docker_run():
    """Accumulating an array and not expanding it would look exactly like a fix."""

    assert LAUNCHER_MOUNT_ARRAY in _docker_run_block(
        SERVO_LAUNCHER.read_text(encoding="utf-8")
    ), "the per-input mounts are accumulated but never passed to docker run"


def test_the_launcher_feeds_its_mounts_from_the_measurement_it_exports():
    """One python3 call, so the digest and the mount list describe one set."""

    text = SERVO_LAUNCHER.read_text(encoding="utf-8")
    assert "--launch-manifest" in text
    assert text.count("z_manip_runtime_fingerprint.py") == 1, (
        "a second measurement re-opens the window where the hashed set and the "
        "mounted set disagree"
    )
    assert f'-v "${LAUNCHER_MANIFEST_VAR}:${LAUNCHER_MANIFEST_VAR}:ro"' in text


def test_the_launch_manifest_leads_with_the_same_digest_as_a_plain_run(
    synthetic_host,
):
    """Line 1 is the digest; the launcher parses by position."""

    _workspace, _stack, module = synthetic_host
    import contextlib
    import io

    plain = io.StringIO()
    with contextlib.redirect_stdout(plain):
        assert module.main([]) == 0
    manifest = io.StringIO()
    with contextlib.redirect_stdout(manifest):
        assert module.main([module.LAUNCH_MANIFEST_FLAG]) == 0

    lines = manifest.getvalue().splitlines()
    assert lines[0] == plain.getvalue().strip() == module.runtime_fingerprint()
    assert lines[1:] == [str(path) for path in module.external_runtime_inputs()]


def test_a_path_that_cannot_be_a_bind_mount_suppresses_the_whole_manifest(
    tmp_path,
):
    """Dropping one mount and still printing the digest is the original bug.

    ``docker run -v SRC:DST:MODE`` is colon-delimited.  A hashed input whose
    path contains a colon cannot be handed across the container boundary, so the
    only honest answers are "no manifest" or a permanently wrong ``mutated``.
    """

    import contextlib
    import io

    workspace = tmp_path / "host"
    workspace.mkdir()
    stack = _build_workspace(workspace, with_external_urdf=True)
    module = _load_fingerprint_module(stack)
    assert module.external_runtime_inputs(), "premise: there IS an external input"

    # A checkout under a directory whose name contains a colon.  Substituted
    # rather than created on disk so the test does not depend on the filesystem
    # tolerating one.
    colon_path = Path(f"{workspace}:evil") / "go2w_sensored.urdf"
    module.external_runtime_inputs = lambda: (colon_path,)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = module.main([module.LAUNCH_MANIFEST_FLAG])
    assert code == 3
    assert out.getvalue() == "", (
        "a digest printed without its mount list is exactly the state that "
        "reported mutated=True on every run"
    )
    assert "bind mounts" in err.getvalue()


def test_a_manifest_failure_yields_no_fingerprint_rather_than_a_wrong_one():
    """An unusable measurement must degrade to ``unverified``, not to a mismatch.

    The launcher reads line 1 as the digest and exports it.  If the tool cannot
    run it prints nothing, the export is empty, and ``runtime_fingerprint_state``
    reports ``unverified`` -- the neutral state the design already defines --
    instead of fabricating the alarm this repair removes.
    """

    text = SERVO_LAUNCHER.read_text(encoding="utf-8")
    assert 'RUNTIME_FINGERPRINT=""' in text
    assert "2>/dev/null || true" in text
    assert 'if [[ -z "$RUNTIME_FINGERPRINT" ]]; then' in text
