"""Pins the entry points that actually run, so a refactor cannot move them silently.

Everything here is read from checked-in units, runner scripts and launch files.
Nothing queries docker, systemd or ROS, so the manifest still holds on a machine
with no robot attached.
"""

import ast
import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT_DIR = ROOT / "configs"
RUNNER_SCRIPT = ROOT / "scripts/runtime/go2w_perception_lab.sh"
WORKBENCH_UNIT = UNIT_DIR / "z-manip-planning-workbench.service"
FINGERPRINT_SCRIPT = ROOT / "scripts/runtime/z_manip_runtime_fingerprint.py"

# Repository path prefix the systemd units hard-code for the host stack.
UNIT_REPO_PREFIX = "%h/Z-Robotics-Lab/Z-Mobile-manip/"

# Every unit that starts a file out of this checkout, and what it starts.
UNIT_ENTRY_POINTS = {
    "z-manip-planning-workbench.service": frozenset({
        "scripts/runtime/go2w_planning_control.py",
        "scripts/runtime/go2w_planning_session.sh",
        "scripts/runtime/piper_home_remote.sh",
        "scripts/runtime/piper_full_grasp_remote.py",
        "scripts/runtime/go2w_depth_servo.sh",
        "scripts/runtime/piper_wrist_search_remote.sh",
        "web/debug_dashboard/index.html",
    }),
    "z-manip-runtime-observer.service": frozenset({
        "scripts/runtime/go2w_runtime_observer.sh",
    }),
    "z-mobile-manip-go2w-posture-intent-live.service": frozenset({
        "scripts/runtime/go2w_posture_intent_bridge.sh",
    }),
    "z-mobile-manip-go2w-posture-intent-shadow.service": frozenset({
        "scripts/runtime/go2w_posture_intent_bridge.sh",
    }),
}

# The workbench unit fans out to the child scripts it drives; this is the
# authoritative wiring, the argparse defaults are only a fallback.
WORKBENCH_CHILD_SCRIPTS = {
    "--session-script": "scripts/runtime/go2w_planning_session.sh",
    "--home-script": "scripts/runtime/piper_home_remote.sh",
    "--grasp-script": "scripts/runtime/piper_full_grasp_remote.py",
    "--approach-script": "scripts/runtime/go2w_depth_servo.sh",
    "--wrist-search-script": "scripts/runtime/piper_wrist_search_remote.sh",
    "--index": "web/debug_dashboard/index.html",
}

# Host files the resident perception/planning containers execute, and the
# in-container program name each is bound to.
CONTAINER_ENTRY_POINTS = {
    "scripts/runtime/go2w_perception_dry_run.py":
        "/usr/local/bin/z-manip-go2w-perception-dry-run",
    "scripts/runtime/go2w_perception_worker.py":
        "/usr/local/bin/z-manip-go2w-perception-worker",
    "scripts/runtime/piper_planning_dry_run.py":
        "/usr/local/bin/z-manip-piper-planning-dry-run",
    "scripts/runtime/piper_planning_worker.py":
        "/usr/local/bin/z-manip-piper-planning-worker",
}

# ROS 2 launch files the live stack starts, and the packages they may reach.
LIVE_ROS2_LAUNCHES = {("z_manip_ros", "perception.launch.py")}
LIVE_ROS2_PACKAGES = frozenset({
    "z_manip_edgetam",
    "z_manip_rgbd_bridge",
    "z_manip_ros",
})
DORMANT_ROS2_PACKAGES = frozenset({
    "z_manip_motion",
    "z_manip_navigation",
    "z_manip_place",
    "z_manip_task",
})

# The planner the live planning runner loads, and where it must live so that the
# resident-worker fingerprint can invalidate a warm worker when it changes.
LIVE_PLANNER_MODULE = "z_manip.planning.online_planner"
LIVE_PLANNER_PATH = "z_manip/planning/online_planner.py"


def _unit_repo_paths(unit: Path) -> frozenset[str]:
    started = []
    for line in unit.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ExecStart", "ExecStop", "ExecReload")):
            continue
        started.extend(
            match.rstrip("'\"")
            for match in re.findall(rf"{re.escape(UNIT_REPO_PREFIX)}(\S+)", line)
        )
    return frozenset(started)


def _workbench_exec_start() -> list[str]:
    for line in WORKBENCH_UNIT.read_text(encoding="utf-8").splitlines():
        if line.startswith("ExecStart="):
            return line[len("ExecStart="):].split()
    raise AssertionError("workbench unit has no ExecStart")


def _top_level_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".", 1)[0])
    return modules


def test_units_start_exactly_the_pinned_repository_entry_points():
    found = {}
    for unit in sorted(UNIT_DIR.glob("*.service")):
        paths = _unit_repo_paths(unit)
        if paths:
            found[unit.name] = paths

    assert found == UNIT_ENTRY_POINTS
    for paths in found.values():
        for relative in paths:
            assert (ROOT / relative).is_file(), relative


def test_workbench_unit_drives_exactly_the_pinned_child_scripts():
    tokens = _workbench_exec_start()
    wired = {}
    for flag, value in zip(tokens, tokens[1:]):
        if flag in WORKBENCH_CHILD_SCRIPTS:
            assert value.startswith(UNIT_REPO_PREFIX), value
            wired[flag] = value[len(UNIT_REPO_PREFIX):]

    assert wired == WORKBENCH_CHILD_SCRIPTS
    assert tokens[1].endswith(
        UNIT_REPO_PREFIX + "scripts/runtime/go2w_planning_control.py"
    )


def test_resident_runners_execute_exactly_the_pinned_host_scripts():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    mounted = {
        host: container
        for host, container in re.findall(
            r'-v "\$ROOT_DIR/(scripts/runtime/[^:]+):(/usr/local/bin/[^:]+):ro"',
            source,
        )
    }

    assert mounted == CONTAINER_ENTRY_POINTS
    for relative in mounted:
        assert (ROOT / relative).is_file(), relative


def test_live_stack_launches_only_live_ros2_packages():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    launched = set(re.findall(r"ros2 launch (\S+) (\S+\.launch\.py)", source))

    assert launched == LIVE_ROS2_LAUNCHES
    for package, launch_file in launched:
        path = ROOT / "ros2" / package / "launch" / launch_file
        assert path.is_file(), str(path)
        packages = set(
            re.findall(r"package=['\"]([^'\"]+)['\"]", path.read_text(encoding="utf-8"))
        )
        assert packages <= LIVE_ROS2_PACKAGES, sorted(packages)


def test_no_runtime_script_imports_a_dormant_ros2_package():
    offenders = {}
    for script in sorted((ROOT / "scripts").rglob("*.py")):
        imported = _top_level_imports(script.read_text(encoding="utf-8"))
        dormant = imported & DORMANT_ROS2_PACKAGES
        if dormant:
            offenders[script.relative_to(ROOT).as_posix()] = sorted(dormant)

    assert offenders == {}


def test_live_planner_is_inside_the_resident_worker_fingerprint():
    spec = importlib.util.spec_from_file_location(
        "z_manip_runtime_fingerprint", FINGERPRINT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    fingerprint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fingerprint)
    inputs = {
        path.resolve() for path in fingerprint.runtime_inputs()
    }

    assert (ROOT / LIVE_PLANNER_PATH).resolve() in inputs
    assert (
        ROOT / "ros2/z_manip_task/z_manip_task/planning.py"
    ).resolve() not in inputs


def test_planning_runner_loads_the_planner_from_the_z_manip_library():
    source = (
        ROOT / "scripts/runtime/piper_planning_dry_run.py"
    ).read_text(encoding="utf-8")

    assert f"from {LIVE_PLANNER_MODULE} import OnlinePlanner" in source
    assert "z_manip_task" not in source
