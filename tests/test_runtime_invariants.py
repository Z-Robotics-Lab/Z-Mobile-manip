from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _runtime_python_sources():
    yield from sorted((ROOT / "z_manip").rglob("*.py"))
    for package in sorted((ROOT / "ros2").glob("z_manip_*")):
        runtime_module = package / package.name
        if runtime_module.is_dir():
            yield from sorted(runtime_module.rglob("*.py"))


def test_runtime_never_reads_sim_ground_truth():
    forbidden = (
        "/ground_truth",
        "/objects/",
        "/piper/ee_pose",
        "get_object_gt_pose",
    )
    offenders = []
    for path in _runtime_python_sources():
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not offenders, "ground truth is test-oracle-only:\n" + "\n".join(offenders)


def test_runtime_never_imports_isaac():
    offenders = []
    for path in _runtime_python_sources():
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import isaac", "from isaac")):
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {stripped}")
    assert not offenders, "platform code belongs behind ROS contracts:\n" + "\n".join(offenders)


def test_cancellable_vlm_transport_has_a_declared_runtime_binary():
    dockerfile = (ROOT / "docker/runtime/Dockerfile").read_text()
    smoke = (ROOT / "docker/runtime/smoke.sh").read_text()
    manifest = (ROOT / "ros2/z_manip_ros/package.xml").read_text()

    assert "        curl \\\n" in dockerfile
    assert "command -v curl >/dev/null" in smoke
    assert "<exec_depend>curl</exec_depend>" in manifest
