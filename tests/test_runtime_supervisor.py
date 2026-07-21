"""Process-level tests for singleton mobile-manipulation deployment."""

from __future__ import annotations

import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / 'scripts' / 'runtime' / 'mobile_manipulation_supervisor.py'
SPEC = importlib.util.spec_from_file_location(
    'mobile_manip_supervisor', SUPERVISOR,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FAKE_ROS2 = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

state_path = Path(os.environ['FAKE_ROS_STATE'])
args = sys.argv[1:]


def active_state():
    try:
        state = json.loads(state_path.read_text())
        os.killpg(int(state['pgid']), 0)
        return state
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ProcessLookupError):
        return None


if args[:2] == ['node', 'list']:
    state = active_state()
    if state:
        namespace = state['namespace']
        prefix = '' if namespace == '/' else namespace
        for name in os.environ['FAKE_ROS_CRITICAL_NODES'].split(','):
            print(f'{prefix}/{name}')
    raise SystemExit(0)

if args[:2] == ['topic', 'info']:
    if active_state():
        print('Type: test_msgs/msg/Status')
        print(f"Publisher count: {os.environ.get('FAKE_ROS_PUBLISHER_COUNT', '1')}")
        print('Subscription count: 0')
        raise SystemExit(0)
    print(f"Unknown topic '{args[2]}'", file=sys.stderr)
    raise SystemExit(1)

if args[:1] != ['launch']:
    print(f'unsupported fake ros2 arguments: {args}', file=sys.stderr)
    raise SystemExit(2)

namespace = '/'
for argument in args:
    if argument.startswith('namespace:='):
        namespace = argument.split(':=', 1)[1]
pgid = os.getpgrp()
state_path.write_text(json.dumps({'pgid': pgid, 'namespace': namespace}))
child = subprocess.Popen(
    [
        sys.executable,
        '-c',
        'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'signal.signal(signal.SIGHUP, signal.SIG_IGN); time.sleep(300)',
    ],
    close_fds=False,
)
try:
    while True:
        time.sleep(1.0)
finally:
    child.wait()
"""


FAKE_RCLPY = r"""from __future__ import annotations

import json
import os
from pathlib import Path
import time


class _SignalHandlerOptions:
    NO = object()


class _Signals:
    SignalHandlerOptions = _SignalHandlerOptions


signals = _Signals()


def _active_state():
    try:
        state = json.loads(Path(os.environ['FAKE_ROS_STATE']).read_text())
        os.killpg(int(state['pgid']), 0)
        return state
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ProcessLookupError):
        return None


class Context:
    def __init__(self):
        self._ok = False

    def ok(self):
        return self._ok

    def shutdown(self):
        self._ok = False


class _Node:
    def __init__(self, name):
        self._name = name
        self._active = False
        self._active_spins = 0

    def _spin(self):
        active = _active_state() is not None
        if active and not self._active:
            self._active_spins = 0
        self._active = active
        if active:
            self._active_spins += 1

    def _hidden(self):
        empty_spins = int(os.environ.get('FAKE_RCLPY_EMPTY_ACTIVE_SPINS', '0'))
        return self._active and self._active_spins <= empty_spins

    def get_node_names_and_namespaces(self):
        state = _active_state()
        if state is None or self._hidden():
            return [(self._name, '/')]
        namespace = state['namespace']
        default_count = int(os.environ.get('FAKE_ROS_NODE_COUNT', '1'))
        missing = os.environ.get('FAKE_ROS_MISSING_NODE', '')
        duplicate = os.environ.get('FAKE_ROS_DUPLICATE_NODE', '')
        discovered = [(self._name, '/')]
        for name in os.environ['FAKE_ROS_CRITICAL_NODES'].split(','):
            count = 0 if name == missing else default_count
            if name == duplicate:
                count += 1
            discovered.extend((name, namespace) for _ in range(count))
        return discovered

    def get_publishers_info_by_topic(self, topic):
        if _active_state() is None or self._hidden():
            return []
        count = int(os.environ.get('FAKE_ROS_PUBLISHER_COUNT', '1'))
        if topic == os.environ.get('FAKE_ROS_MISSING_TOPIC', ''):
            count = 0
        elif topic == os.environ.get('FAKE_ROS_DUPLICATE_TOPIC', ''):
            count += 1
        return [object() for _ in range(count)]

    def destroy_node(self):
        pass


def init(*, args, context, signal_handler_options):
    del args, signal_handler_options
    context._ok = True


def create_node(name, **_kwargs):
    return _Node(name)


"""


FAKE_RCLPY_EXECUTORS = r"""import time


class SingleThreadedExecutor:
    def __init__(self, *, context):
        self._context = context
        self._node = None

    def add_node(self, node):
        self._node = node

    def remove_node(self, node):
        if self._node is node:
            self._node = None

    def spin_once(self, *, timeout_sec):
        time.sleep(timeout_sec)
        self._node._spin()

    def shutdown(self, *, timeout_sec):
        del timeout_sec
        return True
"""


def _make_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    ros2 = fake_bin / 'ros2'
    ros2.write_text(FAKE_ROS2)
    ros2.chmod(0o755)
    fake_python = tmp_path / 'python'
    rclpy = fake_python / 'rclpy'
    rclpy.mkdir(parents=True)
    (rclpy / '__init__.py').write_text(FAKE_RCLPY)
    (rclpy / 'executors.py').write_text(FAKE_RCLPY_EXECUTORS)
    environment = dict(os.environ)
    environment.update({
        'PATH': f'{fake_bin}{os.pathsep}{environment.get("PATH", "")}',
        'PYTHONPATH': (
            f'{fake_python}{os.pathsep}{environment.get("PYTHONPATH", "")}'
        ),
        'ROS_DOMAIN_ID': '184',
        'Z_MANIP_LOCK_DIR': str(tmp_path / 'locks'),
        'FAKE_ROS_STATE': str(tmp_path / 'graph.json'),
        'FAKE_ROS_CRITICAL_NODES': ','.join(MODULE.DEFAULT_CRITICAL_NODES),
        'PYTHONUNBUFFERED': '1',
    })
    return environment


def _command(*extra: str) -> list[str]:
    return [
        sys.executable,
        str(SUPERVISOR),
        '--startup-timeout', '2.0',
        '--shutdown-timeout', '0.2',
        '--graph-spin-time', '0.01',
        '--graph-preflight-observation-time', '0.05',
        '--graph-ready-stability-time', '0.03',
        *extra,
    ]


def _start(environment: dict[str, str], *extra: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _command(*extra),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_state(environment: dict[str, str]) -> dict[str, object]:
    path = Path(environment['FAKE_ROS_STATE'])
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            state = json.loads(path.read_text())
            os.killpg(int(state['pgid']), 0)
            return state
        except (FileNotFoundError, json.JSONDecodeError, ProcessLookupError):
            time.sleep(0.02)
    raise AssertionError('fake launch did not become visible')


def _wait_for_stdout(process: subprocess.Popen[str], expected: str) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + 3.0
    output = ''
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.05)
        if not readable:
            continue
        line = process.stdout.readline()
        if not line:
            break
        output += line
        if expected in output:
            return output
    raise AssertionError(f'{expected!r} not observed in supervisor output: {output!r}')


def _wait_for_group_exit(pgid: int) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _group_has_live_processes(pgid):
            return
        time.sleep(0.02)
    raise AssertionError(f'process group {pgid} survived teardown')


def _group_has_live_processes(pgid: int) -> bool:
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            line = (entry / 'stat').read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        close = line.rfind(')')
        fields = line[close + 2:].split() if close >= 0 else ()
        if len(fields) < 3:
            continue
        if int(fields[2]) == pgid and fields[0] != 'Z':
            return True
    return False


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _wait_for_group_exit(pgid)


@pytest.mark.parametrize(
    ('owner_kind', 'owner'),
    [
        *[('node', name) for name in MODULE.DEFAULT_CRITICAL_NODES],
        *[('topic', name) for name in MODULE.DEFAULT_CRITICAL_TOPICS],
    ],
)
@pytest.mark.parametrize(
    ('fault_count', 'expected_state'),
    ((0, 'incomplete'), (2, 'duplicate')),
)
def test_each_critical_owner_must_be_one_exactly(
    owner_kind: str,
    owner: str,
    fault_count: int,
    expected_state: str,
) -> None:
    """Every required node and publisher independently gates readiness."""
    nodes = {name: 1 for name in MODULE.DEFAULT_CRITICAL_NODES}
    topics = {name: 1 for name in MODULE.DEFAULT_CRITICAL_TOPICS}
    (nodes if owner_kind == 'node' else topics)[owner] = fault_count

    assert MODULE._graph_readiness_state(nodes, topics) == expected_state


def test_same_scope_second_start_refuses_and_signal_cleans_owned_group(tmp_path):
    environment = _make_environment(tmp_path)
    first = _start(environment, '--namespace', '/robot_a')
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        second = subprocess.run(
            _command('--namespace', '/robot_a'),
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        assert second.returncode == 73
        assert 'deployment lock is held' in second.stderr
        os.killpg(pgid, 0)

        first.terminate()
        first.wait(timeout=3.0)
        _wait_for_group_exit(pgid)
    finally:
        if first.poll() is None:
            first.kill()
        _kill_group(pgid)


def test_post_start_duplicate_producer_fails_and_cleans_owned_group(tmp_path):
    environment = _make_environment(tmp_path)
    environment['FAKE_ROS_PUBLISHER_COUNT'] = '2'
    process = _start(environment)
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        stdout, stderr = process.communicate(timeout=3.0)
        assert process.returncode == 70
        assert not stdout
        assert 'critical producer uniqueness violated' in stderr
        _wait_for_group_exit(pgid)
    finally:
        if process.poll() is None:
            process.kill()
        _kill_group(pgid)


def test_post_start_missing_moveit_scene_never_reports_ready(tmp_path):
    """A live launch root cannot mask a missing MoveIt planning scene."""
    environment = _make_environment(tmp_path)
    environment['FAKE_ROS_MISSING_TOPIC'] = '/monitored_planning_scene'
    process = _start(environment, '--startup-timeout', '0.15')
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        stdout, stderr = process.communicate(timeout=3.0)
        assert process.returncode == 70
        assert not stdout
        assert 'critical producers did not become uniquely ready' in stderr
        assert 'topic:/monitored_planning_scene=0' in stderr
        _wait_for_group_exit(pgid)
    finally:
        if process.poll() is None:
            process.kill()
        _kill_group(pgid)


def test_post_start_duplicate_place_owner_fails_immediately(tmp_path):
    """A second observed-place node violates singleton readiness."""
    environment = _make_environment(tmp_path)
    environment['FAKE_ROS_DUPLICATE_NODE'] = 'z_manip_observed_placement'
    process = _start(environment)
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        stdout, stderr = process.communicate(timeout=3.0)
        assert process.returncode == 70
        assert not stdout
        assert 'critical producer uniqueness violated' in stderr
        assert 'node:/z_manip_observed_placement=2' in stderr
        _wait_for_group_exit(pgid)
    finally:
        if process.poll() is None:
            process.kill()
        _kill_group(pgid)


def test_persistent_observer_survives_transient_empty_readiness_snapshots(tmp_path):
    environment = _make_environment(tmp_path)
    environment['FAKE_RCLPY_EMPTY_ACTIVE_SPINS'] = '8'
    process = _start(environment)
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        output = _wait_for_stdout(process, '[z-manip-supervisor] uniquely ready:')
        assert 'ROS_DOMAIN_ID=184' in output
        assert process.poll() is None

        process.terminate()
        process.wait(timeout=3.0)
        _wait_for_group_exit(pgid)
    finally:
        if process.poll() is None:
            process.kill()
        _kill_group(pgid)


def test_inherited_lock_blocks_restart_after_supervisor_and_launch_root_die(tmp_path):
    environment = _make_environment(tmp_path)
    first = _start(environment)
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    try:
        first.kill()
        first.wait(timeout=2.0)
        os.kill(pgid, signal.SIGKILL)
        time.sleep(0.05)
        os.killpg(pgid, 0)

        second = subprocess.run(
            _command(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        assert second.returncode == 73
        assert 'deployment lock is held' in second.stderr
    finally:
        _kill_group(pgid)


def test_preflight_refuses_external_producer_without_terminating_it(tmp_path):
    environment = _make_environment(tmp_path)
    environment['FAKE_RCLPY_EMPTY_ACTIVE_SPINS'] = '3'
    external = subprocess.Popen(
        ['ros2', 'launch', 'fake', 'external.launch.py', 'namespace:=/'],
        env=environment,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state = _wait_for_state(environment)
    pgid = int(state['pgid'])
    assert pgid == external.pid
    try:
        result = subprocess.run(
            _command(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        assert result.returncode == 70
        assert 'preflight found an existing critical producer' in result.stderr
        os.killpg(pgid, 0)
        assert external.poll() is None
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        external.wait(timeout=2.0)
        _wait_for_group_exit(pgid)


def test_namespace_and_domain_are_validated_before_launch(tmp_path):
    environment = _make_environment(tmp_path)
    environment['ROS_DOMAIN_ID'] = 'not-a-domain'
    invalid_domain = subprocess.run(
        _command(), env=environment, capture_output=True, text=True, timeout=2.0,
    )
    assert invalid_domain.returncode == 70
    assert 'ROS_DOMAIN_ID must be a decimal integer' in invalid_domain.stderr

    environment['ROS_DOMAIN_ID'] = '184'
    invalid_namespace = subprocess.run(
        _command('--namespace', '/invalid-name'),
        env=environment,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    assert invalid_namespace.returncode == 70
    assert 'valid ROS name segments' in invalid_namespace.stderr


def test_runtime_image_installs_supervisor_entrypoint():
    dockerfile = (ROOT / 'docker' / 'runtime' / 'Dockerfile').read_text()
    dockerignore = (ROOT / 'docker' / 'runtime' / 'Dockerfile.dockerignore').read_text()
    smoke = (ROOT / 'docker' / 'runtime' / 'smoke.sh').read_text()
    assert 'scripts/runtime/mobile_manipulation_supervisor.py' in dockerfile
    assert '/usr/local/bin/z-manip-mobile-manipulation' in dockerfile
    assert '!scripts/runtime/mobile_manipulation_supervisor.py' in dockerignore
    assert 'command -v z-manip-mobile-manipulation' in smoke
    assert SUPERVISOR.stat().st_mode & 0o111
