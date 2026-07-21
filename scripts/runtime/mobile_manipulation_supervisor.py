#!/usr/bin/env python3
"""Own exactly one mobile-manipulation launch in a ROS deployment scope."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Sequence


DEFAULT_CRITICAL_NODES = (
    'z_manip_urdf_root_alias',
    'vlm_edgetam_bridge',
    'z_manip_edgetam',
    'z_manip_complete_joint_state',
    'z_manip_robot_state_publisher',
    'z_manip_coarse_navigation',
    'z_manip_observed_placement',
    'z_manip_task_runtime',
)
DEFAULT_CRITICAL_TOPICS = (
    '/z_manip/perception/status',
    '/track_3d/frame_manifest',
    '/z_manip/motion/complete_joint_states',
    '/monitored_planning_scene',
    '/z_manip/navigation/status',
    '/z_manip/place/status',
    '/z_manip/task/status',
)


class SupervisorError(RuntimeError):
    """A fail-closed bringup or teardown error."""


class LockHeldError(SupervisorError):
    """Another owner or one of its surviving descendants retains the lock."""


def _domain_id(environment: dict[str, str]) -> int:
    raw = environment.get('ROS_DOMAIN_ID', '0').strip()
    if not raw.isdecimal():
        raise SupervisorError('ROS_DOMAIN_ID must be a decimal integer')
    value = int(raw)
    if value > 232:
        raise SupervisorError('ROS_DOMAIN_ID must be in [0, 232]')
    return value


def _canonical_namespace(value: str) -> str:
    clean = str(value).strip()
    if not clean or clean == '/':
        return '/'
    if not clean.startswith('/'):
        clean = f'/{clean}'
    clean = clean.rstrip('/')
    segments = clean[1:].split('/')
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', item) for item in segments):
        raise SupervisorError(
            'namespace must contain only valid ROS name segments',
        )
    return clean


def _qualified_node(namespace: str, name: str) -> str:
    clean = str(name).strip()
    if not clean:
        raise SupervisorError('critical node names must not be empty')
    if clean.startswith('/'):
        return clean.rstrip('/')
    prefix = '' if namespace == '/' else namespace
    return f'{prefix}/{clean}'


class DeploymentLock:
    """A nonblocking flock whose descriptor is inherited by the launch root."""

    def __init__(self, lock_dir: Path, domain_id: int, namespace: str) -> None:
        """Build the deterministic lock path for one ROS deployment scope."""
        digest = hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:16]
        self.path = lock_dir / (
            f'mobile-manipulation.domain-{domain_id}.namespace-{digest}.lock'
        )
        self._fd: int | None = None
        self._domain_id = domain_id
        self._namespace = namespace

    @property
    def fd(self) -> int:
        """Return the held descriptor that must be inherited by launch."""
        if self._fd is None:
            raise SupervisorError('deployment lock is not held')
        return self._fd

    def acquire(self) -> None:
        """Acquire the deployment lock without waiting for another owner."""
        try:
            self.path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise SupervisorError(
                f'could not open deployment lock {self.path}: {type(error).__name__}',
            ) from error
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SupervisorError(f'lock path is not a regular file: {self.path}')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                owner = os.pread(fd, 4096, 0).decode('utf-8', errors='replace').strip()
                detail = f'; owner={owner}' if owner else ''
                raise LockHeldError(
                    f'deployment lock is held for ROS_DOMAIN_ID={self._domain_id} '
                    f'namespace={self._namespace}{detail}',
                ) from error
            os.set_inheritable(fd, True)
            self._fd = fd
            self.write_metadata(child_pid=None)
        except BaseException:
            os.close(fd)
            raise

    def write_metadata(self, *, child_pid: int | None) -> None:
        """Record bounded owner metadata for a refused second bringup."""
        payload = json.dumps(
            {
                'schema': 'z_manip.runtime_lock.v1',
                'supervisor_pid': os.getpid(),
                'launch_pgid': child_pid,
                'ros_domain_id': self._domain_id,
                'namespace': self._namespace,
            },
            separators=(',', ':'),
        ).encode('utf-8')
        os.ftruncate(self.fd, 0)
        os.pwrite(self.fd, payload + b'\n', 0)
        os.fsync(self.fd)

    def release(self) -> None:
        """Release this process's deployment lock descriptor."""
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


class RosGraphProbe:
    """Observe singleton producers through one persistent DDS participant."""

    def __init__(self, *, spin_time_s: float) -> None:
        """Create one graph observer whose discovery cache survives snapshots."""
        self._spin_time_s = spin_time_s
        self._context: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor

            self._context = rclpy.Context()
            rclpy.init(
                args=[],
                context=self._context,
                signal_handler_options=rclpy.signals.SignalHandlerOptions.NO,
            )
            self._node = rclpy.create_node(
                f'_z_manip_graph_probe_{os.getpid()}',
                context=self._context,
                use_global_arguments=False,
                enable_rosout=False,
                start_parameter_services=False,
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
        except Exception as error:
            self.close()
            raise SupervisorError(
                f'could not create persistent ROS graph observer: '
                f'{type(error).__name__}',
            ) from error

    def close(self) -> None:
        """Destroy the observer and its dedicated rclpy context."""
        executor, self._executor = self._executor, None
        node, self._node = self._node, None
        context, self._context = self._context, None
        if executor is not None:
            try:
                if node is not None:
                    executor.remove_node(node)
                executor.shutdown(timeout_sec=0.2)
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if context is not None:
            try:
                if context.ok():
                    context.shutdown()
            except Exception:
                pass

    def _spin(self) -> None:
        if self._executor is None or self._node is None:
            raise SupervisorError('persistent ROS graph observer is closed')
        try:
            self._executor.spin_once(timeout_sec=self._spin_time_s)
        except Exception as error:
            raise SupervisorError(
                f'persistent ROS graph observer failed: {type(error).__name__}',
            ) from error

    def node_counts(self) -> Counter[str]:
        """Return discovered fully qualified ROS node-name counts."""
        if self._node is None:
            raise SupervisorError('persistent ROS graph observer is closed')
        try:
            discovered = self._node.get_node_names_and_namespaces()
        except Exception as error:
            raise SupervisorError(
                f'ROS node graph query failed: {type(error).__name__}',
            ) from error
        counts: Counter[str] = Counter()
        for name, namespace in discovered:
            prefix = '' if namespace == '/' else str(namespace).rstrip('/')
            counts[f'{prefix}/{str(name).lstrip("/")}'] += 1
        return counts

    def publisher_count(self, topic: str) -> int:
        """Return the discovered publisher count for one absolute topic."""
        if self._node is None:
            raise SupervisorError('persistent ROS graph observer is closed')
        try:
            return len(self._node.get_publishers_info_by_topic(topic))
        except Exception as error:
            raise SupervisorError(
                f'ROS publisher graph query failed for {topic}: '
                f'{type(error).__name__}',
            ) from error

    def snapshot(
        self,
        nodes: Sequence[str],
        topics: Sequence[str],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Observe all critical node and producer counts once."""
        self._spin()
        discovered = self.node_counts()
        node_counts = {name: discovered[name] for name in nodes}
        topic_counts = {topic: self.publisher_count(topic) for topic in topics}
        return node_counts, topic_counts


def _format_counts(nodes: dict[str, int], topics: dict[str, int]) -> str:
    fields = [f'node:{name}={count}' for name, count in nodes.items()]
    fields.extend(f'topic:{name}={count}' for name, count in topics.items())
    return ', '.join(fields)


def _graph_readiness_state(
    nodes: dict[str, int],
    topics: dict[str, int],
) -> str:
    """Classify an exact singleton graph as ready, incomplete, or duplicate."""
    counts = (*nodes.values(), *topics.values())
    if any(value > 1 for value in counts):
        return 'duplicate'
    if counts and all(value == 1 for value in counts):
        return 'ready'
    return 'incomplete'


def _require_empty_graph(
    probe: RosGraphProbe,
    nodes: Sequence[str],
    topics: Sequence[str],
    observation_time_s: float,
    stop_requested: Callable[[], bool],
) -> None:
    """Reject any producer discovered during a bounded persistent warm-up."""
    deadline = time.monotonic() + observation_time_s
    while True:
        if stop_requested():
            raise SupervisorError('preflight interrupted by a termination signal')
        node_counts, topic_counts = probe.snapshot(nodes, topics)
        if any(node_counts.values()) or any(topic_counts.values()):
            raise SupervisorError(
                'preflight found an existing critical producer; refusing to start or '
                f'terminate it: {_format_counts(node_counts, topic_counts)}',
            )
        if time.monotonic() >= deadline:
            return


def _wait_for_unique_graph(
    probe: RosGraphProbe,
    nodes: Sequence[str],
    topics: Sequence[str],
    child: subprocess.Popen[bytes],
    timeout_s: float,
    stability_time_s: float,
    stop_requested: Callable[[], bool],
) -> None:
    deadline = time.monotonic() + timeout_s
    last = 'no graph observation'
    unique_since: float | None = None
    while time.monotonic() < deadline:
        if stop_requested():
            raise SupervisorError('startup interrupted by a termination signal')
        returncode = child.poll()
        if returncode is not None:
            raise SupervisorError(
                f'mobile-manipulation launch exited before readiness ({returncode})',
            )
        node_counts, topic_counts = probe.snapshot(nodes, topics)
        last = _format_counts(node_counts, topic_counts)
        readiness = _graph_readiness_state(node_counts, topic_counts)
        if readiness == 'duplicate':
            raise SupervisorError(
                f'critical producer uniqueness violated: {last}',
            )
        if readiness == 'ready':
            now = time.monotonic()
            if unique_since is None:
                unique_since = now
            elif now - unique_since >= stability_time_s:
                return
        else:
            unique_since = None
    raise SupervisorError(
        f'critical producers did not become uniquely ready in {timeout_s:.1f}s: {last}',
    )


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise SupervisorError(f'cannot inspect owned launch process group {pgid}') from error
    live_members = _linux_group_has_live_members(pgid)
    return True if live_members is None else live_members


def _linux_group_has_live_members(pgid: int) -> bool | None:
    """Ignore terminated zombies when a container PID 1 does not reap them."""
    proc = Path('/proc')
    if not proc.is_dir():
        return None
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
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
        try:
            member_pgid = int(fields[2])
        except ValueError:
            continue
        if member_pgid != pgid:
            continue
        if fields[0] != 'Z':
            return True
    return False


def _stop_owned_group(
    child: subprocess.Popen[bytes],
    pgid: int,
    timeout_s: float,
) -> None:
    """Terminate only the process group created by this supervisor."""
    if pgid != child.pid:
        raise SupervisorError(
            f'refusing teardown because owned launch PGID changed ({pgid} != {child.pid})',
        )
    if _group_exists(pgid):
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while _group_exists(pgid) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.02)
    if _group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + min(2.0, timeout_s)
        while _group_exists(pgid) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.02)
    try:
        child.wait(timeout=1.0)
    except subprocess.TimeoutExpired as error:
        raise SupervisorError(f'owned launch process group {pgid} did not exit') from error


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fail-closed singleton owner for the complete manipulation launch.',
    )
    parser.add_argument(
        '--namespace',
        default=os.environ.get('Z_MANIP_ROS_NAMESPACE', '/'),
        help='ROS namespace and singleton lock scope (default: /).',
    )
    parser.add_argument('--startup-timeout', type=float, default=20.0)
    parser.add_argument('--shutdown-timeout', type=float, default=8.0)
    parser.add_argument('--graph-spin-time', type=float, default=0.2)
    parser.add_argument(
        '--graph-preflight-observation-time', type=float, default=2.0,
        help='Persistent discovery window required before an empty preflight passes.',
    )
    parser.add_argument(
        '--graph-ready-stability-time', type=float, default=1.0,
        help='Continuous exact-uniqueness window required before readiness.',
    )
    parser.add_argument(
        '--critical-node', action='append', default=None,
        help='Critical node name; repeat to replace the default owner set.',
    )
    parser.add_argument(
        '--critical-topic', action='append', default=None,
        help='Critical producer topic; repeat to replace the default owner set.',
    )
    parser.add_argument(
        'launch_arguments', nargs=argparse.REMAINDER,
        help='Arguments passed to mobile_manipulation.launch.py after --.',
    )
    parsed = parser.parse_args(argv)
    for name in (
        'startup_timeout', 'shutdown_timeout', 'graph_spin_time',
        'graph_preflight_observation_time', 'graph_ready_stability_time',
    ):
        value = float(getattr(parsed, name))
        if not 0.0 < value <= 300.0:
            parser.error(f'--{name.replace("_", "-")} must be in (0, 300]')
    if parsed.launch_arguments[:1] == ['--']:
        parsed.launch_arguments = parsed.launch_arguments[1:]
    if any(str(value).startswith('namespace:=') for value in parsed.launch_arguments):
        parser.error('pass the deployment namespace with --namespace only')
    return parsed


def run(argv: Sequence[str] | None = None) -> int:
    """Own one launch until it exits or this supervisor receives a signal."""
    arguments = _arguments(argv)
    environment = dict(os.environ)
    domain_id = _domain_id(environment)
    namespace = _canonical_namespace(arguments.namespace)
    configured_nodes = arguments.critical_node or DEFAULT_CRITICAL_NODES
    critical_nodes = tuple(
        _qualified_node(namespace, name) for name in configured_nodes
    )
    critical_topics = tuple(arguments.critical_topic or DEFAULT_CRITICAL_TOPICS)
    if any(not str(topic).startswith('/') for topic in critical_topics):
        raise SupervisorError('critical producer topics must be absolute ROS names')

    lock_dir = Path(environment.get('Z_MANIP_LOCK_DIR', '/tmp/z-manip-runtime'))
    deployment_lock = DeploymentLock(lock_dir, domain_id, namespace)
    probe: RosGraphProbe | None = None
    child: subprocess.Popen[bytes] | None = None
    child_pgid: int | None = None
    received_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        deployment_lock.acquire()
        probe = RosGraphProbe(spin_time_s=float(arguments.graph_spin_time))
        _require_empty_graph(
            probe,
            critical_nodes,
            critical_topics,
            float(arguments.graph_preflight_observation_time),
            lambda: received_signal is not None,
        )
        command = [
            'ros2', 'launch', '--noninteractive',
            'z_manip_task', 'mobile_manipulation.launch.py',
            f'namespace:={namespace}',
            *arguments.launch_arguments,
        ]
        try:
            child = subprocess.Popen(
                command,
                start_new_session=True,
                pass_fds=(deployment_lock.fd,),
            )
        except OSError as error:
            raise SupervisorError(
                f'could not start mobile-manipulation launch: {type(error).__name__}',
            ) from error
        child_pgid = os.getpgid(child.pid)
        if child_pgid != child.pid:
            raise SupervisorError(
                'launch did not enter the supervisor-owned process group',
            )
        deployment_lock.write_metadata(child_pid=child_pgid)
        _wait_for_unique_graph(
            probe,
            critical_nodes,
            critical_topics,
            child,
            float(arguments.startup_timeout),
            float(arguments.graph_ready_stability_time),
            lambda: received_signal is not None,
        )
        print(
            '[z-manip-supervisor] uniquely ready: '
            f'ROS_DOMAIN_ID={domain_id} namespace={namespace} pgid={child_pgid}',
            flush=True,
        )
        while received_signal is None:
            returncode = child.poll()
            if returncode is not None:
                return returncode
            time.sleep(0.1)
        return 128 + received_signal
    finally:
        cleanup_error: BaseException | None = None
        if probe is not None:
            probe.close()
        if child is not None and child_pgid is not None:
            try:
                _stop_owned_group(
                    child,
                    child_pgid,
                    float(arguments.shutdown_timeout),
                )
            except BaseException as error:
                cleanup_error = error
        deployment_lock.release()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


def main() -> None:
    """Run the supervisor with stable fail-closed exit codes."""
    try:
        raise SystemExit(run())
    except LockHeldError as error:
        print(f'[z-manip-supervisor] refused: {error}', file=sys.stderr, flush=True)
        raise SystemExit(73) from error
    except SupervisorError as error:
        print(f'[z-manip-supervisor] failed: {error}', file=sys.stderr, flush=True)
        raise SystemExit(70) from error


if __name__ == '__main__':
    main()
