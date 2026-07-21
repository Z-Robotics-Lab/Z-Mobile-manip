"""Runtime executor contract tests."""

from types import SimpleNamespace

from z_manip_task import node as task_node


def test_main_keeps_tf_listener_live_during_exact_stamp_lookup(
    monkeypatch,
) -> None:
    """Use two executor workers so exact-stamp TF waits can make progress."""
    events: list[object] = []
    runtime = SimpleNamespace(
        destroy_node=lambda: events.append('destroy_node'),
    )

    class _Executor:
        def __init__(self, *, num_threads: int) -> None:
            events.append(('executor', num_threads))

        def shutdown(self) -> None:
            events.append('executor_shutdown')

    monkeypatch.setattr(
        task_node.rclpy,
        'init',
        lambda *, args: events.append(('init', args)),
    )
    monkeypatch.setattr(
        task_node,
        'MobileManipulationRuntime',
        lambda: runtime,
    )
    monkeypatch.setattr(task_node, 'MultiThreadedExecutor', _Executor)
    monkeypatch.setattr(
        task_node.rclpy,
        'spin',
        lambda node, *, executor: events.append(('spin', node, executor)),
    )
    monkeypatch.setattr(
        task_node.rclpy,
        'shutdown',
        lambda: events.append('shutdown'),
    )

    task_node.main(['--test'])

    assert events[0:2] == [('init', ['--test']), ('executor', 2)]
    assert events[2][0:2] == ('spin', runtime)
    assert events[3:] == ['executor_shutdown', 'destroy_node', 'shutdown']
