"""ROS package and no-ground-truth contract tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ros_graph_contract_and_no_truth_inputs():
    source = (ROOT / 'z_manip_navigation' / 'node.py').read_text()
    config = (ROOT / 'config' / 'navigation.yaml').read_text()
    required = (
        '/way_point',
        '/goal_reached',
        '/cancel_goal',
        '/state_estimation',
        '/z_manip/perception/target_3d',
        '/z_manip/navigation/coarse_ready',
    )
    assert all(topic in source + config for topic in required)
    assert '/objects/' not in source
    assert '/ground_truth' not in source
    assert "Bool, self._topic('cancel_topic')" in source
    assert "PointStamped, self._topic('waypoint_topic')" in source
    assert 'Buffer(node=self)' in source


def test_package_installs_launch_config_and_entrypoint():
    setup = (ROOT / 'setup.py').read_text()
    manifest = (ROOT / 'package.xml').read_text()
    assert "'coarse_navigation = z_manip_navigation.node:main'" in setup
    assert "glob('config/*.yaml')" in setup
    assert "glob('launch/*.launch.py')" in setup
    assert '<exec_depend>vision_msgs</exec_depend>' in manifest


def test_launch_exposes_sim_or_real_clock_selection():
    source = (ROOT / 'launch' / 'coarse_navigation.launch.py').read_text()
    assert "DeclareLaunchArgument('use_sim_time', default_value='true')" in source
    assert "{'use_sim_time': use_sim_time}" in source
