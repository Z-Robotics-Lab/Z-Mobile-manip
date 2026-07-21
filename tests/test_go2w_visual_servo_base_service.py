from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/go2w_visual_servo_base_nuc.sh"
UNIT = ROOT / "configs/z-manip-go2w-base-control.service"


def test_base_control_keeps_guard_between_cmd_vel_and_webrtc():
    script = SCRIPT.read_text(encoding="utf-8")
    unit = UNIT.read_text(encoding="utf-8")
    assert "ROS_DOMAIN_ID=20" in script
    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in script
    assert "CYCLONEDDS_URI=" in script
    assert "go2w-nuc/bringup/cyclonedds.xml" in script
    assert "ros2 launch cmd_vel_guard" in script
    assert "max_linear_mps:=0.20" in script
    assert "ros2 launch unitree_webrtc_ros unitree_control.launch.py" in script
    assert "robot_ip:=192.168.123.161" in script
    assert "connection_method:=LocalSTA" in script
    assert "control_mode:=sport_cmd" in script
    assert "go2w_visual_servo_base_nuc.sh" in unit
    assert "Restart=on-failure" in unit


def test_base_control_script_only_stops_processes_it_started():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "guard_pid=$!" in script
    assert "control_pid=$!" in script
    assert "kill \"$guard_pid\" \"$control_pid\"" in script
    assert "pkill" not in script
    assert "killall" not in script
