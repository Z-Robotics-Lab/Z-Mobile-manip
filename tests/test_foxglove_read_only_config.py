from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "configs" / "foxglove_read_only.yaml"
SERVICE = ROOT / "configs" / "foxglove-bridge-read-only.service"


def test_foxglove_bridge_has_no_remote_mutation_capability():
    document = yaml.safe_load(PARAMETERS.read_text(encoding="utf-8"))
    values = document["foxglove_bridge"]["ros__parameters"]

    assert values["capabilities"] == ["connectionGraph"]
    assert "clientPublish" not in values["capabilities"]
    assert "parameters" not in values["capabilities"]
    assert "services" not in values["capabilities"]
    assert "assets" not in values["capabilities"]
    assert values["client_topic_whitelist"] == ["^$"]
    assert values["service_whitelist"] == ["^$"]
    assert values["param_whitelist"] == ["^$"]
    assert values["asset_uri_allowlist"] == ["^$"]
    assert values["remote_access"] is False
    assert values["sysinfo"] is False
    assert values["publish_client_count"] is False


def test_foxglove_topic_allowlist_excludes_actuators():
    document = yaml.safe_load(PARAMETERS.read_text(encoding="utf-8"))
    topics = document["foxglove_bridge"]["ros__parameters"]["topic_whitelist"]
    rendered = "\n".join(topics)

    assert "camera" in rendered
    assert "z_manip/perception" in rendered
    assert "piper/state" in rendered
    assert "cmd_vel" not in rendered
    assert "joint_trajectory" not in rendered
    assert "gripper" not in rendered


def test_service_loads_only_the_reviewed_parameter_file():
    source = SERVICE.read_text(encoding="utf-8")

    assert "foxglove_read_only.yaml" in source
    assert "-p capabilities" not in source
    assert "clientPublish" not in source
