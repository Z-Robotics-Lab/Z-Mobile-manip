"""R9 / H5 / H7 -- contracts for how this repository is DEPLOYED, not run.

Everything here is a file-level contract because none of it can be executed on a
development host: the units run under systemd on a NUC, the QoS profile is
consumed by ``ros2 bag record`` inside a container, and the launcher drives
docker and ssh.  A contract test is the strongest check available, and each one
below is anchored on a recorded failure rather than on style.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
QOS = CONFIGS / "rosbag_sensor_qos.yaml"
SERVO_LAUNCHER = ROOT / "scripts" / "runtime" / "go2w_depth_servo.sh"
TRANSPORT_PREFLIGHT = (
    ROOT / "scripts" / "runtime" / "go2w_base_transport_preflight.sh"
)
DRY_RUN = ROOT / "scripts" / "runtime" / "go2w_perception_dry_run.py"


def _unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False)
    # systemd keys are case sensitive and repeat (Environment=, ExecStart=).
    parser.optionxform = str
    parser.read_string((CONFIGS / name).read_text(encoding="utf-8"))
    return parser


# ---------------------------------------------------------------------------
# H7.  The recording profile contradicted its own publisher.
#
# A BEST_EFFORT rosbag2 subscription against a RELIABLE writer CONNECTS -- QoS
# is request<=offered -- so nothing ever errored.  It simply stopped NACKing,
# and every sample the Wi-Fi link dropped was gone from the bag.  The bag is the
# only artifact a hang is reproduced from, so the result was a replay that
# concluded "perception never reached the controller" about a run in which it
# did.
# ---------------------------------------------------------------------------

#: topic -> (publisher source file, the QoS symbol it is constructed from).
#: Every point-cloud topic in the recording profile, with the publisher that
#: actually writes it.  Verified by reading the sources below, not asserted.
_CLOUD_PUBLISHERS = {
    "/track_3d/selected_target_pointcloud": (
        "ros2/z_manip_edgetam/z_manip_edgetam/node.py",
        "selected_cloud_topic",
        "reliable",
    ),
    "/z_manip/perception/target_pointcloud": (
        "ros2/z_manip_ros/z_manip_ros/vlm_edgetam_bridge.py",
        "validated_cloud_topic",
        "reliable",
    ),
    "/z_manip/perception/scene_pointcloud": (
        "ros2/z_manip_edgetam/z_manip_edgetam/node.py",
        "scene_cloud_topic",
        "qos_profile_sensor_data",
    ),
}


@pytest.mark.parametrize("topic", sorted(_CLOUD_PUBLISHERS))
def test_the_bag_profile_matches_the_publisher_it_records(topic):
    source_name, topic_parameter, qos_symbol = _CLOUD_PUBLISHERS[topic]
    source = (ROOT / source_name).read_text(encoding="utf-8")
    # The publisher really is built from the QoS symbol this test claims.
    publisher = source[source.index(f"'{topic_parameter}'"):]
    assert qos_symbol in publisher[: publisher.index(")") + 40]
    # And the topic parameter really does default to this topic name.
    assert f"'{topic_parameter}': '{topic}'" in source

    profile = yaml.safe_load(QOS.read_text(encoding="utf-8"))[topic]
    expected = "best_effort" if qos_symbol == "qos_profile_sensor_data" else "reliable"
    assert profile["reliability"] == expected, (
        f"{topic} is published {expected.upper()} but recorded "
        f"{profile['reliability'].upper()}; a bag-based reproduction of a hang "
        "will silently under-report it"
    )


def test_the_reliable_clouds_are_recorded_at_the_publishers_depth():
    profile = yaml.safe_load(QOS.read_text(encoding="utf-8"))
    for topic in (
        "/track_3d/selected_target_pointcloud",
        "/z_manip/perception/target_pointcloud",
    ):
        assert profile[topic]["depth"] == 10


# ---------------------------------------------------------------------------
# H5.  Four units inherited the default 5-per-10 s start limiter and nothing in
# the repository ever called reset-failed.  Under `set -e` a tripped limiter
# aborted the preflight, and the operator's only clue was a launcher that
# stopped.
#
# The decision is PER UNIT and each one is justified in the unit file itself.
# ---------------------------------------------------------------------------

#: unit -> the explicit limiter it must declare, and why in one word.
#: ``None`` interval means StartLimitIntervalSec=0 (retry forever).
_START_LIMITS = {
    # NUC, single CAN owner.  A crash loop must latch (it fights the passive
    # bridge for the bus) but operator-driven starts must not trip it.
    "z-mobile-manip-piper-reactive-view.service": ("60", "10"),
    # NUC, zero-TX evidence producer, stopped/started by seven call sites
    # around every arm motion -- the highest hand-restart rate in the system.
    "z-manip-piper-passive-feedback.service": ("60", "15"),
    # NUC, single owner of the ONLY channel that reaches the base.
    "z-mobile-manip-go2w-reactive-live.service": ("60", "10"),
    # PC, owns no exclusive hardware, and IS the operator's only interface: a
    # latched workbench is a browser tab that stops updating with no second
    # channel to notice it through.  Follows the d435i / ffs-ir-throttle
    # precedent deliberately.
    "z-manip-planning-workbench.service": ("0", None),
}


@pytest.mark.parametrize("name", sorted(_START_LIMITS))
def test_every_restarting_unit_declares_its_start_limiter_explicitly(name):
    interval, burst = _START_LIMITS[name]
    unit = _unit(name)
    assert unit["Unit"]["StartLimitIntervalSec"] == interval
    if burst is None:
        assert "StartLimitBurst" not in unit["Unit"]
    else:
        assert unit["Unit"]["StartLimitBurst"] == burst


@pytest.mark.parametrize("name", sorted(_START_LIMITS))
def test_a_start_limiter_decision_is_justified_where_it_is_written(name):
    text = (CONFIGS / name).read_text(encoding="utf-8")
    assert "# H5." in text, (
        "an inherited-default override with no recorded reason is how the next "
        "reader deletes it"
    )


def test_a_crash_loop_still_latches_under_every_kept_limiter():
    """A wider limiter must not become no limiter.

    Restart= plus RestartSec= gives the crash-loop start rate; the limiter must
    still be reachable from it, or these units would retry a broken CAN bus or a
    broken WebRTC bridge forever.
    """

    for name, (interval, burst) in _START_LIMITS.items():
        if burst is None:
            continue
        restart_sec = float(_unit(name)["Service"]["RestartSec"])
        # Time for ``burst`` crash-loop starts, which must fit in the window.
        assert restart_sec * (int(burst) - 1) < float(interval), (
            f"{name} widened its limiter past its own crash-loop rate, so a "
            "genuine fault would restart forever"
        )


def test_the_units_that_retry_forever_are_only_the_justified_ones():
    """StartLimitIntervalSec=0 is a deliberate, enumerated choice."""

    # PARSED, not grepped: three of these files discuss StartLimitIntervalSec=0
    # in a comment explaining why they did NOT choose it.
    forever = set()
    for path in sorted(CONFIGS.glob("*.service")):
        parser = configparser.ConfigParser(strict=False)
        parser.optionxform = str
        parser.read_string(path.read_text(encoding="utf-8"))
        if parser.has_option("Unit", "StartLimitIntervalSec") and (
            parser["Unit"]["StartLimitIntervalSec"] == "0"
        ):
            forever.add(path.name)
    assert forever == {
        # Pre-existing and justified in their own files.
        "d435i.service",
        "ffs-ir-throttle.service",
        # H5, justified in its own file.
        "z-manip-planning-workbench.service",
    }


def test_both_launch_paths_diagnose_and_clear_a_tripped_start_limiter():
    """Nothing in this repository ever called reset-failed."""

    for script in (SERVO_LAUNCHER, TRANSPORT_PREFLIGHT):
        text = script.read_text(encoding="utf-8")
        assert "start-limit-hit" in text, script.name
        assert "reset-failed" in text, script.name
        assert "NRestarts" in text, script.name


def test_clearing_a_limiter_does_not_weaken_the_postcondition_that_gates_it():
    """The clear is a DIAGNOSIS aid, not a bypass.

    reset-failed only clears systemd's refusal to try.  Both call sites still
    verify the unit afterwards, so a genuinely broken unit still fails the
    launch -- it just says why instead of dying on a `set -e` non-zero.
    """

    servo = SERVO_LAUNCHER.read_text(encoding="utf-8")
    # The is-active postcondition and the passive-owner restore are intact.
    assert (
        "systemctl --user is-active --quiet z-mobile-manip-piper-reactive-view.service"
        in servo
    )
    assert servo.count("restore_passive") >= 3

    preflight = TRANSPORT_PREFLIGHT.read_text(encoding="utf-8")
    # The bounded readiness loop still decides the verdict.
    assert 'for _ in $(seq 1 24); do' in preflight
    assert "Go2W transport preflight failed" in preflight
    # And the restart itself is now checked instead of aborting on `set -e`.
    assert 'if ! "${SSH[@]}" "systemctl --user restart' in preflight
    # The clear is scoped to the limiter, never unconditional.
    assert 'if [[ "$limiter" == start-limit-hit ]]; then' in preflight


# ---------------------------------------------------------------------------
# R9.  A capture whose window closed before its own bundle wait began.
# ---------------------------------------------------------------------------


def test_the_dry_run_rejects_a_window_that_closed_before_its_bundle_wait():
    source = DRY_RUN.read_text(encoding="utf-8")
    # A unix anchor is taken with the monotonic one: the passive report's
    # interval is unix-epoch nanoseconds and bundle_wait_started is not.
    assert "bundle_wait_started_unix_ns = time.time_ns()" in source
    assert "stale_passive_window_reason(" in source
    # The oldest bundle in hand -- not a guessed constant -- is what proves no
    # bundle can overlap.
    assert "oldest_bundle_stamp_ns=min(common)," in source
    # It becomes a NAMED failure, bounded by its own grace.
    assert "PASSIVE_WINDOW_STALE_GRACE_S" in source
    assert 'perception_failure = stale_reason' in source
    # And it is reported on BOTH the failure and the success report, so a run
    # that recovered inside the grace is still an uncensored sample.
    assert source.count('"stale_passive_window": stale_passive_window_detail,') == 2


def test_the_stale_window_verdict_never_widens_the_zero_tx_gate():
    """Fix by RE-CAPTURE, never by relaxation.

    The brief's explicit prohibition: the passive-window gate is the pipeline's
    zero-TX evidence, and dropping it starts base motion on a bundle the
    pipeline could not validate.
    """

    source = DRY_RUN.read_text(encoding="utf-8")
    # The overlap gate's own edges are untouched.
    assert "capture.start_unix_ns - 250_000_000" in source
    assert "capture.end_unix_ns + 250_000_000" in source
    # The stale verdict is reached only AFTER the gate has already rejected
    # every supported bundle, i.e. it can never admit one the gate refused.
    gate_at = source.index("passive_window_rejections += 1")
    stale_at = source.index("stale_reason = stale_passive_window_reason(")
    assert gate_at < stale_at
