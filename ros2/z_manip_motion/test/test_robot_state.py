import pytest

from z_manip_motion.contracts import ContractError
from z_manip_motion.robot_state import (
    ClockHandoverGuard,
    CompleteJointStateAssembler,
    movable_joint_names_from_urdf,
)


URDF = """
<robot name="mobile_manipulator">
  <joint name="fixed_mount" type="fixed"/>
  <joint name="leg" type="revolute"/>
  <joint name="wheel" type="continuous"/>
  <joint name="arm" type="prismatic"/>
  <joint name="finger_copy" type="prismatic"><mimic joint="arm"/></joint>
</robot>
"""


def update(
    assembler,
    names,
    positions,
    *,
    source="state",
    stamp_s=1.0,
    received_at=1.0,
    reference_stamp_s=None,
):
    return assembler.update(
        names,
        positions,
        source=source,
        stamp_ns=round(stamp_s * 1_000_000_000),
        received_at=received_at,
        reference_stamp_ns=(
            None
            if reference_stamp_s is None
            else round(reference_stamp_s * 1_000_000_000)
        ),
    )


def test_required_state_is_derived_from_independent_movable_urdf_joints():
    assert movable_joint_names_from_urdf(URDF) == ("leg", "wheel", "arm")


def test_assembler_requires_every_joint_and_preserves_urdf_order():
    assembler = CompleteJointStateAssembler(
        movable_joint_names_from_urdf(URDF),
        max_age_s=0.25,
    )
    update(
        assembler,
        ["arm"],
        [0.3],
        source="arm",
        stamp_s=10.0,
        received_at=1.0,
    )
    update(
        assembler,
        ["unused", "wheel", "leg"],
        [9.0, 0.2, 0.1],
        source="platform",
        stamp_s=10.1,
        received_at=1.1,
    )

    complete = assembler.snapshot(now=1.2)
    assert complete is not None
    assert complete.names == ("leg", "wheel", "arm")
    assert complete.positions == pytest.approx((0.1, 0.2, 0.3))
    assert complete.stamp_ns == 10_000_000_000


def test_assembler_fails_closed_for_missing_stale_or_malformed_state():
    assembler = CompleteJointStateAssembler(("leg", "arm"), max_age_s=0.25)
    update(assembler, ["arm"], [0.3], source="arm", received_at=1.0)
    assert assembler.readiness(now=1.1).missing == ("leg",)
    assert assembler.snapshot(now=1.1) is None

    update(assembler, ["leg"], [0.1], source="platform", received_at=1.1)
    assert assembler.readiness(now=1.3).stale == ("arm",)
    assert assembler.snapshot(now=1.3) is None

    with pytest.raises(ContractError, match="duplicate"):
        update(assembler, ["arm", "arm"], [0.1, 0.2], received_at=1.3)
    with pytest.raises(ContractError, match="non-finite"):
        update(assembler, ["arm"], [float("nan")], received_at=1.3)

    bounded = CompleteJointStateAssembler(
        ("arm",),
        max_age_s=0.25,
        expected_sources=("arm",),
    )
    with pytest.raises(ContractError, match="unexpected"):
        update(bounded, ["arm"], [0.1], source="other")


def test_assembler_uses_oldest_coherent_input_stamp_and_only_advances():
    assembler = CompleteJointStateAssembler(
        ("leg", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.25,
    )
    update(
        assembler,
        ["arm"],
        [0.3],
        source="arm",
        stamp_s=10.0,
        received_at=1.0,
    )
    update(
        assembler,
        ["leg"],
        [0.1],
        source="platform",
        stamp_s=10.1,
        received_at=1.01,
    )
    first = assembler.next_snapshot(now=1.02)
    assert first is not None
    assert first.stamp_ns == 10_000_000_000

    # A slower source advances the conservative snapshot stamp and can publish
    # immediately without making its sample look newer than it is.
    update(
        assembler,
        ["arm"],
        [0.4],
        source="arm",
        stamp_s=10.05,
        received_at=1.03,
    )
    arm_update = assembler.next_snapshot(now=1.03)
    assert arm_update is not None
    assert arm_update.stamp_ns == 10_050_000_000
    update(
        assembler,
        ["leg"],
        [0.2],
        source="platform",
        stamp_s=10.15,
        received_at=1.04,
    )
    assert assembler.next_snapshot(now=1.04) is None
    update(
        assembler,
        ["arm"],
        [0.5],
        source="arm",
        stamp_s=10.2,
        received_at=1.05,
    )
    second = assembler.next_snapshot(now=1.05)
    assert second is not None
    assert second.stamp_ns == 10_150_000_000
    assert second.positions == pytest.approx((0.2, 0.5))


def test_assembler_blocks_unstamped_or_incoherent_measurements():
    assembler = CompleteJointStateAssembler(
        ("leg", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.1,
    )
    update(
        assembler,
        ["arm"],
        [0.3],
        source="arm",
        stamp_s=0.0,
        received_at=1.0,
    )
    update(
        assembler,
        ["leg"],
        [0.1],
        source="platform",
        stamp_s=1.0,
        received_at=1.0,
    )
    assert assembler.readiness(now=1.01).unstamped == ("arm",)
    assert assembler.next_snapshot(now=1.01) is None

    update(
        assembler,
        ["arm"],
        [0.3],
        source="arm",
        stamp_s=0.5,
        received_at=1.02,
    )
    assert assembler.readiness(now=1.02).inconsistent == ("arm",)
    assert assembler.next_snapshot(now=1.02) is None


def test_sim_assembler_waits_for_first_explicit_clock_sample():
    assembler = CompleteJointStateAssembler(
        ("arm",),
        max_age_s=0.5,
        require_clock=True,
    )
    assert not update(assembler, ["arm"], [1.0], stamp_s=10.0)
    assert assembler.snapshot(now=1.0) is None

    assert not assembler.observe_clock(10_000_000_000)
    assert update(assembler, ["arm"], [1.0], stamp_s=10.0)
    assert assembler.snapshot(now=1.0) is not None


def test_real_assembler_rejects_stale_future_and_non_ros_time_stamps():
    assembler = CompleteJointStateAssembler(("arm",), max_age_s=0.25)
    system_time_s = 1_800_000_000.0

    assert not update(
        assembler,
        ["arm"],
        [1.0],
        stamp_s=system_time_s - 3600.0,
        reference_stamp_s=system_time_s,
    )
    assert not update(
        assembler,
        ["arm"],
        [2.0],
        stamp_s=system_time_s + 1.0,
        reference_stamp_s=system_time_s,
    )
    assert not update(
        assembler,
        ["arm"],
        [3.0],
        stamp_s=1234.0,
        reference_stamp_s=system_time_s,
    )
    assert assembler.snapshot(now=1.0) is None

    assert update(
        assembler,
        ["arm"],
        [4.0],
        stamp_s=system_time_s,
        reference_stamp_s=system_time_s,
    )
    complete = assembler.snapshot(now=1.0)
    assert complete is not None
    assert complete.positions == pytest.approx((4.0,))


def test_assembler_recovers_after_explicit_ros_clock_reset_without_mixing_epochs():
    assembler = CompleteJointStateAssembler(
        ("leg", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.1,
        expected_sources=("arm", "platform"),
    )
    assert not assembler.observe_clock(100_100_000_000)
    update(
        assembler,
        ["arm"],
        [0.3],
        source="arm",
        stamp_s=100.0,
        received_at=1.0,
    )
    update(
        assembler,
        ["leg"],
        [0.1],
        source="platform",
        stamp_s=100.05,
        received_at=1.0,
    )
    before_reset = assembler.next_snapshot(now=1.0)
    assert before_reset is not None
    assert before_reset.epoch == 0

    assert assembler.observe_clock(0)
    assert assembler.snapshot(now=1.1) is None

    # The bounded post-reset quarantine rejects both early new data and late
    # old-epoch data while the long-lived readers wait for a coherent clock.
    assert not update(
        assembler,
        ["arm"],
        [0.4],
        source="arm",
        stamp_s=0.01,
        received_at=1.1,
    )
    assert not update(
        assembler,
        ["leg"],
        [9.9],
        source="platform",
        stamp_s=100.1,
        received_at=1.105,
    )
    # Forward clock data from the retired writer is ignored during handover.
    assert not assembler.observe_clock(100_250_000_000)
    assert not update(
        assembler,
        ["arm"],
        [99.0],
        source="arm",
        stamp_s=100.2,
        received_at=1.106,
    )
    assembler.resume_clock(110_000_000)
    assert update(
        assembler,
        ["arm"],
        [0.4],
        source="arm",
        stamp_s=0.10,
        received_at=1.11,
    )
    assert update(
        assembler,
        ["leg"],
        [0.2],
        source="platform",
        stamp_s=0.11,
        received_at=1.11,
    )
    after_reset = assembler.next_snapshot(now=1.11)
    assert after_reset is not None
    assert after_reset.epoch == 1
    assert after_reset.stamp_ns == 100_000_000
    assert after_reset.positions == pytest.approx((0.2, 0.4))

    # Coherent delayed old frames cannot revive the old epoch after recovery.
    assert not update(
        assembler,
        ["arm"],
        [99.0],
        source="arm",
        stamp_s=100.1,
        received_at=1.12,
    )
    assert not update(
        assembler,
        ["leg"],
        [99.0],
        source="platform",
        stamp_s=100.15,
        received_at=1.12,
    )
    preserved = assembler.snapshot(now=1.12)
    assert preserved is not None
    assert preserved.positions == pytest.approx((0.2, 0.4))


def test_assembler_does_not_treat_one_regressed_source_as_a_clock_reset():
    assembler = CompleteJointStateAssembler(
        ("leg", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.25,
        expected_sources=("arm", "platform"),
    )
    update(assembler, ["arm"], [1.0], source="arm", stamp_s=0.20)
    update(assembler, ["leg"], [2.0], source="platform", stamp_s=0.25)
    initial = assembler.next_snapshot(now=1.0)
    assert initial is not None

    assert not update(
        assembler,
        ["arm"],
        [10.0],
        source="arm",
        stamp_s=0.10,
        received_at=1.01,
    )
    update(
        assembler,
        ["leg"],
        [2.6],
        source="platform",
        stamp_s=0.26,
        received_at=1.02,
    )
    current = assembler.snapshot(now=1.02)
    assert current is not None
    assert current.epoch == 0
    assert current.positions == pytest.approx((2.6, 1.0))

    # A later valid monotonic sample resumes the source in the same epoch.
    assert update(
        assembler,
        ["arm"],
        [1.1],
        source="arm",
        stamp_s=0.21,
        received_at=1.03,
    )
    resumed = assembler.next_snapshot(now=1.03)
    assert resumed is not None
    assert resumed.epoch == 0
    assert resumed.stamp_ns == 210_000_000
    assert resumed.positions == pytest.approx((2.6, 1.1))


def test_assembler_switches_small_clock_rollback_atomically():
    assembler = CompleteJointStateAssembler(
        ("leg", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.25,
        expected_sources=("arm", "platform"),
    )
    assembler.observe_clock(250_000_000)
    update(assembler, ["arm"], [1.0], source="arm", stamp_s=0.20)
    update(assembler, ["leg"], [2.0], source="platform", stamp_s=0.25)
    assert assembler.next_snapshot(now=1.0) is not None

    assert assembler.observe_clock(220_000_000)
    assert not assembler.observe_clock(240_000_000)
    assert not update(
        assembler,
        ["arm"],
        [10.0],
        source="arm",
        stamp_s=0.23,
        received_at=1.01,
    )
    assert assembler.snapshot(now=1.01) is None

    assembler.resume_clock(260_000_000)
    assert update(
        assembler,
        ["arm"],
        [10.0],
        source="arm",
        stamp_s=0.25,
        received_at=1.02,
    )
    assert update(
        assembler,
        ["leg"],
        [20.0],
        source="platform",
        stamp_s=0.26,
        received_at=1.02,
    )
    reset = assembler.next_snapshot(now=1.02)
    assert reset is not None
    assert reset.epoch == 1
    assert reset.stamp_ns == 250_000_000
    assert reset.positions == pytest.approx((20.0, 10.0))


def test_assembler_accumulates_partial_sources_after_clock_reset():
    assembler = CompleteJointStateAssembler(
        ("leg", "wheel", "arm"),
        max_age_s=0.5,
        max_stamp_skew_s=0.1,
        expected_sources=("arm", "platform"),
    )
    assembler.observe_clock(1_000_000_000)
    update(assembler, ["arm"], [1.0], source="arm", stamp_s=1.0)
    update(
        assembler,
        ["leg", "wheel"],
        [2.0, 3.0],
        source="platform",
        stamp_s=1.0,
    )
    assert assembler.next_snapshot(now=1.0) is not None

    assert assembler.observe_clock(0)
    assembler.resume_clock(110_000_000)
    assert update(assembler, ["leg"], [20.0], source="platform", stamp_s=0.10)
    assert update(assembler, ["wheel"], [30.0], source="platform", stamp_s=0.10)
    assert update(assembler, ["arm"], [10.0], source="arm", stamp_s=0.11)
    complete = assembler.next_snapshot(now=1.1)
    assert complete is not None
    assert complete.epoch == 1
    assert complete.positions == pytest.approx((20.0, 30.0, 10.0))


def test_assembler_handles_repeated_explicit_clock_handover():
    assembler = CompleteJointStateAssembler(
        ("arm",),
        max_age_s=0.5,
        max_stamp_skew_s=0.1,
        require_clock=True,
    )
    assembler.observe_clock(10_000_000_000)
    assert update(assembler, ["arm"], [1.0], stamp_s=10.0)
    assert assembler.next_snapshot(now=1.0).epoch == 0

    assert assembler.observe_clock(0)
    assembler.resume_clock(110_000_000)
    assert update(assembler, ["arm"], [2.0], stamp_s=0.11)
    assert assembler.next_snapshot(now=1.1).epoch == 1

    assert assembler.observe_clock(10_000_000)
    assembler.resume_clock(120_000_000)
    assert update(assembler, ["arm"], [3.0], stamp_s=0.12)
    final = assembler.next_snapshot(now=1.2)
    assert final is not None
    assert final.epoch == 2
    assert final.positions == pytest.approx((3.0,))


def test_clock_handover_rejects_old_new_epoch_interleaving_until_stable():
    guard = ClockHandoverGuard(quiet_s=0.5)
    old_gid = (1,)
    new_gid = (2,)
    guard.begin(30_000_000, now=0.0)

    assert not guard.observe(
        100_200_000_000,
        now=0.6,
        publisher_gids={old_gid, new_gid},
    )
    assert not guard.observe(40_000_000, now=0.7, publisher_gids={new_gid})
    assert not guard.observe(50_000_000, now=1.21, publisher_gids={new_gid})
    assert guard.observe(60_000_000, now=1.22, publisher_gids={new_gid})
    guard.finish()
    assert not guard.pending


def test_clock_handover_pause_needs_two_post_pause_progress_samples():
    guard = ClockHandoverGuard(quiet_s=0.5)
    gid = (7, 8)
    guard.begin(30, now=0.0)

    assert not guard.observe(40, now=2.0, publisher_gids={gid})
    assert not guard.observe(50, now=2.51, publisher_gids={gid})
    assert guard.observe(60, now=2.52, publisher_gids={gid})


def test_clock_handover_publisher_change_restarts_coherence_window():
    guard = ClockHandoverGuard(quiet_s=0.5)
    guard.begin(10, now=0.0)
    assert not guard.observe(20, now=0.1, publisher_gids={(1,)})
    assert not guard.observe(30, now=0.7, publisher_gids={(2,)})
    assert not guard.observe(40, now=1.21, publisher_gids={(2,)})
    assert guard.observe(50, now=1.22, publisher_gids={(2,)})


def test_clock_handover_requires_one_publisher_and_valid_inputs():
    guard = ClockHandoverGuard(quiet_s=0.5)
    guard.begin(10, now=0.0)
    assert not guard.observe(20, now=1.0, publisher_gids=set())
    assert not guard.observe(30, now=2.0, publisher_gids={(1,), (2,)})
    assert not guard.observe(40, now=3.0, publisher_gids={()})
    with pytest.raises(ContractError):
        guard.observe(-1, now=2.1, publisher_gids={(1,)})
    with pytest.raises(ContractError):
        ClockHandoverGuard(quiet_s=0.0)


def test_clock_handover_regression_after_arming_restarts_quiet_window():
    guard = ClockHandoverGuard(quiet_s=0.5)
    gid = (9,)
    guard.begin(10, now=0.0)
    assert not guard.observe(20, now=0.1, publisher_gids={gid})
    assert not guard.observe(30, now=0.61, publisher_gids={gid})
    assert not guard.observe(25, now=0.62, publisher_gids={gid})
    assert not guard.observe(35, now=1.13, publisher_gids={gid})
    assert guard.observe(45, now=1.14, publisher_gids={gid})


def test_clock_handover_completion_requires_a_fresh_graph_probe():
    guard = ClockHandoverGuard(quiet_s=0.5)
    gid = (4,)
    guard.begin(10, now=0.0)
    assert not guard.observe(20, now=0.1, publisher_gids={gid})
    assert not guard.observe(30, now=0.61, publisher_gids={gid})
    assert not guard.observe(40, now=0.62, publisher_gids=None)
    assert guard.observe(50, now=0.63, publisher_gids={gid})


def test_multidof_urdf_requires_an_explicit_state_adapter():
    with pytest.raises(ContractError, match="floating"):
        movable_joint_names_from_urdf(
            '<robot name="r"><joint name="base" type="floating"/></robot>',
        )
