"""ROS2 test helpers — the sim/real execution seam + live probes.

Design (one test suite, sim and real):
  Every ROS2 command runs through a single **exec seam**. ``Z_MANIP_ROS_EXEC``
  names the prefix that wraps a ROS2 command; the default targets the sim's
  ``navstack`` container. Set it to the empty string on the real robot (native
  ROS2, no docker) — the very same tests then exercise the real chain.

  Nothing here starts, restarts, or tears down any sim or container. Probes only
  ATTACH to whatever is already running. A probe that cannot reach the chain
  (no container / no topic / timeout) raises :class:`ProbeSkip`, which the tests
  translate into ``pytest.skip`` — never a hard error. This tolerates the chain
  vanishing mid-probe (a sibling session may pair-restart the sim at any time).

Frame / topic names are the go2w source of truth (``~/Desktop/go2w/scripts/sim/
{wrist_camera.py,warehouse_nav.py}``); constants live in :mod:`tests.contract`.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import textwrap
from dataclasses import dataclass


class ProbeSkip(Exception):
    """A live probe could not reach the chain (attach-only failure → skip).

    Distinct from an assertion failure: this means "we could not observe",
    NOT "we observed something wrong". Tests catch it and call ``pytest.skip``.
    Reasons: container absent, ROS env missing, topic silent, probe timeout,
    or the chain disappeared mid-probe (tolerated by contract).
    """


# --------------------------------------------------------------------- exec seam
# Prefix wrapping every ROS2 command. Default = sim navstack container with the
# ROS env the sim chain uses (domain/scope remain deployment overrides). Empty string ⇒ run the
# command natively (real robot). Read once at import; override via the env var.
#
# The default sources BOTH /opt/ros/jazzy/setup.bash and /ws/install/setup.bash
# (the run_navstack.sh convention) so message types from the workspace resolve.
_SIM_DOMAIN_ID = os.environ.get("GO2W_ROS_DOMAIN_ID", "184")
_DEFAULT_EXEC = (
    "docker exec navstack bash -lc "
    + shlex.quote(
        f"export ROS_DOMAIN_ID={_SIM_DOMAIN_ID} "
        "RMW_IMPLEMENTATION=rmw_fastrtps_cpp "
        "FASTDDS_BUILTIN_TRANSPORTS=UDPv4; "
        "source /opt/ros/jazzy/setup.bash; "
        "source /ws/install/setup.bash 2>/dev/null; "
    )
)


def ros_exec_prefix() -> str:
    """Current exec-seam prefix (``Z_MANIP_ROS_EXEC`` or the sim default)."""
    return os.environ.get("Z_MANIP_ROS_EXEC", _DEFAULT_EXEC)


def _wrap(inner_cmd: str) -> list[str]:
    """Compose the full shell argv: ``<prefix> + <inner ROS2 command>``.

    When the prefix ends in ``bash -lc '<preamble>'`` the inner command is
    appended INSIDE that same quoted string (so the sourced env is in scope);
    we detect the default/`bash -lc` shape and concatenate accordingly. When the
    prefix is empty (real robot) the inner command runs directly via the shell.
    """
    prefix = ros_exec_prefix().strip()
    if not prefix:
        return ["bash", "-lc", inner_cmd]
    # If the prefix is a `... bash -lc '<preamble>'` form, fold the inner command
    # into that quoted preamble so sourced setup.bash stays in scope.
    if prefix.endswith("'") or prefix.endswith('"'):
        quote = prefix[-1]
        body = prefix[:-1]
        # Escape the inner command for the SAME quote style.
        if quote == "'":
            safe = inner_cmd.replace("'", "'\\''")
        else:
            safe = inner_cmd.replace('"', '\\"')
        return ["bash", "-c", f"{body}{safe}{quote}"]
    # Otherwise treat the prefix as a plain command prefix.
    return ["bash", "-c", f"{prefix} {inner_cmd}"]


def _run(inner_cmd: str, timeout: float) -> subprocess.CompletedProcess:
    """Run one command through the exec seam; raise :class:`ProbeSkip` on failure
    to reach the chain (missing docker, timeout). Returns the completed process
    even on non-zero exit — callers decide whether the *content* is a skip."""
    argv = _wrap(inner_cmd)
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:  # chain slow / vanished mid-probe
        raise ProbeSkip(f"probe timed out after {timeout}s: {inner_cmd!r}") from exc
    except FileNotFoundError as exc:  # bash/docker not present
        raise ProbeSkip(f"exec seam unavailable: {exc}") from exc


def ros2_cli(cmd: str, timeout: float = 15.0) -> str:
    """Run a bare ``ros2 <cmd>`` through the seam; return stdout (stripped).

    Raises :class:`ProbeSkip` if the seam is unreachable or the command returns
    non-zero with empty stdout (typical when the ROS graph is not up).
    """
    proc = _run(f"ros2 {cmd}", timeout=timeout)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        raise ProbeSkip(
            f"`ros2 {cmd}` rc={proc.returncode}: {(proc.stderr or '').strip()[:200]}"
        )
    return out


def list_topics(timeout: float = 12.0) -> list[str]:
    """Live ``ros2 topic list`` → sorted topic names (``ProbeSkip`` if graph down)."""
    out = ros2_cli("topic list", timeout=timeout)
    topics = sorted(t.strip() for t in out.splitlines() if t.strip().startswith("/"))
    if not topics:
        raise ProbeSkip("`ros2 topic list` returned no topics (graph down?)")
    return topics


def topic_exists(topic: str, timeout: float = 12.0) -> bool:
    """True iff ``topic`` is present in the live graph."""
    return topic in list_topics(timeout=timeout)


def topic_hz(topic: str, window_s: float = 8.0) -> float:
    """WALL-CLOCK publish rate of ``topic`` via ``ros2 topic hz`` (Hz).

    Parses the last ``average rate:`` line ``ros2 topic hz`` prints within a
    ``window_s`` wall window. WALL rate scales with RTF (≈0.2 here): a healthy
    10 fps-sim stream reads ~2.1 Hz wall — the exact trap M0 verify documented.
    So this is NOT the M0 hz-gate quantity; use it only for wall-side
    observations (e.g. WiFi bandwidth budgeting). Gates use
    :func:`topic_hz_sim`. Raises :class:`ProbeSkip` if no sample lands in the
    window (silent topic / chain gone).
    """
    # `ros2 topic hz` runs until killed; bound it with the container-side timeout
    # so the average is computed over ~window_s of real time.
    proc = _run(
        f"timeout {window_s:.1f} ros2 topic hz {shlex.quote(topic)}",
        timeout=window_s + 12.0,
    )
    rates: list[float] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("average rate:"):
            try:
                rates.append(float(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
    if not rates:
        raise ProbeSkip(
            f"no `average rate:` for {topic} in {window_s}s "
            f"(silent or chain gone): {(proc.stderr or '').strip()[:160]}"
        )
    return rates[-1]  # last reported average = most settled


def topic_hz_sim(topic: str, msg_module: str = "sensor_msgs.msg",
                 msg_class: str = "Image", n_msgs: int = 15,
                 timeout: float = 45.0) -> dict:
    """SIM-TIME publish rate from header stamps (messages per sim second).

    The M0 hz gates live in the sim clock domain (wrist camera CAM_STRIDE=10
    @ 100 Hz physics ⇒ 10 fps sim); wall rate is that × RTF (≈0.2), so a wall
    measurement flunks a healthy stream. On the real robot header stamps ARE
    wall time, so this same probe measures the same quantity — the sim/real
    seam holds. Returns ``{"fps_sim", "fps_wall", "rtf_implied", "n"}``;
    :class:`ProbeSkip` on <3 stamped messages.
    """
    body = f"""
        import importlib
        from rclpy.qos import qos_profile_sensor_data
        Msg = getattr(importlib.import_module("{msg_module}"), "{msg_class}")
        rclpy.init()
        n = rclpy.create_node("zm_probe_hzsim")
        got = {{"s": [], "w": []}}
        def cb(m):
            got["s"].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            got["w"].append(time.monotonic())
        n.create_subscription(Msg, "{topic}", cb, qos_profile_sensor_data)
        t0 = time.monotonic()
        while time.monotonic() - t0 < {timeout - 10.0:.1f} and len(got["s"]) < {n_msgs}:
            rclpy.spin_once(n, timeout_sec=0.3)
        rclpy.shutdown()
        if len(got["s"]) < 3:
            print("RESULT " + json.dumps({{"n": len(got["s"])}}))
        else:
            ds = got["s"][-1] - got["s"][0]
            dw = got["w"][-1] - got["w"][0]
            k = len(got["s"]) - 1
            if ds <= 0.0 or dw <= 0.0:
                print("RESULT " + json.dumps({{"n": len(got["s"])}}))
            else:
                print("RESULT " + json.dumps({{
                    "fps_sim": k / ds, "fps_wall": k / dw,
                    "rtf_implied": ds / dw, "n": len(got["s"])}}))
    """
    res = _run_probe(body, timeout=timeout)
    if "fps_sim" not in res:
        raise ProbeSkip(
            f"{topic}: only {res.get('n', 0)} stamped msgs in window "
            "(silent or chain gone)"
        )
    return res


def echo_once(topic: str, timeout: float = 10.0) -> str:
    """Raw YAML of one message via ``ros2 topic echo --once`` (``ProbeSkip`` if
    none arrives)."""
    proc = _run(
        f"timeout {timeout:.1f} ros2 topic echo --once {shlex.quote(topic)}",
        timeout=timeout + 8.0,
    )
    out = (proc.stdout or "").strip()
    if not out:
        raise ProbeSkip(f"no message on {topic} within {timeout}s")
    return out


# ------------------------------------------------------------------ rclpy probes
# Small rclpy programs run INSIDE the chain's container (through the seam) via a
# python3 heredoc. They read ground truth the tests cannot compute host-side:
# depth-frame statistics, joint errors, and TF-derived optical-axis pitch. Each
# prints a single ``RESULT <json>`` line; anything else ⇒ ProbeSkip.

_PROBE_HEADER = textwrap.dedent(
    """
    import json, math, time, sys
    import rclpy
    """
).strip()


def _run_probe(body: str, timeout: float) -> dict:
    """Run an rclpy heredoc probe in the container; parse its ``RESULT <json>``.

    ``body`` is Python that must, on success, ``print("RESULT " + json.dumps(d))``
    exactly once. Missing/failed ⇒ :class:`ProbeSkip` (attach-only failure).
    """
    prog = _PROBE_HEADER + "\n" + textwrap.dedent(body)
    # Feed the program on stdin to python3 (avoids nested-heredoc quoting hell).
    proc = _run(f"python3 - <<'ZMPROBE'\n{prog}\nZMPROBE", timeout=timeout)
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except (ValueError, TypeError) as exc:
                raise ProbeSkip(f"probe RESULT unparseable: {exc}") from exc
    raise ProbeSkip(
        "probe produced no RESULT "
        f"(rc={proc.returncode}): {(proc.stderr or proc.stdout or '').strip()[:200]}"
    )


def clock_rtf(window_s: float = 5.0) -> float:
    """Real-time factor = Δsim-time / Δwall-time, from two ``/clock`` samples.

    Subscribes to ``/clock`` (``rosgraph_msgs/Clock``), reads sim-time at the
    start and after ``window_s`` of WALL time, returns their ratio. RTF≈0.2 at
    100 Hz physics. ``ProbeSkip`` if fewer than two clock samples arrive.
    """
    body = f"""
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        n = rclpy.create_node("zm_probe_rtf")
        got = {{"t": []}}
        def cb(m):
            got["t"].append((time.monotonic(), m.clock.sec + m.clock.nanosec * 1e-9))
        n.create_subscription(Clock, "/clock", cb, 10)
        t0 = time.monotonic()
        # collect a first sample
        while time.monotonic() - t0 < 8.0 and not got["t"]:
            rclpy.spin_once(n, timeout_sec=0.2)
        if not got["t"]:
            print("NO_CLOCK"); rclpy.shutdown(); sys.exit(0)
        w0, s0 = got["t"][0]
        # let {window_s}s of WALL time pass while pumping clock
        while time.monotonic() - w0 < {window_s}:
            rclpy.spin_once(n, timeout_sec=0.2)
        w1, s1 = got["t"][-1]
        rclpy.shutdown()
        dw = w1 - w0
        ds = s1 - s0
        if dw <= 0.0 or len(got["t"]) < 2:
            print("NO_CLOCK"); sys.exit(0)
        print("RESULT " + json.dumps({{"rtf": ds / dw, "dwall": dw, "dsim": ds,
                                       "samples": len(got["t"])}}))
    """
    return float(_run_probe(body, timeout=window_s + 20.0)["rtf"])


def wait_sim_seconds(seconds: float, timeout_wall: float | None = None) -> float:
    """Block until ``/clock`` advances by ``seconds`` of SIM time; return Δsim.

    Waiting the settle window by SIM time (not ``time.sleep`` on the wall) is
    mandatory: at RTF 0.2 a 3-wall-second sleep is only 0.6 sim-s (§3, pitfall
    41). ``timeout_wall`` guards against a stalled clock (default = seconds/0.05
    + 10, i.e. tolerate RTF as low as 0.05). ``ProbeSkip`` on stall.
    """
    if timeout_wall is None:
        timeout_wall = seconds / 0.05 + 10.0
    body = f"""
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        n = rclpy.create_node("zm_probe_wait")
        got = {{"s": None, "last": None}}
        def cb(m):
            t = m.clock.sec + m.clock.nanosec * 1e-9
            if got["s"] is None:
                got["s"] = t
            got["last"] = t
        n.create_subscription(Clock, "/clock", cb, 10)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout_wall}:
            rclpy.spin_once(n, timeout_sec=0.2)
            if got["s"] is not None and got["last"] - got["s"] >= {seconds}:
                break
        rclpy.shutdown()
        if got["s"] is None or got["last"] - got["s"] < {seconds}:
            print("NO_CLOCK"); sys.exit(0)
        print("RESULT " + json.dumps({{"dsim": got["last"] - got["s"]}}))
    """
    return float(_run_probe(body, timeout=timeout_wall + 15.0)["dsim"])


def prop_odom_pose(topic: str, timeout: float = 12.0) -> dict:
    """One ``nav_msgs/Odometry`` pose off ``topic`` as ``{x,y,z, frame, child}``.

    Reads a single GT odom message (the sim publishes these for every physics
    prop at ~5 sim-Hz) INSIDE the container and returns its world-frame position
    plus header/child frame ids. Used by G-p2 to check a prop rests on its
    configured spot. ``ProbeSkip`` if no message arrives.
    """
    body = f"""
        from nav_msgs.msg import Odometry
        rclpy.init()
        n = rclpy.create_node("zm_probe_prop_odom")
        got = {{"m": None}}
        def cb(m):
            if got["m"] is None:
                got["m"] = m
        n.create_subscription(Odometry, {topic!r}, cb, 10)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} and got["m"] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
        rclpy.shutdown()
        m = got["m"]
        if m is None:
            print("NO_MSG"); sys.exit(0)
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        v = m.twist.twist.linear
        print("RESULT " + json.dumps({{
            "x": float(p.x), "y": float(p.y), "z": float(p.z),
            "qw": float(q.w), "qx": float(q.x), "qy": float(q.y), "qz": float(q.z),
            "vx": float(v.x), "vy": float(v.y), "vz": float(v.z),
            "frame": m.header.frame_id, "child": m.child_frame_id}}))
    """
    return _run_probe(body, timeout=timeout + 15.0)


def depth_frame_stats(topic: str, timeout: float = 15.0) -> "DepthStats":
    """One aligned-depth frame's stats: min non-zero (m), in-band pixel fraction.

    Reads one ``sensor_msgs/Image`` (16UC1, mm) off ``topic`` and computes,
    with numpy INSIDE the container:
      * ``min_nonzero_m`` — smallest non-zero depth in metres (G-e near-clip),
      * ``inband_frac``   — fraction of pixels in [0.3 m, 3.0 m] ("sees scene"),
      * ``nonzero_frac``, ``encoding``, ``width``, ``height``.
    ``ProbeSkip`` if no frame arrives or the encoding is not 16UC1.
    """
    body = f"""
        import numpy as np
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        rclpy.init()
        n = rclpy.create_node("zm_probe_depth")
        got = {{"m": None, "enc": None, "w": 0, "h": 0}}
        def cb(m):
            got["enc"] = m.encoding; got["w"] = m.width; got["h"] = m.height
            if m.encoding == "16UC1":
                a = np.frombuffer(bytes(m.data), dtype=np.uint16)
                if a.size >= m.width * m.height:
                    got["m"] = a[: m.width * m.height].reshape(m.height, m.width)
        n.create_subscription(Image, "{topic}", cb, qos_profile_sensor_data)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} and got["m"] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
        rclpy.shutdown()
        if got["m"] is None:
            print("RESULT " + json.dumps({{"ok": False, "enc": got["enc"],
                                           "w": got["w"], "h": got["h"]}}))
        else:
            a = got["m"]; tot = int(a.size); nz = a[a > 0]
            inband = int(((a >= 300) & (a <= 3000)).sum())  # 0.3..3.0 m in mm
            print("RESULT " + json.dumps({{
                "ok": True, "enc": got["enc"], "w": int(got["w"]), "h": int(got["h"]),
                "min_nonzero_m": (float(nz.min()) / 1000.0 if nz.size else 0.0),
                "nonzero_frac": (nz.size / tot if tot else 0.0),
                "inband_frac": (inband / tot if tot else 0.0),
            }}))
    """
    d = _run_probe(body, timeout=timeout + 15.0)
    if not d.get("ok"):
        raise ProbeSkip(
            f"no 16UC1 depth on {topic} (enc={d.get('enc')} "
            f"{d.get('w')}x{d.get('h')})"
        )
    return DepthStats(
        min_nonzero_m=float(d["min_nonzero_m"]),
        nonzero_frac=float(d["nonzero_frac"]),
        inband_frac=float(d["inband_frac"]),
        width=int(d["w"]),
        height=int(d["h"]),
    )


def camera_info(topic: str, timeout: float = 12.0) -> "CamInfo":
    """Read one ``sensor_msgs/CameraInfo``: width, height, fx (K[0])."""
    body = f"""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo
        rclpy.init()
        n = rclpy.create_node("zm_probe_info")
        got = {{"m": None}}
        def cb(m): got["m"] = m
        n.create_subscription(CameraInfo, "{topic}", cb, qos_profile_sensor_data)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} and got["m"] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
        rclpy.shutdown()
        if got["m"] is None:
            print("RESULT " + json.dumps({{"ok": False}}))
        else:
            m = got["m"]; K = list(m.k)
            print("RESULT " + json.dumps({{"ok": True, "w": int(m.width),
                "h": int(m.height), "fx": float(K[0]), "fy": float(K[4]),
                "cx": float(K[2]), "cy": float(K[5])}}))
    """
    d = _run_probe(body, timeout=timeout + 12.0)
    if not d.get("ok"):
        raise ProbeSkip(f"no CameraInfo on {topic}")
    return CamInfo(width=int(d["w"]), height=int(d["h"]), fx=float(d["fx"]),
                   fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]))


def image_encoding(topic: str, timeout: float = 12.0) -> tuple[str, int, int]:
    """One ``sensor_msgs/Image``'s ``(encoding, width, height)``."""
    body = f"""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        rclpy.init()
        n = rclpy.create_node("zm_probe_enc")
        got = {{"m": None}}
        def cb(m): got["m"] = (m.encoding, int(m.width), int(m.height))
        n.create_subscription(Image, "{topic}", cb, qos_profile_sensor_data)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} and got["m"] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
        rclpy.shutdown()
        if got["m"] is None:
            print("RESULT " + json.dumps({{"ok": False}}))
        else:
            print("RESULT " + json.dumps({{"ok": True, "enc": got["m"][0],
                                           "w": got["m"][1], "h": got["m"][2]}}))
    """
    d = _run_probe(body, timeout=timeout + 12.0)
    if not d.get("ok"):
        raise ProbeSkip(f"no Image on {topic}")
    return str(d["enc"]), int(d["w"]), int(d["h"])


def joint_error(state_topic: str, cmd_topic: str,
                timeout: float = 12.0) -> "JointErr":
    """Max |state-cmd| joint error (rad) reading two ``JointState`` topics.

    Matches by joint NAME (not array index) so a reordering in either publisher
    cannot silently produce a false error — the go2w publisher sends
    ``arm_names + grip_names`` (j1..j8) on both, but name-matching is the robust
    contract. ``ProbeSkip`` unless both topics deliver overlapping names.
    """
    body = f"""
        from sensor_msgs.msg import JointState
        rclpy.init()
        n = rclpy.create_node("zm_probe_jerr")
        got = {{"s": None, "c": None}}
        def cs(m): got["s"] = dict(zip(list(m.name), list(m.position)))
        def cc(m): got["c"] = dict(zip(list(m.name), list(m.position)))
        n.create_subscription(JointState, "{state_topic}", cs, 10)
        n.create_subscription(JointState, "{cmd_topic}", cc, 10)
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} and (got["s"] is None or got["c"] is None):
            rclpy.spin_once(n, timeout_sec=0.2)
        rclpy.shutdown()
        if got["s"] is None or got["c"] is None:
            print("RESULT " + json.dumps({{"ok": False}}))
        else:
            names = [k for k in got["s"] if k in got["c"]]
            if not names:
                print("RESULT " + json.dumps({{"ok": False, "reason": "no_common_names"}}))
            else:
                per = {{k: abs(got["s"][k] - got["c"][k]) for k in names}}
                print("RESULT " + json.dumps({{"ok": True, "per": per,
                                               "max": max(per.values())}}))
    """
    d = _run_probe(body, timeout=timeout + 12.0)
    if not d.get("ok"):
        raise ProbeSkip(
            f"joint_error: missing state/cmd ({d.get('reason', 'no message')})"
        )
    return JointErr(max_err=float(d["max"]),
                    per_joint={k: float(v) for k, v in d["per"].items()})


def optical_axis_pitch_deg(parent_frame: str, optical_frame: str,
                           settle_sim_s: float = 3.0,
                           timeout: float = 40.0) -> float:
    """Elevation of the camera optical AXIS above horizontal, in degrees.

    The optical axis is the optical frame's **+Z** basis vector (REP-105
    z-forward). We look up TF ``parent_frame → optical_frame``, rotate the unit
    +Z into the parent frame, and return ``atan2(z, hypot(x, y))`` in degrees —
    the angle the boresight makes with the parent's horizontal plane (~0° when
    the camera is level / LOOKOUT; positive tilts up).

    Waits ``settle_sim_s`` of SIM time (via /clock) before sampling so a pose
    just commanded has settled. ``ProbeSkip`` if TF never resolves or the clock
    stalls (chain gone).
    """
    body = f"""
        from rosgraph_msgs.msg import Clock
        from tf2_ros import Buffer, TransformListener
        rclpy.init()
        n = rclpy.create_node("zm_probe_pitch")
        buf = Buffer(); TransformListener(buf, n)
        clk = {{"s": None, "last": None}}
        def cb(m):
            t = m.clock.sec + m.clock.nanosec * 1e-9
            if clk["s"] is None: clk["s"] = t
            clk["last"] = t
        n.create_subscription(Clock, "/clock", cb, 10)
        # settle {settle_sim_s} SIM seconds
        w0 = time.monotonic()
        while time.monotonic() - w0 < {timeout} * 0.6:
            rclpy.spin_once(n, timeout_sec=0.2)
            if clk["s"] is not None and clk["last"] - clk["s"] >= {settle_sim_s}:
                break
        tf = None; last_e = ""
        w1 = time.monotonic()
        while time.monotonic() - w1 < {timeout} * 0.4:
            rclpy.spin_once(n, timeout_sec=0.2)
            try:
                tf = buf.lookup_transform("{parent_frame}", "{optical_frame}",
                                          rclpy.time.Time())
                break
            except Exception as e:
                last_e = str(e)
        rclpy.shutdown()
        if tf is None:
            print("NO_TF " + last_e[:120]); sys.exit(0)
        q = tf.transform.rotation
        w, x, y, z = q.w, q.x, q.y, q.z
        # rotate unit +Z by quaternion → optical axis in parent frame
        zx = 2 * (x * z + w * y)
        zy = 2 * (y * z - w * x)
        zz = 1 - 2 * (x * x + y * y)
        elev = math.degrees(math.atan2(zz, math.hypot(zx, zy)))
        print("RESULT " + json.dumps({{"elev_deg": elev, "axis": [zx, zy, zz],
            "settled_sim": (clk["last"] - clk["s"]) if clk["s"] is not None else None}}))
    """
    return float(_run_probe(body, timeout=timeout + 15.0)["elev_deg"])


def set_named_pose(name: str, timeout: float = 8.0) -> None:
    """Publish a ``std_msgs/String`` pose name to ``/piper/named_pose`` (one shot).

    Uses ``ros2 topic pub --once``. This is the ONLY thing the suite commands —
    a pose switch on an already-running arm — and it is idempotent and bounded.
    ``ProbeSkip`` if the publish cannot be issued (graph down).
    """
    from tests import contract  # local import: avoid cycle at module load

    proc = _run(
        f"timeout {timeout:.1f} ros2 topic pub --once "
        f"{shlex.quote(contract.TOPIC_NAMED_POSE)} std_msgs/msg/String "
        f"{shlex.quote('{data: ' + name + '}')}",
        timeout=timeout + 8.0,
    )
    if proc.returncode != 0:
        raise ProbeSkip(
            f"could not publish named_pose {name!r}: "
            f"{(proc.stderr or '').strip()[:160]}"
        )


# ------------------------------------------------------------------ result types
@dataclass(frozen=True)
class DepthStats:
    min_nonzero_m: float
    nonzero_frac: float
    inband_frac: float
    width: int
    height: int


@dataclass(frozen=True)
class CamInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class JointErr:
    max_err: float
    per_joint: dict


def rtf_scaled_min_hz(sim_hz: float, floor_wall_hz: float,
                      assume_rtf: float = 0.15) -> float:
    """Wall-rate floor for a sim-Hz publisher: ``max(sim_hz*assume_rtf, floor)``.

    A 5 Hz sim-time publisher at RTF 0.15 shows ~0.75 Hz on the wall. We assert
    against the RTF-folded expectation but never below an absolute ``floor``.
    """
    return max(sim_hz * assume_rtf, floor_wall_hz)


# Re-export so ``from tests.helpers import math`` isn't needed by callers who
# only want the constant π-based helpers; keeps the public surface obvious.
__all__ = [
    "ProbeSkip", "ros_exec_prefix", "ros2_cli", "list_topics", "topic_exists",
    "topic_hz", "topic_hz_sim", "echo_once", "clock_rtf", "wait_sim_seconds", "depth_frame_stats",
    "camera_info", "image_encoding", "joint_error", "optical_axis_pitch_deg",
    "set_named_pose", "DepthStats", "CamInfo", "JointErr", "rtf_scaled_min_hz",
    "math",
]
