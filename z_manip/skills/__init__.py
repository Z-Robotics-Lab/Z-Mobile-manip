"""L3 skill layer — find / approach / align / pick / carry / place.

Each skill is a composable unit with four fixed parts (``docs/plan.md`` §3):

    entry condition · action · a DETERMINISTIC verify predicate · timeout
    (sim-s) + retry / degrade budget (§3a).

Skills only emit ROS2 actions/topics and drive L2 primitives; they contain no
algorithm. They map onto the skill state machine
SEARCH→APPROACH→ALIGN→GRASP→VERIFY→CARRY→PLACE→RECOVER; every timeout is on sim
time (RTF 0.2 amplifies wall-clock 5× — §3, pitfall 41).

Skills (see each module):

- :mod:`z_manip.skills.find`     — find(X): SEARCH; scan + detect + VLM →
                                 stable 3D pose.
- :mod:`z_manip.skills.approach` — approach(X): two-stage servo to standoff.
- :mod:`z_manip.skills.align`    — align(X): lock EdgeTAM mask + yaw-align + base
                                 pose gate (|pitch|≤12°, |roll|≤10°).
- :mod:`z_manip.skills.pick`     — pick(X): grasp candidate → IK/plan filter →
                                 MoveIt2-RRT → Cartesian sting → close → lift.
- :mod:`z_manip.skills.carry`    — carry: CARRY pose + navigate to destination,
                                 aperture held >0 throughout.
- :mod:`z_manip.skills.place`    — place(X): place pose → lower → release →
                                 hand-off; verify object no longer follows.

The verify predicate is the moat (invariant 1): it reads ground truth the actor
cannot author, and M0-M3 record only FREE debug signals — never a completion /
acceptance verdict.

Skeleton only (M0). Every skill raises ``NotImplementedError``.
"""

from __future__ import annotations

__all__: list[str] = []
