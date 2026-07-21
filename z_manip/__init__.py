"""Z-Mobile-Manip — thin orchestration for mobile grasping.

Layered mobile-manipulation runtime for a Go2W wheel-legged base + a
back-mounted PiPER 6-DoF arm + a wrist D435i. This package is the **thin
orchestration layer only**: it speaks a ROS2 contract (topics / actions +
an agent-bridge HTTP face) and NEVER imports Isaac. Platform differences are
isolated in :mod:`z_manip.adapters` (``isaac_adapter`` ↔ ``real_adapter``), so
sim→real migration is a DDS-config change, not a code change.

Layer map (design authority: ``docs/plan.md`` §2 L0-L4):

- :mod:`z_manip.skills`      L3 — find/approach/align/pick/carry/place; each
                            skill = entry condition + action + a deterministic
                            verify predicate + timeout (sim-s) + retry budget.
- :mod:`z_manip.primitives`  L2 — scan/servo_base/arm_goto/track/grasp_exec;
                            ROS2 action contracts, no algorithms.
- :mod:`z_manip.models`      L1 — Detector / VLM / Tracker / GraspSource /
                            Planner, all hidden behind replaceable Protocols.
- :mod:`z_manip.ik`          IK near-limit four-stage pipeline (symmetry-expand
                            → reachability filter → solve → DLS fallback).
- :mod:`z_manip.adapters`    L0 — isaac_adapter (consumes topics only) /
                            real_adapter (D435i + piper_sdk).

This module is a **skeleton** (M0 bootstrap): the code below defines only the
contracts (Protocols / ABCs + docstrings). No algorithm is implemented here;
concrete backends land in later milestones (M1-M5, see ``docs/plan.md`` §5).
"""

from __future__ import annotations

__all__: list[str] = []

# Semantic version of the orchestration contract. Bumped when a Protocol /
# frozen dataclass in this package changes; those changes are additive
# (new field last, with a default) per the repo's frozen-schema rule.
__version__ = "0.0.0.dev0"
