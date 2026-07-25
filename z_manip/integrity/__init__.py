"""Canonical content-identity digests for wire messages.

Distinct from :mod:`z_manip.verification` (sensor-based manipulation
*outcome* verification): this package is about stable content hashing for
replay/integrity checks, not judging whether a grasp succeeded.
"""

from __future__ import annotations

from .trajectory_digest import canonical_joint_trajectory_sha256

__all__ = ["canonical_joint_trajectory_sha256"]
