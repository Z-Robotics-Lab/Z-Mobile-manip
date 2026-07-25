"""Deprecated import path — moved to :mod:`z_manip.integrity.trajectory_digest`.

Kept as a thin re-export shim so existing ``from z_manip.trajectory_digest
import ...`` call sites keep working. New code should import from
:mod:`z_manip.integrity` directly.
"""

from __future__ import annotations

from z_manip.integrity.trajectory_digest import canonical_joint_trajectory_sha256

__all__ = ["canonical_joint_trajectory_sha256"]
