"""Cross-cutting concurrency/cancellation primitives.

Shared by :mod:`z_manip.planning`, :mod:`z_manip.collision`, and
:mod:`z_manip.kinematics`; lives outside all three to avoid inverting any of
their dependency directions.
"""

from __future__ import annotations
