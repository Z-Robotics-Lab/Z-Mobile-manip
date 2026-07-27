"""Compatibility alias for the planner that now lives in ``z_manip.planning``.

The module object itself is aliased rather than its names re-exported: callers
monkeypatch planner collaborators through this module path, and only an alias
keeps those patches visible to the code that reads them.
"""

import sys

from z_manip.planning import online_planner as _online_planner

sys.modules[__name__] = _online_planner
