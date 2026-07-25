"""Deprecated import path — moved to :mod:`z_manip.config`.

Kept as a thin re-export shim so existing ``from z_manip.configuration
import ...`` call sites keep working. New code should import from
:mod:`z_manip.config` directly.
"""

from __future__ import annotations

from z_manip.config import (
    RobotModelConfig,
    StackConfig,
    TopicConfig,
    ToolGeometryConfig,
    load_stack_config,
)

__all__ = [
    "RobotModelConfig",
    "StackConfig",
    "TopicConfig",
    "ToolGeometryConfig",
    "load_stack_config",
]
