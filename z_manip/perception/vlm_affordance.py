"""Deprecated import path — moved to :mod:`z_manip.perception.vlm_openrouter`
and its sibling modules (:mod:`z_manip.perception.vlm_errors`,
:mod:`z_manip.perception.vlm_types`, :mod:`z_manip.perception.vlm_schema`,
:mod:`z_manip.perception.vlm_transport`,
:mod:`z_manip.perception.vlm_result_parsing`).

Kept as a thin re-export shim so existing ``from
z_manip.perception.vlm_affordance import ...`` call sites keep working. New
code should import from those modules directly.
"""

from __future__ import annotations

from z_manip.perception.vlm_errors import (
    VLMCancellationError,
    VLMError,
    VLMTransportError,
)
from z_manip.perception.vlm_openrouter import DEFAULT_MODELS, OpenRouterVLM
from z_manip.perception.vlm_transport import (  # noqa: F401 (white-box test seam)
    _curl_transport,
    _terminate_process_group,
)
from z_manip.perception.vlm_types import (
    GROUNDING_SCOPES,
    AffordanceResult,
    AvoidRegion,
    NormalizedBox,
    PlacementVerification,
    VLMAttemptEvent,
)

__all__ = [
    "GROUNDING_SCOPES",
    "AffordanceResult",
    "AvoidRegion",
    "DEFAULT_MODELS",
    "NormalizedBox",
    "OpenRouterVLM",
    "PlacementVerification",
    "VLMAttemptEvent",
    "VLMCancellationError",
    "VLMError",
    "VLMTransportError",
]
