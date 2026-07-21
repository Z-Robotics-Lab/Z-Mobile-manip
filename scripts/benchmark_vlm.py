#!/usr/bin/env python3
"""Run the configured VLM candidates on one captured wrist-camera frame."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z_manip.perception.vlm_affordance import (
    GROUNDING_SCOPES,
    OpenRouterVLM,
    VLMError,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("instruction")
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--scope",
        choices=sorted(GROUNDING_SCOPES),
        default="grasp_only",
    )
    args = parser.parse_args()
    image = args.image.read_bytes()
    results = []
    for model in args.models:
        try:
            result = OpenRouterVLM(
                models=(model,),
                timeout_s=args.timeout,
            ).locate_and_reason(
                image,
                args.instruction,
                grounding_scope=args.scope,
            )
            results.append({"ok": True, **asdict(result)})
        except VLMError as error:
            results.append({"ok": False, "model": model, "error": str(error)})
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if any(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
