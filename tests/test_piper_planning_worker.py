from __future__ import annotations

import importlib.util
from argparse import Namespace
from collections import OrderedDict
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/runtime/piper_planning_worker.py"
    spec = importlib.util.spec_from_file_location("piper_planning_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contained_rejects_sibling_and_parent_escape(tmp_path):
    module = _module()
    root = tmp_path / "root"
    root.mkdir()
    assert module._contained(root / "session" / "perception", root)
    assert not module._contained(tmp_path / "root-sibling" / "capture", root)
    assert not module._contained(root / ".." / "outside", root)


def test_client_requires_fixed_planner_arguments():
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["client"])


def test_serve_rejects_trailing_planner_arguments():
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["serve", "--", "--artifacts", "/tmp"])


def test_same_source_all_ik_failure_runs_exhaustive_search_only_once(tmp_path):
    module = _module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("grasp_candidates.npz", "target_points.npy", "scene_points.npy"):
        (artifacts / name).write_bytes(name.encode())
    output1 = tmp_path / "output-1"
    output2 = tmp_path / "output-2"
    calls = 0

    class FakeDryRun:
        @staticmethod
        def _arguments(argv):
            return Namespace(
                artifacts=artifacts,
                output=Path(argv[argv.index("--output") + 1]),
            )

        @staticmethod
        def main(argv):
            nonlocal calls
            calls += 1
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir()
            report = {
                "plan_valid": False,
                "source_stamp_ns": 456,
                "candidate_count": 64,
                "rejection_count": 64,
                "rejections_truncated": False,
                "rejections": [{"stage": "ik"}] * 64,
                "timings_s": {"search": 5.6, "total": 6.2},
            }
            (output / "planning_report.json").write_text(json.dumps(report))
            return 1

    cache = OrderedDict()
    first_argv = ["--artifacts", str(artifacts), "--output", str(output1)]
    second_argv = ["--artifacts", str(artifacts), "--output", str(output2)]

    first_code, _ = module._run_validated_request(
        FakeDryRun, first_argv, "pinocchio", cache,
    )
    second_code, _ = module._run_validated_request(
        FakeDryRun, second_argv, "pinocchio", cache,
    )

    assert first_code == second_code == 1
    assert calls == 1
    second = json.loads((output2 / "planning_report.json").read_text())
    assert second["planning_disposition"] == "NEED_BASE_APPROACH"
    assert second["cached_all_ik_failure"] is True
    assert second["timings_s"]["search"] == 0.0
