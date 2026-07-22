from __future__ import annotations

import importlib.util
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
