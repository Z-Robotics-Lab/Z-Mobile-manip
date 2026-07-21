import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_wrist_search_executor.py"
SPEC = importlib.util.spec_from_file_location("piper_wrist_search_executor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_views_change_only_j4_j5_and_stay_in_limits():
    home = np.asarray(json.loads((ROOT / "configs/piper_home.example.json").read_text())["joint_radians"])
    targets = MODULE.fixed_view_targets(home)
    assert len(targets) > 5
    for target in targets:
        changed = set(np.flatnonzero(np.abs(target - home) > 1e-9))
        assert changed.issubset({3, 4})
        assert np.all(target >= MODULE.executor.JOINT_LIMITS_RAD[:, 0])
        assert np.all(target <= MODULE.executor.JOINT_LIMITS_RAD[:, 1])


def test_smooth_path_is_rest_to_rest_and_small_sampled():
    start = np.zeros(6)
    target = start.copy()
    target[3] = np.radians(18)
    path, times = MODULE.smooth_path(start, target)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], target)
    assert np.all(np.diff(times) > 0)
    assert np.max(np.abs(np.diff(path, axis=0))) < np.radians(1.0)
    assert 1.12 <= times[-1] <= 1.13


def test_cli_is_dry_run_by_default_and_rejects_arbitrary_view(tmp_path):
    home = tmp_path / "piper_home.json"
    home_payload = json.loads((ROOT / "configs/piper_home.example.json").read_text(encoding="utf-8"))
    home_payload["capture_zero_can_tx_verified"] = True
    home_payload["captured_at"] = "test-fixture"
    home.write_text(json.dumps(home_payload), encoding="utf-8")
    good = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(home), "--view-index", "0"],
        text=True, capture_output=True, check=False,
    )
    assert good.returncode == 0
    assert json.loads(good.stdout)["commands_sent"] == 0
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(home), "--view-index", "999"],
        text=True, capture_output=True, check=False,
    )
    assert bad.returncode != 0
