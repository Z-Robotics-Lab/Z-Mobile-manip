from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_mount_ui.py"
PAGE = ROOT / "web/mount_dashboard/index.html"
SPEC = importlib.util.spec_from_file_location("piper_mount_ui", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(path: Path):
    path.write_text(json.dumps({
        "schema": MODULE.SCHEMA,
        "calibrated": False,
        "read_only": True,
        "urdf_modified": False,
        "motion_commands_published": 0,
    }))


def test_load_report_rejects_changed_urdf(tmp_path):
    path = tmp_path / "report.json"
    report(path)
    value = json.loads(path.read_text())
    value["urdf_modified"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ReportError, match="unchanged-URDF"):
        MODULE.load_report(path)


def test_loopback_server_is_read_only(tmp_path):
    path = tmp_path / "report.json"
    report(path)
    server = MODULE.create_server(path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/api/report", timeout=2) as response:
            assert json.load(response)["motion_commands_published"] == 0
            assert response.headers["Cache-Control"] == "no-store"
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + "/api/report", method="POST"), timeout=2)
        assert caught.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_page_renders_mount_and_residual_views():
    source = PAGE.read_text(encoding="utf-8")
    assert "top" in source and "side" in source and "chart" in source
    assert "nominal_from_calibrated_delta" in source
    assert "fetch('/api/report'" in source
    assert "/piper/cmd" not in source
