from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "runtime" / "go2w_perception_worker.py"
    spec = importlib.util.spec_from_file_location("perception_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDryRun:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []
        self.context_starts = 0
        self.context_stops = 0

    def start_resident_context(self):
        self.context_starts += 1
        return object()

    def stop_resident_context(self, _node):
        self.context_stops += 1

    @staticmethod
    def _arguments(argv):
        values = list(argv)

        def value(name: str) -> str:
            return values[values.index(name) + 1]

        return SimpleNamespace(
            output=Path(value("--output")),
            passive_window=Path(value("--passive-window")),
            selected_passive_window=Path(value("--selected-passive-window")),
            learned_endpoint="",
        )

    def main(self, argv, *, manage_rclpy_context=True):
        assert manage_rclpy_context is False
        self.calls.append(tuple(argv))
        args = self._arguments(argv)
        args.output.mkdir(parents=True, exist_ok=True)
        payload = b"exact-identity-age-0.5-six-artifacts-64-candidates"
        (args.output / "grasp_candidates.npz").write_bytes(payload)
        print(json.dumps({"candidate_count": 64, "exact_identity": True}))
        return 0


def _argv(output: Path) -> list[str]:
    return [
        "--instruction", "white adapter",
        "--output", str(output),
        "--passive-window", str(output / "live_passive_joint_report.json"),
        "--selected-passive-window", str(output / "selected_passive_joint_report.json"),
        "--timeout", "15",
        "--min-bundle-target-points", "400",
        "--reuse-valid-tracking",
        "--tracking-reuse-max-age", "0.5",
    ]


def test_worker_reuses_one_import_and_preserves_candidate_bytes(tmp_path, monkeypatch):
    module = _module()
    artifact_root = tmp_path / "artifacts"
    socket_path = artifact_root / ".worker.sock"
    fake = _FakeDryRun()
    load_calls = []

    def load_once():
        load_calls.append(1)
        return fake

    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "_load_dry_run", load_once)
    worker = threading.Thread(
        target=module._serve,
        args=(socket_path,),
        kwargs={"max_requests": 2},
        daemon=True,
    )
    worker.start()
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert socket_path.exists()

    outputs = [artifact_root / "one", artifact_root / "two"]
    assert module._client(socket_path, _argv(outputs[0])) == 0
    assert module._client(socket_path, _argv(outputs[1])) == 0
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert load_calls == [1]
    assert fake.context_starts == 1
    assert fake.context_stops == 1
    assert len(fake.calls) == 2
    assert all("--reuse-valid-tracking" in call for call in fake.calls)
    assert all(
        call[call.index("--tracking-reuse-max-age") + 1] == "0.5"
        for call in fake.calls
    )
    hashes = [
        hashlib.sha256((output / "grasp_candidates.npz").read_bytes()).hexdigest()
        for output in outputs
    ]
    assert hashes[0] == hashes[1]


def test_simulated_warm_lifecycle_removes_repeated_heavy_import():
    requests = 5
    import_delay_s = 0.025
    argv = ["--synthetic"]

    def load():
        time.sleep(import_delay_s)
        return SimpleNamespace(main=lambda _argv: 0)

    cold_started = time.perf_counter()
    cold_results = [load().main(argv) for _ in range(requests)]
    cold_elapsed = time.perf_counter() - cold_started

    warm_started = time.perf_counter()
    resident = load()
    warm_results = [resident.main(argv) for _ in range(requests)]
    warm_elapsed = time.perf_counter() - warm_started

    assert cold_results == warm_results == [0] * requests
    assert warm_elapsed < cold_elapsed * 0.35
