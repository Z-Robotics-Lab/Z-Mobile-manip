from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image as PilImage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_charuco_tool.py"
SPEC = importlib.util.spec_from_file_location("piper_charuco_tool", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHARUCO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHARUCO)


def test_generate_a4_board_with_physical_dpi_metadata(tmp_path):
    image_path = tmp_path / "board.png"
    metadata_path = tmp_path / "board.json"

    metadata = CHARUCO.generate_board(image_path, metadata_path)
    image = PilImage.open(image_path)

    assert metadata["page"] == "A4"
    assert metadata["orientation"] == "landscape"
    assert metadata["content"] == "board_only"
    assert metadata["dpi"] == 300
    assert metadata["pixel_size"] == [3508, 2480]
    assert metadata["board_pixel_size"] == [2894, 2067]
    assert metadata["board_physical_size_m"] == pytest.approx([0.245, 0.175])
    assert image.size == tuple(metadata["pixel_size"])
    assert image.info["dpi"][0] == pytest.approx(300, abs=0.01)
    pixels = np.asarray(image)
    left = (image.width - metadata["board_pixel_size"][0]) // 2
    top = (image.height - metadata["board_pixel_size"][1]) // 2
    right = left + metadata["board_pixel_size"][0]
    bottom = top + metadata["board_pixel_size"][1]
    outside_board = np.ones(pixels.shape, dtype=bool)
    outside_board[top:bottom, left:right] = False
    assert np.all(pixels[outside_board] == 255)
    assert np.any(pixels[top:bottom, left:right] == 0)
    specification = CHARUCO.load_board_metadata(metadata_path)
    assert specification == {
        "squares_x": 7,
        "squares_y": 5,
        "square_length_m": pytest.approx(0.035),
        "marker_length_m": pytest.approx(0.025666666666666667),
        "dictionary_name": "DICT_4X4_50",
    }


def test_board_metadata_rejects_unknown_schema(tmp_path):
    metadata_path = tmp_path / "board.json"
    metadata_path.write_text('{"schema":"unknown"}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        CHARUCO.load_board_metadata(metadata_path)


def test_detect_generated_board_pose():
    board, dictionary = CHARUCO.make_board(
        squares_x=7,
        squares_y=5,
        square_length_m=0.0254,
        marker_length_m=0.018626666666666667,
        dictionary_name="DICT_4X4_50",
    )
    rendered = board.draw((1400, 1000), 20, 1)
    image = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
    camera_matrix = np.array(((1800.0, 0.0, 700.0), (0.0, 1800.0, 500.0), (0.0, 0.0, 1.0)))

    result = CHARUCO.detect_board_pose(
        image,
        camera_matrix,
        np.zeros(5),
        board,
        dictionary,
        min_corners=12,
    )

    assert result["charuco_corner_count"] >= 12
    assert result["camera_from_target"][2, 3] > 0.0
    assert result["reprojection_rmse_px"] < 1.0


def test_capture_source_has_no_publish_or_actuator_transport():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_imports = {"can", "socket", "piper_sdk", "pyAgxArm"}
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"create_publisher", "publish", "send", "sendto"}
    }

    assert imports.isdisjoint(forbidden_imports)
    assert calls == set()
