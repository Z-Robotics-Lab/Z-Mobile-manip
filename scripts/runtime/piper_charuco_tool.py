#!/usr/bin/env python3
"""Generate or passively observe the PiPER wrist-camera ChArUco board."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image as PilImage


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}


def load_board_metadata(path: Path) -> dict[str, object]:
    """Load the exact physical board specification used during printing."""

    document = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "z_manip.charuco_board.v1":
        raise ValueError("unsupported ChArUco board metadata schema")
    try:
        specification = {
            "squares_x": int(document["squares_x"]),
            "squares_y": int(document["squares_y"]),
            "square_length_m": float(document["square_length_m"]),
            "marker_length_m": float(document["marker_length_m"]),
            "dictionary_name": str(document["dictionary"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid ChArUco board metadata") from error
    values = (
        specification["square_length_m"],
        specification["marker_length_m"],
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("ChArUco board dimensions must be finite")
    # Reuse all geometric and dictionary validation from the board constructor.
    make_board(**specification)
    return specification


def make_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
):
    if dictionary_name not in DICTIONARIES:
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}")
    if squares_x < 3 or squares_y < 3:
        raise ValueError("ChArUco board must have at least 3x3 squares")
    if not 0.0 < marker_length_m < square_length_m:
        raise ValueError("marker length must be positive and smaller than a square")
    dictionary = cv2.aruco.getPredefinedDictionary(
        DICTIONARIES[dictionary_name],
    )
    return cv2.aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_length_m,
        marker_length_m,
        dictionary,
    ), dictionary


def generate_board(
    output: Path,
    metadata_output: Path,
    *,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length_m: float = 0.035,
    marker_length_m: float = 0.025666666666666667,
    dictionary_name: str = "DICT_4X4_50",
    dpi: int = 300,
) -> dict[str, object]:
    board, _dictionary = make_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
    )
    # A 7x5 board is landscape.  Render it on an actual A4 landscape canvas so
    # print dialogs do not rotate, crop, or scale a portrait page unexpectedly.
    page_width_px = round(297.0 / 25.4 * dpi)
    page_height_px = round(210.0 / 25.4 * dpi)
    board_width_px = round(squares_x * square_length_m / 0.0254 * dpi)
    board_height_px = round(squares_y * square_length_m / 0.0254 * dpi)
    minimum_margin_px = round(10.0 / 25.4 * dpi)
    if (
        board_width_px > page_width_px - 2 * minimum_margin_px
        or board_height_px > page_height_px - 2 * minimum_margin_px
    ):
        raise ValueError("board does not fit on A4 with 10 mm printer margins")
    board_image = board.draw((board_width_px, board_height_px), 0, 1)
    page = PilImage.new("L", (page_width_px, page_height_px), color=255)
    left = (page_width_px - board_width_px) // 2
    top = (page_height_px - board_height_px) // 2
    page.paste(PilImage.fromarray(board_image), (left, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, dpi=(dpi, dpi))
    metadata = {
        "schema": "z_manip.charuco_board.v1",
        "dictionary": dictionary_name,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "page": "A4",
        "orientation": "landscape",
        "content": "board_only",
        "dpi": dpi,
        "pixel_size": [page_width_px, page_height_px],
        "board_pixel_size": [board_width_px, board_height_px],
        "board_physical_size_m": [
            squares_x * square_length_m,
            squares_y * square_length_m,
        ],
        "print_scale_percent": 100,
        "verification": {
            "single_square_m": square_length_m,
            "board_width_m": squares_x * square_length_m,
            "board_height_m": squares_y * square_length_m,
        },
        "warning": "Print A4 landscape at actual size/100%; disable fit-to-page.",
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def detect_board_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    board: object,
    dictionary: object,
    *,
    min_corners: int,
) -> dict[str, object]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
        gray,
        dictionary,
    )
    if marker_ids is None or len(marker_ids) < 2:
        raise ValueError("fewer than two ChArUco markers were detected")
    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        minMarkers=2,
    )
    if charuco_ids is None or int(count) < min_corners:
        raise ValueError(
            f"only {int(count)} ChArUco corners were detected; need {min_corners}",
        )
    ok, rotation_vector, translation_vector = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        distortion,
        None,
        None,
    )
    if not ok:
        raise ValueError("ChArUco pose estimation failed")
    camera_from_target = np.eye(4)
    camera_from_target[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    camera_from_target[:3, 3] = np.asarray(translation_vector).reshape(3)
    if not np.all(np.isfinite(camera_from_target)) or camera_from_target[2, 3] <= 0.0:
        raise ValueError("ChArUco pose is non-finite or behind the camera")
    identifiers = np.asarray(charuco_ids, dtype=int).reshape(-1)
    object_points = np.asarray(board.chessboardCorners, dtype=float)[identifiers]
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    residuals = projected.reshape(-1, 2) - np.asarray(charuco_corners).reshape(-1, 2)
    reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    annotated = image_bgr.copy()
    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(
        annotated,
        charuco_corners,
        charuco_ids,
    )
    cv2.drawFrameAxes(
        annotated,
        camera_matrix,
        distortion,
        rotation_vector,
        translation_vector,
        float(board.getSquareLength()) * 2.0,
    )
    return {
        "camera_from_target": camera_from_target,
        "marker_count": len(marker_ids),
        "charuco_corner_count": int(count),
        "reprojection_rmse_px": reprojection_rmse_px,
        "annotated": annotated,
    }


def _stamp_ns(message: object) -> int:
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def capture(args: argparse.Namespace) -> int:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    if args.board_metadata is None:
        board_specification = {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length_m,
            "marker_length_m": args.marker_length_m,
            "dictionary_name": args.dictionary,
        }
    else:
        board_specification = load_board_metadata(args.board_metadata)
    board, dictionary = make_board(**board_specification)
    rclpy.init()
    node = Node("piper_charuco_read_only_capture")
    bridge = CvBridge()
    images: dict[int, Image] = {}
    infos: dict[int, CameraInfo] = {}

    def callback(cache: dict[int, object]):
        def receive(message: object) -> None:
            cache[_stamp_ns(message)] = message
            while len(cache) > 90:
                cache.pop(next(iter(cache)))

        return receive

    node.create_subscription(
        Image,
        args.image_topic,
        callback(images),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        CameraInfo,
        args.camera_info_topic,
        callback(infos),
        qos_profile_sensor_data,
    )
    deadline = time.monotonic() + args.timeout
    last_error = "no exact RGB/CameraInfo pair arrived"
    result = None
    selected_stamp = None
    selected_image = None
    selected_info = None
    while time.monotonic() < deadline and result is None:
        rclpy.spin_once(node, timeout_sec=0.2)
        common = sorted(images.keys() & infos.keys(), reverse=True)
        for stamp in common[:4]:
            image_message = images[stamp]
            info_message = infos[stamp]
            image_bgr = bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
            camera_matrix = np.asarray(info_message.k, dtype=float).reshape(3, 3)
            distortion = np.asarray(info_message.d, dtype=float)
            if distortion.size == 0:
                distortion = np.zeros(5)
            try:
                candidate = detect_board_pose(
                    image_bgr,
                    camera_matrix,
                    distortion,
                    board,
                    dictionary,
                    min_corners=args.min_corners,
                )
            except ValueError as error:
                last_error = str(error)
                continue
            if float(candidate["reprojection_rmse_px"]) > args.max_reprojection_rmse_px:
                last_error = (
                    f"reprojection RMSE {candidate['reprojection_rmse_px']:.3f}px "
                    f"exceeds {args.max_reprojection_rmse_px:.3f}px"
                )
                continue
            result = candidate
            selected_stamp = stamp
            selected_image = image_message
            selected_info = info_message
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        report = {
            "schema": "z_manip.charuco_camera_sample.v1",
            "read_only": True,
            "valid": False,
            "error": last_error,
        }
        (args.output_dir / "camera_sample.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        node.destroy_node()
        rclpy.shutdown()
        return 1
    cv2.imwrite(
        str(args.output_dir / "charuco_detection.png"),
        np.asarray(result["annotated"]),
    )
    report = {
        "schema": "z_manip.charuco_camera_sample.v1",
        "read_only": True,
        "valid": True,
        "source_stamp_ns": selected_stamp,
        "camera_frame": selected_image.header.frame_id,
        "target_frame": args.target_frame,
        "camera_from_target": np.asarray(result["camera_from_target"]).tolist(),
        "marker_count": result["marker_count"],
        "charuco_corner_count": result["charuco_corner_count"],
        "reprojection_rmse_px": result["reprojection_rmse_px"],
        "image_size": [selected_info.width, selected_info.height],
        "board": {
            "dictionary": board_specification["dictionary_name"],
            "squares_x": board_specification["squares_x"],
            "squares_y": board_specification["squares_y"],
            "square_length_m": board_specification["square_length_m"],
            "marker_length_m": board_specification["marker_length_m"],
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "camera_sample.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    node.destroy_node()
    rclpy.shutdown()
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--squares-x", type=int, default=7)
    common.add_argument("--squares-y", type=int, default=5)
    common.add_argument("--square-length-m", type=float, default=0.035)
    common.add_argument("--marker-length-m", type=float, default=0.025666666666666667)
    common.add_argument("--dictionary", choices=sorted(DICTIONARIES), default="DICT_4X4_50")

    generate_parser = subparsers.add_parser("generate", parents=[common])
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--metadata", type=Path, required=True)
    generate_parser.add_argument("--dpi", type=int, default=300)

    capture_parser = subparsers.add_parser("capture", parents=[common])
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument(
        "--board-metadata",
        type=Path,
        help="generated board.json; overrides manual board geometry options",
    )
    capture_parser.add_argument("--image-topic", default="/camera/color/image_raw")
    capture_parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    capture_parser.add_argument("--target-frame", default="charuco_board")
    capture_parser.add_argument("--timeout", type=float, default=30.0)
    capture_parser.add_argument("--min-corners", type=int, default=12)
    capture_parser.add_argument("--max-reprojection-rmse-px", type=float, default=1.0)
    values = parser.parse_args()
    if values.squares_x < 3 or values.squares_y < 3:
        parser.error("board must have at least 3x3 squares")
    if not 0.0 < values.marker_length_m < values.square_length_m:
        parser.error("marker length must be positive and smaller than a square")
    if values.command == "generate" and values.dpi < 150:
        parser.error("board DPI must be at least 150")
    if values.command == "capture" and (
        values.timeout <= 1.0
        or values.min_corners < 4
        or not math.isfinite(values.max_reprojection_rmse_px)
        or values.max_reprojection_rmse_px <= 0.0
    ):
        parser.error("invalid capture quality limits")
    return values


def main() -> int:
    args = _arguments()
    if args.command == "generate":
        metadata = generate_board(
            args.output,
            args.metadata,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_m=args.square_length_m,
            marker_length_m=args.marker_length_m,
            dictionary_name=args.dictionary,
            dpi=args.dpi,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
