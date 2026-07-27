from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm thu m?c g?c d? án vào Python path.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from app.core.camera_config import CameraConfig
from app.modules.face_verification.sources.base import (
    FaceImageSourceError,
)
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)


WINDOW_NAME = "Logitech C922 Preview"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xem truc tiep hinh anh tu Logitech C922."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Mac dinh: 0.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Chieu rong. Mac dinh: 1280.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Chieu cao. Mac dinh: 720.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS mong muon. Mac dinh: 30.",
    )

    return parser.parse_args()


def draw_information(
    frame,
    camera_index: int,
) -> None:
    height, width = frame.shape[:2]

    lines = [
        (
            f"Camera index: {camera_index} | "
            f"Resolution: {width}x{height}"
        ),
        "S: Save image",
        "Q or ESC: Exit",
    ]

    y_position = 35

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        y_position += 35


def main() -> None:
    args = parse_arguments()

    output_dir = ROOT_DIR / "storage" / "debug"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = CameraConfig(
        device_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=cv2.CAP_DSHOW,
        fourcc="MJPG",
        warmup_frames=15,
    )

    source = OpenCVCameraSource(config)

    print("=" * 60)
    print("LOGITECH C922 PREVIEW")
    print("=" * 60)
    print(f"Camera index: {args.camera}")
    print(
        f"Requested resolution: "
        f"{args.width}x{args.height}"
    )
    print(f"Requested FPS: {args.fps}")
    print()
    print("Nhan S de luu anh.")
    print("Nhan Q hoac ESC de thoat.")

    saved_image_number = 0

    try:
        source.open()

        actual_width, actual_height = (
            source.get_actual_resolution()
        )

        actual_fps = source.get_actual_fps()

        print()
        print(
            f"Actual resolution: "
            f"{actual_width}x{actual_height}"
        )
        print(f"Actual FPS: {actual_fps:.1f}")
        print("Camera da mo thanh cong.")

        while True:
            frame = source.capture_frame()

            display_frame = frame.copy()

            draw_information(
                frame=display_frame,
                camera_index=args.camera,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                saved_image_number += 1

                output_path = (
                    output_dir
                    / f"c922_capture_{saved_image_number:03d}.jpg"
                )

                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if saved:
                    print(f"Da luu anh: {output_path}")
                else:
                    print("[ERROR] Khong luu duoc anh.")

    except FaceImageSourceError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    finally:
        source.close()
        cv2.destroyAllWindows()
        print("Da dong camera.")


if __name__ == "__main__":
    main()
