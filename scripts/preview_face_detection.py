from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.detector import InsightFaceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview phát hiện khuôn mặt bằng InsightFace."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, mặc định là 0.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Chiều rộng camera.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Chiều cao camera.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = ROOT_DIR / "storage" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INSIGHTFACE FACE DETECTION")
    print("=" * 60)
    print(f"Camera index: {args.camera}")
    print("Provider: CPUExecutionProvider")
    print("Model: buffalo_l")
    print()
    print("Lần đầu chạy có thể tải model InsightFace.")
    print("S: lưu ảnh")
    print("Q hoặc ESC: thoát")
    print("=" * 60)

    print("\nĐang khởi tạo InsightFace...")

    detector = InsightFaceDetector(
        model_name="buffalo_l",
        detection_size=(640, 640),
        confidence_threshold=0.60,
        providers=["CPUExecutionProvider"],
    )

    print("Khởi tạo InsightFace thành công.")

    capture = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW,
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Không thể mở camera index {args.camera}."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, 30)

    saved_count = 0
    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                print("Không đọc được frame từ camera.")
                break

            faces = detector.detect(frame)
            preview = detector.draw_faces(frame, faces)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            status = (
                f"Faces: {len(faces)} | "
                f"FPS: {fps:.1f} | "
                f"Provider: CPU"
            )

            cv2.putText(
                preview,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "InsightFace Detection - Logitech C922",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                saved_count += 1

                detection_path = (
                    output_dir
                    / f"face_detection_{saved_count:03d}.jpg"
                )

                cv2.imwrite(
                    str(detection_path),
                    preview,
                )

                print(
                    f"Đã lưu ảnh detection: {detection_path}"
                )

                if faces:
                    face_crop = detector.crop_face(
                        frame,
                        faces[0],
                        margin_ratio=0.15,
                    )

                    crop_path = (
                        output_dir
                        / f"face_crop_{saved_count:03d}.jpg"
                    )

                    cv2.imwrite(
                        str(crop_path),
                        face_crop,
                    )

                    print(
                        f"Đã lưu khuôn mặt: {crop_path}"
                    )
                else:
                    print("Không có khuôn mặt để crop.")

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print("\nĐã đóng camera.")


if __name__ == "__main__":
    main()