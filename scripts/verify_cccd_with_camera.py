from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.core.camera_config import CameraConfig
from app.core.face_verification_provider import FaceVerificationProvider
from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.quality import FaceQualityEvaluator
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đối chiếu chân dung CCCD với khuôn mặt từ webcam."
    )
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh mặt trước CCCD.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--match-threshold",
        "--threshold",
        dest="match_threshold",
        type=float,
        default=0.50,
        help="Ngưỡng MATCH, mặc định 0.50.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.40,
        help="Ngưỡng REVIEW, mặc định 0.40.",
    )
    return parser.parse_args()


def capture_live_face(source: OpenCVCameraSource) -> np.ndarray | None:
    print("\nĐưa khuôn mặt vào giữa khung hình và nhìn thẳng.")
    print("Nhấn S để chụp; Q hoặc ESC để hủy.")

    while True:
        frame = source.capture_frame()
        preview = frame.copy()
        height, width = preview.shape[:2]
        box_width = int(width * 0.36)
        box_height = int(height * 0.70)
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            "Look straight | S: Capture | Q/ESC: Exit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("CCCD Face Verification", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            frames = [frame]
            for _ in range(6):
                frames.append(source.capture_frame())
            return max(frames, key=lambda item: _center_sharpness(item))
        if key in (ord("q"), 27):
            return None


def _center_sharpness(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    crop = frame[
        int(height * 0.15):int(height * 0.90),
        int(width * 0.28):int(width * 0.72),
    ]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh CCCD: {image_path}")

    card_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if card_image is None or card_image.size == 0:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")

    if args.review_threshold >= args.match_threshold:
        raise ValueError("review-threshold phải nhỏ hơn match-threshold.")

    print("=" * 68)
    print("CCCD FACE VERIFICATION - INSIGHTFACE BUFFALO_L")
    print("=" * 68)
    print(f"Ảnh CCCD: {image_path}")
    print(f"Camera: {args.camera} | {args.width}x{args.height}@{args.fps}")
    print(
        f"Ngưỡng: MATCH >= {args.match_threshold:.2f} | "
        f"REVIEW >= {args.review_threshold:.2f}"
    )
    print("Đang nạp model InsightFace...")

    provider = FaceVerificationProvider.instance()
    service = FaceVerificationService(
        portrait_extractor=provider.portrait_extractor,
        embedder=provider.embedder,
        quality_evaluator=FaceQualityEvaluator(),
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
    )

    camera_config = CameraConfig(
        device_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=cv2.CAP_DSHOW,
        fourcc="MJPG",
        warmup_frames=15,
    )
    source = OpenCVCameraSource(camera_config)
    try:
        source.open()
        actual_width, actual_height = source.get_actual_resolution()
        print(
            f"Camera thực tế: {actual_width}x{actual_height} "
            f"@{source.get_actual_fps():.1f} FPS"
        )
        webcam_image = capture_live_face(source)
    finally:
        source.close()
        cv2.destroyAllWindows()

    if webcam_image is None:
        print("Đã hủy xác minh.")
        return

    try:
        output = service.verify(card_image, webcam_image)
    except FaceVerificationError as exc:
        print("\nKẾT QUẢ: ẢNH KHÔNG ĐẠT YÊU CẦU")
        print(f"Mã lỗi: {exc.error_code}")
        print(f"Lý do: {exc}")
        if exc.details:
            print(json.dumps(exc.details, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    output_dir = ROOT_DIR / "storage" / "debug" / "cccd_face_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "cccd_face.jpg"), output.artifacts.cccd_portrait)
    cv2.imwrite(str(output_dir / "webcam_face.jpg"), output.artifacts.webcam_face)
    cv2.imwrite(str(output_dir / "webcam_frame.jpg"), webcam_image)
    result_path = output_dir / "verification_result.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(output.result.to_dict(), file, ensure_ascii=False, indent=2)

    result = output.result
    print("\n" + "=" * 68)
    print("KẾT QUẢ XÁC MINH")
    print("=" * 68)
    print(f"Trạng thái: {result.status.upper()}")
    print(f"Cosine similarity: {result.similarity:.4f}")
    print(f"CCCD detection: {result.cccd_detection_score:.4f}")
    print(f"Webcam detection: {result.webcam_detection_score:.4f}")
    print(f"Chất lượng CCCD: {result.cccd_quality.status.upper()}")
    print(f"Chất lượng webcam: {result.webcam_quality.status.upper()}")
    print(f"Thời gian xử lý: {result.processing_time_ms:.2f} ms")
    print(f"JSON: {result_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
