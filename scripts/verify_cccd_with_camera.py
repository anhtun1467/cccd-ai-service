from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.embedding import InsightFaceEmbedder
from app.modules.face_verification.matcher import CosineFaceMatcher
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So khớp chân dung CCCD với khuôn mặt webcam."
    )

    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Đường dẫn ảnh mặt trước CCCD.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, mặc định là 0.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Ngưỡng cosine similarity.",
    )

    return parser.parse_args()


def capture_live_face(
    camera: cv2.VideoCapture,
) -> object | None:
    print()
    print("Đưa khuôn mặt vào giữa camera.")
    print("Nhấn S để chụp.")
    print("Nhấn Q hoặc ESC để hủy.")

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            raise RuntimeError(
                "Không đọc được frame từ webcam."
            )

        preview = frame.copy()

        height, width = preview.shape[:2]

        center_x = width // 2
        center_y = height // 2

        box_width = int(width * 0.32)
        box_height = int(height * 0.62)

        x1 = center_x - box_width // 2
        y1 = center_y - box_height // 2
        x2 = center_x + box_width // 2
        y2 = center_y + box_height // 2

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

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

        cv2.imshow(
            "CCCD Face Verification",
            preview,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            return frame.copy()

        if key in (ord("q"), 27):
            return None


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh CCCD: {image_path}"
        )

    card_image = cv2.imread(str(image_path))

    if card_image is None:
        raise ValueError(
            f"OpenCV không đọc được ảnh: {image_path}"
        )

    output_dir = (
        ROOT_DIR
        / "storage"
        / "debug"
        / "cccd_face_verification"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 64)
    print("CCCD FACE VERIFICATION")
    print("=" * 64)
    print(f"CCCD image: {image_path}")
    print(f"Camera index: {args.camera}")
    print(f"Threshold: {args.threshold:.4f}")
    print("Provider: CPUExecutionProvider")
    print("Model: buffalo_l")
    print("=" * 64)

    print("\n[1/5] Khởi tạo Portrait Extractor...")

    portrait_extractor = CCCDPortraitExtractor()

    print("[2/5] Trích xuất chân dung từ CCCD...")

    portrait_result = portrait_extractor.extract(
        card_image
    )

    portrait_image = portrait_result.portrait

    portrait_path = output_dir / "cccd_portrait.jpg"
    portrait_debug_path = output_dir / "cccd_portrait_debug.jpg"

    portrait_debug = portrait_extractor.draw_result(
        card_image,
        portrait_result,
    )

    cv2.imwrite(
        str(portrait_path),
        portrait_image,
    )

    cv2.imwrite(
        str(portrait_debug_path),
        portrait_debug,
    )

    print(
        f"Portrait method: "
        f"{portrait_result.extraction_method}"
    )

    print(
        f"Portrait score: "
        f"{portrait_result.detection_score:.4f}"
    )

    print("[3/5] Khởi tạo Face Embedder...")

    embedder = InsightFaceEmbedder(
        model_name="buffalo_l",
        detection_size=(640, 640),
        confidence_threshold=0.50,
        providers=["CPUExecutionProvider"],
    )

    print("[4/5] Trích xuất embedding CCCD...")

    cccd_embedding_result = embedder.extract_single(
        portrait_image
    )

    if cccd_embedding_result is None:
        raise RuntimeError(
            "Không thể trích xuất embedding từ chân dung CCCD."
        )

    camera = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Không thể mở camera index {args.camera}."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    try:
        webcam_image = capture_live_face(camera)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if webcam_image is None:
        print("Đã hủy xác minh.")
        return

    webcam_path = output_dir / "webcam_face.jpg"

    cv2.imwrite(
        str(webcam_path),
        webcam_image,
    )

    print("[5/5] Trích xuất embedding webcam và so khớp...")

    webcam_embedding_result = embedder.extract_single(
        webcam_image
    )

    if webcam_embedding_result is None:
        raise RuntimeError(
            "Không phát hiện được khuôn mặt trong ảnh webcam."
        )

    matcher = CosineFaceMatcher(
        threshold=args.threshold,
    )

    match_result = matcher.compare(
        cccd_embedding_result.embedding,
        webcam_embedding_result.embedding,
    )

    print()
    print("=" * 64)
    print("KẾT QUẢ XÁC MINH")
    print("=" * 64)
    print(
        f"CCCD detection score: "
        f"{cccd_embedding_result.detection_score:.4f}"
    )
    print(
        f"Webcam detection score: "
        f"{webcam_embedding_result.detection_score:.4f}"
    )
    print(
        f"Cosine similarity: "
        f"{match_result.similarity:.4f}"
    )
    print(
        f"Threshold: "
        f"{match_result.threshold:.4f}"
    )
    print(
        f"Distance: "
        f"{match_result.distance:.4f}"
    )

    if match_result.is_match:
        print("Kết luận: MATCH - Khuôn mặt trùng khớp")
    else:
        print("Kết luận: NOT MATCH - Khuôn mặt không trùng khớp")

    print("=" * 64)
    print(f"CCCD portrait: {portrait_path}")
    print(f"Webcam image: {webcam_path}")
    print(f"Debug image: {portrait_debug_path}")


if __name__ == "__main__":
    main()
