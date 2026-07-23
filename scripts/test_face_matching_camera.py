from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.embedding import (
    InsightFaceEmbedder,
)
from app.modules.face_verification.matcher import (
    CosineFaceMatcher,
)


def capture_image(
    camera: cv2.VideoCapture,
    window_name: str,
    instruction: str,
):
    """
    Hiển thị camera và chụp ảnh khi nhấn S.
    """

    print()
    print(instruction)
    print("Nhấn S để chụp.")
    print("Nhấn Q hoặc ESC để thoát.")

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            raise RuntimeError(
                "Không đọc được frame từ camera."
            )

        preview = frame.copy()

        cv2.putText(
            preview,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            "S: Capture | Q/ESC: Exit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            return frame.copy()

        if key in (ord("q"), 27):
            return None


def main() -> None:
    output_dir = (
        ROOT_DIR
        / "storage"
        / "debug"
        / "face_matching"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("FACE EMBEDDING AND MATCHING TEST")
    print("=" * 60)
    print("Model: buffalo_l")
    print("Provider: CPUExecutionProvider")
    print("Threshold ban đầu: 0.45")
    print("=" * 60)

    print("\nĐang khởi tạo InsightFace...")

    embedder = InsightFaceEmbedder(
        model_name="buffalo_l",
        detection_size=(640, 640),
        confidence_threshold=0.60,
        providers=["CPUExecutionProvider"],
    )

    matcher = CosineFaceMatcher(
        threshold=0.45,
    )

    print("Khởi tạo thành công.")

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Không thể mở camera index 0."
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
        image_a = capture_image(
            camera=camera,
            window_name="Capture Face A",
            instruction="Capture image A",
        )

        if image_a is None:
            print("Đã hủy kiểm thử.")
            return

        image_b = capture_image(
            camera=camera,
            window_name="Capture Face B",
            instruction="Capture image B",
        )

        if image_b is None:
            print("Đã hủy kiểm thử.")
            return

    finally:
        camera.release()
        cv2.destroyAllWindows()

    image_a_path = output_dir / "face_a.jpg"
    image_b_path = output_dir / "face_b.jpg"

    cv2.imwrite(
        str(image_a_path),
        image_a,
    )

    cv2.imwrite(
        str(image_b_path),
        image_b,
    )

    print("\nĐang trích xuất embedding ảnh A...")

    result_a = embedder.extract_single(
        image_a,
    )

    print("Đang trích xuất embedding ảnh B...")

    result_b = embedder.extract_single(
        image_b,
    )

    if result_a is None:
        print(
            "Không tìm thấy khuôn mặt trong ảnh A."
        )
        return

    if result_b is None:
        print(
            "Không tìm thấy khuôn mặt trong ảnh B."
        )
        return

    match_result = matcher.compare(
        result_a.embedding,
        result_b.embedding,
    )

    print()
    print("=" * 60)
    print("KẾT QUẢ SO KHỚP")
    print("=" * 60)
    print(
        f"Embedding A: {result_a.dimension} chiều"
    )
    print(
        f"Embedding B: {result_b.dimension} chiều"
    )
    print(
        f"Detection score A: "
        f"{result_a.detection_score:.4f}"
    )
    print(
        f"Detection score B: "
        f"{result_b.detection_score:.4f}"
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
        print("Kết luận: MATCH - Cùng một người")
    else:
        print("Kết luận: NOT MATCH - Khác người")

    print("=" * 60)
    print(f"Ảnh A: {image_a_path}")
    print(f"Ảnh B: {image_b_path}")


if __name__ == "__main__":
    main()