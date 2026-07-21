from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.field_cropper import CCCDFieldCropper


def choose_card_image() -> Path:
    """
    Chọn đúng ảnh CCCD đã được detect và perspective transform.

    Ưu tiên:
    1. Ảnh card theo tên file cố định.
    2. Nếu không tồn tại, tự lấy ảnh *_card.jpg mới nhất.
    """

    output_dir = (
        ROOT_DIR
        / "storage"
        / "outputs"
    )

    preferred_image = (
        output_dir
        / "f05464331278439fbc014550ce6b194b_card.jpg"
    )

    if preferred_image.exists():
        return preferred_image

    card_images = list(
        output_dir.glob("*_card.jpg")
    )

    if not card_images:
        raise FileNotFoundError(
            "Không tìm thấy ảnh *_card.jpg trong:\n"
            f"{output_dir}\n\n"
            "Hãy gọi API OCR trước để tạo ảnh card."
        )

    latest_card_image = max(
        card_images,
        key=lambda path: path.stat().st_mtime,
    )

    return latest_card_image


def validate_field_image(
    field_path: Path,
) -> tuple[bool, int, int]:
    """
    Kiểm tra ảnh vùng cắt có đọc được và không bị rỗng.

    Returns:
        tuple:
        - valid
        - width
        - height
    """

    field_image = cv2.imread(
        str(field_path),
        cv2.IMREAD_UNCHANGED,
    )

    if field_image is None or field_image.size == 0:
        return False, 0, 0

    height, width = field_image.shape[:2]

    return True, width, height


def main() -> None:
    card_image_path = choose_card_image()

    output_dir = (
        ROOT_DIR
        / "tests"
        / "ocr"
        / "output"
        / "fields"
    )

    cropper = CCCDFieldCropper()

    results = cropper.crop_fields_from_path(
        image_path=str(card_image_path),
        output_dir=str(output_dir),
    )

    print("=" * 80)
    print("FIELD CROPPER TEST")
    print("=" * 80)
    print(f"Input : {card_image_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    passed_count = 0
    failed_count = 0

    for field_name, result in results.items():
        if field_name == "_debug":
            continue

        processed_path = Path(
            result["imagePath"]
        )

        raw_path = Path(
            result["rawImagePath"]
        )

        processed_valid, processed_width, processed_height = (
            validate_field_image(processed_path)
        )

        raw_valid, raw_width, raw_height = (
            validate_field_image(raw_path)
        )

        passed = (
            processed_valid
            and raw_valid
        )

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{field_name:<20} "
            f"raw={raw_width}x{raw_height} "
            f"processed={processed_width}x{processed_height}"
        )

        print(
            f"       Raw      : {raw_path}"
        )

        print(
            f"       Processed: {processed_path}"
        )

    debug_result = results.get(
        "_debug",
        {},
    )

    normalized_image_path = debug_result.get(
        "normalizedImagePath",
        "",
    )

    debug_image_path = debug_result.get(
        "debugImagePath",
        "",
    )

    print("=" * 80)
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)

    print(
        "Ảnh CCCD chuẩn hóa:",
        normalized_image_path,
    )

    print(
        "Ảnh kiểm tra vùng cắt:",
        debug_image_path,
    )

    print("=" * 80)

    if failed_count > 0:
        raise AssertionError(
            f"Có {failed_count} vùng cắt không hợp lệ."
        )

    print("Tất cả vùng cắt đã được tạo thành công.")


if __name__ == "__main__":
    main()