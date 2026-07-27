from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PixelRegion:
    """
    Vùng cắt trên ảnh CCCD chuẩn hóa 1000 × 630 pixel.

    x1, y1: góc trên bên trái.
    x2, y2: góc dưới bên phải.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class CCCDFieldCropper:
    """
    Cắt các trường thông tin trên mặt trước CCCD Việt Nam.

    Quy trình:
    1. Đọc ảnh CCCD đã được perspective transform.
    2. Chuẩn hóa chiều xoay.
    3. Resize về kích thước cố định 1000 × 630.
    4. Cắt từng trường bằng tọa độ pixel.
    5. Tiền xử lý riêng từng vùng phục vụ OCR.
    6. Lưu ảnh debug để kiểm tra.
    """

    CANONICAL_WIDTH = 1000
    CANONICAL_HEIGHT = 630

    # Tọa độ được tính trên ảnh chuẩn 1000 × 630.
    #
    # Các vùng bên dưới tập trung vào PHẦN GIÁ TRỊ,
    # hạn chế lấy nhãn tiếng Việt và tiếng Anh.
    FIELD_REGIONS: dict[str, PixelRegion] = {
        # Số CCCD: 001208031482
        "idNumber": PixelRegion(
            x1=335,
            y1=205,
            x2=740,
            y2=285,
        ),

        # Họ tên: NGUYEN TUAN MINH
        "fullName": PixelRegion(
            x1=285,
            y1=270,
            x2=790,
            y2=355,
        ),

        # Ngày sinh: 16/04/2008
        "dateOfBirth": PixelRegion(
            x1=575,
            y1=330,
            x2=850,
            y2=410,
        ),

        # Giới tính: Nam
        "gender": PixelRegion(
            x1=285,
            y1=375,
            x2=505,
            y2=445,
        ),

        # Quốc tịch: Viet Nam
        "nationality": PixelRegion(
            x1=570,
            y1=375,
            x2=900,
            y2=445,
        ),

        # Quê quán
        "placeOfOrigin": PixelRegion(
            x1=275,
            y1=425,
            x2=940,
            y2=515,
        ),

        # Nơi thường trú
        "placeOfResidence": PixelRegion(
            x1=270,
            y1=495,
            x2=960,
            y2=600,
        ),

        # Ngày hết hạn phía dưới ảnh chân dung
        "dateOfExpiry": PixelRegion(
            x1=15,
            y1=490,
            x2=285,
            y2=590,
        ),

        # Ảnh chân dung
        "portrait": PixelRegion(
            x1=20,
            y1=200,
            x2=270,
            y2=505,
        ),
    }

    def crop_fields_from_path(
        self,
        image_path: str,
        output_dir: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Đọc ảnh từ đường dẫn và cắt toàn bộ trường.

        Args:
            image_path:
                Nên truyền ảnh *_card.jpg thay vì *_enhanced.jpg.

            output_dir:
                Thư mục lưu các vùng đã cắt.
        """

        source_path = Path(image_path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh CCCD: {source_path}"
            )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None or image.size == 0:
            raise ValueError(
                f"Không thể đọc ảnh CCCD: {source_path}"
            )

        return self.crop_fields(
            image=image,
            output_dir=output_dir,
            source_image_path=str(source_path),
        )

    def crop_fields(
        self,
        image: np.ndarray,
        output_dir: str,
        source_image_path: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Chuẩn hóa ảnh và cắt các trường CCCD.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Ảnh CCCD đầu vào không hợp lệ"
            )

        output_path = Path(output_dir)

        # Xóa kết quả cũ để tránh xem nhầm ảnh.
        if output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalized_card = self.normalize_card_image(image)

        normalized_card_path = (
            output_path / "00_normalized_card.jpg"
        )

        if not cv2.imwrite(
            str(normalized_card_path),
            normalized_card,
        ):
            raise IOError(
                f"Không thể lưu ảnh chuẩn hóa: "
                f"{normalized_card_path}"
            )

        results: dict[str, dict[str, Any]] = {}

        for field_name, region in self.FIELD_REGIONS.items():
            self.validate_region(region)

            cropped_image = normalized_card[
                region.y1:region.y2,
                region.x1:region.x2,
            ].copy()

            if cropped_image.size == 0:
                raise ValueError(
                    f"Vùng cắt {field_name} bị rỗng: {region}"
                )

            # Lưu cả ảnh màu nguyên bản của vùng cắt.
            raw_field_path = (
                output_path / f"{field_name}_raw.jpg"
            )

            if not cv2.imwrite(
                str(raw_field_path),
                cropped_image,
            ):
                raise IOError(
                    f"Không thể lưu ảnh raw của {field_name}"
                )

            processed_image = self.preprocess_field(
                field_name=field_name,
                image=cropped_image,
            )

            processed_field_path = (
                output_path / f"{field_name}.jpg"
            )

            if not cv2.imwrite(
                str(processed_field_path),
                processed_image,
            ):
                raise IOError(
                    f"Không thể lưu ảnh xử lý của {field_name}"
                )

            results[field_name] = {
                "fieldName": field_name,
                "imagePath": str(processed_field_path),
                "rawImagePath": str(raw_field_path),
                "box": [
                    [region.x1, region.y1],
                    [region.x2, region.y1],
                    [region.x2, region.y2],
                    [region.x1, region.y2],
                ],
                "rawWidth": int(cropped_image.shape[1]),
                "rawHeight": int(cropped_image.shape[0]),
                "processedWidth": int(
                    processed_image.shape[1]
                ),
                "processedHeight": int(
                    processed_image.shape[0]
                ),
            }

        debug_image = self.draw_regions(normalized_card)

        debug_output_path = (
            output_path / "fields_debug.jpg"
        )

        if not cv2.imwrite(
            str(debug_output_path),
            debug_image,
        ):
            raise IOError(
                f"Không thể lưu ảnh debug: "
                f"{debug_output_path}"
            )

        results["_debug"] = {
            "sourceImagePath": source_image_path,
            "normalizedImagePath": str(
                normalized_card_path
            ),
            "debugImagePath": str(
                debug_output_path
            ),
            "imageWidth": self.CANONICAL_WIDTH,
            "imageHeight": self.CANONICAL_HEIGHT,
        }

        return results

    def normalize_card_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Chuẩn hóa ảnh thẻ về 1000 × 630.

        Nếu ảnh đang nằm dọc thì tự xoay 90 độ.
        """

        card = image.copy()

        height, width = card.shape[:2]

        if height > width:
            card = cv2.rotate(
                card,
                cv2.ROTATE_90_CLOCKWISE,
            )

        card = cv2.resize(
            card,
            (
                self.CANONICAL_WIDTH,
                self.CANONICAL_HEIGHT,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        return card

    def preprocess_field(
        self,
        field_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Tiền xử lý từng vùng sau khi đã cắt.

        Ảnh portrait giữ nguyên màu.
        Các trường chữ được phóng lớn, chuyển xám,
        tăng tương phản và làm nét nhẹ.
        """

        if field_name == "portrait":
            return image.copy()

        scale_factor = self.get_scale_factor(
            field_name
        )

        enlarged = cv2.resize(
            image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(
                enlarged,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = enlarged

        # Khử nhiễu nhẹ, vẫn giữ nét chữ.
        denoised = cv2.bilateralFilter(
            gray,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )

        # Tăng tương phản cục bộ.
        clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(denoised)

        # Làm nét bằng unsharp mask.
        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            sigmaX=0.8,
        )

        sharpened = cv2.addWeighted(
            enhanced,
            1.6,
            blurred,
            -0.6,
            0,
        )

        return sharpened

    @staticmethod
    def get_scale_factor(
        field_name: str,
    ) -> float:
        """
        Các trường nhỏ cần phóng lớn hơn.
        """

        scale_factors = {
            "idNumber": 3.0,
            "fullName": 3.0,
            "dateOfBirth": 3.5,
            "gender": 3.5,
            "nationality": 3.0,
            "placeOfOrigin": 2.7,
            "placeOfResidence": 2.7,
            "dateOfExpiry": 3.5,
        }

        return scale_factors.get(
            field_name,
            3.0,
        )

    def validate_region(
        self,
        region: PixelRegion,
    ) -> None:
        """
        Kiểm tra vùng cắt có nằm trong ảnh chuẩn hay không.
        """

        if not (
            0 <= region.x1 < region.x2
            <= self.CANONICAL_WIDTH
        ):
            raise ValueError(
                f"Tọa độ X không hợp lệ: {region}"
            )

        if not (
            0 <= region.y1 < region.y2
            <= self.CANONICAL_HEIGHT
        ):
            raise ValueError(
                f"Tọa độ Y không hợp lệ: {region}"
            )

    def draw_regions(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Vẽ các vùng cắt lên ảnh CCCD chuẩn hóa.
        """

        debug_image = image.copy()

        for field_name, region in self.FIELD_REGIONS.items():
            cv2.rectangle(
                debug_image,
                (region.x1, region.y1),
                (region.x2, region.y2),
                (0, 255, 0),
                2,
            )

            label_y = max(
                region.y1 - 8,
                20,
            )

            cv2.putText(
                debug_image,
                field_name,
                (region.x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return debug_image
