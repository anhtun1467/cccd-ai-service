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
    """

    x1: int
    y1: int
    x2: int
    y2: int


class CCCDFieldCropper:
    """
    Cắt các trường trên mặt trước CCCD Việt Nam.

    Vùng cắt được hiệu chỉnh theo mẫu CCCD gắn chip có dòng:
    - Quê quán nằm dưới giới tính/quốc tịch.
    - Nơi thường trú nằm sát đáy thẻ.
    - Có giá trị đến nằm dưới ảnh chân dung.
    """

    CANONICAL_WIDTH = 1000
    CANONICAL_HEIGHT = 630

    FIELD_REGIONS: dict[str, PixelRegion] = {
        "idNumber": PixelRegion(
            x1=390,
            y1=255,
            x2=790,
            y2=325,
        ),
        "fullName": PixelRegion(
            x1=295,
            y1=340,
            x2=700,
            y2=405,
        ),
        "dateOfBirth": PixelRegion(
            x1=580,
            y1=390,
            x2=830,
            y2=440,
        ),
        "gender": PixelRegion(
            x1=455,
            y1=425,
            x2=590,
            y2=475,
        ),
        "nationality": PixelRegion(
            x1=790,
            y1=425,
            x2=995,
            y2=475,
        ),
        "placeOfOrigin": PixelRegion(
            x1=290,
            y1=490,
            x2=760,
            y2=550,
        ),
        "placeOfResidence": PixelRegion(
            x1=290,
            y1=535,
            x2=995,
            y2=630,
        ),
        "dateOfExpiry": PixelRegion(
            x1=15,
            y1=545,
            x2=300,
            y2=615,
        ),
        "portrait": PixelRegion(
            x1=20,
            y1=200,
            x2=270,
            y2=520,
        ),
    }

    def crop_fields_from_path(
        self,
        image_path: str,
        output_dir: str,
    ) -> dict[str, dict[str, Any]]:
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
        if image is None or image.size == 0:
            raise ValueError(
                "Ảnh CCCD đầu vào không hợp lệ"
            )

        output_path = Path(output_dir)

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
        card = image.copy()
        height, width = card.shape[:2]

        if height > width:
            card = cv2.rotate(
                card,
                cv2.ROTATE_90_CLOCKWISE,
            )

        return cv2.resize(
            card,
            (
                self.CANONICAL_WIDTH,
                self.CANONICAL_HEIGHT,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    def preprocess_field(
        self,
        field_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        if field_name == "portrait":
            return image.copy()

        scale_factor = self.get_scale_factor(field_name)

        enlarged = cv2.resize(
            image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = (
            cv2.cvtColor(
                enlarged,
                cv2.COLOR_BGR2GRAY,
            )
            if len(enlarged.shape) == 3
            else enlarged
        )

        denoised = cv2.bilateralFilter(
            gray,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )

        clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(denoised)

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            sigmaX=0.8,
        )

        return cv2.addWeighted(
            enhanced,
            1.6,
            blurred,
            -0.6,
            0,
        )

    @staticmethod
    def get_scale_factor(
        field_name: str,
    ) -> float:
        scale_factors = {
            "idNumber": 3.0,
            "fullName": 3.0,
            "dateOfBirth": 3.5,
            "gender": 3.5,
            "nationality": 3.0,
            "placeOfOrigin": 3.0,
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
        debug_image = image.copy()

        for field_name, region in self.FIELD_REGIONS.items():
            cv2.rectangle(
                debug_image,
                (region.x1, region.y1),
                (region.x2, region.y2),
                (0, 255, 0),
                2,
            )

            label_y = max(region.y1 - 8, 20)

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