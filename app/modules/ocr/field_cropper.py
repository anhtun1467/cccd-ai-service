from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PixelRegion:
    """Vùng cắt trên ảnh CCCD chuẩn hóa 1000 × 630 pixel."""

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
            x1=255,
            y1=230,
            x2=930,
            y2=330,
        ),
        "fullName": PixelRegion(
            x1=270,
            y1=315,
            x2=975,
            y2=395,
        ),
        "dateOfBirth": PixelRegion(
            x1=270,
            y1=370,
            x2=900,
            y2=450,
        ),
        "gender": PixelRegion(
            x1=270,
            y1=405,
            x2=650,
            y2=485,
        ),
        "nationality": PixelRegion(
            x1=620,
            y1=405,
            x2=995,
            y2=485,
        ),
        "placeOfOrigin": PixelRegion(
            x1=265,
            y1=460,
            x2=995,
            y2=560,
        ),
        "placeOfResidence": PixelRegion(
            x1=265,
            y1=515,
            x2=995,
            y2=630,
        ),
        "dateOfExpiry": PixelRegion(
            x1=0,
            y1=515,
            x2=345,
            y2=630,
        ),
        "portrait": PixelRegion(
            x1=20,
            y1=200,
            x2=270,
            y2=520,
        ),
    }

    # Vùng rộng ở trên giữ nhãn để parser có ngữ cảnh. Các vùng dưới đây
    # tập trung vào phần giá trị, giúp EasyOCR không phải đồng thời đọc
    # nhãn nhỏ, hoa văn bảo an và trường kế bên trên ảnh hơi mờ.
    VALUE_REGIONS: dict[str, PixelRegion] = {
        "idNumber": PixelRegion(330, 245, 850, 320),
        # Dừng trước dòng ngày sinh để chữ nhỏ của nhãn kế tiếp không bị
        # ghép vào họ tên trên ảnh mờ.
        "fullName": PixelRegion(270, 330, 985, 420),
        "dateOfBirth": PixelRegion(500, 380, 860, 455),
        "gender": PixelRegion(400, 410, 625, 490),
        "nationality": PixelRegion(730, 410, 1000, 490),
        "placeOfOrigin": PixelRegion(280, 465, 1000, 555),
        "placeOfResidence": PixelRegion(280, 520, 1000, 630),
        # Phải bao trọn TIGHT_VALUE_REGIONS["dateOfExpiry"]. Vùng hẹp
        # trải từ y=520 và đến x=345 để giữ đủ dãy DD/MM/YYYY.
        "dateOfExpiry": PixelRegion(15, 520, 345, 630),
    }

    # Vùng ngày hết hạn chỉ chứa dãy DD/MM/YYYY. OCR numeric trên vùng
    # này ổn định hơn nhiều so với crop có cả chân dung và hai dòng nhãn.
    TIGHT_VALUE_REGIONS: dict[str, PixelRegion] = {
        "dateOfExpiry": PixelRegion(105, 520, 345, 615),
    }

    BINARY_RETRY_FIELDS = {
        "idNumber",
        "dateOfBirth",
        "dateOfExpiry",
    }

    DETAIL_RETRY_FIELDS = {
        "fullName",
        "placeOfOrigin",
        "placeOfResidence",
    }

    def crop_fields_from_path(
        self,
        image_path: str,
        output_dir: str,
        layout_y_offset: float = 0.0,
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
            layout_y_offset=layout_y_offset,
        )

    def crop_fields(
        self,
        image: np.ndarray,
        output_dir: str,
        source_image_path: str | None = None,
        layout_y_offset: float = 0.0,
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

        for field_name, base_region in self.FIELD_REGIONS.items():
            region = self.shift_region(
                field_name,
                base_region,
                layout_y_offset,
            )
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

            variant_image_paths: list[dict[str, str]] = []
            tight_base_region = self.TIGHT_VALUE_REGIONS.get(field_name)
            if tight_base_region is not None:
                tight_region = self.shift_region(
                    field_name,
                    tight_base_region,
                    layout_y_offset,
                )
                self.validate_region(tight_region)
                tight_image = normalized_card[
                    tight_region.y1:tight_region.y2,
                    tight_region.x1:tight_region.x2,
                ].copy()
                if tight_image.size == 0:
                    raise ValueError(
                        f"Vùng tight {field_name} bị rỗng: {tight_region}"
                    )

                tight_raw_path = (
                    output_path / f"{field_name}_tight_raw.png"
                )
                if not cv2.imwrite(str(tight_raw_path), tight_image):
                    raise IOError(
                        f"Không thể lưu ảnh tight raw của {field_name}"
                    )

                tight_processed = self.preprocess_field(
                    field_name=field_name,
                    image=tight_image,
                )
                tight_processed_path = (
                    output_path / f"{field_name}_tight.png"
                )
                if not cv2.imwrite(
                    str(tight_processed_path),
                    tight_processed,
                ):
                    raise IOError(
                        f"Không thể lưu ảnh tight của {field_name}"
                    )

                tight_binary = self.preprocess_binary_field(
                    field_name=field_name,
                    image=tight_image,
                )
                tight_binary_path = (
                    output_path / f"{field_name}_tight_binary.png"
                )
                if not cv2.imwrite(str(tight_binary_path), tight_binary):
                    raise IOError(
                        f"Không thể lưu ảnh tight binary của {field_name}"
                    )

                variant_image_paths.extend([
                    {
                        "variant": "tight_raw",
                        "imagePath": str(tight_raw_path),
                    },
                    {
                        "variant": "tight_processed",
                        "imagePath": str(tight_processed_path),
                    },
                    {
                        "variant": "tight_binary",
                        "imagePath": str(tight_binary_path),
                    },
                ])

            value_base_region = self.VALUE_REGIONS.get(field_name)
            if value_base_region is not None:
                value_region = self.shift_region(
                    field_name,
                    value_base_region,
                    layout_y_offset,
                )
                self.validate_region(value_region)
                value_image = normalized_card[
                    value_region.y1:value_region.y2,
                    value_region.x1:value_region.x2,
                ].copy()
                if value_image.size == 0:
                    raise ValueError(
                        f"Vùng giá trị {field_name} bị rỗng: {value_region}"
                    )

                value_raw_path = output_path / f"{field_name}_value_raw.jpg"
                if not cv2.imwrite(str(value_raw_path), value_image):
                    raise IOError(
                        f"Không thể lưu ảnh value raw của {field_name}"
                    )

                value_processed = self.preprocess_field(
                    field_name=field_name,
                    image=value_image,
                )
                value_processed_path = (
                    output_path / f"{field_name}_value.jpg"
                )
                if not cv2.imwrite(
                    str(value_processed_path),
                    value_processed,
                ):
                    raise IOError(
                        f"Không thể lưu ảnh value của {field_name}"
                    )

                variant_image_paths.extend([
                    {
                        "variant": "value_processed",
                        "imagePath": str(value_processed_path),
                    },
                    {
                        "variant": "value_raw",
                        "imagePath": str(value_raw_path),
                    },
                ])

                if field_name in self.DETAIL_RETRY_FIELDS:
                    detail_image = self.preprocess_detail_field(
                        field_name=field_name,
                        image=value_image,
                    )
                    detail_path = (
                        output_path / f"{field_name}_value_detail.png"
                    )
                    if not cv2.imwrite(str(detail_path), detail_image):
                        raise IOError(
                            f"Không thể lưu ảnh detail của {field_name}"
                        )
                    variant_image_paths.append({
                        "variant": "value_detail",
                        "imagePath": str(detail_path),
                    })

                if field_name in self.BINARY_RETRY_FIELDS:
                    binary_image = self.preprocess_binary_field(
                        field_name=field_name,
                        image=value_image,
                    )
                    binary_path = (
                        output_path / f"{field_name}_value_binary.jpg"
                    )
                    if not cv2.imwrite(str(binary_path), binary_image):
                        raise IOError(
                            f"Không thể lưu ảnh binary của {field_name}"
                        )
                    variant_image_paths.append({
                        "variant": "value_binary",
                        "imagePath": str(binary_path),
                    })

            variant_image_paths.extend([
                {
                    "variant": "wide_processed",
                    "imagePath": str(processed_field_path),
                },
                {
                    "variant": "wide_raw",
                    "imagePath": str(raw_field_path),
                },
            ])

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
                "variantImagePaths": variant_image_paths,
            }

        debug_image = self.draw_regions(
            normalized_card,
            layout_y_offset=layout_y_offset,
        )
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

        value_debug_image = self.draw_value_regions(
            normalized_card,
            layout_y_offset=layout_y_offset,
        )
        value_debug_output_path = (
            output_path / "fields_value_debug.jpg"
        )
        if not cv2.imwrite(
            str(value_debug_output_path),
            value_debug_image,
        ):
            raise IOError(
                f"Không thể lưu ảnh debug vùng giá trị: "
                f"{value_debug_output_path}"
            )

        results["_debug"] = {
            "sourceImagePath": source_image_path,
            "normalizedImagePath": str(
                normalized_card_path
            ),
            "debugImagePath": str(
                debug_output_path
            ),
            "valueDebugImagePath": str(
                value_debug_output_path
            ),
            "imageWidth": self.CANONICAL_WIDTH,
            "imageHeight": self.CANONICAL_HEIGHT,
            "layoutYOffset": round(float(layout_y_offset), 2),
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
            d=3,
            sigmaColor=25,
            sigmaSpace=25,
        )

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(denoised)

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            sigmaX=0.7,
        )

        return cv2.addWeighted(
            enhanced,
            1.30,
            blurred,
            -0.30,
            0,
        )

    def preprocess_binary_field(
        self,
        field_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        """Tạo biến thể nhị phân riêng cho số CCCD và các trường ngày."""
        scale_factor = self.get_scale_factor(field_name)
        enlarged = cv2.resize(
            image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = (
            cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            if enlarged.ndim == 3
            else enlarged
        )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(
            clipLimit=1.6,
            tileGridSize=(8, 8),
        ).apply(gray)
        _, binary = cv2.threshold(
            clahe,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary

    def preprocess_detail_field(
        self,
        field_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        """Tăng tương phản/nét nhẹ, tránh tạo viền kép trên chữ mờ."""
        scale_factor = self.get_scale_factor(field_name)
        enlarged = cv2.resize(
            image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LANCZOS4,
        )
        gray = (
            cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            if enlarged.ndim == 3
            else enlarged
        )
        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        contrast = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(denoised)
        soft = cv2.GaussianBlur(contrast, (0, 0), sigmaX=0.9)
        return cv2.addWeighted(
            contrast,
            1.45,
            soft,
            -0.45,
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

    def shift_region(
        self,
        field_name: str,
        region: PixelRegion,
        layout_y_offset: float,
    ) -> PixelRegion:
        """Dịch vùng cắt theo mẫu in được định vị từ cụm số CCCD.

        Một số lô thẻ có cả khối chữ lệch lên/xuống hơn 40 px sau khi
        chuẩn hóa. Vùng cố định khi đó đọc sang hàng kế bên. Hai trường
        nằm sát đáy được giữ đến biên dưới để không cắt mất dòng thứ hai.
        """
        delta = int(round(max(-55.0, min(55.0, layout_y_offset))))
        y1 = max(0, min(self.CANONICAL_HEIGHT - 1, region.y1 + delta))
        shifted_y2 = region.y2 + delta
        if field_name in {"placeOfResidence", "dateOfExpiry"}:
            y2 = self.CANONICAL_HEIGHT
        else:
            y2 = max(y1 + 1, min(self.CANONICAL_HEIGHT, shifted_y2))

        return PixelRegion(
            x1=region.x1,
            y1=y1,
            x2=region.x2,
            y2=y2,
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
        layout_y_offset: float = 0.0,
    ) -> np.ndarray:
        debug_image = image.copy()

        for field_name, base_region in self.FIELD_REGIONS.items():
            region = self.shift_region(
                field_name,
                base_region,
                layout_y_offset,
            )
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

    def draw_value_regions(
        self,
        image: np.ndarray,
        layout_y_offset: float = 0.0,
    ) -> np.ndarray:
        """Vẽ riêng các vùng giá trị được ưu tiên khi OCR ảnh mờ."""
        debug_image = image.copy()
        for field_name, base_region in self.VALUE_REGIONS.items():
            region = self.shift_region(
                field_name,
                base_region,
                layout_y_offset,
            )
            cv2.rectangle(
                debug_image,
                (region.x1, region.y1),
                (region.x2, region.y2),
                (255, 120, 0),
                2,
            )
            cv2.putText(
                debug_image,
                f"value:{field_name}",
                (region.x1, max(region.y1 - 6, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return debug_image
