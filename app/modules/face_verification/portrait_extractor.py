from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import cv2
import numpy as np

from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.models import (
    FaceEmbeddingResult,
    PortraitExtractionResult,
)


class FaceAnalyzer(Protocol):
    def extract(self, image: np.ndarray) -> list[FaceEmbeddingResult]: ...


class CCCDPortraitExtractor:
    """Tìm chân dung in trên mặt trước CCCD và giữ lại embedding gốc.

    Mỗi khuôn mặt chỉ được đưa qua InsightFace một lần. Nếu detector không
    thấy khuôn mặt trên toàn thẻ, hệ thống thử vùng chân dung bên trái,
    tăng tương phản nhẹ và tự xoay thẻ trước khi kết luận thất bại.
    """

    def __init__(
        self,
        analyzer: FaceAnalyzer | None = None,
        face_margin_ratio: float = 0.22,
        minimum_portrait_size: int = 32,
        enable_rotation_retry: bool = True,
    ) -> None:
        if face_margin_ratio < 0:
            raise ValueError("face_margin_ratio không được nhỏ hơn 0.")
        if minimum_portrait_size <= 0:
            raise ValueError("minimum_portrait_size phải lớn hơn 0.")

        if analyzer is None:
            from app.modules.face_verification.embedding import (
                InsightFaceEmbedder,
            )

            analyzer = InsightFaceEmbedder(
                model_name="buffalo_l",
                detection_size=(640, 640),
                confidence_threshold=0.50,
                providers=["CPUExecutionProvider"],
            )

        self.analyzer = analyzer
        self.face_margin_ratio = face_margin_ratio
        self.minimum_portrait_size = minimum_portrait_size
        self.enable_rotation_retry = enable_rotation_retry

    def extract(self, card_image: np.ndarray) -> PortraitExtractionResult:
        self._validate_image(card_image)

        attempts: list[str] = []
        for rotation_degrees, oriented_card in self._orientation_candidates(
            card_image
        ):
            image_height, image_width = oriented_card.shape[:2]
            aspect_ratio = image_width / max(1, image_height)

            if not 1.15 <= aspect_ratio <= 2.30:
                attempts.append(f"rotation_{rotation_degrees}:invalid_ratio")
                continue

            roi_result = self._extract_from_portrait_roi(
                oriented_card,
                rotation_degrees=rotation_degrees,
            )
            if roi_result is not None:
                return roi_result
            attempts.append(f"rotation_{rotation_degrees}:roi_not_found")

            full_faces = self.analyzer.extract(oriented_card)
            selected_face = self._select_card_portrait_face(
                faces=full_faces,
                image_width=image_width,
                image_height=image_height,
            )
            if selected_face is None:
                attempts.append(f"rotation_{rotation_degrees}:full_not_found")
                continue

            portrait = self._crop_face(
                image=oriented_card,
                face=selected_face,
            )
            return PortraitExtractionResult(
                portrait=portrait,
                bbox=(
                    selected_face.x1,
                    selected_face.y1,
                    selected_face.x2,
                    selected_face.y2,
                ),
                detection_score=selected_face.detection_score,
                extraction_method="full_card_face_detection",
                source_width=image_width,
                source_height=image_height,
                embedding_result=selected_face,
                detection_image=oriented_card,
                rotation_degrees=rotation_degrees,
                detected_face_count=len(full_faces),
            )

        raise FaceVerificationError(
            "CCCD_FACE_NOT_FOUND",
            "Không phát hiện được ảnh chân dung trên mặt trước CCCD.",
            details={
                "attempts": attempts,
                "suggestion": (
                    "Chụp trọn mặt trước CCCD, không lóa và để thẻ nằm ngang."
                ),
            },
        )

    def draw_result(
        self,
        card_image: np.ndarray,
        result: PortraitExtractionResult,
    ) -> np.ndarray:
        output = self._rotate_image(card_image, result.rotation_degrees).copy()
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = (
            f"{result.extraction_method} | "
            f"score={result.detection_score:.3f} | "
            f"rotate={result.rotation_degrees}"
        )
        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return output

    def _extract_from_portrait_roi(
        self,
        oriented_card: np.ndarray,
        *,
        rotation_degrees: int,
    ) -> PortraitExtractionResult | None:
        roi, roi_bbox = self._extract_portrait_roi(oriented_card)
        image_height, image_width = oriented_card.shape[:2]
        for variant_name, detection_image, scale in self._iter_roi_variants(roi):
            faces = self.analyzer.extract(detection_image)
            valid_faces = [
                face
                for face in faces
                if min(face.width, face.height) >= self.minimum_portrait_size
            ]
            if not valid_faces:
                continue

            selected_face = max(
                valid_faces,
                key=lambda face: (face.area, face.detection_score),
            )
            portrait = self._crop_face(detection_image, selected_face)

            roi_x1, roi_y1, _, _ = roi_bbox
            local_x1 = int(round(selected_face.x1 / scale))
            local_y1 = int(round(selected_face.y1 / scale))
            local_x2 = int(round(selected_face.x2 / scale))
            local_y2 = int(round(selected_face.y2 / scale))
            global_bbox = (
                max(0, roi_x1 + local_x1),
                max(0, roi_y1 + local_y1),
                min(image_width, roi_x1 + local_x2),
                min(image_height, roi_y1 + local_y2),
            )

            return PortraitExtractionResult(
                portrait=portrait,
                bbox=global_bbox,
                detection_score=selected_face.detection_score,
                extraction_method=f"portrait_roi_{variant_name}",
                source_width=image_width,
                source_height=image_height,
                embedding_result=selected_face,
                detection_image=detection_image,
                rotation_degrees=rotation_degrees,
                detected_face_count=len(faces),
            )

        return None

    def _select_card_portrait_face(
        self,
        faces: list[FaceEmbeddingResult],
        image_width: int,
        image_height: int,
    ) -> FaceEmbeddingResult | None:
        image_area = max(1, image_width * image_height)
        valid_faces: list[FaceEmbeddingResult] = []

        for face in faces:
            center_x = (face.x1 + face.x2) / 2.0
            center_y = (face.y1 + face.y2) / 2.0
            area_ratio = face.area / image_area

            if not (
                center_x <= image_width * 0.48
                and image_height * 0.12 <= center_y <= image_height * 0.94
                and min(face.width, face.height) >= self.minimum_portrait_size
                and 0.001 <= area_ratio <= 0.18
            ):
                continue
            valid_faces.append(face)

        if not valid_faces:
            return None
        return max(
            valid_faces,
            key=lambda face: (face.area, face.detection_score),
        )

    @staticmethod
    def _extract_portrait_roi(
        card_image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        image_height, image_width = card_image.shape[:2]
        x1 = int(image_width * 0.01)
        y1 = int(image_height * 0.12)
        x2 = int(image_width * 0.41)
        y2 = int(image_height * 0.97)

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))

        roi = card_image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            raise ValueError("Không thể tạo vùng chân dung từ ảnh CCCD.")
        return roi, (x1, y1, x2, y2)

    @staticmethod
    def _iter_roi_variants(
        roi: np.ndarray,
    ) -> Iterator[tuple[str, np.ndarray, float]]:
        # Generator để không tạo CLAHE/upscale nếu ảnh gốc đã phát hiện tốt.
        yield "original", roi, 1.0

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced_lightness = clahe.apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((enhanced_lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
        yield "clahe", enhanced, 1.0

        longest_side = max(roi.shape[:2])
        scale = min(2.2, max(1.0, 640.0 / max(1, longest_side)))
        if scale >= 1.15:
            upscaled = cv2.resize(
                enhanced,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            yield "clahe_upscaled", upscaled, scale

    def _orientation_candidates(
        self,
        image: np.ndarray,
    ) -> Iterator[tuple[int, np.ndarray]]:
        # Xoay theo nhu cầu; ảnh đúng chiều không phải tạo thêm ba bản sao.
        yield 0, image
        if not self.enable_rotation_retry:
            return
        yield 180, cv2.rotate(image, cv2.ROTATE_180)
        yield 90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        yield 270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _rotate_image(image: np.ndarray, degrees_value: int) -> np.ndarray:
        if degrees_value == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if degrees_value == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if degrees_value == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _crop_face(
        self,
        image: np.ndarray,
        face: FaceEmbeddingResult,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        margin_x = int(face.width * self.face_margin_ratio)
        margin_top = int(face.height * self.face_margin_ratio)
        margin_bottom = int(face.height * 0.38)

        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_top)
        x2 = min(image_width, face.x2 + margin_x)
        y2 = min(image_height, face.y2 + margin_bottom)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box khuôn mặt CCCD không hợp lệ.")
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Ảnh CCCD không được là None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Ảnh CCCD phải là numpy.ndarray.")
        if image.size == 0:
            raise ValueError("Ảnh CCCD đầu vào bị rỗng.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh CCCD phải là ảnh BGR 3 kênh.")


__all__ = [
    "CCCDPortraitExtractor",
    "PortraitExtractionResult",
]
