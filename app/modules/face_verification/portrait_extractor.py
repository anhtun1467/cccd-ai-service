from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.modules.face_verification.detector import (
    DetectedFace,
    InsightFaceDetector,
)


@dataclass(frozen=True)
class PortraitExtractionResult:
    """
    K?t qu? trích xu?t ?nh chân dung t? CCCD.
    """

    portrait: np.ndarray
    bbox: tuple[int, int, int, int]
    detection_score: float
    extraction_method: str
    source_width: int
    source_height: int


class CCCDPortraitExtractor:
    """
    Trích xu?t ?nh chân dung t? m?t tru?c CCCD.

    Chi?n lu?c:
    1. Phát hi?n khuôn m?t trên toàn b? ?nh th?.
    2. N?u th?t b?i, c?t vùng chân dung u?c lu?ng bên trái.
    3. Phát hi?n l?i khuôn m?t trong vùng u?c lu?ng.
    4. N?u v?n th?t b?i, tr? v? vùng chân dung u?c lu?ng.
    """

    def __init__(
        self,
        detector: InsightFaceDetector | None = None,
        face_margin_ratio: float = 0.25,
        minimum_portrait_size: int = 40,
    ) -> None:
        if face_margin_ratio < 0:
            raise ValueError(
                "face_margin_ratio không du?c nh? hon 0."
            )

        if minimum_portrait_size <= 0:
            raise ValueError(
                "minimum_portrait_size ph?i l?n hon 0."
            )

        self.detector = detector or InsightFaceDetector(
            model_name="buffalo_l",
            detection_size=(1024, 1024),
            confidence_threshold=0.40,
            providers=["CPUExecutionProvider"],
        )

        self.face_margin_ratio = face_margin_ratio
        self.minimum_portrait_size = minimum_portrait_size

    def extract(
        self,
        card_image: np.ndarray,
    ) -> PortraitExtractionResult:
        """
        Trích xu?t ?nh chân dung t? ?nh CCCD dă crop và can th?ng.
        """

        self._validate_image(card_image)

        image_height, image_width = card_image.shape[:2]

        full_image_faces = self.detector.detect(card_image)

        selected_face = self._select_card_portrait_face(
            faces=full_image_faces,
            image_width=image_width,
            image_height=image_height,
        )

        if selected_face is not None:
            portrait, bbox = self._crop_with_margin(
                image=card_image,
                face=selected_face,
            )

            return PortraitExtractionResult(
                portrait=portrait,
                bbox=bbox,
                detection_score=selected_face.score,
                extraction_method="full_card_face_detection",
                source_width=image_width,
                source_height=image_height,
            )

        roi, roi_bbox = self._extract_portrait_roi(card_image)

        roi_faces = self.detector.detect(roi)

        if roi_faces:
            selected_roi_face = roi_faces[0]

            portrait, local_bbox = self._crop_with_margin(
                image=roi,
                face=selected_roi_face,
            )

            roi_x1, roi_y1, _, _ = roi_bbox
            local_x1, local_y1, local_x2, local_y2 = local_bbox

            global_bbox = (
                roi_x1 + local_x1,
                roi_y1 + local_y1,
                roi_x1 + local_x2,
                roi_y1 + local_y2,
            )

            return PortraitExtractionResult(
                portrait=portrait,
                bbox=global_bbox,
                detection_score=selected_roi_face.score,
                extraction_method="portrait_roi_face_detection",
                source_width=image_width,
                source_height=image_height,
            )

        return PortraitExtractionResult(
            portrait=roi.copy(),
            bbox=roi_bbox,
            detection_score=0.0,
            extraction_method="portrait_roi_fallback",
            source_width=image_width,
            source_height=image_height,
        )

    def draw_result(
        self,
        card_image: np.ndarray,
        result: PortraitExtractionResult,
    ) -> np.ndarray:
        """
        V? vùng ?nh chân dung trên CCCD d? debug.
        """

        output = card_image.copy()

        x1, y1, x2, y2 = result.bbox

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        label = (
            f"{result.extraction_method} "
            f"| score={result.detection_score:.3f}"
        )

        text_y = max(25, y1 - 10)

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return output

    def _select_card_portrait_face(
        self,
        faces: list[DetectedFace],
        image_width: int,
        image_height: int,
    ) -> DetectedFace | None:
        """
        Ch?n khuôn m?t có kh? nang là ?nh chân dung CCCD nh?t.
        """

        valid_faces: list[DetectedFace] = []

        for face in faces:
            center_x = (face.x1 + face.x2) / 2
            center_y = (face.y1 + face.y2) / 2

            is_left_side = center_x <= image_width * 0.55

            is_inside_vertical_region = (
                image_height * 0.15
                <= center_y
                <= image_height * 0.92
            )

            is_large_enough = (
                face.width >= self.minimum_portrait_size
                and face.height >= self.minimum_portrait_size
            )

            if (
                is_left_side
                and is_inside_vertical_region
                and is_large_enough
            ):
                valid_faces.append(face)

        if not valid_faces:
            return None

        valid_faces.sort(
            key=lambda face: (
                face.area,
                face.score,
            ),
            reverse=True,
        )

        return valid_faces[0]

    def _extract_portrait_roi(
        self,
        card_image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        L?y vùng u?c lu?ng ch?a ?nh chân dung trên CCCD.
        """

        image_height, image_width = card_image.shape[:2]

        x1 = int(image_width * 0.02)
        y1 = int(image_height * 0.25)
        x2 = int(image_width * 0.37)
        y2 = int(image_height * 0.94)

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))

        roi = card_image[y1:y2, x1:x2]

        if roi.size == 0:
            raise ValueError(
                "Không th? t?o vùng chân dung t? ?nh CCCD."
            )

        return roi.copy(), (x1, y1, x2, y2)

    def _crop_with_margin(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        image_height, image_width = image.shape[:2]

        margin_x = int(face.width * self.face_margin_ratio)
        margin_y = int(face.height * self.face_margin_ratio)

        lower_margin = int(face.height * 0.45)

        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_y)
        x2 = min(image_width, face.x2 + margin_x)
        y2 = min(image_height, face.y2 + lower_margin)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "Bounding box khuôn m?t không h?p l?."
            )

        portrait = image[y1:y2, x1:x2].copy()

        return portrait, (x1, y1, x2, y2)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError(
                "?nh CCCD không du?c là None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "?nh CCCD ph?i là numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError(
                "?nh CCCD d?u vào r?ng."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "?nh CCCD ph?i có d?nh d?ng BGR 3 kênh."
            )
