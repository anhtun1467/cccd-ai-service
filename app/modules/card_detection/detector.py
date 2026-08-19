from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import BadRequestException
from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.enhancer import ImageEnhancer
from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.perspective_transformer import PerspectiveTransformer
from app.modules.card_detection.preprocessor import ImagePreprocessor


class CardDetector:
    """
    Pipeline phát hiện và chuẩn hóa ảnh CCCD.
    """

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        self.contour_detector = ContourDetector()
        self.transformer = PerspectiveTransformer()
        self.enhancer = ImageEnhancer()

    def _normalized_corner_distance(
        self,
        first: np.ndarray,
        second: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> float:
        first_rect = self.transformer.order_points(first)
        second_rect = self.transformer.order_points(second)
        height, width = image_shape[:2]
        diagonal = max(float(np.hypot(width, height)), 1.0)
        return float(
            np.mean(np.linalg.norm(first_rect - second_rect, axis=1))
            / diagonal
        )

    @staticmethod
    def _expand_source_corners(
        points: np.ndarray,
        image_shape: tuple[int, ...],
        scale: float,
    ) -> np.ndarray:
        corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
        center = corners.mean(axis=0)
        expanded = center + (corners - center) * float(scale)
        height, width = image_shape[:2]
        expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
        return expanded.reshape(4, 1, 2).astype(np.float32)

    @staticmethod
    def _scale_source_corners(
        points: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        corners = np.asarray(points, dtype=np.float32).reshape(4, 2).copy()
        corners[:, 0] *= float(scale_x)
        corners[:, 1] *= float(scale_y)
        return corners.reshape(4, 1, 2).astype(np.float32)

    def build_geometry_candidates(
        self,
        image: np.ndarray,
        primary_corners: np.ndarray,
        primary_geometry: dict,
        source_candidates: list[dict] | None = None,
    ) -> list[dict]:
        """Tạo các cách nắn an toàn để tầng OCR chọn bằng dữ liệu thật."""
        height, width = image.shape[:2]
        frame_aspect = max(width, height) / max(min(width, height), 1)
        aspect_error = abs(
            frame_aspect - self.transformer.CARD_ASPECT_RATIO
        ) / self.transformer.CARD_ASPECT_RATIO
        source_coverage = float(
            primary_geometry.get("sourceCoverageRatio", 0.0)
        )
        bounding_coverage = float(
            primary_geometry.get("sourceBoundingCoverageRatio", 0.0)
        )
        frame_deviation = float(
            primary_geometry.get("frameCornerDeviation", 1.0)
        )
        severity = float(
            primary_geometry.get("perspectiveSeverity", 0.0)
        )
        edge_touch_count = int(
            primary_geometry.get("edgeTouchCount", 0)
        )

        frame_available = bool(
            aspect_error <= 0.18
            and (
                source_coverage >= 0.48
                or bounding_coverage >= 0.70
                or edge_touch_count >= 3
            )
        )
        frame_recommended = bool(
            frame_available
            and (
                frame_deviation >= 0.018
                or severity >= 0.050
            )
        )
        primary_geometry["frameAspectErrorRatio"] = round(
            aspect_error,
            4,
        )
        primary_geometry["fullFrameCandidateAvailable"] = frame_available
        primary_geometry["fullFrameCandidateRecommended"] = (
            frame_recommended
        )

        # Detector đã xác nhận đủ bốn cạnh và nội dung trải toàn thẻ thì
        # không tạo thêm ảnh nắn gần như giống hệt. Đây là đường nhanh cho
        # ảnh chụp đúng khung và ảnh scan nguyên thẻ.
        if (
            bool(primary_geometry.get("wholeCardReliable"))
            and not source_candidates
        ):
            primary_geometry["geometryCandidateCount"] = 1
            primary_geometry["geometryCandidatesSkippedReason"] = (
                "PRIMARY_WHOLE_CARD_RELIABLE"
            )
            return []

        candidates: list[dict] = []
        candidate_errors: list[str] = []
        seen_corners: list[np.ndarray] = [
            np.asarray(primary_corners, dtype=np.float32)
        ]

        def add_perspective_candidate(
            name: str,
            corners: np.ndarray,
            extra_geometry: dict | None = None,
        ) -> None:
            if any(
                self._normalized_corner_distance(
                    corners,
                    existing,
                    image.shape,
                )
                < 0.004
                for existing in seen_corners
            ):
                return
            try:
                candidate_image, geometry = (
                    self.transformer.transform_with_metadata(
                        image,
                        corners,
                    )
                )
            except (TypeError, ValueError, cv2.error) as error:
                candidate_errors.append(f"{name}: {error}")
                return
            geometry["candidateName"] = name
            if extra_geometry:
                geometry.update(extra_geometry)
            candidates.append(
                {
                    "name": name,
                    "cardImage": candidate_image,
                    "geometry": geometry,
                }
            )
            seen_corners.append(np.asarray(corners, dtype=np.float32))

        # Ứng viên bốn cạnh Hough được ưu tiên trước các phép nới hình học.
        # Đây thường là quad đúng khi ảnh chụp xa làm threshold sáng dính
        # thẻ vào bàn tay, màn hình hoặc mặt bàn.
        for source_candidate in source_candidates or []:
            if not isinstance(source_candidate, dict):
                continue
            corners = source_candidate.get("corners")
            if corners is None:
                continue
            name = str(
                source_candidate.get("name")
                or "hough_quadrilateral"
            )
            detection = source_candidate.get("detection")
            add_perspective_candidate(
                name,
                np.asarray(corners, dtype=np.float32),
                {
                    "candidateName": name,
                    "detectionMethod": "hough_quadrilateral",
                    "detectionMetrics": (
                        detection if isinstance(detection, dict) else {}
                    ),
                    "wholeCardReliable": bool(
                        isinstance(detection, dict)
                        and detection.get("wholeCardCandidate")
                    ),
                },
            )

        if frame_available and (
            frame_recommended or frame_deviation >= 0.008
        ):
            try:
                frame_image, frame_geometry = (
                    self.transformer.normalize_full_frame_with_metadata(
                        image
                    )
                )
                candidates.append(
                    {
                        "name": "full_frame",
                        "cardImage": frame_image,
                        "geometry": frame_geometry,
                    }
                )
                seen_corners.append(
                    np.asarray(
                        frame_geometry["sourceCorners"],
                        dtype=np.float32,
                    )
                )
            except (TypeError, ValueError, cv2.error) as error:
                candidate_errors.append(f"full_frame: {error}")

        expanded_corners = self._expand_source_corners(
            primary_corners,
            image.shape,
            scale=1.045,
        )
        add_perspective_candidate(
            "expanded_contour",
            expanded_corners,
            {"sourceExpansionScale": 1.045},
        )

        rotated_rectangle = cv2.boxPoints(
            cv2.minAreaRect(
                np.asarray(primary_corners, dtype=np.float32).reshape(-1, 2)
            )
        ).reshape(4, 1, 2)
        add_perspective_candidate(
            "rotated_rectangle",
            rotated_rectangle,
        )

        if candidate_errors:
            primary_geometry["candidateErrors"] = candidate_errors
        primary_geometry["geometryCandidateCount"] = len(candidates) + 1
        return candidates[:3]

    def detect_from_path(
        self,
        image_path: str,
        output_dir: str | None = None,
    ) -> dict:
        image = self.loader.load(image_path)

        resized, ratio = self.preprocessor.resize(image)

        multiple_cards, multiple_mask = (
            self.contour_detector.find_multiple_card_contours_from_image(
                resized
            )
        )

        if len(multiple_cards) >= 2:
            if output_dir:
                self.save_multiple_card_debug(
                    resized,
                    multiple_mask,
                    multiple_cards,
                    output_dir,
                )
            raise BadRequestException(
                "Phát hiện nhiều CCCD trong cùng một ảnh",
                data={
                    "errorCode": "MULTIPLE_CARDS",
                    "stage": "CARD_COUNT",
                    "reason": "Phát hiện nhiều CCCD trong cùng một ảnh",
                    "cardCount": len(multiple_cards),
                    "debugImages": {
                        "candidates": str(
                            Path(output_dir) / "detector_00_multiple_cards.jpg"
                        ) if output_dir else None,
                        "mask": str(
                            Path(output_dir) / "detector_00_multiple_mask.jpg"
                        ) if output_dir else None,
                    },
                    "suggestion": (
                        "Vui lòng chỉ chụp một CCCD trong mỗi ảnh."
                    ),
                },
            )

        (
            card_contour,
            mask,
            contours,
            contour_detection,
        ) = self.contour_detector.find_card_contour_candidates_from_image(
            resized
        )

        if card_contour is None:
            raise BadRequestException("Không phát hiện được vùng CCCD")

        alternate_candidates = contour_detection.get(
            "alternateCandidates",
            [],
        )
        if not isinstance(alternate_candidates, list):
            alternate_candidates = []

        # Phát hiện trên ảnh cao 700 px để nhanh, nhưng nếu ảnh gốc lớn hơn
        # thì perspective transform trực tiếp từ ảnh gốc. Ảnh chụp xa nhờ
        # vậy không mất thêm chi tiết chữ trước khi vào OCR.
        use_original_resolution = bool(
            image.shape[0] > resized.shape[0]
            or image.shape[1] > resized.shape[1]
        )
        geometry_image = image if use_original_resolution else resized
        scale_x = (
            image.shape[1] / float(resized.shape[1])
            if use_original_resolution
            else 1.0
        )
        scale_y = (
            image.shape[0] / float(resized.shape[0])
            if use_original_resolution
            else 1.0
        )
        geometry_corners = self._scale_source_corners(
            card_contour,
            scale_x,
            scale_y,
        )
        scaled_alternates: list[dict] = []
        for candidate in alternate_candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_corners = candidate.get("corners")
            if candidate_corners is None:
                continue
            scaled_alternates.append({
                **candidate,
                "corners": self._scale_source_corners(
                    np.asarray(candidate_corners, dtype=np.float32),
                    scale_x,
                    scale_y,
                ),
            })

        warped, geometry = self.transformer.transform_with_metadata(
            geometry_image,
            geometry_corners,
        )
        detection_method = str(
            contour_detection.get("detectionMethod")
            or "brightness_contour"
        )
        if detection_method == "hough_quadrilateral":
            geometry["candidateName"] = "hough_quadrilateral"
        geometry["detectionMethod"] = detection_method
        geometry["detectionScore"] = contour_detection.get(
            "detectionScore"
        )
        geometry["detectionMetrics"] = contour_detection.get(
            "detectionMetrics",
            {},
        )
        geometry["primaryAreaRatio"] = contour_detection.get(
            "primaryAreaRatio"
        )
        geometry["primaryEdgeTouchCount"] = contour_detection.get(
            "primaryEdgeTouchCount"
        )
        geometry["wholeCardReliable"] = bool(
            contour_detection.get("wholeCardReliable")
        )
        geometry["houghFallbackEvaluated"] = bool(
            contour_detection.get("houghFallbackEvaluated")
        )
        geometry["houghSkippedReason"] = contour_detection.get(
            "houghSkippedReason"
        )
        geometry["relaxedForegroundContour"] = contour_detection.get(
            "relaxedForegroundContour"
        )
        geometry["contourReplacement"] = contour_detection.get(
            "contourReplacement"
        )
        geometry["geometrySource"] = (
            "original_image" if use_original_resolution else "resized_image"
        )
        geometry["detectorResizeScaleX"] = round(scale_x, 6)
        geometry["detectorResizeScaleY"] = round(scale_y, 6)
        geometry["detectorSourceCorners"] = (
            self.transformer.order_points(card_contour).tolist()
        )
        geometry_candidates = self.build_geometry_candidates(
            geometry_image,
            geometry_corners,
            geometry,
            source_candidates=scaled_alternates,
        )
        enhanced_images = self.enhancer.enhance(warped)

        result = {
            "success": True,
            "message": "Phát hiện CCCD thành công",
            "resizeRatio": ratio,
            "geometry": geometry,
            "cardCount": 1,
            "cardImage": warped,
            "enhancedImage": enhanced_images["final"],
            "cardCandidates": geometry_candidates,
            "debug": {
                "resized": resized,
                "mask": mask,
                "warped": warped,
                "enhanced": enhanced_images,
            },
        }

        if output_dir:
            self.save_debug_images(result, output_dir)

        return result

    @staticmethod
    def save_multiple_card_debug(
        resized: np.ndarray,
        mask: np.ndarray,
        card_contours: list[np.ndarray],
        output_dir: str,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        overlay = resized.copy()
        debug_contours = [
            np.rint(contour).astype(np.int32)
            for contour in card_contours
        ]
        cv2.drawContours(overlay, debug_contours, -1, (0, 0, 255), 5)
        for index, contour in enumerate(debug_contours, start=1):
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.putText(
                overlay,
                f"CCCD {index}",
                (x, max(35, y + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            str(output_path / "detector_00_multiple_mask.jpg"),
            mask,
        )
        cv2.imwrite(
            str(output_path / "detector_00_multiple_cards.jpg"),
            overlay,
        )

    def save_debug_images(self, result: dict, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(output_path / "detector_01_resized.jpg"),
            result["debug"]["resized"],
        )
        cv2.imwrite(str(output_path / "detector_02_mask.jpg"), result["debug"]["mask"])
        cv2.imwrite(
            str(output_path / "detector_03_warped.jpg"),
            result["debug"]["warped"],
        )
        cv2.imwrite(
            str(output_path / "detector_04_enhanced.jpg"),
            result["debug"]["enhanced"]["final"],
        )

        for index, candidate in enumerate(
            result.get("cardCandidates", []),
            start=1,
        ):
            candidate_image = candidate.get("cardImage")
            if candidate_image is None:
                continue
            candidate_name = str(candidate.get("name", f"candidate_{index}"))
            safe_name = "".join(
                character
                for character in candidate_name
                if character.isalnum() or character in ("_", "-")
            )
            cv2.imwrite(
                str(
                    output_path
                    / f"detector_03_candidate_{index:02d}_{safe_name}.jpg"
                ),
                candidate_image,
            )
