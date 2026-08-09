from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.api import face_verification as face_api
from app.services.face_session_store import FaceSession


def make_session() -> FaceSession:
    now = datetime.now(timezone.utc)
    return FaceSession(
        version=1,
        session_id="s" * 43,
        ocr_request_id="ocrrequest123",
        card_image_path="storage/outputs/card.jpg",
        portrait_image_path="storage/debug/request/fields/portrait.jpg",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=30)).isoformat(),
        max_attempts=5,
    )


def quality(source: str) -> dict[str, object]:
    return {
        "source": source,
        "status": "pass",
        "is_acceptable": True,
        "sharpness": 120.0,
        "brightness": 128.0,
        "face_width": 150,
        "face_height": 180,
        "face_height_ratio": 0.4,
        "center_offset": 0.05,
        "errors": [],
        "warnings": [],
    }


def verification_payload() -> dict[str, object]:
    return {
        "success": True,
        "request_id": "face-request",
        "status": "match",
        "is_match": True,
        "needs_review": False,
        "message": "Khuôn mặt trùng khớp với ảnh chân dung trên CCCD.",
        "similarity": 0.78,
        "distance": 0.22,
        "match_threshold": 0.50,
        "review_threshold": 0.40,
        "processing_time_ms": 120.0,
        "cccd_detection_score": 0.95,
        "webcam_detection_score": 0.96,
        "portrait_method": "ocr_portrait_crop",
        "portrait_bbox": (10, 10, 110, 150),
        "portrait_rotation_degrees": 0,
        "cccd_face_count": 1,
        "webcam_face_count": 1,
        "embedding_dimension": 512,
        "model_name": "buffalo_l",
        "quality_adjusted": False,
        "liveness_checked": False,
        "cccd_quality": quality("cccd"),
        "webcam_quality": quality("webcam"),
        "reference_source": "ocr_portrait_crop",
    }


def test_verify_from_ocr_uses_server_reference_and_only_selfie_upload(
    monkeypatch,
) -> None:
    session = make_session()
    updated = replace(
        session,
        status="verified",
        attempts_used=1,
        last_verification_status="match",
        last_capture_source="camera",
    )
    pipeline_output = SimpleNamespace(
        verification=SimpleNamespace(status="match"),
        to_dict=verification_payload,
    )

    @contextmanager
    def fake_lease(value):
        yield session

    monkeypatch.setattr(
        face_api.face_session_store,
        "verification_lease",
        fake_lease,
    )
    monkeypatch.setattr(
        face_api.face_session_store,
        "resolve_card_image_path",
        lambda value: Path("server-card.jpg"),
    )
    monkeypatch.setattr(
        face_api.face_session_store,
        "resolve_portrait_image_path",
        lambda value: Path("server-portrait.jpg"),
    )
    monkeypatch.setattr(
        face_api.face_session_store,
        "record_attempt",
        lambda *args, **kwargs: updated,
    )
    received: dict[str, object] = {}

    def fake_process(**kwargs):
        received.update(kwargs)
        return pipeline_output

    monkeypatch.setattr(
        face_api.face_verification_pipeline,
        "process_from_ocr_paths",
        fake_process,
    )

    upload = UploadFile(
        file=BytesIO(b"selfie-bytes"),
        filename="selfie.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )
    response = asyncio.run(
        face_api.verify_face_from_ocr(
            session_id=session.session_id,
            selfie_image=upload,
            capture_source="camera",
        )
    )

    payload = response.model_dump(mode="json")
    assert payload["ocr_session_id"] == session.session_id
    assert payload["capture_source"] == "camera"
    assert payload["session"]["status"] == "verified"
    assert received["card_image_path"] == Path("server-card.jpg")
    assert received["portrait_image_path"] == Path("server-portrait.jpg")
    assert received["webcam_image_bytes"] == b"selfie-bytes"
