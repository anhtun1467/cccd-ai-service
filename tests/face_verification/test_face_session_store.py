from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.face_session_store import (
    FaceSessionError,
    FaceSessionStore,
)


def build_ocr_result(
    project_root: Path,
    *,
    with_portrait: bool = True,
) -> dict[str, object]:
    card_path = project_root / "storage" / "outputs" / "request_card.jpg"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_bytes(b"card-image")

    portrait: dict[str, object] | None = None
    if with_portrait:
        portrait_path = (
            project_root
            / "storage"
            / "debug"
            / "request"
            / "fields"
            / "portrait_raw.jpg"
        )
        portrait_path.parent.mkdir(parents=True, exist_ok=True)
        portrait_path.write_bytes(b"portrait-image")
        portrait = {
            "rawImagePath": str(portrait_path),
            "success": True,
        }

    return {
        "status": "OCR_SUCCESS",
        "metadata": {"cardImage": str(card_path)},
        "portrait": portrait,
    }


def make_store(tmp_path: Path, *, max_attempts: int = 3) -> FaceSessionStore:
    return FaceSessionStore(
        project_root=tmp_path,
        session_dir=tmp_path / "storage" / "face_sessions",
        ttl_seconds=1800,
        max_attempts=max_attempts,
        expired_retention_seconds=86400,
    )


def test_create_session_keeps_reference_paths_server_side(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session = store.create_from_ocr_result(
        build_ocr_result(tmp_path),
        ocr_request_id="request123",
    )

    public = session.to_public_dict()
    assert public["can_verify"] is True
    assert public["remaining_attempts"] == 3
    assert "card_image_path" not in public
    assert "portrait_image_path" not in public
    assert store.resolve_card_image_path(session).is_file()
    assert store.resolve_portrait_image_path(session).is_file()


def test_match_consumes_session(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session = store.create_from_ocr_result(
        build_ocr_result(tmp_path),
        ocr_request_id="request123",
    )

    updated = store.record_attempt(
        session.session_id,
        verification_status="match",
        capture_source="camera",
    )

    assert updated.status == "verified"
    assert updated.attempts_used == 1
    assert updated.last_capture_source == "camera"
    with pytest.raises(FaceSessionError) as context:
        store.require_verifiable(session.session_id)
    assert context.value.error_code == "FACE_SESSION_ALREADY_VERIFIED"


def test_failed_attempts_are_limited(tmp_path: Path) -> None:
    store = make_store(tmp_path, max_attempts=2)
    session = store.create_from_ocr_result(
        build_ocr_result(tmp_path, with_portrait=False),
        ocr_request_id="request123",
    )

    first = store.record_attempt(
        session.session_id,
        verification_status="not_match",
        capture_source="upload",
    )
    second = store.record_attempt(
        session.session_id,
        verification_status="error",
        capture_source="camera",
        error_code="WEBCAM_FACE_NOT_FOUND",
    )

    assert first.status == "active"
    assert first.remaining_attempts == 1
    assert second.status == "exhausted"
    assert second.remaining_attempts == 0
    with pytest.raises(FaceSessionError) as context:
        store.require_verifiable(session.session_id)
    assert context.value.status_code == 429


def test_expired_session_returns_gone(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session = store.create_from_ocr_result(
        build_ocr_result(tmp_path),
        ocr_request_id="request123",
    )
    expired = replace(
        session,
        expires_at=(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat(),
    )
    with store._lock:
        store._write_unlocked(expired)

    with pytest.raises(FaceSessionError) as context:
        store.require_verifiable(session.session_id)
    assert context.value.error_code == "FACE_SESSION_EXPIRED"
    assert context.value.status_code == 410


def test_reference_path_outside_ocr_storage_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    result = {
        "status": "OCR_SUCCESS",
        "metadata": {"cardImage": str(outside)},
    }

    with pytest.raises(FaceSessionError) as context:
        make_store(tmp_path).create_from_ocr_result(
            result,
            ocr_request_id="request123",
        )
    assert context.value.error_code == "OCR_REFERENCE_PATH_INVALID"


def test_invalid_session_id_is_rejected_before_file_access(tmp_path: Path) -> None:
    with pytest.raises(FaceSessionError) as context:
        make_store(tmp_path).get_public("../../secret")
    assert context.value.error_code == "INVALID_FACE_SESSION_ID"


def test_same_session_cannot_run_two_verifications_at_once(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    session = store.create_from_ocr_result(
        build_ocr_result(tmp_path),
        ocr_request_id="request123",
    )

    with store.verification_lease(session.session_id):
        with pytest.raises(FaceSessionError) as context:
            with store.verification_lease(session.session_id):
                pass

    assert context.value.error_code == "FACE_SESSION_BUSY"
    # Lease phải được nhả kể cả khi block bên trong kết thúc.
    with store.verification_lease(session.session_id) as leased:
        assert leased.session_id == session.session_id


def test_session_lease_is_shared_between_store_instances(
    tmp_path: Path,
) -> None:
    first_store = make_store(tmp_path)
    second_store = make_store(tmp_path)
    session = first_store.create_from_ocr_result(
        build_ocr_result(tmp_path),
        ocr_request_id="request123",
    )

    with first_store.verification_lease(session.session_id):
        with pytest.raises(FaceSessionError) as context:
            with second_store.verification_lease(session.session_id):
                pass

    assert context.value.error_code == "FACE_SESSION_BUSY"
