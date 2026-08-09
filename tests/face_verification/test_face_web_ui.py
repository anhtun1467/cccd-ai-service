from __future__ import annotations

from pathlib import Path


UI_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "web"
    / "face_verification.html"
)


def test_face_ui_contains_camera_and_upload_workflows() -> None:
    content = UI_PATH.read_text(encoding="utf-8")

    assert "navigator.mediaDevices.getUserMedia" in content
    assert 'id="quick-open-camera"' in content
    assert 'id="start-camera" class="primary"' in content
    assert 'id="selfie-section"' in content
    assert 'elements.quickOpenCamera.addEventListener("click", openCameraNow)' in content
    assert 'id="selfie-file"' in content
    assert 'id="cccd-file"' in content
    assert "/api/face-verification/verify-from-ocr" in content
    assert 'form.append("session_id", state.sessionId)' in content
    assert 'form.append("selfie_image"' in content


def test_face_ui_never_appends_cccd_to_face_request() -> None:
    content = UI_PATH.read_text(encoding="utf-8")
    face_request_start = content.index("async function verifyFace()")
    face_request_end = content.index(
        'elements.ocrButton.addEventListener',
        face_request_start,
    )
    face_request = content[face_request_start:face_request_end]

    assert 'form.append("file"' not in face_request
    assert 'form.append("card_image"' not in face_request


def test_main_app_exposes_easy_camera_entry_points() -> None:
    main_path = UI_PATH.parents[1] / "main.py"
    content = main_path.read_text(encoding="utf-8")

    assert '"/face-verification"' in content
    assert '"/camera"' in content
    assert '"/"' in content
    assert "MỞ GIAO DIỆN CAMERA TRỰC TIẾP" in content
