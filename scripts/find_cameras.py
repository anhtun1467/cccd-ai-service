from __future__ import annotations

import time

import cv2


def test_camera(
    index: int,
    backend: int,
    backend_name: str,
) -> bool:
    """
    Ki?m tra m?t camera index.

    Returns:
        True n?u camera m? và d?c du?c h́nh ?nh.
    """
    camera = cv2.VideoCapture(index, backend)

    try:
        if not camera.isOpened():
            return False

        time.sleep(0.3)

        valid_frame = None

        for _ in range(10):
            success, frame = camera.read()

            if success and frame is not None:
                valid_frame = frame

        if valid_frame is None:
            return False

        height, width = valid_frame.shape[:2]

        fps = camera.get(cv2.CAP_PROP_FPS)

        print(
            f"[OK] index={index} | "
            f"backend={backend_name} | "
            f"resolution={width}x{height} | "
            f"fps={fps:.1f}"
        )

        return True

    finally:
        camera.release()


def find_available_cameras(
    max_index: int = 10,
) -> list[int]:
    """
    T́m toàn b? camera có th? s? d?ng.
    """
    available_indices: set[int] = set()

    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MediaFoundation"),
    ]

    for index in range(max_index):
        for backend, backend_name in backends:
            try:
                found = test_camera(
                    index=index,
                    backend=backend,
                    backend_name=backend_name,
                )
            except cv2.error as error:
                print(
                    f"[WARNING] Camera {index}: {error}"
                )
                found = False

            if found:
                available_indices.add(index)
                break

    return sorted(available_indices)


def main() -> None:
    print("=" * 70)
    print("TIM CAMERA TREN WINDOWS")
    print("=" * 70)

    camera_indices = find_available_cameras(
        max_index=10
    )

    print("=" * 70)

    if not camera_indices:
        print("[ERROR] Khong tim thay camera nao.")
        print("")
        print("Kiem tra:")
        print("1. Logitech C922 da cam vao USB.")
        print("2. Da dong Camera, OBS, Zoom, Discord.")
        print("3. Windows cho phep ung dung desktop dung camera.")
        return

    print(
        "Camera index kha dung: "
        f"{camera_indices}"
    )


if __name__ == "__main__":
    main()
