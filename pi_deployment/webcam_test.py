from __future__ import annotations

import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "ANVIKSA AI - Webcam Test"


def main() -> None:
    print("=" * 55)
    print("ANVIKSA AI - USB WEBCAM TEST")
    print("=" * 55)

    print(f"[INFO] Opening webcam index {CAMERA_INDEX}...")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Webcam could not be opened.")
        print("[INFO] Try another camera index such as 1 or 2.")
        return

    # Request a reasonable resolution.
    # The webcam may choose the nearest supported resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("[OK] Webcam opened successfully.")
    print(f"[INFO] Resolution : {width} x {height}")
    print(f"[INFO] Camera FPS : {fps:.2f}")
    print()
    print("Press Q to close the webcam window.")

    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print("[ERROR] Failed to read frame from webcam.")
            break

        cv2.putText(
            frame,
            "ANVIKSA AI - USB Webcam",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[OK] Webcam released.")
    print("[OK] Webcam test completed.")


if __name__ == "__main__":
    main()