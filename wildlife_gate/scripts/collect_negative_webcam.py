from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "wildlife_gate" / "raw_non_wildlife"

VALID_CATEGORIES = (
    "human",
    "empty_scene",
    "indoor_objects",
    "outdoor_background",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect NON_WILDLIFE webcam images for the ANVIKSA Wildlife Gate."
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=VALID_CATEGORIES,
        help="Type of negative scene being collected.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV webcam index. Default: 0",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Minimum seconds between automatic captures. Default: 1.5",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of images to save. Default: 100",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = RAW_ROOT / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. "
            "Try another camera index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved = 0
    auto_capture = False
    last_capture = 0.0

    print("=" * 60)
    print("ANVIKSA AI - NON-WILDLIFE DATA COLLECTOR")
    print("=" * 60)
    print(f"Category   : {args.category}")
    print(f"Output     : {output_dir}")
    print(f"Interval   : {args.interval:.1f} seconds")
    print(f"Max images : {args.max_images}")
    print()
    print("Controls:")
    print("  S = save one frame")
    print("  A = start/stop automatic capture")
    print("  Q = quit")
    print()
    print("Vary pose, distance, angle, lighting and background.")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[ERROR] Webcam frame could not be read.")
                break

            preview = frame.copy()

            status = "AUTO ON" if auto_capture else "AUTO OFF"

            cv2.putText(
                preview,
                f"{args.category} | Saved: {saved}/{args.max_images}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                preview,
                f"{status} | S=Save  A=Auto  Q=Quit",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            now = time.monotonic()

            should_save = (
                auto_capture
                and saved < args.max_images
                and now - last_capture >= args.interval
            )

            if should_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = output_dir / f"{args.category}_{timestamp}.jpg"

                if cv2.imwrite(str(output_path), frame):
                    saved += 1
                    last_capture = now
                    print(f"[SAVED {saved:03d}] {output_path.name}")
                else:
                    print("[WARNING] Failed to save frame.")

                if saved >= args.max_images:
                    auto_capture = False
                    print("[INFO] Maximum image count reached.")

            cv2.imshow("ANVIKSA AI - Negative Data Collector", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("a"):
                auto_capture = not auto_capture
                last_capture = 0.0
                print(
                    "[INFO] Automatic capture "
                    + ("STARTED" if auto_capture else "STOPPED")
                )

            if key == ord("s") and saved < args.max_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = output_dir / f"{args.category}_{timestamp}.jpg"

                if cv2.imwrite(str(output_path), frame):
                    saved += 1
                    print(f"[SAVED {saved:03d}] {output_path.name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print(f"[DONE] Saved {saved} image(s) to:")
    print(output_dir)


if __name__ == "__main__":
    main()