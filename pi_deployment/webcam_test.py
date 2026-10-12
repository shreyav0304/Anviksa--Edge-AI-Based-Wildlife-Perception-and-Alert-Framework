"""USB webcam preview using Tkinter; compatible with opencv-python-headless."""
from __future__ import annotations
import argparse
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    try:
        import cv2
        import tkinter as tk
        from PIL import Image, ImageTk
    except (ImportError, OSError) as exc:
        print(f"Preview dependencies unavailable: {exc}. Activate the app environment.", file=sys.stderr)
        return 1
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        print(f"Preview display unavailable: {exc}. Launch from the Pi desktop.", file=sys.stderr)
        return 1
    capture = None
    try:
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            print(f"Cannot open camera {args.camera}; check the device or try --camera 1.", file=sys.stderr)
            return 1
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        window.title("Anviksa - USB Webcam Test")
        label = tk.Label(window, bg="#101915")
        label.pack()
        failed = False
        def show_frame():
            nonlocal failed
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera disconnected or frame read failed")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((640, 480))
                label.image = ImageTk.PhotoImage(image)
                label.configure(image=label.image)
                window.after(30, show_frame)
            except (RuntimeError, cv2.error) as exc:
                failed = True
                print(str(exc), file=sys.stderr)
                window.destroy()
        window.bind("<q>", lambda _event: window.destroy())
        window.after(0, show_frame)
        window.mainloop()
        return 1 if failed else 0
    except (cv2.error, OSError, RuntimeError) as exc:
        print(f"Webcam preview failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        try:
            window.destroy()
        except tk.TclError:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
