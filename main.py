"""
Tiresias - main.py
====================
Phase 1: The Eyes (Basic Webcam Feed)

Opens the default webcam, captures frames in real-time,
and displays them in a window. Press 'q' to quit.
"""

import cv2
import sys


def initialize_camera(camera_index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Initialize and configure the webcam."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Is a webcam connected?")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"[Tiresias] Camera initialized at {width}x{height}")
    return cap


def run():
    """Main loop: capture and display webcam frames."""
    cap = initialize_camera()

    print("[Tiresias] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        cv2.imshow("Tiresias - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Tiresias] Camera released. Goodbye.")


if __name__ == "__main__":
    run()
