"""
Tiresias - main.py
====================
Phase 2: The Eyes + Depth Perception

Opens the webcam, captures frames in real-time, and displays
both the live feed and a MiDaS depth map side by side.
Press 'q' to quit.
"""

import cv2
import sys
from depth_estimation import DepthEstimator


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
    """Main loop: capture frames, estimate depth, and display both."""
    cap = initialize_camera()
    depth_estimator = DepthEstimator()

    print("[Tiresias] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        # Generate colorized depth map
        depth_colorized = depth_estimator.estimate(frame)

        # Display both windows
        cv2.imshow("Tiresias - Live Feed", frame)
        cv2.imshow("Tiresias - Depth Map", depth_colorized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Tiresias] Camera released. Goodbye.")


if __name__ == "__main__":
    run()
