"""
Tiresias - main.py
====================
Phase 3: The Eyes + Depth Perception + Object Detection

Opens the webcam, captures frames in real-time, and displays:
  1. Live feed with YOLOv8 bounding boxes, labels, and zone info.
  2. Colorized MiDaS depth map (RED = near, BLUE = far).
Press 'q' to quit.
"""

import cv2
import sys
from depth_estimation import DepthEstimator
from object_detection import ObjectDetector


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
    """Main loop: capture frames, detect objects, estimate depth, and display."""
    cap = initialize_camera()
    depth_estimator = DepthEstimator()
    detector = ObjectDetector()

    print("[Tiresias] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        # Run object detection
        detections = detector.detect(frame)

        # Draw bounding boxes, labels, center points, and zones on the frame
        annotated_frame = detector.draw_detections(frame, detections)

        # Generate colorized depth map
        depth_colorized = depth_estimator.estimate(frame)

        # Display both windows
        cv2.imshow("Tiresias - Live Feed", annotated_frame)
        cv2.imshow("Tiresias - Depth Map", depth_colorized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Tiresias] Camera released. Goodbye.")


if __name__ == "__main__":
    run()
