"""
Tiresias - main.py
====================
Phase 5: Sensor Fusion + Audio Alerts

Opens the webcam, captures frames in real-time, and:
  1. Detects objects with YOLO-World (open-vocabulary).
  2. Estimates depth with MiDaS.
  3. Fuses both: samples depth at each detection box → NEAR/MID/FAR.
  4. Speaks audio alerts for nearby obstacles (non-blocking TTS).
  5. Displays annotated live feed + depth map.
Press 'q' to quit.
"""

import cv2
import numpy as np
import sys
from audio_engine import AudioEngine
from depth_estimation import DepthEstimator
from object_detection import ObjectDetector

# Depth thresholds on MiDaS normalized 0-255 scale (HIGH = near)
DEPTH_NEAR = 170   # > 170 → NEAR (top ~33%)
DEPTH_MID = 85     # 85-170 → MID  (middle ~33%)
                    # < 85  → FAR   (bottom ~33%)


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


def classify_depth(value: int) -> str:
    """Map a normalized depth value (0-255) to a proximity category."""
    if value > DEPTH_NEAR:
        return "NEAR"
    elif value > DEPTH_MID:
        return "MID"
    return "FAR"


def fuse_detections_with_depth(
    detections: list[dict], depth_map: np.ndarray
) -> list[dict]:
    """
    Enrich each detection with a proximity estimate from the depth map.

    For each bounding box, slices the depth ROI, takes the median,
    and maps it to NEAR / MID / FAR.
    """
    h, w = depth_map.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        # Clamp coordinates to depth map boundaries
        x1c = max(0, min(x1, w - 1))
        x2c = max(0, min(x2, w))
        y1c = max(0, min(y1, h - 1))
        y2c = max(0, min(y2, h))

        roi = depth_map[y1c:y2c, x1c:x2c]

        if roi.size == 0:
            det["proximity"] = "FAR"
        else:
            median_depth = int(np.median(roi))
            det["proximity"] = classify_depth(median_depth)

    return detections


def run():
    """Main loop: capture → detect → depth → fuse → alert → display."""
    cap = initialize_camera()
    depth_estimator = DepthEstimator()
    detector = ObjectDetector()
    audio = AudioEngine()

    print("[Tiresias] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame. Exiting.")
            break

        # Run object detection
        detections = detector.detect(frame)

        # Get raw depth map (grayscale 0-255, HIGH = near)
        depth_raw = depth_estimator.get_raw_depth(frame)

        # Fuse: sample depth at each detection → add 'proximity' field
        detections = fuse_detections_with_depth(detections, depth_raw)

        # Trigger audio alerts for nearby obstacles
        for det in detections:
            audio.alert(det["label"], det["zone"], det["proximity"])

        # Draw annotated frame with proximity labels
        annotated_frame = detector.draw_detections(frame, detections)

        # Generate colorized depth map for display
        depth_colorized = depth_estimator.estimate(frame)

        # Display both windows
        cv2.imshow("Tiresias - Live Feed", annotated_frame)
        cv2.imshow("Tiresias - Depth Map", depth_colorized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    audio.shutdown()
    cap.release()
    cv2.destroyAllWindows()
    print("[Tiresias] Camera released. Goodbye.")


if __name__ == "__main__":
    run()
