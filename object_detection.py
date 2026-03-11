"""
Tiresias - object_detection.py
================================
Phase 3: The Brain (Object Detection)

Uses YOLOv8 (Ultralytics) to detect objects in webcam frames.
Draws bounding boxes, labels, and calculates center points
to determine if objects are in the user's path.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """Wraps YOLOv8 for real-time object detection with spatial awareness."""

    # Frame zones: divide width into thirds (Left, Center, Right)
    ZONE_LEFT = "Left"
    ZONE_CENTER = "Center"
    ZONE_RIGHT = "Right"

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Args:
            model_name: YOLOv8 model variant. 'n' = nano (fastest), 's' = small,
                        'm' = medium, 'l' = large, 'x' = extra-large (most accurate).
            confidence: Minimum confidence threshold for detections (0.0 - 1.0).
        """
        print(f"[Tiresias] Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        print("[Tiresias] YOLOv8 model loaded successfully.")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run object detection on a frame.

        Args:
            frame: BGR image (H, W, 3) from cv2.VideoCapture.

        Returns:
            List of detection dicts, each containing:
                - 'label': str (e.g., 'person', 'chair', 'cup')
                - 'confidence': float (0.0 - 1.0)
                - 'box': tuple (x1, y1, x2, y2) pixel coordinates
                - 'center': tuple (cx, cy) center point of the box
                - 'zone': str ('Left', 'Center', 'Right')
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Calculate center point of the bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Determine which zone the center falls in
                zone = self._get_zone(cx, frame.shape[1])

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "zone": zone,
                })

        return detections

    def _get_zone(self, cx: int, frame_width: int) -> str:
        """Determine if the center point is in the Left, Center, or Right third."""
        third = frame_width // 3

        if cx < third:
            return self.ZONE_LEFT
        elif cx < third * 2:
            return self.ZONE_CENTER
        else:
            return self.ZONE_RIGHT

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes, labels, center points, and zone info on the frame.

        Args:
            frame: BGR image to draw on (will be modified in-place).
            detections: List of detection dicts from detect().

        Returns:
            The frame with annotations drawn.
        """
        h, w = frame.shape[:2]
        third = w // 3

        # Draw zone divider lines
        cv2.line(frame, (third, 0), (third, h), (255, 255, 255), 1)
        cv2.line(frame, (third * 2, 0), (third * 2, h), (255, 255, 255), 1)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            label = det["label"]
            conf = det["confidence"]
            zone = det["zone"]

            # Color based on zone: Center = Red (danger), sides = Green (safe)
            color = (0, 0, 255) if zone == self.ZONE_CENTER else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Draw label with confidence and zone
            text = f"{label} {conf:.0%} [{zone}]"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
