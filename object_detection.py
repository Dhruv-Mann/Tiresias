"""
Tiresias - object_detection.py
================================
Phase 4: The Brain (Object Detection — YOLO-World)

Uses YOLO-World (Ultralytics) for open-vocabulary object detection.
Unlike YOLOv8's fixed 80 COCO classes, YOLO-World can detect ANY
object described by a text prompt — critical for blind assistance
where we need to detect stairs, curbs, potholes, traffic lights, etc.
Draws bounding boxes, labels, and calculates center points
to determine if objects are in the user's path.
"""

import cv2
import numpy as np
from ultralytics import YOLOWorld


# Classes relevant to blind/visually-impaired navigation.
# YOLO-World is open-vocabulary — add or remove classes as needed.
DEFAULT_CLASSES = [
    # People & animals
    "person", "dog", "cat", "bicycle",
    # Vehicles
    "car", "bus", "truck", "motorcycle", "scooter",
    # Indoor obstacles
    "chair", "table", "couch", "bed", "door", "cabinet",
    # Outdoor obstacles
    "bench", "pole", "fire hydrant", "trash can", "cone",
    "barrier", "fence", "wall",
    # Navigation hazards
    "stairs", "curb", "pothole",
    # Traffic
    "traffic light", "stop sign", "crosswalk",
    # Common objects
    "bag", "suitcase", "umbrella", "stroller",
    "wheelchair", "shopping cart",
]


class ObjectDetector:
    """Wraps YOLO-World for real-time open-vocabulary object detection with spatial awareness."""

    # Frame zones: divide width into thirds (Left, Center, Right)
    ZONE_LEFT = "Left"
    ZONE_CENTER = "Center"
    ZONE_RIGHT = "Right"

    def __init__(
        self,
        model_name: str = "yolov8x-worldv2.pt",
        confidence: float = 0.15,
        classes: list[str] | None = None,
    ):
        """
        Args:
            model_name: YOLO-World model variant.
                        'yolov8s-worldv2.pt' = small (fastest),
                        'yolov8m-worldv2.pt' = medium,
                        'yolov8l-worldv2.pt' = large,
                        'yolov8x-worldv2.pt' = extra-large (most accurate).
            confidence: Minimum confidence threshold for detections (0.0 - 1.0).
            classes:    List of text class prompts for open-vocabulary detection.
                        If None, uses DEFAULT_CLASSES (optimized for blind assistance).
        """
        print(f"[Tiresias] Loading YOLO-World model: {model_name}...")
        self.model = YOLOWorld(model_name)
        self.confidence = confidence

        # Set open-vocabulary classes
        self.classes = classes or DEFAULT_CLASSES
        self.model.set_classes(self.classes)
        print(f"[Tiresias] YOLO-World loaded with {len(self.classes)} custom classes.")
        print(f"[Tiresias] Classes: {', '.join(self.classes)}")

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
            zone = det["zone"]
            proximity = det.get("proximity", "")

            # Color based on proximity + zone
            if proximity == "NEAR" and zone == self.ZONE_CENTER:
                color = (0, 0, 255)       # RED — danger: near + center
            elif proximity == "NEAR":
                color = (0, 140, 255)     # ORANGE — near but off to the side
            elif proximity == "MID":
                color = (0, 255, 255)     # YELLOW — moderate distance
            else:
                color = (0, 255, 0)       # GREEN — far / safe

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Draw label with proximity and zone
            text = f"{label} - {proximity} [{zone}]" if proximity else f"{label} [{zone}]"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
