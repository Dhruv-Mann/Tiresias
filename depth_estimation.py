"""
Tiresias - depth_estimation.py
================================
Phase 2: Depth Perception (MiDaS)

Loads the MiDaS v2.1 Small model from PyTorch Hub and provides
a function to convert an RGB frame into a colorized depth map.
Near objects appear RED, far objects appear BLUE.
"""

import cv2
import numpy as np
import torch


class DepthEstimator:
    """Wraps the MiDaS Small model for real-time monocular depth estimation."""

    def __init__(self):
        print("[Tiresias] Loading MiDaS Small model...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Tiresias] Using device: {self.device}")

        # Load MiDaS Small from PyTorch Hub
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load the appropriate transform for the small model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

        print("[Tiresias] MiDaS model loaded successfully.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Takes a BGR frame from OpenCV, returns a colorized depth map.

        Args:
            frame: BGR image (H, W, 3) from cv2.VideoCapture.

        Returns:
            Colorized depth map (H, W, 3) where RED = near, BLUE = far.
        """
        # Convert BGR (OpenCV) -> RGB (MiDaS expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform: resizes and normalizes the image
        input_batch = self.transform(rgb_frame).to(self.device)

        # Run inference with no gradient computation (saves memory and speed)
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize prediction back to original frame dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Move to CPU and convert to NumPy
        depth_map = prediction.cpu().numpy()

        # Normalize to 0-255 range for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)

        # Apply colormap: JET maps low values to BLUE, high values to RED
        # MiDaS outputs higher values for CLOSER objects, so:
        #   RED = close (high depth value), BLUE = far (low depth value)
        colorized = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        return colorized

    def get_raw_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the raw normalized depth map (0-255, uint8) without colormap.
        Useful for Phase 4 when we need depth values for distance checks.

        Args:
            frame: BGR image (H, W, 3) from cv2.VideoCapture.

        Returns:
            Grayscale depth map (H, W) where HIGH = near, LOW = far.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb_frame).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        return depth_map.astype(np.uint8)
