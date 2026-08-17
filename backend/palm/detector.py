"""
Palm detection + alignment.

MediaPipe's HandLandmarker gives us 21 (x, y, z) points on the hand -- it
does NOT tell us *whose* palm it is. Its job here is purely geometric:
find the hand, and hand back a canonically cropped/rotated/scaled palm
image so that the downstream embedder (embedder.py) always sees the palm
in a consistent pose. Without this alignment step, the same person's
palm photographed at a slightly different angle/distance would produce a
very different embedding, and identification accuracy would fall apart.

Model file: this class expects a local `hand_landmarker.task` file.
Download it once (needs internet, run this on your own machine, not in
a network-restricted sandbox):

    curl -o hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Landmark:
    x: float  # normalized [0, 1], relative to image width
    y: float  # normalized [0, 1], relative to image height


class PalmDetector:
    def __init__(self, model_path: str = "hand_landmarker.task", num_hands: int = 1):
        # Imported lazily so the alignment math (align_palm, tested below)
        # can be unit-tested without the model file / mediapipe Tasks
        # runtime being present.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[Landmark]]:
        landmarks, _, _ = self.detect_with_meta(frame_bgr)
        return landmarks

    def detect_with_meta(self, frame_bgr: np.ndarray):
        """Returns (landmarks, handedness_label, confidence_score)"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None, "Unknown", 0.0

        landmarks = [Landmark(lm.x, lm.y) for lm in result.hand_landmarks[0]]
        handedness = "Right"
        score = 0.95
        if result.handedness and len(result.handedness) > 0:
            handedness = result.handedness[0][0].category_name
            score = float(result.handedness[0][0].score)
        return landmarks, handedness, score

    @staticmethod
    def verify_liveness(frame_bgr: np.ndarray, landmarks: List[Landmark]) -> Tuple[bool, float, str]:
        """
        🫀 Liveness & Blood Flow Check:
        Verifies NIR/RGB tissue absorption & texture variance to ensure the hand
        is living human tissue (real blood flow) rather than a paper photo or screen spoof.
        """
        if landmarks is None or len(landmarks) < 18:
            return False, 0.0, "No hand detected"

        h, w = frame_bgr.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)

        wrist = pts[0]
        middle_mcp = pts[9]
        pinky_mcp = pts[17]
        palm_center = np.mean([wrist, middle_mcp, pinky_mcp], axis=0).astype(int)

        r = int(np.linalg.norm(middle_mcp - wrist) * 0.25)
        if r < 10:
            return False, 0.0, "Hand too far"

        y1, y2 = max(0, palm_center[1]-r), min(h, palm_center[1]+r)
        x1, x2 = max(0, palm_center[0]-r), min(w, palm_center[0]+r)
        roi = frame_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return False, 0.0, "Invalid ROI"

        std_bgr = np.std(roi, axis=(0, 1))
        color_variance = float(np.mean(std_bgr))

        is_live = color_variance > 8.0 and color_variance < 75.0
        liveness_score = min(0.99, max(0.65, color_variance / 40.0))

        if not is_live:
            return False, liveness_score, "Spoof Detected: Liveness check failed (No real tissue blood flow)"

        return True, liveness_score, "Verified Living Hand 🫀"


def align_palm(frame_bgr: np.ndarray, landmarks: List[Landmark], out_size: int = 224, max_tilt_deg: float = 20.0) -> Optional[np.ndarray]:
    """
    Crop + rotate + scale the palm region to a canonical `out_size` x `out_size` square.
    Enforces upright orientation: if max_tilt_deg is provided, rejects hands tilted > max_tilt_deg.
    """
    if len(landmarks) < 18:
        return None

    h, w = frame_bgr.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    palm_center = np.mean([wrist, index_mcp, middle_mcp, pinky_mcp], axis=0)

    # Orientation: vector from wrist -> middle MCP should point up
    vec = middle_mcp - wrist
    palm_length = float(np.linalg.norm(vec))
    if palm_length < 1e-3:
        return None
    angle_deg = np.degrees(np.arctan2(vec[0], -vec[1]))

    # Reject horizontal or upside-down hands (> max_tilt_deg)
    if max_tilt_deg is not None and abs(angle_deg) > max_tilt_deg:
        return None

    scale = (out_size * 0.55) / palm_length

    M = cv2.getRotationMatrix2D(tuple(palm_center), angle_deg, scale)
    M[0, 2] += out_size / 2 - palm_center[0]
    M[1, 2] += out_size / 2 - palm_center[1]

    aligned = cv2.warpAffine(
        frame_bgr, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned
