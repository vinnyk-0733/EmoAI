# Facial Landmark Analyzer — 68-Point LBF Model
# Uses OpenCV's FacemarkLBF (from opencv-contrib-python) for rich geometric
# emotion features: MAR, EAR, brow position, smile ratio, brow furrow, lip angle.

import cv2  # type: ignore[import-untyped]
import cv2.face  # type: ignore[import-untyped]
import numpy as np
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 68-point landmark index reference (iBUG convention)
#   Jaw:         0–16
#   Right Brow: 17–21
#   Left Brow:  22–26
#   Nose:       27–35
#   Right Eye:  36–41
#   Left Eye:   42–47
#   Outer Lip:  48–59
#   Inner Lip:  60–67
# ──────────────────────────────────────────────

class LBFLandmarkAnalyzer:
    """
    68-point facial landmark analyzer using OpenCV FacemarkLBF.
    Computes geometric metrics and returns emotion boost scores.
    """

    def __init__(self):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "lbfmodel.yaml")

        if not os.path.exists(model_path):
            logger.error(f"LBF model not found at {model_path}. "
                         "Download from: https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml")
            self.facemark = None
            return

        try:
            self.facemark = cv2.face.createFacemarkLBF()  # type: ignore[attr-defined]
            self.facemark.loadModel(model_path)  # type: ignore[union-attr]
            logger.info("LBF 68-point landmark model loaded")
        except Exception as e:
            logger.error(f"Failed to load LBF model: {e}")
            self.facemark = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # ── Core detection ──────────────────────────

    def _detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect 68 landmarks. Returns shape (68, 2) or None."""
        if self.facemark is None:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Need a face rect for FacemarkLBF
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        # FacemarkLBF expects faces as a numpy array
        faces_arr = np.array(faces)
        ok, landmarks_list = self.facemark.fit(gray, faces_arr)  # type: ignore[union-attr]
        if not ok or len(landmarks_list) == 0:
            return None

        # landmarks_list[0] has shape (1, 68, 2) — take the first face
        pts = landmarks_list[0].reshape(68, 2)  # type: ignore[index]
        return pts

    # ── Normalization helper ────────────────────

    @staticmethod
    def _interocular_distance(pts: np.ndarray) -> float:
        """Distance between outer eye corners (points 36 & 45) for normalization."""
        d = np.linalg.norm(pts[36] - pts[45])
        return max(d, 1.0)  # avoid division by zero

    # ── Geometric metrics ───────────────────────

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray) -> float:
        """
        Average EAR for both eyes.
        EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        Higher → eyes more open (surprise/fear)
        Lower  → eyes more closed (neutral/sad)
        """
        def ear_one_eye(p):
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])
            return (v1 + v2) / (2.0 * max(h, 1.0))

        right_eye = pts[36:42]
        left_eye = pts[42:48]
        return (ear_one_eye(right_eye) + ear_one_eye(left_eye)) / 2.0

    @staticmethod
    def _mouth_aspect_ratio(pts: np.ndarray) -> float:
        """
        MAR using inner lip points (60–67).
        MAR = (|p61-p67| + |p62-p66| + |p63-p65|) / (3 * |p60-p64|)
        Higher → mouth more open (surprise/happy-laugh)
        """
        v1 = np.linalg.norm(pts[61] - pts[67])
        v2 = np.linalg.norm(pts[62] - pts[66])
        v3 = np.linalg.norm(pts[63] - pts[65])
        h = np.linalg.norm(pts[60] - pts[64])
        return (v1 + v2 + v3) / (3.0 * max(h, 1.0))

    @staticmethod
    def _smile_ratio(pts: np.ndarray, iod: float) -> float:
        """
        Smile detection via:
        1. Mouth width (48↔54) normalized by IOD
        2. Lip corner height relative to mouth center
        Higher → wider smile → happy
        """
        mouth_width = np.linalg.norm(pts[48] - pts[54])
        normalized_width = mouth_width / iod

        # Lip corners (48, 54) vs bottom of upper lip center (62)
        corner_avg_y = (pts[48][1] + pts[54][1]) / 2.0
        center_y = pts[62][1]
        # If corners are ABOVE center → smile (upward curve)
        # In image coords, y increases downward, so corners < center means smile
        corner_lift = (center_y - corner_avg_y) / iod

        return normalized_width + corner_lift

    @staticmethod
    def _brow_eye_distance(pts: np.ndarray, iod: float) -> float:
        """
        Average distance from brow to eye, normalized by IOD.
        Higher → raised brows (surprise)
        Lower  → lowered brows (angry)
        """
        # Right brow center (19) to right eye center
        right_brow_center = pts[19]
        right_eye_center = (pts[37] + pts[38] + pts[40] + pts[41]) / 4.0
        d_right = np.linalg.norm(right_brow_center - right_eye_center)

        # Left brow center (24) to left eye center
        left_brow_center = pts[24]
        left_eye_center = (pts[43] + pts[44] + pts[46] + pts[47]) / 4.0
        d_left = np.linalg.norm(left_brow_center - left_eye_center)

        return ((d_right + d_left) / 2.0) / iod

    @staticmethod
    def _brow_furrow(pts: np.ndarray, iod: float) -> float:
        """
        Distance between inner brow points (21, 22) normalized by IOD.
        Smaller → brows drawn together → angry/fear
        """
        return np.linalg.norm(pts[21] - pts[22]) / iod

    @staticmethod
    def _lip_corner_angle(pts: np.ndarray) -> float:
        """
        Angle of lip corners relative to horizontal.
        Negative → corners turned down → sad
        Positive → corners turned up → happy
        """
        left_corner = pts[48]
        right_corner = pts[54]
        mid_bottom = pts[57]  # Bottom lip center

        # Average corner height vs bottom center
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2.0
        # In image coords, positive y is down
        # If corners are higher (smaller y) than mid_bottom → upward → happy
        angle = mid_bottom[1] - corner_avg_y
        return angle

    # ── Main analysis ───────────────────────────

    def analyze(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Analyze facial landmarks and return emotion boost scores.

        Returns dict like: {'happy': 0.75, 'surprise': 0.6}
        Same interface as the old GenericLandmarkAnalyzer.
        """
        if image is None or image.size == 0:
            return None

        # Skip tiny images (landmarks won't be reliable)
        h, w = image.shape[:2]
        if h < 60 or w < 60:
            return None

        pts = self._detect_landmarks(image)
        if pts is None:
            return None

        iod = self._interocular_distance(pts)
        if iod < 5:  # face too small for reliable metrics
            return None

        # Compute all metrics
        ear = self._eye_aspect_ratio(pts)
        mar = self._mouth_aspect_ratio(pts)
        smile = self._smile_ratio(pts, iod)
        brow_dist = self._brow_eye_distance(pts, iod)
        furrow = self._brow_furrow(pts, iod)
        lip_angle = self._lip_corner_angle(pts)

        logger.info(f"Landmarks → EAR={ear:.3f} MAR={mar:.3f} Smile={smile:.3f} "
                    f"BrowDist={brow_dist:.3f} Furrow={furrow:.3f} LipAngle={lip_angle:.1f}")

        boosts: Dict[str, float] = {}

        # ── HAPPY ──
        # Strong smile (wide mouth + upturned corners) AND mouth not extremely open
        if smile > 0.95 and lip_angle > 3.0 and mar < 0.6:
            confidence = float(min(1.0, 0.5 + (smile - 0.95) * 2.0 + (lip_angle - 3.0) * 0.05))
            boosts['happy'] = int(confidence * 1000) / 1000

        # ── SURPRISE ──
        # Wide open eyes AND open mouth AND raised brows
        if ear > 0.30 and mar > 0.5 and brow_dist > 0.28:
            confidence = float(min(1.0, 0.5 + (ear - 0.30) * 3.0 + (mar - 0.5) * 1.0))
            boosts['surprise'] = int(confidence * 1000) / 1000

        # ── ANGRY ──
        # Lowered/furrowed brows AND compressed lips (low MAR)
        if brow_dist < 0.20 and furrow < 0.12 and mar < 0.25:
            confidence = float(min(1.0, 0.5 + (0.20 - brow_dist) * 5.0 + (0.12 - furrow) * 3.0))
            boosts['angry'] = int(confidence * 1000) / 1000

        # ── SAD ──
        # Downturned lip corners AND low brows (but not furrowed like angry)
        if lip_angle < -2.0 and brow_dist < 0.24:
            confidence = float(min(1.0, 0.5 + abs(lip_angle + 2.0) * 0.05 + (0.24 - brow_dist) * 3.0))
            boosts['sad'] = int(confidence * 1000) / 1000

        # ── FEAR ──
        # Wide open eyes AND raised brows AND tight/slightly open mouth (not as open as surprise)
        if ear > 0.28 and brow_dist > 0.26 and 0.15 < mar < 0.45 and furrow < 0.14:
            confidence = float(min(1.0, 0.5 + (ear - 0.28) * 3.0 + (brow_dist - 0.26) * 3.0))
            boosts['fear'] = int(confidence * 1000) / 1000

        # ── NEUTRAL ──
        # Nothing extreme — all metrics in mid range
        is_neutral = (0.20 <= ear <= 0.32 and
                      mar < 0.3 and
                      0.20 <= brow_dist <= 0.30 and
                      abs(lip_angle) < 3.0 and
                      furrow > 0.10)
        if is_neutral and len(boosts) == 0:
            boosts['neutral'] = 0.6

        if boosts:
            logger.info(f"Landmark boosts: {boosts}")

        return boosts if boosts else None


# ── Singleton ──
landmark_analyzer = LBFLandmarkAnalyzer()
