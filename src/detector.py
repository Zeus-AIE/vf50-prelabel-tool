"""Face detection (YuNet + CLAHE IR enhancement) and rotation coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocessor import apply_photometric_variant


VALID_ROTATIONS = ("0", "90cw", "90ccw", "180", "auto")
VALID_DRIVER_SIDES = ("largest", "left", "right", "center")
VALID_DETECTORS = ("scrfd10g", "yunet")


@dataclass
class FaceDetection:
    bbox: List[float]        # [x, y, w, h] in the upright (rotated) image
    score: float
    landmarks5: List[List[float]]  # 5 face keypoints [[x,y], ...]
    used_clahe: bool = False


def rotate_image(image: np.ndarray, angle: str) -> np.ndarray:
    """Rotate image upright before running face detector / landmark model."""
    if angle == "0":
        return image
    if angle == "90cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == "90ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def unrotate_points(
    pts: np.ndarray, angle: str, orig_w: int, orig_h: int
) -> np.ndarray:
    """Map (N, 2) coordinates from rotated upright image back to original image."""
    out = np.array(pts, dtype=np.float64).copy()
    if angle == "0":
        return out
    xr = out[:, 0].copy()
    yr = out[:, 1].copy()
    if angle == "90cw":
        out[:, 0] = yr
        out[:, 1] = orig_h - xr
    elif angle == "90ccw":
        out[:, 0] = orig_w - yr
        out[:, 1] = xr
    elif angle == "180":
        out[:, 0] = orig_w - xr
        out[:, 1] = orig_h - yr
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")
    return out


def rotate_points(
    pts: np.ndarray, angle: str, orig_w: int, orig_h: int
) -> np.ndarray:
    """Map (N, 2) coordinates from original image into rotated upright image."""
    out = np.array(pts, dtype=np.float64).copy()
    if angle == "0":
        return out
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    if angle == "90cw":
        out[:, 0] = orig_h - y
        out[:, 1] = x
    elif angle == "90ccw":
        out[:, 0] = y
        out[:, 1] = orig_w - x
    elif angle == "180":
        out[:, 0] = orig_w - x
        out[:, 1] = orig_h - y
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")
    return out


def rotate_bbox(
    bbox: Sequence[float], angle: str, orig_w: int, orig_h: int
) -> List[float]:
    """Transform [x, y, w, h] from original image into rotated upright image."""
    x, y, w, h = [float(v) for v in bbox]
    if angle == "0":
        return [x, y, w, h]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    rc = rotate_points(corners, angle, orig_w, orig_h)
    x1, y1 = rc.min(axis=0)
    x2, y2 = rc.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def unrotate_bbox(
    bbox: Sequence[float], angle: str, orig_w: int, orig_h: int
) -> List[float]:
    """Transform [x, y, w, h] from rotated upright image back to original image."""
    x, y, w, h = [float(v) for v in bbox]
    if angle == "0":
        return [x, y, w, h]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    oc = unrotate_points(corners, angle, orig_w, orig_h)
    x1, y1 = oc.min(axis=0)
    x2, y2 = oc.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def apply_clahe_bgr(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """Enhance local contrast for dark/blurry IR cabin images."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class BaseFaceDetector:
    """Shared driver selection and cardinal rotation helpers across face detectors."""

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        raise NotImplementedError

    @staticmethod
    def select_driver_face(
        faces: List[FaceDetection], img_w: int, img_h: int, driver_side: str = "largest"
    ) -> Tuple[Optional[FaceDetection], bool]:
        """Pick the driver's face and indicate if multiple valid faces exist."""
        if not faces:
            return None, False

        multi_face = len(faces) > 1
        if len(faces) == 1:
            return faces[0], False

        if driver_side == "largest":
            chosen = max(faces, key=lambda f: (f.bbox[2] * f.bbox[3]) * (0.5 + f.score))
        elif driver_side == "left":
            chosen = min(faces, key=lambda f: f.bbox[0] + f.bbox[2] * 0.5)
        elif driver_side == "right":
            chosen = max(faces, key=lambda f: f.bbox[0] + f.bbox[2] * 0.5)
        elif driver_side == "center":
            cx, cy = img_w * 0.5, img_h * 0.5
            chosen = min(
                faces,
                key=lambda f: ((f.bbox[0] + f.bbox[2] * 0.5 - cx) ** 2 + (f.bbox[1] + f.bbox[3] * 0.5 - cy) ** 2),
            )
        else:
            raise ValueError(f"Unsupported driver_side: {driver_side}")
        return chosen, multi_face

    def detect_candidates_all_rotations(
        self,
        orig_image: np.ndarray,
        driver_side: str = "largest",
    ) -> List[Tuple[str, np.ndarray, FaceDetection, bool]]:
        """Collect best driver detection for each of the 4 cardinal rotations."""
        candidates: List[Tuple[str, np.ndarray, FaceDetection, bool]] = []
        for angle in ("0", "90cw", "90ccw", "180"):
            rot_img = rotate_image(orig_image, angle)
            faces = self.detect_faces(rot_img)
            rh, rw = rot_img.shape[:2]
            chosen, multi = self.select_driver_face(faces, rw, rh, driver_side)
            if chosen is not None:
                candidates.append((angle, rot_img, chosen, multi))
        return candidates

    def detect_with_rotation(
        self,
        orig_image: np.ndarray,
        rotate_mode: str = "0",
        driver_side: str = "largest",
    ) -> Tuple[str, np.ndarray, Optional[FaceDetection], bool]:
        """Rotate image according to mode (or auto-detect best upright angle) and detect driver face."""
        if rotate_mode not in VALID_ROTATIONS:
            raise ValueError(f"Invalid rotate_mode: {rotate_mode}")

        if rotate_mode != "auto":
            rot_img = rotate_image(orig_image, rotate_mode)
            faces = self.detect_faces(rot_img)
            rh, rw = rot_img.shape[:2]
            chosen, multi = self.select_driver_face(faces, rw, rh, driver_side)
            return rotate_mode, rot_img, chosen, multi

        candidates = self.detect_candidates_all_rotations(orig_image, driver_side=driver_side)
        if not candidates:
            return "0", orig_image, None, False

        best_tuple: Optional[Tuple[float, str, np.ndarray, FaceDetection, bool]] = None
        for angle, rot_img, chosen, multi in candidates:
            rh, rw = rot_img.shape[:2]
            lm5 = chosen.landmarks5
            eye_y = 0.5 * (lm5[0][1] + lm5[1][1])
            mouth_y = 0.5 * (lm5[3][1] + lm5[4][1])
            bx, by, bw, bh = chosen.bbox
            in_bounds_bonus = 0.0 if (bx < 0 or by < 0 or bx + bw > rw or by + bh > rh) else 0.08
            upright_bonus = 0.25 if mouth_y > eye_y + bh * 0.15 else -0.3
            rank_score = chosen.score + upright_bonus + in_bounds_bonus + (0.05 if angle == "0" else 0.0)
            if best_tuple is None or rank_score > best_tuple[0]:
                best_tuple = (rank_score, angle, rot_img, chosen, multi)

        assert best_tuple is not None
        _, angle, rot_img, chosen, multi = best_tuple
        return angle, rot_img, chosen, multi


class FaceDetector(BaseFaceDetector):
    """OpenCV YuNet wrapper with IR CLAHE fallback and driver selection."""

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.3,
        min_face_size: int = 36,
    ) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
        )

    def _run_yunet(self, image: np.ndarray, used_clahe: bool) -> List[FaceDetection]:
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, raw_faces = self._detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        results: List[FaceDetection] = []
        for row in raw_faces:
            bx, by, bw, bh = [float(v) for v in row[:4]]
            score = float(row[14])
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            lm5 = [[float(row[4 + 2 * i]), float(row[5 + 2 * i])] for i in range(5)]
            results.append(FaceDetection([bx, by, bw, bh], score, lm5, used_clahe=used_clahe))
        return results

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces on raw image first, falling back to CLAHE and photometric variants if empty."""
        faces = self._run_yunet(image, used_clahe=False)
        if faces:
            return faces
        clahe_img = apply_clahe_bgr(image)
        faces = self._run_yunet(clahe_img, used_clahe=True)
        if faces:
            return faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        p50_img = float(np.median(gray))
        primary_var = "dark_lift" if p50_img < 120.0 else "highlight_recover"
        for var_name in (primary_var, "shadow_highlight"):
            faces = self._run_yunet(apply_photometric_variant(image, var_name), used_clahe=True)
            if faces:
                return faces
        return []
