"""Synthetic Near-Infrared (NIR/IR) Cabin Augmentation for 256x256 face crops."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class SyntheticIRAugmentor:
    """Transform RGB 256x256 face crops into realistic blurry/IR cabin images without moving landmarks."""

    def __init__(
        self,
        prob: float = 0.85,
        glasses_prob: float = 0.30,
        occlusion_prob: float = 0.25,
    ) -> None:
        self.prob = prob
        self.glasses_prob = glasses_prob
        self.occlusion_prob = occlusion_prob

    @staticmethod
    def _apply_nir_spectrum(
        bgr: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Convert BGR to NIR-like monochrome weighted strongly toward Red (lip-skin low contrast)."""
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)

        w_r = float(rng.uniform(0.65, 0.85))
        w_g = float(rng.uniform(0.10, 0.25))
        w_b = max(0.05, 1.0 - w_r - w_g)
        total = w_r + w_g + w_b
        gray = (w_r * r + w_g * g + w_b * b) / total

        # Optional mild bright-pupil / dark-iris IR shift and gamma variation
        gamma = float(rng.uniform(0.55, 1.75))
        gray_norm = np.clip(gray / 255.0, 0.0, 1.0)
        gray = np.power(gray_norm, gamma) * 255.0

        # Non-uniform A-pillar IR LED illumination gradient + dark cabin contrast shift
        h, w = gray.shape
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        grad = np.cos(angle) * xs + np.sin(angle) * ys
        strength = float(rng.uniform(0.10, 0.38))
        bias = float(rng.uniform(-22.0, 12.0))
        gray = gray * (1.0 + strength * grad) + bias

        # Subtle contrast compression (typical of low-bit IR cabin frames)
        alpha = float(rng.uniform(0.75, 1.10))
        mean_val = float(np.mean(gray))
        gray = mean_val + alpha * (gray - mean_val)
        gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    @staticmethod
    def _apply_glasses_glare(
        img: np.ndarray,
        landmarks_256: Optional[np.ndarray],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Synthesize thin dark eyeglass rims + bridge + soft lens glare over GT eye region."""
        if landmarks_256 is None or len(landmarks_256) < 76:
            return img

        out = img.copy()
        eye_l = landmarks_256[60:68]
        eye_r = landmarks_256[68:76]
        if np.all(eye_l == 0) or np.all(eye_r == 0):
            return out

        h, w = out.shape[:2]
        frame_val = int(rng.randint(15, 55))
        rim_thick = int(rng.choice([1, 2]))

        centers = []
        radii = []
        for eye_pts in (eye_l, eye_r):
            cx = float(np.mean(eye_pts[:, 0]))
            cy = float(np.mean(eye_pts[:, 1]))
            ew = max(float(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])), 10.0)
            rx = int(round(ew * rng.uniform(0.75, 1.05)))
            ry = int(round(ew * rng.uniform(0.45, 0.72)))
            centers.append((int(round(cx)), int(round(cy))))
            radii.append((rx, ry))

            # Draw rim ellipse
            cv2.ellipse(
                out,
                (int(round(cx)), int(round(cy))),
                (rx, ry),
                0,
                0,
                360,
                (frame_val, frame_val, frame_val),
                thickness=rim_thick,
                lineType=cv2.LINE_AA,
            )

            # Soft IR lens reflection patch
            if rng.rand() < 0.75:
                glare_layer = out.copy()
                gx = int(np.clip(cx + rng.uniform(-0.35, 0.35) * rx, 0, w - 1))
                gy = int(np.clip(cy + rng.uniform(-0.35, 0.35) * ry, 0, h - 1))
                grx = max(3, int(rx * rng.uniform(0.25, 0.50)))
                gry = max(2, int(ry * rng.uniform(0.25, 0.45)))
                g_val = int(rng.randint(185, 245))
                cv2.ellipse(
                    glare_layer,
                    (gx, gy),
                    (grx, gry),
                    float(rng.uniform(-25, 25)),
                    0,
                    360,
                    (g_val, g_val, g_val),
                    -1,
                    lineType=cv2.LINE_AA,
                )
                glare_layer = cv2.GaussianBlur(glare_layer, (7, 7), 0)
                alpha_g = float(rng.uniform(0.20, 0.45))
                out = cv2.addWeighted(glare_layer, alpha_g, out, 1.0 - alpha_g, 0)

        # Nose bridge connecting inner rims & temple arms
        (lx, ly), (rx_c, ry_c) = centers
        (lrx, _), (rrx, _) = radii
        p_bridge_l = (lx + lrx - 1, ly - 1)
        p_bridge_r = (rx_c - rrx + 1, ry_c - 1)
        cv2.line(out, p_bridge_l, p_bridge_r, (frame_val, frame_val, frame_val), rim_thick, cv2.LINE_AA)
        cv2.line(out, (lx - lrx, ly), (0, max(0, ly - int(rng.randint(2, 12)))), (frame_val, frame_val, frame_val), rim_thick, cv2.LINE_AA)
        cv2.line(out, (rx_c + rrx, ry_c), (w - 1, max(0, ry_c - int(rng.randint(2, 12)))), (frame_val, frame_val, frame_val), rim_thick, cv2.LINE_AA)
        return out

    @staticmethod
    def _apply_degradation(
        img: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply sensor noise, blur/motion blur, low-res down-up scaling, and JPEG block artifacts."""
        h, w = img.shape[:2]
        out = img.copy()

        # 1. Gaussian blur or directional motion blur
        r_blur = rng.rand()
        if r_blur < 0.45:
            k = int(rng.choice([3, 5, 7]))
            out = cv2.GaussianBlur(out, (k, k), float(rng.uniform(0.6, 1.8)))
        elif r_blur < 0.75:
            k = int(rng.choice([5, 7, 9]))
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.rand() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                np.fill_diagonal(kernel, 1.0 / k)
            M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), float(rng.uniform(-45, 45)), 1.0)
            kernel = cv2.warpAffine(kernel, M, (k, k))
            s = float(kernel.sum())
            if s > 1e-6:
                kernel /= s
                out = cv2.filter2D(out, -1, kernel)

        # 2. Downsample -> Upsample (simulates zoomed low-res cabin ROI like 56..160px -> 256px)
        if rng.rand() < 0.80:
            low_sz = int(rng.randint(56, 165))
            interp_down = cv2.INTER_AREA
            interp_up = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
            small = cv2.resize(out, (low_sz, low_sz), interpolation=interp_down)
            out = cv2.resize(small, (w, h), interpolation=interp_up)

        # 3. Sensor Gaussian noise
        sigma = float(rng.uniform(2.0, 9.5))
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        # 4. JPEG compression blockiness (Q = 15..55, matches compressed CVAT IR frames)
        if rng.rand() < 0.85:
            q = int(rng.randint(15, 56))
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    out = dec

        return out

    @staticmethod
    def _apply_cabin_occlusion(
        img: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Simulate dark cabin occlusion (seatbelt, steering wheel edge, hand shadow)."""
        h, w = img.shape[:2]
        out = img.copy()
        rw = int(rng.uniform(0.14, 0.34) * w)
        rh = int(rng.uniform(0.14, 0.34) * h)
        x1 = int(rng.randint(0, max(1, w - rw)))
        y1 = int(rng.randint(0, max(1, h - rh)))
        val = int(rng.randint(5, 45))
        patch = out.copy()
        cv2.rectangle(patch, (x1, y1), (x1 + rw, y1 + rh), (val, val, val), -1)
        patch = cv2.GaussianBlur(patch, (9, 9), 0)
        alpha = float(rng.uniform(0.70, 0.95))
        return cv2.addWeighted(patch, alpha, out, 1.0 - alpha, 0)

    def __call__(
        self,
        bgr_256: np.ndarray,
        landmarks_256: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Apply synthetic cabin IR augmentation to a 256x256 BGR uint8 crop."""
        if rng is None:
            rng = np.random.RandomState()

        if rng.rand() >= self.prob:
            return bgr_256

        out = self._apply_nir_spectrum(bgr_256, rng)
        if rng.rand() < self.glasses_prob:
            out = self._apply_glasses_glare(out, landmarks_256, rng)
        if rng.rand() < self.occlusion_prob:
            out = self._apply_cabin_occlusion(out, rng)
        out = self._apply_degradation(out, rng)
        return out
