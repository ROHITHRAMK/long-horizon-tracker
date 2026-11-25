"""Camera Motion Compensation (CMC) via foreground-masked optical flow affine warp.

Estimates frame-to-frame camera ego-motion using Shi-Tomasi background keypoints,
forward-backward Lucas-Kanade optical flow, and RANSAC partial affine estimation.
Provides exact geometric warp utilities for bounding boxes, point clouds, velocity
vectors, and 8D Kalman filter states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


@dataclass
class CMCResult:
    """Result of a single frame-to-frame camera motion estimation."""

    matrix: np.ndarray  # (2, 3) float32 affine warp prev -> curr
    dx: float = 0.0
    dy: float = 0.0
    dtheta_deg: float = 0.0
    scale: float = 1.0
    inlier_ratio: float = 1.0
    num_keypoints: int = 0
    num_inliers: int = 0
    used_fallback: bool = False


class CameraMotionCompensator:
    """Foreground-masked sparse optical flow CMC estimator."""

    def __init__(
        self,
        downscale: float = 0.5,
        max_corners: int = 350,
        quality_level: float = 0.01,
        min_distance: int = 12,
        block_size: int = 7,
        fb_max_error: float = 1.5,
        ransac_reproj_thresh: float = 3.0,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.35,
        max_translation_frac: float = 0.25,
        max_scale_delta: float = 0.15,
        max_rotation_deg: float = 15.0,
        mask_expand_px: int = 10,
    ) -> None:
        self.downscale = float(np.clip(downscale, 0.2, 1.0))
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.fb_max_error = fb_max_error
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_translation_frac = max_translation_frac
        self.max_scale_delta = max_scale_delta
        self.max_rotation_deg = max_rotation_deg
        self.mask_expand_px = mask_expand_px

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_boxes: Optional[np.ndarray] = None
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self) -> None:
        """Reset temporal state (e.g., on video cut or new sequence)."""
        self._prev_gray = None
        self._prev_boxes = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3
            else frame_bgr
        )
        if self.downscale != 1.0:
            h, w = gray.shape[:2]
            new_w = max(32, int(round(w * self.downscale)))
            new_h = max(32, int(round(h * self.downscale)))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def _build_background_mask(
        self,
        shape_hw: Tuple[int, int],
        boxes: Optional[np.ndarray],
    ) -> np.ndarray:
        h, w = shape_hw
        mask = np.full((h, w), 255, dtype=np.uint8)
        if boxes is None or len(boxes) == 0:
            return mask

        pad = int(round(self.mask_expand_px * self.downscale))
        scaled = np.asarray(boxes, dtype=np.float32) * self.downscale
        for box in scaled:
            x1 = max(0, int(np.floor(box[0])) - pad)
            y1 = max(0, int(np.floor(box[1])) - pad)
            x2 = min(w, int(np.ceil(box[2])) + pad)
            y2 = min(h, int(np.ceil(box[3])) + pad)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def update(
        self,
        frame_bgr: np.ndarray,
        foreground_boxes: Optional[np.ndarray] = None,
    ) -> CMCResult:
        """Estimate partial-affine ego-motion from previous frame to current frame."""
        curr_gray = self._preprocess(frame_bgr)
        curr_boxes = (
            np.asarray(foreground_boxes, dtype=np.float32).reshape(-1, 4)
            if foreground_boxes is not None and len(foreground_boxes) > 0
            else None
        )

        if self._prev_gray is None or self._prev_gray.shape != curr_gray.shape:
            self._prev_gray = curr_gray
            self._prev_boxes = curr_boxes
            return CMCResult(matrix=IDENTITY_AFFINE.copy(), used_fallback=True)

        prev_gray = self._prev_gray
        prev_mask = self._build_background_mask(prev_gray.shape[:2], self._prev_boxes)
        curr_mask = self._build_background_mask(curr_gray.shape[:2], curr_boxes)

        # Save state for next step immediately so early returns still advance frame.
        self._prev_gray = curr_gray
        self._prev_boxes = curr_boxes

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=prev_mask,
        )
        if pts_prev is None or len(pts_prev) < self.min_inliers:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                num_keypoints=0 if pts_prev is None else len(pts_prev),
                used_fallback=True,
            )

        # Forward-backward Lucas-Kanade optical flow for robust outlier rejection
        pts_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_prev, None, **self._lk_params
        )
        if pts_curr is None or status_fwd is None:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                num_keypoints=len(pts_prev),
                used_fallback=True,
            )

        pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, pts_curr, None, **self._lk_params
        )
        if pts_back is None or status_bwd is None:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                num_keypoints=len(pts_prev),
                used_fallback=True,
            )

        fb_err = np.linalg.norm(
            pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1
        )
        valid = (
            (status_fwd.reshape(-1) == 1)
            & (status_bwd.reshape(-1) == 1)
            & (fb_err <= self.fb_max_error)
        )

        # Also discard points that landed inside a current foreground object mask
        h_s, w_s = curr_gray.shape[:2]
        pts_c_flat = pts_curr.reshape(-1, 2)
        in_bounds = (
            (pts_c_flat[:, 0] >= 0)
            & (pts_c_flat[:, 0] < w_s)
            & (pts_c_flat[:, 1] >= 0)
            & (pts_c_flat[:, 1] < h_s)
        )
        valid &= in_bounds
        if np.any(valid):
            cx_i = pts_c_flat[valid, 0].astype(np.int32)
            cy_i = pts_c_flat[valid, 1].astype(np.int32)
            valid_idx = np.where(valid)[0]
            bg_ok = curr_mask[cy_i, cx_i] > 0
            valid[valid_idx[~bg_ok]] = False

        src = pts_prev.reshape(-1, 2)[valid]
        dst = pts_c_flat[valid]
        num_matched = len(src)

        if num_matched < self.min_inliers:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                num_keypoints=num_matched,
                used_fallback=True,
            )

        M_small, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
            maxIters=2000,
            confidence=0.99,
        )
        if M_small is None or inliers is None:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                num_keypoints=num_matched,
                used_fallback=True,
            )

        num_inliers = int(np.sum(inliers))
        inlier_ratio = num_inliers / max(1, num_matched)

        # Rescale affine matrix back to original full-resolution pixel space
        M = M_small.astype(np.float32).copy()
        M[0, 2] /= self.downscale
        M[1, 2] /= self.downscale

        # Decompose similarity parameters for sanity validation & telemetry
        a, b = float(M[0, 0]), float(M[0, 1])
        scale = float(np.hypot(a, b))
        dtheta_deg = float(np.degrees(np.arctan2(-b, a)))
        dx = float(M[0, 2])
        dy = float(M[1, 2])

        full_h, full_w = frame_bgr.shape[:2]
        max_shift = self.max_translation_frac * max(full_w, full_h)

        is_sane = (
            num_inliers >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
            and abs(scale - 1.0) <= self.max_scale_delta
            and abs(dtheta_deg) <= self.max_rotation_deg
            and np.hypot(dx, dy) <= max_shift
            and np.all(np.isfinite(M))
        )

        if not is_sane:
            return CMCResult(
                matrix=IDENTITY_AFFINE.copy(),
                inlier_ratio=inlier_ratio,
                num_keypoints=num_matched,
                num_inliers=num_inliers,
                used_fallback=True,
            )

        return CMCResult(
            matrix=M,
            dx=dx,
            dy=dy,
            dtheta_deg=dtheta_deg,
            scale=scale,
            inlier_ratio=inlier_ratio,
            num_keypoints=num_matched,
            num_inliers=num_inliers,
            used_fallback=False,
        )


def warp_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp 2D points of shape (N, 2) by a (2, 3) affine matrix."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.copy()
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    A = M[:, :2].astype(np.float32)
    t = M[:, 2].astype(np.float32)
    warped = pts @ A.T + t
    return warped[0] if single else warped


def warp_vectors(vectors: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Rotate/scale 2D velocity vectors of shape (N, 2) without adding translation."""
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.size == 0:
        return vecs.copy()
    single = vecs.ndim == 1
    if single:
        vecs = vecs.reshape(1, 2)
    A = M[:, :2].astype(np.float32)
    warped = vecs @ A.T
    return warped[0] if single else warped


def warp_boxes(
    boxes: np.ndarray,
    M: np.ndarray,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Warp (N, 4) xyxy boxes via all 4 corners and return enclosing xyxy boxes."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4).copy()
    arr = arr.reshape(-1, 4)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    corners = np.stack(
        [
            np.stack([x1, y1], axis=-1),
            np.stack([x2, y1], axis=-1),
            np.stack([x2, y2], axis=-1),
            np.stack([x1, y2], axis=-1),
        ],
        axis=1,
    )  # (N, 4, 2)
    A = M[:, :2].astype(np.float32)
    t = M[:, 2].astype(np.float32)
    warped = corners @ A.T + t
    out = np.stack(
        [
            warped[:, :, 0].min(axis=1),
            warped[:, :, 1].min(axis=1),
            warped[:, :, 0].max(axis=1),
            warped[:, :, 1].max(axis=1),
        ],
        axis=-1,
    )
    if frame_shape is not None:
        h, w = frame_shape[:2]
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(w))
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(h))
    return out


def warp_box(
    box: np.ndarray,
    M: np.ndarray,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Warp a single (4,) xyxy bounding box."""
    return warp_boxes(np.asarray(box, dtype=np.float32).reshape(1, 4), M, frame_shape)[0]


def warp_kalman_state(
    mean: np.ndarray,
    cov: np.ndarray,
    M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp an 8D Kalman state [cx, cy, w, h, vx, vy, vw, vh] and its 8x8 covariance."""
    A = M[:, :2].astype(np.float64)
    t = M[:, 2].astype(np.float64)
    scale = float(np.sqrt(max(1e-6, abs(np.linalg.det(A)))))

    J = np.eye(8, dtype=np.float64)
    J[0:2, 0:2] = A
    J[2, 2] = scale
    J[3, 3] = scale
    J[4:6, 4:6] = A
    J[6, 6] = scale
    J[7, 7] = scale

    new_mean = J @ mean.astype(np.float64)
    new_mean[0:2] += t
    new_cov = J @ cov.astype(np.float64) @ J.T
    return new_mean, new_cov
