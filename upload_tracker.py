"""Maximum-Accuracy 2-Pass Clothes-Invariant Biometric AI Video Tracker (Streamlit).

Pipeline:
- Pose & Person Detector (cached via @st.cache_resource):
    * `yolo11l-pose.pt` (default) / `yolo11x-pose.pt` (Ultra-Accurate) at imgsz=960, conf=0.35
      extracting person bounding boxes AND 17 COCO keypoints
    * Standard YOLO models (`yolo11l.pt`, `yolo11x.pt`, `yolov8n.pt`) also available when
      Clothes-Invariant Biometric Mode is toggled OFF
- Facial Biometrics (cached via @st.cache_resource):
    * `insightface.app.FaceAnalysis(name="buffalo_l")` (CUDA/CPU ONNX auto-selection, graceful fallback)
      extracting 512-D L2-normalized ArcFace embeddings from visible head regions
- Multi-Part DINOv2 (`dinov2_vitb14`, fallback `dinov2_vits14`, cached via @st.cache_resource):
    * Biometric Mode ON (Default, 0% clothing color weight):
        - Head + Hair + Ears crop strictly above shoulder keypoints (+ horizontal flip avg)
        - Desaturated grayscale body-shape crop (clothing color stripped)
        - Scale-invariant 8D skeletal body shape ratios (running per-component median)
        - Lighting-normalized CIE Lab (a*, b*) skin chroma histogram + median (L* dropped)
        - Pass-1 embedding: L2Norm([0.45*head_hair, 0.25*gray_shape, 0.15*body_ratio, 0.15*skin_chroma])
    * Biometric Mode OFF:
        - Standard appearance fusion: L2Norm([0.75*dinov2_multi, 0.25*2-zone_clothing_hist])
- Depth Occlusion Guard (cached via @st.cache_resource):
    * Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`)
- Pass 1: LongHorizonTracker with 60s (1800f) ghost memory, +14.0 px/f corridor,
  max_active_miss=35, AGENTS.md-locked >=15f probation & margin-ratio abstention
  (stitch_min_sim=0.72, margin_min_diff=0.08), plus per-track biometric profile & face gallery
- Pass 2: Hierarchical Biometric Consolidation
    (a) Retroactive Probation Back-Fill for STITCH_COMMIT (and globally merged) tracklets
    (b) Tier 1 Direct Face Lock: ArcFace top-3 cosine sim >= 0.42 & mutual-best/margin >= 0.06
        -> logs BIOMETRIC_FACE_LOCK
    (c) Tier 2 Head/Hair + Body Shape + Skin Tone + Grayscale Contour:
        0.45*head_hair + 0.25*body_ratio + 0.15*skin_tone + 0.15*gray_contour >= 0.68 &
        mutual-best/margin >= 0.05 -> logs BIOMETRIC_HEAD_BODY_SKIN
    (d) Sequential #1, #2, #3... re-numbering, <6f blip filtering, optional 17-keypoint skeleton overlay
- Final Render: Browser-playable H.264 (avc1 / yuv420p + faststart) via imageio-ffmpeg with
  continuous WebSocket keep-alive ticks

Launch with:
    .venv\\Scripts\\streamlit.exe run upload_tracker.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import streamlit as st

from run_tracker import (
    AppearanceEncoder,
    AsyncVideoWriter,
    DepthEstimator,
    EVENT_COLORS,
    draw_dashed_rect,
    draw_label_pill,
    extract_box_depth,
    resolve_device,
    track_color_bgr,
    upload_frame_gpu,
    warp_depth_map,
)
from tracker import (
    Detection,
    GhostConfig,
    LongHorizonTracker,
    TrackerConfig,
)


ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = ROOT / "output" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# COCO 17-keypoint skeleton connectivity for optional overlay rendering
COCO_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),         # Face / head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Shoulders & arms
    (5, 11), (6, 12), (11, 12),             # Torso
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
)

# Anthropometric priors & scale normalizers for the 8 scale-invariant body shape ratios:
# [shoulder/torso, hip/shoulder, upper_arm/torso, forearm/upper_arm,
#  thigh/torso, calf/thigh, head_ears/shoulder, box_aspect_ratio]
BODY_RATIO_PRIORS = np.array([0.75, 0.80, 0.55, 0.85, 0.78, 0.90, 0.42, 0.42], dtype=np.float32)
BODY_RATIO_SCALES = np.array([0.22, 0.22, 0.16, 0.22, 0.22, 0.22, 0.15, 0.20], dtype=np.float32)


st.set_page_config(
    page_title="Long-Horizon Biometric MOT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _even_dims(width: int, height: int) -> Tuple[int, int]:
    """Guarantee strictly even frame dimensions (>= 2x2) for yuv420p H.264 compatibility."""
    ew = max(2, int(width) - (int(width) % 2))
    eh = max(2, int(height) - (int(height) % 2))
    return ew, eh


def _ensure_even_frame(frame_bgr: np.ndarray, even_w: int, even_h: int) -> np.ndarray:
    """Crop or resize frame cleanly to exact even (even_w, even_h) dimensions."""
    fh, fw = frame_bgr.shape[:2]
    if fw == even_w and fh == even_h:
        return frame_bgr
    if fw >= even_w and fh >= even_h and (fw - even_w) <= 2 and (fh - even_h) <= 2:
        return frame_bgr[:even_h, :even_w]
    return cv2.resize(frame_bgr, (even_w, even_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Biometric & Appearance Feature Extractors (Pose, Face, Head/Shape, Skin)
# ---------------------------------------------------------------------------


class PosePersonDetector:
    """Ultralytics YOLO11-Pose / YOLO detector extracting boxes AND 17 COCO keypoints."""

    def __init__(
        self,
        model_name: str = "yolo11l-pose.pt",
        imgsz: int = 960,
        classes: Optional[Sequence[int]] = (0,),
        conf: float = 0.35,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model_name = model_name
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes is not None else None
        self.conf = float(conf)
        self.device = device
        self.is_pose_model = "pose" in model_name.lower()
        self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes_xyxy [N,4], scores [N], class_ids [N], keypoints_xyc [N,17,3])."""
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 17, 3), dtype=np.float32),
            )
        r0 = results[0]
        b = r0.boxes
        boxes = b.xyxy.cpu().numpy().astype(np.float32)
        scores = b.conf.cpu().numpy().astype(np.float32)
        cls_ids = b.cls.cpu().numpy().astype(np.int32)

        n = len(boxes)
        kpts = np.zeros((n, 17, 3), dtype=np.float32)
        if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
            try:
                raw_k = r0.keypoints.data.cpu().numpy().astype(np.float32)
                if raw_k.ndim == 3 and raw_k.shape[0] == n and raw_k.shape[1] >= 17:
                    if raw_k.shape[2] >= 3:
                        kpts = raw_k[:, :17, :3]
                    elif raw_k.shape[2] == 2:
                        kpts[:, :, :2] = raw_k[:, :17, :2]
                        kpts[:, :, 2] = 1.0
            except Exception:
                pass
        return boxes, scores, cls_ids, kpts


class ArcFaceBiometricAnalyzer:
    """InsightFace `buffalo_l` ArcFace 512-D facial biometric extractor with graceful fallback."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.available: bool = False
        self.provider_label: str = "Unavailable"
        self._app = None
        self._init_insightface(device)

    def _init_insightface(self, device: str) -> None:
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            avail_providers = set(ort.get_available_providers())
            use_cuda = device == "cuda" and "CUDAExecutionProvider" in avail_providers
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            try:
                app.prepare(ctx_id=0 if use_cuda else -1, det_size=(320, 320), det_thresh=0.35)
            except TypeError:
                app.prepare(ctx_id=0 if use_cuda else -1, det_size=(320, 320))
                if getattr(app, "det_model", None) is not None and hasattr(app.det_model, "det_thresh"):
                    app.det_model.det_thresh = 0.35
            self._app = app
            self.available = True
            self.provider_label = "ArcFace buffalo_l (CUDA ONNX)" if use_cuda else "ArcFace buffalo_l (CPU ONNX)"
        except Exception as exc:
            print(f"[ArcFaceBiometricAnalyzer] Unavailable, falling back cleanly ({exc})")
            self._app = None
            self.available = False
            self.provider_label = "ArcFace unavailable (Head/Body/Skin fallback)"

    @staticmethod
    def prepare_upscaled_head_crop(
        frame_bgr: np.ndarray,
        head_box: np.ndarray,
        pad_frac: float = 0.35,
        upscale_trigger_px: int = 160,
        target_min_px: int = 224,
    ) -> Optional[np.ndarray]:
        """Pad a head crop by 35% and bicubic-upscale small overhead CCTV crops (<160px) to >=224px."""
        h, w = frame_bgr.shape[:2]
        hx1, hy1, hx2, hy2 = [float(v) for v in head_box]
        hw = hx2 - hx1
        hh = hy2 - hy1
        if hw < 14.0 or hh < 14.0:
            return None

        pad_x = pad_frac * hw
        pad_y = pad_frac * hh
        cx1 = max(0, int(math.floor(hx1 - pad_x)))
        cy1 = max(0, int(math.floor(hy1 - pad_y)))
        cx2 = min(w, int(math.ceil(hx2 + pad_x)))
        cy2 = min(h, int(math.ceil(hy2 + pad_y)))
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 18 or ch < 18:
            return None

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        min_side = min(cw, ch)
        if min_side < upscale_trigger_px:
            scale = float(target_min_px) / float(max(1, min_side))
            new_w = max(target_min_px, int(round(cw * scale)))
            new_h = max(target_min_px, int(round(ch * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def extract_from_head_box(
        self,
        frame_bgr: np.ndarray,
        head_box: np.ndarray,
        min_det_score: float = 0.35,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Extract 512-D L2-normalized ArcFace embedding from a padded, upscaled head region."""
        if not self.available or self._app is None:
            return None

        crop = self.prepare_upscaled_head_crop(
            frame_bgr,
            head_box,
            pad_frac=0.35,
            upscale_trigger_px=160,
            target_min_px=224,
        )
        if crop is None:
            return None

        try:
            faces = self._app.get(crop)
        except Exception:
            return None
        if not faces:
            return None

        # Choose the highest-confidence / largest detected face inside the head region
        best_face = None
        best_rank = -1.0
        for f in faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < min_det_score:
                continue
            bbox = getattr(f, "bbox", None)
            area = 1.0
            if bbox is not None and len(bbox) >= 4:
                area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            rank = score * math.sqrt(area)
            if rank > best_rank:
                best_rank = rank
                best_face = f

        if best_face is None:
            return None

        emb = getattr(best_face, "normed_embedding", None)
        if emb is None:
            emb = getattr(best_face, "embedding", None)
        if emb is None:
            return None

        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        vec = vec / norm
        return vec, float(getattr(best_face, "det_score", 0.5))


def compute_head_box_above_shoulders(
    box: np.ndarray,
    kpts: np.ndarray,
    frame_shape: Tuple[int, int],
    conf_thresh: float = 0.35,
) -> np.ndarray:
    """Anchor a Head + Hair + Ears crop strictly ABOVE the shoulder keypoints (zero clothing)."""
    h, w = frame_shape[:2]
    x1 = float(np.clip(box[0], 0.0, max(0.0, float(w) - 2.0)))
    y1 = float(np.clip(box[1], 0.0, max(0.0, float(h) - 2.0)))
    x2 = float(np.clip(box[2], x1 + 2.0, float(w)))
    y2 = float(np.clip(box[3], y1 + 2.0, float(h)))
    bw = x2 - x1
    bh = y2 - y1

    # Strict upper shoulder boundary (indices 5=L_shoulder, 6=R_shoulder)
    sh_ys = [float(kpts[idx, 1]) for idx in (5, 6) if kpts.shape[0] >= 17 and kpts[idx, 2] >= conf_thresh]
    if sh_ys:
        # Use highest (smallest y) shoulder minus small collar margin so zero shirt pixels enter
        shoulder_y = min(sh_ys) - 0.02 * bh
        hy2 = float(np.clip(shoulder_y, y1 + max(10.0, 0.10 * bh), y2))
    else:
        hy2 = float(np.clip(y1 + 0.24 * bh, y1 + 10.0, y2))

    # Top of head (include hair crown)
    head_pts = [kpts[idx, :2] for idx in range(5) if kpts.shape[0] >= 17 and kpts[idx, 2] >= conf_thresh]
    if head_pts:
        pts_arr = np.asarray(head_pts, dtype=np.float32)
        min_kx, max_kx = float(np.min(pts_arr[:, 0])), float(np.max(pts_arr[:, 0]))
        min_ky = float(np.min(pts_arr[:, 1]))
        cx = 0.5 * (min_kx + max_kx)
        span_x = max(max_kx - min_kx, 0.22 * bw)
        half_w = max(0.85 * span_x, 0.20 * bw)
        hx1 = float(np.clip(cx - half_w, x1, x2 - 4.0))
        hx2 = float(np.clip(cx + half_w, hx1 + 4.0, x2))
        hy1 = float(np.clip(min(y1, min_ky - 0.45 * max(12.0, hy2 - min_ky)), 0.0, max(0.0, hy2 - 8.0)))
    else:
        hx1 = float(np.clip(x1 + 0.18 * bw, 0.0, max(0.0, float(w) - 4.0)))
        hx2 = float(np.clip(x2 - 0.18 * bw, hx1 + 4.0, float(w)))
        hy1 = y1

    return np.array([hx1, hy1, hx2, hy2], dtype=np.float32)


def extract_body_shape_vector(
    box: np.ndarray,
    kpts: np.ndarray,
    conf_thresh: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scale-invariant 8D skeletal body shape ratios from confident keypoints (conf >= 0.4).

    Ratios:
      0: shoulder_width / torso_len
      1: hip_width / shoulder_width
      2: upper_arm / torso_len
      3: forearm / upper_arm
      4: thigh / torso_len
      5: calf / thigh
      6: head_width_ears / shoulder_width
      7: box_aspect_ratio

    Returns:
      (raw_ratios [8], valid_mask [8] bool, normalized_l2_vec [8])
    """
    ratios = BODY_RATIO_PRIORS.copy()
    valid = np.zeros((8,), dtype=bool)

    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    ratios[7] = float(np.clip(bw / bh, 0.12, 1.50))
    valid[7] = True

    if kpts is not None and kpts.shape[0] >= 17:
        def ok(i: int) -> bool:
            return float(kpts[i, 2]) >= conf_thresh

        def dist(i: int, j: int) -> Optional[float]:
            if ok(i) and ok(j):
                return float(math.hypot(float(kpts[i, 0] - kpts[j, 0]), float(kpts[i, 1] - kpts[j, 1])))
            return None

        sh_w = dist(5, 6)
        hip_w = dist(11, 12)
        ear_w = dist(3, 4)

        # Torso length: mid-shoulder to mid-hip (or single side shoulder-to-hip if partially occluded)
        torso_len: Optional[float] = None
        if ok(5) and ok(6) and ok(11) and ok(12):
            mid_sh = 0.5 * (kpts[5, :2] + kpts[6, :2])
            mid_hip = 0.5 * (kpts[11, :2] + kpts[12, :2])
            torso_len = float(math.hypot(float(mid_sh[0] - mid_hip[0]), float(mid_sh[1] - mid_hip[1])))
        else:
            side_ts = [d for d in (dist(5, 11), dist(6, 12)) if d is not None]
            if side_ts:
                torso_len = float(np.mean(side_ts))

        if torso_len is not None and torso_len < 8.0:
            torso_len = None

        # 0: shoulder_width / torso_len
        if sh_w is not None and torso_len is not None:
            ratios[0] = float(np.clip(sh_w / torso_len, 0.25, 1.60))
            valid[0] = True

        # 1: hip_width / shoulder_width
        if hip_w is not None and sh_w is not None and sh_w >= 6.0:
            ratios[1] = float(np.clip(hip_w / sh_w, 0.35, 1.60))
            valid[1] = True

        # 2 & 3: upper_arm / torso_len, forearm / upper_arm
        ua_l, ua_r = dist(5, 7), dist(6, 8)
        fa_l, fa_r = dist(7, 9), dist(8, 10)
        ua_vals = [v for v in (ua_l, ua_r) if v is not None and v >= 4.0]
        if ua_vals and torso_len is not None:
            ratios[2] = float(np.clip(float(np.mean(ua_vals)) / torso_len, 0.20, 1.20))
            valid[2] = True
        fa_ratios = []
        if ua_l is not None and fa_l is not None and ua_l >= 4.0:
            fa_ratios.append(fa_l / ua_l)
        if ua_r is not None and fa_r is not None and ua_r >= 4.0:
            fa_ratios.append(fa_r / ua_r)
        if fa_ratios:
            ratios[3] = float(np.clip(float(np.mean(fa_ratios)), 0.35, 1.50))
            valid[3] = True

        # 4 & 5: thigh / torso_len, calf / thigh
        th_l, th_r = dist(11, 13), dist(12, 14)
        cf_l, cf_r = dist(13, 15), dist(14, 16)
        th_vals = [v for v in (th_l, th_r) if v is not None and v >= 6.0]
        if th_vals and torso_len is not None:
            ratios[4] = float(np.clip(float(np.mean(th_vals)) / torso_len, 0.30, 1.50))
            valid[4] = True
        cf_ratios = []
        if th_l is not None and cf_l is not None and th_l >= 6.0:
            cf_ratios.append(cf_l / th_l)
        if th_r is not None and cf_r is not None and th_r >= 6.0:
            cf_ratios.append(cf_r / th_r)
        if cf_ratios:
            ratios[5] = float(np.clip(float(np.mean(cf_ratios)), 0.40, 1.50))
            valid[5] = True

        # 6: head_width_ears / shoulder_width
        if ear_w is not None and sh_w is not None and sh_w >= 6.0:
            ratios[6] = float(np.clip(ear_w / sh_w, 0.15, 1.00))
            valid[6] = True

    # Convert ratios to centered, L2-normalized embedding vector for Pass-1 cosine similarity
    # Include positive offset so cosine similarity stays well-behaved in [0, 1]
    z = np.clip((ratios - BODY_RATIO_PRIORS) / BODY_RATIO_SCALES, -2.5, 2.5)
    vec = np.concatenate([ratios, 0.35 * z], axis=0).astype(np.float32)  # 16D
    norm = float(np.linalg.norm(vec))
    vec = vec / max(norm, 1e-8)
    return ratios, valid, vec


def extract_skin_chroma_vector(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    kpts: np.ndarray,
    conf_thresh: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Extract lighting-normalized CIE Lab (a*, b*) skin tone vector (L* brightness dropped).

    Samples small interior patches from:
      - Nose & cheeks (from nose/eye/ear keypoints)
      - Exposed wrists / lower forearms (from wrist/elbow keypoints)
    Returns:
      (skin_vec [34D L2-normalized], median_ab [2D float32], has_valid_skin bool)
    """
    h, w = frame_bgr.shape[:2]
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    patch_r = max(3, min(12, int(round(0.025 * min(bw, bh)))))

    sample_centers: List[Tuple[float, float]] = []
    if kpts is not None and kpts.shape[0] >= 17:
        def ok(i: int) -> bool:
            return float(kpts[i, 2]) >= conf_thresh

        # Nose interior
        if ok(0):
            sample_centers.append((float(kpts[0, 0]), float(kpts[0, 1])))
            # Left cheek (midpoint between nose and left eye/ear, shifted slightly downward)
            if ok(1):
                cx = 0.55 * float(kpts[0, 0]) + 0.45 * float(kpts[1, 0])
                cy = float(kpts[0, 1]) + 0.35 * abs(float(kpts[0, 1] - kpts[1, 1]))
                sample_centers.append((cx, cy))
            elif ok(3):
                cx = 0.50 * float(kpts[0, 0]) + 0.50 * float(kpts[3, 0])
                cy = 0.65 * float(kpts[0, 1]) + 0.35 * float(kpts[3, 1])
                sample_centers.append((cx, cy))
            # Right cheek
            if ok(2):
                cx = 0.55 * float(kpts[0, 0]) + 0.45 * float(kpts[2, 0])
                cy = float(kpts[0, 1]) + 0.35 * abs(float(kpts[0, 1] - kpts[2, 1]))
                sample_centers.append((cx, cy))
            elif ok(4):
                cx = 0.50 * float(kpts[0, 0]) + 0.50 * float(kpts[4, 0])
                cy = 0.65 * float(kpts[0, 1]) + 0.35 * float(kpts[4, 1])
                sample_centers.append((cx, cy))

        # Exposed wrists & lower forearms
        for wrist_idx, elbow_idx in ((9, 7), (10, 8)):
            if ok(wrist_idx):
                wx, wy = float(kpts[wrist_idx, 0]), float(kpts[wrist_idx, 1])
                sample_centers.append((wx, wy))
                if ok(elbow_idx):
                    ex, ey = float(kpts[elbow_idx, 0]), float(kpts[elbow_idx, 1])
                    sample_centers.append((0.75 * wx + 0.25 * ex, 0.75 * wy + 0.25 * ey))

    ab_pixels_list: List[np.ndarray] = []
    for cx, cy in sample_centers:
        px1 = max(0, int(round(cx)) - patch_r)
        py1 = max(0, int(round(cy)) - patch_r)
        px2 = min(w, int(round(cx)) + patch_r + 1)
        py2 = min(h, int(round(cy)) + patch_r + 1)
        if px2 - px1 < 3 or py2 - py1 < 3:
            continue
        patch = frame_bgr[py1:py2, px1:px2]
        # Normalize patch exposure to a canonical luminance before Lab conversion so dark/bright
        # lighting shifts preserve identical (a*, b*) skin chromaticity
        pf = patch.astype(np.float32)
        mean_luma = float(np.mean(pf))
        if mean_luma > 8.0:
            pf = np.clip(pf * (145.0 / mean_luma), 0.0, 255.0)
        lab = cv2.cvtColor(pf.astype(np.uint8), cv2.COLOR_BGR2LAB)
        # Drop L* channel (index 0) completely; keep only lighting-normalized (a*, b*) chroma
        ab = lab[:, :, 1:3].reshape(-1, 2)
        ab_pixels_list.append(ab)

    if not ab_pixels_list:
        # Neutral skin fallback when no face/wrist keypoints are visible
        default_ab = np.array([140.0, 142.0], dtype=np.float32)
        neutral = np.zeros((34,), dtype=np.float32)
        neutral[7] = 0.5
        neutral[23] = 0.5
        neutral[32] = (default_ab[0] - 128.0) / 30.0
        neutral[33] = (default_ab[1] - 128.0) / 30.0
        neutral /= max(1e-8, float(np.linalg.norm(neutral)))
        return neutral, default_ab, False

    ab_all = np.concatenate(ab_pixels_list, axis=0).astype(np.float32)
    # Soft chroma gate to reject stray dark background/clothing edge pixels around wrists
    skin_mask = (
        (ab_all[:, 0] >= 120.0) & (ab_all[:, 0] <= 178.0)
        & (ab_all[:, 1] >= 118.0) & (ab_all[:, 1] <= 185.0)
    )
    ab_used = ab_all[skin_mask] if int(np.sum(skin_mask)) >= 12 else ab_all

    a_hist, _ = np.histogram(ab_used[:, 0], bins=16, range=(110.0, 182.0))
    b_hist, _ = np.histogram(ab_used[:, 1], bins=16, range=(110.0, 182.0))
    # Soft-bin Gaussian smoothing across adjacent chroma bins so lighting shifts never fall off bin edges
    kernel = np.array([0.08, 0.22, 0.40, 0.22, 0.08], dtype=np.float32)
    a_h = np.convolve(a_hist.astype(np.float32), kernel, mode="same")
    b_h = np.convolve(b_hist.astype(np.float32), kernel, mode="same")
    a_h /= max(1e-6, float(np.sum(a_h)))
    b_h /= max(1e-6, float(np.sum(b_h)))

    med_ab = np.median(ab_used, axis=0).astype(np.float32)
    med_scaled = np.array(
        [(med_ab[0] - 128.0) / 30.0, (med_ab[1] - 128.0) / 30.0],
        dtype=np.float32,
    )

    hist_32 = np.concatenate([a_h, b_h], axis=0)
    hist_32 /= max(1e-8, float(np.linalg.norm(hist_32)))
    vec = np.concatenate([0.75 * hist_32, 0.25 * med_scaled], axis=0).astype(np.float32)  # 34D
    vec /= max(1e-8, float(np.linalg.norm(vec)))
    return vec, med_ab, True


def extract_two_zone_clothing_hist(frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Standard-mode 2-zone HSV+Lab clothing histogram (140D) from central 60% width."""
    h, w = frame_bgr.shape[:2]
    x1 = float(np.clip(box[0], 0.0, max(0.0, float(w) - 1.0)))
    y1 = float(np.clip(box[1], 0.0, max(0.0, float(h) - 1.0)))
    x2 = float(np.clip(box[2], x1 + 1.0, float(w)))
    y2 = float(np.clip(box[3], y1 + 1.0, float(h)))
    bw = x2 - x1
    bh = y2 - y1

    cx1 = max(0, min(w - 1, int(round(x1 + 0.20 * bw))))
    cx2 = max(cx1 + 1, min(w, int(round(x1 + 0.80 * bw))))

    zone_vecs: List[np.ndarray] = []
    for yf1, yf2 in ((0.15, 0.55), (0.55, 0.90)):
        zy1 = max(0, min(h - 1, int(round(y1 + yf1 * bh))))
        zy2 = max(zy1 + 1, min(h, int(round(y1 + yf2 * bh))))
        crop = frame_bgr[zy1:zy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            zone_vecs.append(np.zeros((70,), dtype=np.float32))
            continue
        if crop.shape[0] > 96 or crop.shape[1] > 64:
            crop = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hists = [
            cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten(),
            cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten(),
            cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten(),
            cv2.calcHist([lab], [0], None, [8], [0, 256]).flatten(),
            cv2.calcHist([lab], [1], None, [12], [0, 256]).flatten(),
            cv2.calcHist([lab], [2], None, [12], [0, 256]).flatten(),
        ]
        chans = []
        for hist in hists:
            hf = hist.astype(np.float32)
            s = float(np.sum(hf))
            if s > 1e-6:
                hf /= s
            chans.append(hf)
        zvec = np.concatenate(chans, axis=0).astype(np.float32)
        zn = float(np.linalg.norm(zvec))
        if zn > 1e-8:
            zvec /= zn
        zone_vecs.append(zvec)

    full_hist = np.concatenate(zone_vecs, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(full_hist))
    if norm > 1e-8:
        full_hist /= norm
    return full_hist


class BiometricDINOv2Encoder:
    """Unified DINOv2 (`dinov2_vitb14`, fallback `dinov2_vits14`) encoder supporting:
      1. Clothes-Invariant Biometric Mode (0% clothing color weight):
         - Head + Hair + Ears crop strictly above shoulders (averaged with horizontal flip)
         - Desaturated grayscale body-shape crop (clothing color stripped)
      2. Standard Appearance Mode:
         - Full+HFlip + Top-55% Torso DINOv2 fused with 2-zone central clothing histogram
    """

    FULL_WEIGHT: float = 0.70
    TORSO_WEIGHT: float = 0.30
    TORSO_HEIGHT_FRAC: float = 0.55
    DINO_FUSION_WEIGHT: float = 0.75
    COLOR_FUSION_WEIGHT: float = 0.25
    COLOR_DIM: int = 140

    def __init__(self, preferred_model: str = "dinov2_vitb14", device: str = "cpu") -> None:
        self.requested_backend = preferred_model
        self.backbone_name = preferred_model
        self.device = device
        self._model = None
        self._torch = None
        self._roi_align = None
        self._mean = None
        self._std = None
        self._out_hw: Tuple[int, int] = (224, 224)
        self._embed_dim: int = 768
        self._load_backbone(preferred_model)

    def _try_load_single(self, torch_mod: Any, variant: str) -> Optional[Any]:
        timm_name = (
            "vit_base_patch14_dinov2.lvd142m"
            if variant == "dinov2_vitb14"
            else "vit_small_patch14_dinov2.lvd142m"
        )
        try:
            return torch_mod.hub.load("facebookresearch/dinov2", variant, trust_repo=True)
        except Exception:
            pass
        try:
            import timm

            return timm.create_model(timm_name, pretrained=True, num_classes=0)
        except Exception:
            return None

    def _load_backbone(self, preferred: str) -> None:
        try:
            import torch
            from torchvision.ops import roi_align

            self._torch = torch
            self._roi_align = roi_align

            order = (
                ["dinov2_vitb14", "dinov2_vits14"]
                if preferred == "dinov2_vitb14"
                else ["dinov2_vits14", "dinov2_vitb14"]
            )
            net = None
            loaded_name = None
            for candidate in order:
                net = self._try_load_single(torch, candidate)
                if net is not None:
                    loaded_name = candidate
                    break

            if net is None:
                raise RuntimeError("Could not load dinov2_vitb14 or dinov2_vits14")

            net.eval().to(self.device)
            self._model = net
            self.backbone_name = loaded_name
            self._embed_dim = 768 if loaded_name == "dinov2_vitb14" else 384
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        except Exception as exc:
            print(f"[BiometricDINOv2Encoder] Falling back to CPU descriptors ({exc})")
            self.backbone_name = "cpu_hog_fallback"
            self._model = None
            self._embed_dim = 256

    @staticmethod
    def _l2_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        return (mat / norms).astype(np.float32)

    def _forward_chunked(self, batch: Any, chunk_size: int = 48) -> Any:
        torch = self._torch
        use_cuda = self.device == "cuda"
        parts = []
        n = batch.shape[0]
        for start in range(0, n, chunk_size):
            sub = batch[start : start + chunk_size]
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._model(sub)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            parts.append(feats)
        return torch.cat(parts, dim=0)

    def extract_biometric_dino(
        self,
        frame_bgr: np.ndarray,
        boxes: np.ndarray,
        head_boxes: np.ndarray,
        frame_gpu: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (head_hair_embs [N,D], grayscale_shape_embs [N,D]) L2-normalized."""
        n = len(boxes)
        if n == 0:
            empty = np.zeros((0, self._embed_dim), dtype=np.float32)
            return empty, empty

        h, w = frame_bgr.shape[:2]
        if self._model is None:
            # CPU fallback using HOG/texture on head crop and desaturated grayscale body crop
            head_vecs, gray_vecs = [], []
            for i in range(n):
                hx1, hy1, hx2, hy2 = [int(round(v)) for v in head_boxes[i]]
                hx1, hy1 = max(0, hx1), max(0, hy1)
                hx2, hy2 = min(w, max(hx1 + 4, hx2)), min(h, max(hy1 + 4, hy2))
                hcrop = frame_bgr[hy1:hy2, hx1:hx2]
                fh = AppearanceEncoder._hsv_hog_embedding(hcrop)
                fhf = AppearanceEncoder._hsv_hog_embedding(cv2.flip(hcrop, 1))
                hv = 0.5 * (fh + fhf)
                hv /= max(1e-8, float(np.linalg.norm(hv)))
                head_vecs.append(hv)

                bx1, by1, bx2, by2 = [int(round(v)) for v in boxes[i]]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, max(bx1 + 4, bx2)), min(h, max(by1 + 4, by2))
                bcrop = frame_bgr[by1:by2, bx1:bx2]
                g = cv2.cvtColor(bcrop, cv2.COLOR_BGR2GRAY)
                g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                gv = AppearanceEncoder._hsv_hog_embedding(g3)
                gray_vecs.append(gv)
            return np.stack(head_vecs, axis=0).astype(np.float32), np.stack(gray_vecs, axis=0).astype(np.float32)

        torch = self._torch
        use_cuda = self.device == "cuda"

        hb = np.asarray(head_boxes, dtype=np.float32).reshape(-1, 4).copy()
        hb[:, 0] = np.clip(hb[:, 0], 0.0, max(0.0, float(w) - 2.0))
        hb[:, 1] = np.clip(hb[:, 1], 0.0, max(0.0, float(h) - 2.0))
        hb[:, 2] = np.clip(hb[:, 2], hb[:, 0] + 1.0, float(w))
        hb[:, 3] = np.clip(hb[:, 3], hb[:, 1] + 1.0, float(h))

        fb = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        fb[:, 0] = np.clip(fb[:, 0], 0.0, max(0.0, float(w) - 2.0))
        fb[:, 1] = np.clip(fb[:, 1], 0.0, max(0.0, float(h) - 2.0))
        fb[:, 2] = np.clip(fb[:, 2], fb[:, 0] + 1.0, float(w))
        fb[:, 3] = np.clip(fb[:, 3], fb[:, 1] + 1.0, float(h))

        rois_np = np.zeros((2 * n, 5), dtype=np.float32)
        rois_np[:n, 1:] = hb
        rois_np[n:, 1:] = fb

        with torch.inference_mode():
            img_t = frame_gpu if frame_gpu is not None else upload_frame_gpu(frame_bgr, self.device)
            rois_t = torch.from_numpy(rois_np).to(self.device, non_blocking=use_cuda)
            aligned = self._roi_align(
                img_t,
                rois_t,
                output_size=self._out_hw,
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )

            head_rgb = aligned[:n]
            body_rgb = aligned[n:]

            # Strip ALL clothing color from body crop via RGB luminance -> 3-channel grayscale
            luma = (
                0.299 * body_rgb[:, 0:1, :, :]
                + 0.587 * body_rgb[:, 1:2, :, :]
                + 0.114 * body_rgb[:, 2:3, :, :]
            )
            gray_body = luma.expand(-1, 3, -1, -1).contiguous()

            # Normalize Head + Head-Flip + Desaturated Grayscale Body with ImageNet stats
            head_norm = (head_rgb - self._mean) / self._std
            head_flip = torch.flip(head_norm, dims=[3])
            gray_norm = (gray_body - self._mean) / self._std

            all_crops = torch.cat([head_norm, head_flip, gray_norm], dim=0)
            all_feats = self._forward_chunked(all_crops, chunk_size=48)

            f_head = all_feats[:n]
            f_head_flip = all_feats[n : 2 * n]
            f_gray = all_feats[2 * n :]

            head_avg = torch.nn.functional.normalize(0.5 * (f_head + f_head_flip), p=2, dim=1)
            gray_vec = torch.nn.functional.normalize(f_gray, p=2, dim=1)

        return (
            head_avg.cpu().numpy().astype(np.float32),
            gray_vec.cpu().numpy().astype(np.float32),
        )

    def extract_standard(
        self,
        frame_bgr: np.ndarray,
        boxes: np.ndarray,
        frame_gpu: Optional[Any] = None,
    ) -> np.ndarray:
        """Standard Appearance Mode (Biometric Mode OFF): DINOv2 multi-part + 2-zone clothing."""
        n = len(boxes)
        if n == 0:
            return np.zeros((0, self._embed_dim * 2 + self.COLOR_DIM), dtype=np.float32)

        color_hists = np.stack(
            [extract_two_zone_clothing_hist(frame_bgr, box) for box in boxes],
            axis=0,
        ).astype(np.float32)
        color_hists = self._l2_rows(color_hists)

        h, w = frame_bgr.shape[:2]
        if self._model is None:
            dino_vecs = []
            for box in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, max(x1 + 4, x2)), min(h, max(y1 + 4, y2))
                fc = frame_bgr[y1:y2, x1:x2]
                ty2 = max(y1 + 2, min(y2, int(round(y1 + self.TORSO_HEIGHT_FRAC * (y2 - y1)))))
                tc = frame_bgr[y1:ty2, x1:x2]
                ff = 0.5 * (AppearanceEncoder._hsv_hog_embedding(fc) + AppearanceEncoder._hsv_hog_embedding(cv2.flip(fc, 1)))
                ff /= max(1e-8, float(np.linalg.norm(ff)))
                tv = AppearanceEncoder._hsv_hog_embedding(tc)
                comb = np.concatenate([self.FULL_WEIGHT * ff, self.TORSO_WEIGHT * tv], axis=0)
                comb /= max(1e-8, float(np.linalg.norm(comb)))
                dino_vecs.append(comb)
            dino_multi = np.stack(dino_vecs, axis=0).astype(np.float32)
        else:
            torch = self._torch
            use_cuda = self.device == "cuda"
            full_b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
            full_b[:, 0] = np.clip(full_b[:, 0], 0.0, max(0.0, float(w) - 2.0))
            full_b[:, 1] = np.clip(full_b[:, 1], 0.0, max(0.0, float(h) - 2.0))
            full_b[:, 2] = np.clip(full_b[:, 2], full_b[:, 0] + 1.0, float(w))
            full_b[:, 3] = np.clip(full_b[:, 3], full_b[:, 1] + 1.0, float(h))

            torso_b = full_b.copy()
            torso_b[:, 3] = np.clip(
                full_b[:, 1] + self.TORSO_HEIGHT_FRAC * (full_b[:, 3] - full_b[:, 1]),
                full_b[:, 1] + 1.0,
                float(h),
            )
            rois_np = np.zeros((2 * n, 5), dtype=np.float32)
            rois_np[:n, 1:] = full_b
            rois_np[n:, 1:] = torso_b

            with torch.inference_mode():
                img_t = frame_gpu if frame_gpu is not None else upload_frame_gpu(frame_bgr, self.device)
                rois_t = torch.from_numpy(rois_np).to(self.device, non_blocking=use_cuda)
                aligned = self._roi_align(
                    img_t, rois_t, output_size=self._out_hw, spatial_scale=1.0, sampling_ratio=2, aligned=True
                )
                aligned.sub_(self._mean).div_(self._std)
                full_crops = aligned[:n]
                torso_crops = aligned[n:]
                flip_crops = torch.flip(full_crops, dims=[3])
                all_feats = self._forward_chunked(
                    torch.cat([full_crops, flip_crops, torso_crops], dim=0), chunk_size=48
                )
                full_flip = torch.nn.functional.normalize(0.5 * (all_feats[:n] + all_feats[n : 2 * n]), p=2, dim=1)
                torso_emb = torch.nn.functional.normalize(all_feats[2 * n :], p=2, dim=1)
                multi = torch.nn.functional.normalize(
                    torch.cat([self.FULL_WEIGHT * full_flip, self.TORSO_WEIGHT * torso_emb], dim=1), p=2, dim=1
                )
                dino_multi = multi.cpu().numpy().astype(np.float32)

        dino_multi = self._l2_rows(dino_multi)
        fused = np.concatenate(
            [self.DINO_FUSION_WEIGHT * dino_multi, self.COLOR_FUSION_WEIGHT * color_hists],
            axis=1,
        ).astype(np.float32)
        return self._l2_rows(fused)


def build_clothes_invariant_embedding(
    head_hair_emb: np.ndarray,
    gray_shape_emb: np.ndarray,
    body_ratio_vec: np.ndarray,
    skin_chroma_vec: np.ndarray,
) -> np.ndarray:
    """Build 0%-clothing-color biometric vector for Pass-1 LongHorizonTracker:
    L2Norm([0.45 * head_hair_dinov2, 0.25 * grayscale_shape_dinov2, 0.15 * body_ratio_vec, 0.15 * skin_chroma_vec]).
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        vf = np.asarray(v, dtype=np.float32).reshape(-1)
        return vf / max(1e-8, float(np.linalg.norm(vf)))

    comb = np.concatenate(
        [
            0.45 * _norm(head_hair_emb),
            0.25 * _norm(gray_shape_emb),
            0.15 * _norm(body_ratio_vec),
            0.15 * _norm(skin_chroma_vec),
        ],
        axis=0,
    ).astype(np.float32)
    return comb / max(1e-8, float(np.linalg.norm(comb)))


# ---------------------------------------------------------------------------
# Independent @st.cache_resource Model Loaders (Loaded ONCE per server session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_pose_detector(detector_weights: str, device: str) -> PosePersonDetector:
    """Load and cache YOLO11-Pose / YOLO detector once per server session."""
    return PosePersonDetector(
        model_name=detector_weights,
        imgsz=960,
        classes=[0],
        conf=0.35,
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_face_analyzer(device: str) -> ArcFaceBiometricAnalyzer:
    """Load and cache InsightFace buffalo_l ArcFace once per server session."""
    return ArcFaceBiometricAnalyzer(device=device)


@st.cache_resource(show_spinner=False)
def load_reid_encoder(reid_variant: str, device: str) -> BiometricDINOv2Encoder:
    """Load and cache DINOv2 ViT-B/14 (or ViT-S/14) once per server session."""
    return BiometricDINOv2Encoder(preferred_model=reid_variant, device=device)


@st.cache_resource(show_spinner=False)
def load_depth_estimator(depth_backend: str, device: str) -> DepthEstimator:
    """Load and cache Depth Anything V2 / MiDaS once per server session."""
    return DepthEstimator(backend=depth_backend, device=device)


def _close_writer_with_keepalive(
    writer: AsyncVideoWriter,
    progress_bar: Any,
    status_placeholder: Any,
) -> None:
    """Finalize AsyncVideoWriter + H.264 ffmpeg transcode while keeping the Streamlit WebSocket alive."""
    err: List[BaseException] = []

    def _run_close() -> None:
        try:
            writer.close()
        except BaseException as exc:  # pragma: no cover
            err.append(exc)

    t0 = time.perf_counter()
    worker = threading.Thread(target=_run_close, name="H264TranscodeKeepAlive", daemon=True)
    worker.start()

    tick = 0
    while worker.is_alive():
        worker.join(timeout=1.0)
        if worker.is_alive():
            tick += 1
            elapsed = time.perf_counter() - t0
            frac = min(0.99, 0.96 + 0.005 * min(tick, 6))
            progress_bar.progress(frac)
            status_placeholder.markdown(
                f"**Finalizing browser-playable H.264 (`avc1` / `yuv420p`) video via `imageio-ffmpeg`...** "
                f"(`{elapsed:.0f}s` elapsed)"
            )

    if err:
        raise err[0]


# ---------------------------------------------------------------------------
# Per-Track Biometric Profiles, Never-Empty Galleries & Timeline Snapshots
# ---------------------------------------------------------------------------


class TopKTrackGallery:
    """Bounded top-K gallery ranked by quality/confidence & area."""

    def __init__(self, max_entries: int = 10) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: List[Tuple[float, int, np.ndarray]] = []

    @staticmethod
    def _rank_score(conf: float, box: Optional[np.ndarray]) -> float:
        if box is None:
            return float(conf)
        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
        return float(conf) * (1.0 + 0.18 * math.log1p((bw * bh) / 1000.0))

    def add(self, embedding: np.ndarray, conf: float, box: Optional[np.ndarray], frame_idx: int) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-8:
            return
        emb = emb / norm
        rank = self._rank_score(conf, box)
        if len(self._items) < self.max_entries:
            self._items.append((rank, frame_idx, emb))
            self._items.sort(key=lambda x: -x[0])
        elif rank > self._items[-1][0]:
            self._items[-1] = (rank, frame_idx, emb)
            self._items.sort(key=lambda x: -x[0])

    def extend_from(self, other: "TopKTrackGallery") -> None:
        for rank, fidx, emb in other._items:
            if len(self._items) < self.max_entries:
                self._items.append((rank, fidx, emb))
            elif rank > self._items[-1][0]:
                self._items[-1] = (rank, fidx, emb)
            self._items.sort(key=lambda x: -x[0])

    def to_matrix(self) -> np.ndarray:
        if not self._items:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([emb for _, _, emb in self._items], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self._items)


VIEWPOINT_BUCKETS: Tuple[str, str, str] = ("FRONT", "SIDE", "BACK")


def classify_viewpoint(kpts: Optional[np.ndarray], face_detected: bool = False) -> str:
    """Classify person orientation per frame into 'FRONT', 'SIDE', or 'BACK' from 17 COCO keypoints.

    - FRONT: InsightFace detects a face inside head box, OR nose (>= 0.35) + at least one eye (>= 0.35).
    - BACK:  nose and both eyes low-confidence/missing (< 0.30) while shoulders or ears are visible (>= 0.30).
    - SIDE:  otherwise (profile view).
    """
    if face_detected:
        return "FRONT"
    if kpts is None or kpts.shape[0] < 17:
        return "SIDE"

    nose_c = float(kpts[0, 2])
    leye_c = float(kpts[1, 2])
    reye_c = float(kpts[2, 2])
    lear_c = float(kpts[3, 2])
    rear_c = float(kpts[4, 2])
    lsh_c = float(kpts[5, 2])
    rsh_c = float(kpts[6, 2])

    if nose_c >= 0.35 and (leye_c >= 0.35 or reye_c >= 0.35):
        return "FRONT"

    if (
        nose_c < 0.30
        and leye_c < 0.30
        and reye_c < 0.30
        and max(lear_c, rear_c, lsh_c, rsh_c) >= 0.30
    ):
        return "BACK"

    return "SIDE"


def extract_torso_color_recent(frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Extract L2-normalized upper-torso HSV+Lab histogram (70D) for short-term (<1.5s) turn bridging ONLY.

    NEVER used for Pass-1 biometric embeddings, Tier-1 Face Lock, or Tier-2 long-horizon matching.
    """
    h, w = frame_bgr.shape[:2]
    x1 = float(np.clip(box[0], 0.0, max(0.0, float(w) - 1.0)))
    y1 = float(np.clip(box[1], 0.0, max(0.0, float(h) - 1.0)))
    x2 = float(np.clip(box[2], x1 + 1.0, float(w)))
    y2 = float(np.clip(box[3], y1 + 1.0, float(h)))
    bw, bh = x2 - x1, y2 - y1

    cx1 = max(0, min(w - 1, int(round(x1 + 0.20 * bw))))
    cx2 = max(cx1 + 1, min(w, int(round(x1 + 0.80 * bw))))
    zy1 = max(0, min(h - 1, int(round(y1 + 0.18 * bh))))
    zy2 = max(zy1 + 1, min(h, int(round(y1 + 0.55 * bh))))
    crop = frame_bgr[zy1:zy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((70,), dtype=np.float32)

    if crop.shape[0] > 96 or crop.shape[1] > 64:
        crop = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hists = [
        cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten(),
        cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten(),
        cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten(),
        cv2.calcHist([lab], [0], None, [8], [0, 256]).flatten(),
        cv2.calcHist([lab], [1], None, [12], [0, 256]).flatten(),
        cv2.calcHist([lab], [2], None, [12], [0, 256]).flatten(),
    ]
    chans = []
    for hist in hists:
        hf = hist.astype(np.float32)
        s = float(np.sum(hf))
        if s > 1e-6:
            hf /= s
        chans.append(hf)
    vec = np.concatenate(chans, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec


@dataclass
class TrackBiometricProfile:
    """Complete per-track biometric + viewpoint-bucketed + kinematic endpoint profile."""

    face_gallery: TopKTrackGallery = field(default_factory=lambda: TopKTrackGallery(max_entries=10))
    head_gallery: Dict[str, TopKTrackGallery] = field(
        default_factory=lambda: {ang: TopKTrackGallery(max_entries=8) for ang in VIEWPOINT_BUCKETS}
    )
    gray_contour_gallery: TopKTrackGallery = field(default_factory=lambda: TopKTrackGallery(max_entries=10))
    skin_chroma_gallery: TopKTrackGallery = field(default_factory=lambda: TopKTrackGallery(max_entries=10))
    best_fused_gallery: TopKTrackGallery = field(default_factory=lambda: TopKTrackGallery(max_entries=10))
    # Running per-component lists for exact running median of skeletal body ratios (8D) & skin Lab (a*, b*)
    ratio_samples: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    skin_ab_samples: List[np.ndarray] = field(default_factory=list)
    # Short-term torso color endpoints (used ONLY by Tier-0 <=45f turn bridge, NEVER for outfit-change Re-ID)
    first_torso_color: Optional[np.ndarray] = None
    last_torso_color: Optional[np.ndarray] = None
    _first_torso_count: int = 0
    # Temporal & kinematic endpoints for Tier-0 Spatio-Temporal Turn & Flicker Bridge
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    first_bbox: Optional[np.ndarray] = None
    last_bbox: Optional[np.ndarray] = None
    first_viewpoint: str = "SIDE"
    last_viewpoint: str = "SIDE"
    recent_frame_boxes: List[Tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def torso_color_recent(self) -> Optional[np.ndarray]:
        return self.last_torso_color

    def has_any_head_gallery(self) -> bool:
        return any(len(g) > 0 for g in self.head_gallery.values())

    def all_head_embeddings(self) -> np.ndarray:
        mats = [g.to_matrix() for g in self.head_gallery.values() if len(g) > 0]
        if not mats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(mats, axis=0).astype(np.float32)

    def last_velocity(self) -> np.ndarray:
        """Average center velocity (vx, vy) in px/frame over up to the last 5 observed frames."""
        if len(self.recent_frame_boxes) < 2:
            return np.zeros((2,), dtype=np.float32)
        pts = sorted(self.recent_frame_boxes, key=lambda x: x[0])[-6:]
        vels: List[np.ndarray] = []
        for idx in range(1, len(pts)):
            f0, b0 = pts[idx - 1]
            f1, b1 = pts[idx]
            df = max(1, f1 - f0)
            c0 = np.array([0.5 * (b0[0] + b0[2]), 0.5 * (b0[1] + b0[3])], dtype=np.float32)
            c1 = np.array([0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])], dtype=np.float32)
            vels.append((c1 - c0) / float(df))
        if not vels:
            return np.zeros((2,), dtype=np.float32)
        return np.mean(np.stack(vels, axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _norm_vec(v: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(arr))
        return (arr / n).astype(np.float32) if n > 1e-8 else None

    def add_observation(
        self,
        frame_idx: int,
        conf: float,
        box: np.ndarray,
        fused_emb: np.ndarray,
        head_hair_emb: Optional[np.ndarray] = None,
        gray_contour_emb: Optional[np.ndarray] = None,
        face_obs: Optional[Tuple[np.ndarray, float]] = None,
        body_ratios: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        skin_obs: Optional[Tuple[np.ndarray, np.ndarray, bool]] = None,
        viewpoint: str = "SIDE",
        torso_color: Optional[np.ndarray] = None,
    ) -> None:
        vp = viewpoint if viewpoint in VIEWPOINT_BUCKETS else "SIDE"
        b_copy = np.asarray(box, dtype=np.float32).reshape(4).copy()

        # Update chronological temporal & spatial endpoints
        if self.first_frame is None or frame_idx < self.first_frame:
            self.first_frame = frame_idx
            self.first_bbox = b_copy.copy()
            self.first_viewpoint = vp
        if self.last_frame is None or frame_idx >= self.last_frame:
            self.last_frame = frame_idx
            self.last_bbox = b_copy.copy()
            self.last_viewpoint = vp

        self.recent_frame_boxes.append((frame_idx, b_copy))
        if len(self.recent_frame_boxes) > 6:
            self.recent_frame_boxes = sorted(self.recent_frame_boxes, key=lambda x: x[0])[-6:]

        # Short-term torso color endpoints (first 5 frames & running recent EMA)
        if torso_color is not None:
            tc = self._norm_vec(torso_color)
            if tc is not None:
                if self._first_torso_count < 5:
                    if self.first_torso_color is None:
                        self.first_torso_color = tc.copy()
                    else:
                        blended = self.first_torso_color * self._first_torso_count + tc
                        normed_f = self._norm_vec(blended)
                        self.first_torso_color = normed_f if normed_f is not None else tc.copy()
                    self._first_torso_count += 1
                if self.last_torso_color is None:
                    self.last_torso_color = tc.copy()
                else:
                    normed_l = self._norm_vec(0.65 * self.last_torso_color + 0.35 * tc)
                    self.last_torso_color = normed_l if normed_l is not None else tc.copy()

        self.best_fused_gallery.add(fused_emb, conf, b_copy, frame_idx)
        if head_hair_emb is not None:
            self.head_gallery[vp].add(head_hair_emb, conf, b_copy, frame_idx)
        if gray_contour_emb is not None:
            self.gray_contour_gallery.add(gray_contour_emb, conf, b_copy, frame_idx)
        if face_obs is not None:
            f_emb, f_score = face_obs
            self.face_gallery.add(f_emb, f_score, b_copy, frame_idx)
        if body_ratios is not None:
            raw_r, valid_m = body_ratios
            if len(self.ratio_samples) < 120:
                self.ratio_samples.append((raw_r.copy(), valid_m.copy()))
        if skin_obs is not None:
            s_vec, med_ab, valid_skin = skin_obs
            if valid_skin:
                self.skin_chroma_gallery.add(s_vec, conf, b_copy, frame_idx)
                if len(self.skin_ab_samples) < 120:
                    self.skin_ab_samples.append(med_ab.copy())

    def merge_from(self, other: "TrackBiometricProfile") -> None:
        self.face_gallery.extend_from(other.face_gallery)
        for ang in VIEWPOINT_BUCKETS:
            self.head_gallery[ang].extend_from(other.head_gallery[ang])
        self.gray_contour_gallery.extend_from(other.gray_contour_gallery)
        self.skin_chroma_gallery.extend_from(other.skin_chroma_gallery)
        self.best_fused_gallery.extend_from(other.best_fused_gallery)
        if other.ratio_samples:
            combined = self.ratio_samples + other.ratio_samples
            self.ratio_samples = combined[-180:]
        if other.skin_ab_samples:
            combined_ab = self.skin_ab_samples + other.skin_ab_samples
            self.skin_ab_samples = combined_ab[-180:]

        # Keep strictly chronological first_* and last_* endpoints & velocity history
        if other.first_frame is not None and (self.first_frame is None or other.first_frame < self.first_frame):
            self.first_frame = other.first_frame
            self.first_bbox = other.first_bbox.copy() if other.first_bbox is not None else self.first_bbox
            self.first_viewpoint = other.first_viewpoint
            if other.first_torso_color is not None:
                self.first_torso_color = other.first_torso_color.copy()
                self._first_torso_count = max(1, other._first_torso_count)
        if other.last_frame is not None and (self.last_frame is None or other.last_frame >= self.last_frame):
            self.last_frame = other.last_frame
            self.last_bbox = other.last_bbox.copy() if other.last_bbox is not None else self.last_bbox
            self.last_viewpoint = other.last_viewpoint
            if other.last_torso_color is not None:
                self.last_torso_color = other.last_torso_color.copy()

        if other.recent_frame_boxes:
            merged_recent = self.recent_frame_boxes + [(f, b.copy()) for f, b in other.recent_frame_boxes]
            self.recent_frame_boxes = sorted(merged_recent, key=lambda x: x[0])[-6:]

    def copy(self) -> "TrackBiometricProfile":
        clone = TrackBiometricProfile()
        clone.merge_from(self)
        return clone

    def body_ratio_median(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return running per-component median (8D) and valid boolean mask (8D)."""
        med = BODY_RATIO_PRIORS.copy()
        valid = np.zeros((8,), dtype=bool)
        if not self.ratio_samples:
            return med, valid
        arr = np.stack([r for r, _ in self.ratio_samples], axis=0)      # (M, 8)
        msk = np.stack([m for _, m in self.ratio_samples], axis=0)      # (M, 8)
        for d in range(8):
            vals = arr[msk[:, d], d]
            if len(vals) > 0:
                med[d] = float(np.median(vals))
                valid[d] = True
        return med, valid

    def skin_ab_median(self) -> Optional[np.ndarray]:
        if not self.skin_ab_samples:
            return None
        arr = np.stack(self.skin_ab_samples, axis=0)
        return np.median(arr, axis=0).astype(np.float32)


def _match_box_to_detection(target_box: np.ndarray, det_boxes: np.ndarray) -> Optional[int]:
    if len(det_boxes) == 0:
        return None
    diffs = np.max(np.abs(det_boxes - target_box[None, :]), axis=1)
    best_l1_idx = int(np.argmin(diffs))
    if float(diffs[best_l1_idx]) < 1e-2:
        return best_l1_idx

    ix1 = np.maximum(target_box[0], det_boxes[:, 0])
    iy1 = np.maximum(target_box[1], det_boxes[:, 1])
    ix2 = np.minimum(target_box[2], det_boxes[:, 2])
    iy2 = np.minimum(target_box[3], det_boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_t = max(1e-6, float((target_box[2] - target_box[0]) * (target_box[3] - target_box[1])))
    area_d = np.maximum(1e-6, (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]))
    ious = inter / np.maximum(1e-6, area_t + area_d - inter)
    best_iou_idx = int(np.argmax(ious))
    if float(ious[best_iou_idx]) >= 0.50:
        return best_iou_idx
    return None


@dataclass
class TrackBoxSnapshot:
    track_id: int
    box: np.ndarray  # (4,) float32 xyxy
    score: float
    depth_ema: float
    time_since_update: int
    is_occluded: bool
    is_depth_occluded: bool
    occluder_id: Optional[int]
    last_clean_reason: str
    stitched_count: int
    vault_len: int
    vault_max: int
    keypoints: Optional[np.ndarray] = None  # (17, 3) float32
    has_face_lock: bool = False
    is_backfilled: bool = False
    is_global_recovered: bool = False


@dataclass
class ProbBoxSnapshot:
    prob_id: int
    box: np.ndarray
    hits: int
    depth_ema: float
    last_veto_reason: Optional[str]
    keypoints: Optional[np.ndarray] = None


@dataclass
class FrameSnapshot:
    frame_idx: int
    detections_count: int
    cmc_dx: float
    cmc_dy: float
    cmc_dtheta: float
    cmc_inlier_ratio: float
    cmc_fallback: bool
    active: List[TrackBoxSnapshot] = field(default_factory=list)
    probationary: List[ProbBoxSnapshot] = field(default_factory=list)
    ghost_count: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass 2: Hierarchical Biometric Consolidation (Tier 1 Face -> Tier 2 Head/Body/Skin)
# ---------------------------------------------------------------------------


def _top3_set_to_set_similarity(gal_a: np.ndarray, gal_b: np.ndarray) -> float:
    """Symmetric set-to-set top-3 mean cosine similarity in [0, 1]."""
    if gal_a.size == 0 or gal_b.size == 0:
        return 0.0
    a = np.asarray(gal_a, dtype=np.float32)
    b = np.asarray(gal_b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)

    sim_mat = np.clip(a @ b.T, 0.0, 1.0)
    row_best = np.max(sim_mat, axis=1)
    col_best = np.max(sim_mat, axis=0)
    ka = min(3, len(row_best))
    kb = min(3, len(col_best))
    return float(np.clip(0.5 * (float(np.mean(np.sort(row_best)[-ka:])) + float(np.mean(np.sort(col_best)[-kb:]))), 0.0, 1.0))


def _body_shape_similarity(prof_a: TrackBiometricProfile, prof_b: TrackBiometricProfile) -> float:
    """Scale-invariant similarity in [0, 1] between two tracks' running-median body ratios."""
    med_a, val_a = prof_a.body_ratio_median()
    med_b, val_b = prof_b.body_ratio_median()
    shared = val_a & val_b
    if not np.any(shared):
        return 0.5
    rel_diff = np.abs(med_a[shared] - med_b[shared]) / BODY_RATIO_SCALES[shared]
    # Exponential kernel over mean normalized ratio error
    return float(np.clip(math.exp(-0.65 * float(np.mean(rel_diff))), 0.0, 1.0))


def _skin_tone_similarity(prof_a: TrackBiometricProfile, prof_b: TrackBiometricProfile) -> float:
    """Lighting-normalized CIE Lab (a*, b*) skin chroma similarity in [0, 1]."""
    ga = prof_a.skin_chroma_gallery.to_matrix()
    gb = prof_b.skin_chroma_gallery.to_matrix()
    med_a = prof_a.skin_ab_median()
    med_b = prof_b.skin_ab_median()
    if ga.size == 0 or gb.size == 0 or med_a is None or med_b is None:
        return 0.5
    hist_sim = _top3_set_to_set_similarity(ga, gb)
    chroma_dist = float(np.linalg.norm(med_a - med_b))
    med_sim = float(math.exp(-chroma_dist / 14.0))
    return float(np.clip(0.55 * hist_sim + 0.45 * med_sim, 0.0, 1.0))


def _box_iou_single(b1: np.ndarray, b2: np.ndarray) -> float:
    ix1 = max(float(b1[0]), float(b2[0]))
    iy1 = max(float(b1[1]), float(b2[1]))
    ix2 = min(float(b1[2]), float(b2[2]))
    iy2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1e-6, float((b1[2] - b1[0]) * (b1[3] - b1[1])))
    a2 = max(1e-6, float((b2[2] - b2[0]) * (b2[3] - b2[1])))
    return float(inter / max(1e-6, a1 + a2 - inter))


def _project_box(box: np.ndarray, vel_xy: np.ndarray, steps: int) -> np.ndarray:
    dx = float(vel_xy[0]) * float(steps)
    dy = float(vel_xy[1]) * float(steps)
    return np.array([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], dtype=np.float32)


def _angle_aware_head_similarity(
    prof_a: TrackBiometricProfile,
    prof_b: TrackBiometricProfile,
) -> Tuple[float, str, bool]:
    """Compare matching viewpoint head buckets first (BACK<->BACK, FRONT<->FRONT, SIDE<->SIDE).

    Falls back to attenuated cross-angle similarity only when no shared angle bucket exists.
    Returns (head_sim [0,1], angle_label, used_same_angle).
    """
    shared_scores: List[Tuple[float, str]] = []
    for ang in VIEWPOINT_BUCKETS:
        ga = prof_a.head_gallery[ang].to_matrix()
        gb = prof_b.head_gallery[ang].to_matrix()
        if ga.size > 0 and gb.size > 0:
            sim = _top3_set_to_set_similarity(ga, gb)
            shared_scores.append((sim, f"{ang.title()}↔{ang.title()}"))

    if shared_scores:
        shared_scores.sort(key=lambda kv: -kv[0])
        return shared_scores[0][0], shared_scores[0][1], True

    # No shared angle bucket: fall back to cross-angle similarity with lower confidence weight
    all_a = prof_a.all_head_embeddings()
    all_b = prof_b.all_head_embeddings()
    if all_a.size == 0 or all_b.size == 0:
        return 0.0, "No-Head", False
    cross_sim = _top3_set_to_set_similarity(all_a, all_b)
    return float(cross_sim), "Cross-Angle", False


def _tier2_biometric_score(
    prof_a: TrackBiometricProfile,
    prof_b: TrackBiometricProfile,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Tier-2 Angle-Aware Clothes-Invariant Biometric Score (0% clothing weight).

    When a shared viewpoint bucket exists (BACK<->BACK, FRONT<->FRONT, SIDE<->SIDE):
      0.45 * head_hair_sim + 0.25 * body_shape_ratio_sim + 0.15 * skin_tone_sim + 0.15 * grayscale_contour_sim
    When NO shared viewpoint bucket exists (cross-angle fallback):
      shifts weight off cross-angle head similarity onto body shape + skin tone + grayscale contour
      (0.28 * head_cross + 0.34 * body_shape_ratio_sim + 0.19 * skin_tone_sim + 0.19 * grayscale_contour_sim).
    """
    head_sim, angle_tag, same_angle = _angle_aware_head_similarity(prof_a, prof_b)
    body_sim = _body_shape_similarity(prof_a, prof_b)
    skin_sim = _skin_tone_similarity(prof_a, prof_b)
    gray_sim = _top3_set_to_set_similarity(prof_a.gray_contour_gallery.to_matrix(), prof_b.gray_contour_gallery.to_matrix())

    if same_angle:
        total = 0.45 * head_sim + 0.25 * body_sim + 0.15 * skin_sim + 0.15 * gray_sim
    else:
        total = 0.28 * head_sim + 0.34 * body_sim + 0.19 * skin_sim + 0.19 * gray_sim

    total = float(np.clip(total, 0.0, 1.0))
    return total, {
        "head_hair_sim": head_sim,
        "head_angle_tag": angle_tag,
        "same_angle": same_angle,
        "body_shape_sim": body_sim,
        "skin_tone_sim": skin_sim,
        "gray_contour_sim": gray_sim,
    }


def _describe_biometric_cues(parts: Dict[str, Any]) -> str:
    """Format human-readable biometric cues contributing to a Tier-2 match."""
    angle_tag = str(parts.get("head_angle_tag", "Head/Hair"))
    cues: List[str] = []
    if float(parts.get("head_hair_sim", 0.0)) >= 0.62:
        cues.append(f"Head/Hair [{angle_tag}] ({float(parts['head_hair_sim']):.2f})")
    if float(parts.get("body_shape_sim", 0.0)) >= 0.66:
        cues.append(f"Body Shape ({float(parts['body_shape_sim']):.2f})")
    if float(parts.get("skin_tone_sim", 0.0)) >= 0.66:
        cues.append(f"Skin Tone ({float(parts['skin_tone_sim']):.2f})")
    if float(parts.get("gray_contour_sim", 0.0)) >= 0.64:
        cues.append(f"Grayscale Build ({float(parts['gray_contour_sim']):.2f})")
    if not cues:
        cues = [
            f"Head/Hair [{angle_tag}] ({float(parts.get('head_hair_sim', 0.0)):.2f})",
            f"Body Shape ({float(parts.get('body_shape_sim', 0.0)):.2f})",
            f"Skin Tone ({float(parts.get('skin_tone_sim', 0.0)):.2f})",
        ]
    return " + ".join(cues)


def run_pass2_biometric_consolidation(
    timeline: List[FrameSnapshot],
    track_profiles: Dict[int, TrackBiometricProfile],
    track_class: Dict[int, int],
    track_frames: Dict[int, Set[int]],
    prob_frames: Dict[int, Set[int]],
    stitch_prob_to_target: Dict[int, int],
    new_track_prob_to_id: Dict[int, int],
    pass1_recoveries: List[Dict[str, Any]],
    src_fps: float,
    biometric_mode: bool = True,
    show_verifying: bool = False,
    min_track_frames: int = 6,
    turn_bridge_max_gap: int = 45,
    face_lock_thresh: float = 0.40,
    face_margin_thresh: float = 0.05,
    tier2_bio_thresh: float = 0.66,
    tier2_margin_thresh: float = 0.05,
    std_merge_thresh: float = 0.65,
    std_margin_thresh: float = 0.05,
) -> Tuple[
    List[FrameSnapshot],
    Dict[int, int],
    List[Dict[str, Any]],
    int,
    int,
    Set[int],
]:
    """Execute Pass-2 3-Tier Consolidation Cascade (run iteratively until convergence):
      - Tier 0: Spatio-Temporal Turn & Flicker Bridge (`ROTATION_SPATIAL_BRIDGE`)
      - Tier 1: Direct Face Lock (`BIOMETRIC_FACE_LOCK`, 0% clothing weight)
      - Tier 2: Angle-Aware Clothes-Invariant Biometrics (`BIOMETRIC_HEAD_BODY_SKIN`, 0% clothing weight)
    """
    # ------------------------------------------------------------------
    # Step 2a: Fold STITCH_COMMIT probation frames into track frame sets
    # ------------------------------------------------------------------
    effective_frames: Dict[int, Set[int]] = {tid: set(fset) for tid, fset in track_frames.items()}
    for pid, target_tid in stitch_prob_to_target.items():
        if target_tid in effective_frames:
            effective_frames[target_tid].update(prob_frames.get(pid, set()))

    all_confirmed = sorted(
        effective_frames.keys(),
        key=lambda tid: (min(effective_frames[tid]) if effective_frames[tid] else 10**9, tid),
    )

    cluster_frames: Dict[int, Set[int]] = {tid: set(effective_frames[tid]) for tid in all_confirmed}
    cluster_class: Dict[int, int] = {tid: track_class.get(tid, 0) for tid in all_confirmed}
    cluster_members: Dict[int, List[int]] = {tid: [tid] for tid in all_confirmed}
    cluster_profiles: Dict[int, TrackBiometricProfile] = {
        tid: (track_profiles[tid].copy() if tid in track_profiles else TrackBiometricProfile())
        for tid in all_confirmed
    }
    member_to_cluster: Dict[int, int] = {tid: tid for tid in all_confirmed}

    # Build per-frame active box index AND per-cluster frame->box lookup for overlap-tolerant Tier-0 handoff
    frame_boxes_index: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    cluster_frame_boxes: Dict[int, Dict[int, Tuple[np.ndarray, bool, float]]] = {tid: {} for tid in all_confirmed}
    for snap in timeline:
        if snap.active:
            frame_boxes_index[snap.frame_idx] = [(a.track_id, a.box) for a in snap.active]
            for a in snap.active:
                if a.track_id in cluster_frame_boxes:
                    is_clean_active = (a.time_since_update == 0) and (not a.is_depth_occluded)
                    cluster_frame_boxes[a.track_id][snap.frame_idx] = (a.box, is_clean_active, float(a.score))
        for p in snap.probationary:
            target_tid = stitch_prob_to_target.get(p.prob_id)
            if target_tid is not None and target_tid in cluster_frame_boxes:
                if snap.frame_idx not in cluster_frame_boxes[target_tid]:
                    cluster_frame_boxes[target_tid][snap.frame_idx] = (p.box, True, 0.75)

    active_clusters: Set[int] = set(all_confirmed)
    raw_global_merges: List[Dict[str, Any]] = []

    def _select_best_pair(
        pair_records: List[Tuple[float, int, int, Dict[str, Any]]],
        sim_thresh: float,
        margin_thresh: float,
    ) -> Optional[Tuple[float, int, int, float, float, bool, Dict[str, Any]]]:
        if not pair_records:
            return None
        per_c: Dict[int, List[Tuple[float, int]]] = {}
        for sim, ce, cl, _ in pair_records:
            per_c.setdefault(ce, []).append((sim, cl))
            per_c.setdefault(cl, []).append((sim, ce))

        best_partner: Dict[int, Optional[int]] = {}
        second_best: Dict[int, float] = {}
        for cid, sc_list in per_c.items():
            sc_list.sort(key=lambda kv: -kv[0])
            best_partner[cid] = sc_list[0][1]
            second_best[cid] = sc_list[1][0] if len(sc_list) > 1 else 0.0

        pair_records.sort(key=lambda x: -x[0])
        for sim, c_early, c_late, details in pair_records:
            if sim < sim_thresh:
                break
            is_mutual = best_partner.get(c_early) == c_late and best_partner.get(c_late) == c_early
            comp_sim = second_best.get(c_late, 0.0)
            margin = sim - comp_sim
            if is_mutual or margin >= margin_thresh:
                return sim, c_early, c_late, comp_sim, margin, is_mutual, details
        return None

    def _commit_merge(c_early: int, c_late: int) -> None:
        cluster_frames[c_early].update(cluster_frames[c_late])
        cluster_profiles[c_early].merge_from(cluster_profiles[c_late])
        # Merge per-frame box lookup (prefer observed ACTIVE over LOCKED/coasting on any overlapping frame)
        early_fb = cluster_frame_boxes[c_early]
        for fidx, (box_l, clean_l, score_l) in cluster_frame_boxes[c_late].items():
            if fidx not in early_fb:
                early_fb[fidx] = (box_l, clean_l, score_l)
            else:
                _, clean_e, score_e = early_fb[fidx]
                if (clean_l, score_l) > (clean_e, score_e):
                    early_fb[fidx] = (box_l, clean_l, score_l)
        for m_tid in cluster_members[c_late]:
            member_to_cluster[m_tid] = c_early
        cluster_members[c_early].extend(cluster_members[c_late])
        active_clusters.discard(c_late)

    # ------------------------------------------------------------------
    # Step 2b: Iterative 3-Tier Cascade (Tier 0 Handoff/Turn Bridge -> Tier 1 Face -> Tier 2 Head/Body/Skin)
    # ------------------------------------------------------------------
    while len(active_clusters) >= 2:
        c_list = sorted(
            active_clusters,
            key=lambda cid: (min(cluster_frames[cid]) if cluster_frames[cid] else 10**9, cid),
        )
        merged_this_round = False

        # --- TIER 0: Overlap-Tolerant Spatial Handoff & Turn Bridge [RUN FIRST] ---
        # Handles both overlapping coasting/locked handoffs (0..45f overlap) and small-gap turns (0..45f gap).
        tier0_candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
        for i in range(len(c_list)):
            ci = c_list[i]
            fi = cluster_frames[ci]
            pi = cluster_profiles[ci]
            if not fi or pi.last_bbox is None or pi.first_bbox is None:
                continue
            for j in range(i + 1, len(c_list)):
                cj = c_list[j]
                if cluster_class[ci] != cluster_class[cj]:
                    continue
                fj = cluster_frames[cj]
                pj = cluster_profiles[cj]
                if not fj or pj.last_bbox is None or pj.first_bbox is None:
                    continue

                ei_first = pi.first_frame if pi.first_frame is not None else min(fi)
                ei_last = pi.last_frame if pi.last_frame is not None else max(fi)
                ej_first = pj.first_frame if pj.first_frame is not None else min(fj)
                ej_last = pj.last_frame if pj.last_frame is not None else max(fj)

                # Require true temporal handoff: first_early < first_late AND last_early < last_late
                if ei_first < ej_first and ei_last < ej_last:
                    c_early, c_late = ci, cj
                    p_early, p_late = pi, pj
                    first_late_f, last_early_f = ej_first, ei_last
                elif ej_first < ei_first and ej_last < ei_last:
                    c_early, c_late = cj, ci
                    p_early, p_late = pj, pi
                    first_late_f, last_early_f = ei_first, ej_last
                else:
                    continue

                overlap_frames = cluster_frames[c_early] & cluster_frames[c_late]
                n_ov = len(overlap_frames)
                shorter_len = max(1, min(len(cluster_frames[c_early]), len(cluster_frames[c_late])))
                gap = first_late_f - last_early_f

                if n_ov > 0:
                    # Subcase A: Overlapping handoff (e.g., locked/coasting cashier #7 overlapping replacement #16)
                    if n_ov > turn_bridge_max_gap or (n_ov / float(shorter_len)) >= 0.35:
                        continue

                    fb_e = cluster_frame_boxes[c_early]
                    fb_l = cluster_frame_boxes[c_late]
                    ious_ov: List[float] = []
                    dists_ov: List[float] = []
                    scales_ov: List[float] = []
                    for ov_f in sorted(overlap_frames):
                        if ov_f not in fb_e or ov_f not in fb_l:
                            continue
                        be = fb_e[ov_f][0]
                        bl = fb_l[ov_f][0]
                        ious_ov.append(_box_iou_single(be, bl))
                        h1 = max(1.0, float(be[3] - be[1]))
                        h2 = max(1.0, float(bl[3] - bl[1]))
                        avg_h = 0.5 * (h1 + h2)
                        c1x, c1y = 0.5 * (float(be[0]) + float(be[2])), 0.5 * (float(be[1]) + float(be[3]))
                        c2x, c2y = 0.5 * (float(bl[0]) + float(bl[2])), 0.5 * (float(bl[1]) + float(bl[3]))
                        dists_ov.append(float(math.hypot(c1x - c2x, c1y - c2y) / max(1.0, avg_h)))
                        scales_ov.append(float(min(h1, h2) / max(h1, h2)))

                    if not ious_ov:
                        continue
                    med_iou = float(np.median(ious_ov))
                    med_dist = float(np.median(dists_ov))
                    med_scale = float(np.median(scales_ov))

                    if not (med_iou >= 0.35 or (med_dist <= 0.35 and med_scale >= 0.65)):
                        continue

                    vp_e = p_early.last_viewpoint.title()
                    vp_l = p_late.first_viewpoint.title()
                    rank_score = float(0.60 * med_iou + 0.40 * max(0.0, 1.0 - med_dist))
                    tier0_candidates.append(
                        (
                            rank_score,
                            c_early,
                            c_late,
                            {
                                "event_type": "HANDOFF_OVERLAP_BRIDGE",
                                "recovery_type": f"Pass 2: HANDOFF_OVERLAP_BRIDGE ({vp_e}↔{vp_l})",
                                "biometric_cue": (
                                    f"Spatial Handoff ({n_ov}f overlap, medIoU={med_iou:.2f}, d={med_dist:.2f})"
                                ),
                                "gap_frames": 0,
                                "overlap_frames": n_ov,
                                "first_late_frame": first_late_f,
                                "bbox_iou": med_iou,
                                "norm_center_dist": med_dist,
                                "scale_ratio": med_scale,
                                "vp_pair": f"{vp_e}↔{vp_l}",
                            },
                        )
                    )
                    continue

                # Subcase B: Small non-overlapping gap (0 <= gap <= 45 frames)
                if not (0 <= gap <= turn_bridge_max_gap):
                    continue

                vel = p_early.last_velocity()
                proj_box = _project_box(p_early.last_bbox, vel, min(max(gap, 0), 8))
                late_box = p_late.first_bbox

                iou_val = _box_iou_single(proj_box, late_box)
                h1 = max(1.0, float(proj_box[3] - proj_box[1]))
                h2 = max(1.0, float(late_box[3] - late_box[1]))
                avg_h = 0.5 * (h1 + h2)
                c1x, c1y = 0.5 * (float(proj_box[0]) + float(proj_box[2])), 0.5 * (float(proj_box[1]) + float(proj_box[3]))
                c2x, c2y = 0.5 * (float(late_box[0]) + float(late_box[2])), 0.5 * (float(late_box[1]) + float(late_box[3]))
                norm_dist = float(math.hypot(c1x - c2x, c1y - c2y) / max(1.0, avg_h))
                scale_ratio = float(min(h1, h2) / max(h1, h2))

                # Short-term continuity signals (used strictly for <=45f gap bridging)
                other_overlaps = False
                for otid, obox in frame_boxes_index.get(first_late_f, []):
                    oroot = member_to_cluster.get(otid, otid)
                    if oroot in (c_early, c_late):
                        continue
                    if _box_iou_single(late_box, obox) >= 0.30:
                        other_overlaps = True
                        break
                isolated_turn = (norm_dist <= 0.25) and (not other_overlaps)

                torso_sim = 0.0
                if p_early.last_torso_color is not None and p_late.first_torso_color is not None:
                    torso_sim = float(np.clip(np.dot(p_early.last_torso_color, p_late.first_torso_color), 0.0, 1.0))
                body_sim = _body_shape_similarity(p_early, p_late)

                direct_iou_pass = (iou_val >= 0.32) and (scale_ratio >= 0.65)
                center_continuity_pass = (
                    norm_dist <= 0.45
                    and scale_ratio >= 0.65
                    and (isolated_turn or torso_sim >= 0.62 or body_sim >= 0.75)
                )
                if not (direct_iou_pass or center_continuity_pass):
                    continue

                reasons: List[str] = []
                if direct_iou_pass:
                    reasons.append(f"IoU={iou_val:.2f}")
                if isolated_turn:
                    reasons.append(f"In-Place Turn (d={norm_dist:.2f})")
                if torso_sim >= 0.62:
                    reasons.append(f"Short-Term Torso ({torso_sim:.2f})")
                if body_sim >= 0.75:
                    reasons.append(f"Body Shape ({body_sim:.2f})")

                rank_score = float(
                    0.45 * iou_val
                    + 0.30 * max(0.0, 1.0 - norm_dist)
                    + 0.25 * max(torso_sim, body_sim, 0.80 if (isolated_turn or direct_iou_pass) else 0.0)
                )
                vp_e = p_early.last_viewpoint.title()
                vp_l = p_late.first_viewpoint.title()
                tier0_candidates.append(
                    (
                        rank_score,
                        c_early,
                        c_late,
                        {
                            "event_type": "ROTATION_SPATIAL_BRIDGE",
                            "recovery_type": f"Pass 2: ROTATION_SPATIAL_BRIDGE ({vp_e}↔{vp_l})",
                            "biometric_cue": f"Turn Bridge ({vp_e}↔{vp_l}): {' + '.join(reasons)}",
                            "gap_frames": gap,
                            "overlap_frames": 0,
                            "first_late_frame": first_late_f,
                            "bbox_iou": iou_val,
                            "norm_center_dist": norm_dist,
                            "scale_ratio": scale_ratio,
                            "torso_color_sim": torso_sim,
                            "body_shape_sim": body_sim,
                            "vp_pair": f"{vp_e}↔{vp_l}",
                            "continuity_reason": " + ".join(reasons),
                        },
                    )
                )

        if tier0_candidates:
            tier0_candidates.sort(key=lambda x: -x[0])
            rank_score, c_early, c_late, det = tier0_candidates[0]
            _commit_merge(c_early, c_late)
            raw_global_merges.append(
                {
                    "frame": det["first_late_frame"],
                    "event_type": det["event_type"],
                    "recovery_type": det["recovery_type"],
                    "biometric_cue": det["biometric_cue"],
                    "early_root": c_early,
                    "late_root": c_late,
                    "gap_frames": det["gap_frames"],
                    "best_sim": float(rank_score),
                    "second_best_sim": 0.0,
                    "margin": float(rank_score),
                    "is_mutual": True,
                    "details": det,
                }
            )
            merged_this_round = True
            continue

        if biometric_mode:
            # --- TIER 1: Direct Face Lock (ArcFace top-3 cosine >= 0.40 & mutual-best/margin >= 0.05, 0% clothing) ---
            tier1_pairs: List[Tuple[float, int, int, Dict[str, Any]]] = []
            for i in range(len(c_list)):
                ci = c_list[i]
                fi = cluster_frames[ci]
                fgal_i = cluster_profiles[ci].face_gallery.to_matrix()
                if not fi or fgal_i.size == 0:
                    continue
                for j in range(i + 1, len(c_list)):
                    cj = c_list[j]
                    if cluster_class[ci] != cluster_class[cj]:
                        continue
                    fj = cluster_frames[cj]
                    fgal_j = cluster_profiles[cj].face_gallery.to_matrix()
                    if not fj or fgal_j.size == 0 or not fi.isdisjoint(fj):
                        continue
                    fsim = _top3_set_to_set_similarity(fgal_i, fgal_j)
                    ce, cl = (ci, cj) if min(fi) <= min(fj) else (cj, ci)
                    tier1_pairs.append((fsim, ce, cl, {"face_sim": fsim}))

            chosen_t1 = _select_best_pair(tier1_pairs, face_lock_thresh, face_margin_thresh)
            if chosen_t1 is not None:
                sim, c_early, c_late, comp_sim, margin, is_mutual, details = chosen_t1
                first_late = min(cluster_frames[c_late])
                prev_frames = [f for f in cluster_frames[c_early] if f < first_late]
                gap_frames = max(0, first_late - max(prev_frames)) if prev_frames else 0

                _commit_merge(c_early, c_late)
                raw_global_merges.append(
                    {
                        "frame": first_late,
                        "event_type": "BIOMETRIC_FACE_LOCK",
                        "recovery_type": "Pass 2: BIOMETRIC_FACE_LOCK",
                        "biometric_cue": f"Face (ArcFace {sim:.2f})",
                        "early_root": c_early,
                        "late_root": c_late,
                        "gap_frames": gap_frames,
                        "best_sim": float(sim),
                        "second_best_sim": float(comp_sim),
                        "margin": float(margin),
                        "is_mutual": is_mutual,
                        "details": details,
                    }
                )
                merged_this_round = True
                continue

            # --- TIER 2: Angle-Aware Clothes-Invariant Biometrics (>= 0.66 & mutual-best/margin >= 0.05, 0% clothing) ---
            tier2_pairs: List[Tuple[float, int, int, Dict[str, Any]]] = []
            for i in range(len(c_list)):
                ci = c_list[i]
                fi = cluster_frames[ci]
                if not fi or not cluster_profiles[ci].has_any_head_gallery():
                    continue
                for j in range(i + 1, len(c_list)):
                    cj = c_list[j]
                    if cluster_class[ci] != cluster_class[cj]:
                        continue
                    fj = cluster_frames[cj]
                    if not fj or not cluster_profiles[cj].has_any_head_gallery() or not fi.isdisjoint(fj):
                        continue
                    score, parts = _tier2_biometric_score(cluster_profiles[ci], cluster_profiles[cj])
                    ce, cl = (ci, cj) if min(fi) <= min(fj) else (cj, ci)
                    tier2_pairs.append((score, ce, cl, parts))

            chosen_t2 = _select_best_pair(tier2_pairs, tier2_bio_thresh, tier2_margin_thresh)
            if chosen_t2 is not None:
                sim, c_early, c_late, comp_sim, margin, is_mutual, parts = chosen_t2
                first_late = min(cluster_frames[c_late])
                prev_frames = [f for f in cluster_frames[c_early] if f < first_late]
                gap_frames = max(0, first_late - max(prev_frames)) if prev_frames else 0

                _commit_merge(c_early, c_late)
                raw_global_merges.append(
                    {
                        "frame": first_late,
                        "event_type": "BIOMETRIC_HEAD_BODY_SKIN",
                        "recovery_type": "Pass 2: BIOMETRIC_HEAD_BODY_SKIN",
                        "biometric_cue": _describe_biometric_cues(parts),
                        "early_root": c_early,
                        "late_root": c_late,
                        "gap_frames": gap_frames,
                        "best_sim": float(sim),
                        "second_best_sim": float(comp_sim),
                        "margin": float(margin),
                        "is_mutual": is_mutual,
                        "details": parts,
                    }
                )
                merged_this_round = True
                continue
        else:
            # --- Standard Appearance Mode (Biometric Mode OFF) ---
            std_pairs: List[Tuple[float, int, int, Dict[str, Any]]] = []
            for i in range(len(c_list)):
                ci = c_list[i]
                fi = cluster_frames[ci]
                gi = cluster_profiles[ci].best_fused_gallery.to_matrix()
                if not fi or gi.size == 0:
                    continue
                for j in range(i + 1, len(c_list)):
                    cj = c_list[j]
                    if cluster_class[ci] != cluster_class[cj]:
                        continue
                    fj = cluster_frames[cj]
                    gj = cluster_profiles[cj].best_fused_gallery.to_matrix()
                    if not fj or gj.size == 0 or not fi.isdisjoint(fj):
                        continue
                    sim = _top3_set_to_set_similarity(gi, gj)
                    ce, cl = (ci, cj) if min(fi) <= min(fj) else (cj, ci)
                    std_pairs.append((sim, ce, cl, {"appearance_sim": sim}))

            chosen_std = _select_best_pair(std_pairs, std_merge_thresh, std_margin_thresh)
            if chosen_std is not None:
                sim, c_early, c_late, comp_sim, margin, is_mutual, details = chosen_std
                first_late = min(cluster_frames[c_late])
                prev_frames = [f for f in cluster_frames[c_early] if f < first_late]
                gap_frames = max(0, first_late - max(prev_frames)) if prev_frames else 0

                _commit_merge(c_early, c_late)
                raw_global_merges.append(
                    {
                        "frame": first_late,
                        "event_type": "GLOBAL_AI_RECOVERY",
                        "recovery_type": "Pass 2: Global AI Recovery",
                        "biometric_cue": "Appearance + Clothing",
                        "early_root": c_early,
                        "late_root": c_late,
                        "gap_frames": gap_frames,
                        "best_sim": float(sim),
                        "second_best_sim": float(comp_sim),
                        "margin": float(margin),
                        "is_mutual": is_mutual,
                        "details": details,
                    }
                )
                merged_this_round = True
                continue

        if not merged_this_round:
            break

    # Map original Pass-1 confirmed ID -> agglomerative cluster root
    cluster_remap: Dict[int, int] = {}
    for root in active_clusters:
        for member_tid in cluster_members[root]:
            cluster_remap[member_tid] = root

    # Probationary back-fill map (all STITCH_COMMIT + globally merged NEW_TRACK probation boxes)
    prob_backfill_target: Dict[int, Tuple[int, bool]] = {}
    for pid, target_tid in stitch_prob_to_target.items():
        prob_backfill_target[pid] = (cluster_remap.get(target_tid, target_tid), False)
    for pid, new_tid in new_track_prob_to_id.items():
        mapped = cluster_remap.get(new_tid, new_tid)
        if mapped != new_tid:
            prob_backfill_target[pid] = (mapped, True)

    # ------------------------------------------------------------------
    # Step 2c: Noise Blip Filtering (<6f) + Sequential #1, #2, #3... Re-Numbering
    # ------------------------------------------------------------------
    prelim_timeline: List[Tuple[FrameSnapshot, Dict[int, TrackBoxSnapshot], List[ProbBoxSnapshot]]] = []
    cluster_obs_frames: Dict[int, List[int]] = {}
    backfilled_frame_boxes = 0

    for snap in timeline:
        by_root: Dict[int, TrackBoxSnapshot] = {}
        for a in snap.active:
            root = cluster_remap.get(a.track_id, a.track_id)
            was_remapped = root != a.track_id
            occ_root = cluster_remap.get(a.occluder_id, a.occluder_id) if a.occluder_id is not None else None
            entry = TrackBoxSnapshot(
                track_id=root,
                box=a.box,
                score=a.score,
                depth_ema=a.depth_ema,
                time_since_update=a.time_since_update,
                is_occluded=a.is_occluded,
                is_depth_occluded=a.is_depth_occluded,
                occluder_id=occ_root,
                last_clean_reason=a.last_clean_reason,
                stitched_count=a.stitched_count + (1 if was_remapped else 0),
                vault_len=a.vault_len,
                vault_max=a.vault_max,
                keypoints=a.keypoints,
                has_face_lock=len(cluster_profiles[root].face_gallery) > 0 if root in cluster_profiles else a.has_face_lock,
                is_backfilled=False,
                is_global_recovered=was_remapped,
            )
            existing = by_root.get(root)
            if existing is None:
                by_root[root] = entry
            else:
                # Frame deduplication: prefer clean ACTIVE higher-confidence box over LOCKED/coasting box
                new_prio = (
                    int(entry.time_since_update == 0 and not entry.is_depth_occluded),
                    int(entry.time_since_update == 0),
                    int(not entry.is_occluded),
                    float(entry.score),
                )
                old_prio = (
                    int(existing.time_since_update == 0 and not existing.is_depth_occluded),
                    int(existing.time_since_update == 0),
                    int(not existing.is_occluded),
                    float(existing.score),
                )
                if new_prio > old_prio:
                    by_root[root] = entry

        rem_prob: List[ProbBoxSnapshot] = []
        for p in snap.probationary:
            bf = prob_backfill_target.get(p.prob_id)
            if bf is not None:
                root, is_global = bf
                backfilled_frame_boxes += 1
                bf_entry = TrackBoxSnapshot(
                    track_id=root,
                    box=p.box,
                    score=1.0,
                    depth_ema=p.depth_ema,
                    time_since_update=0,
                    is_occluded=False,
                    is_depth_occluded=False,
                    occluder_id=None,
                    last_clean_reason="BACKFILLED",
                    stitched_count=1,
                    vault_len=len(cluster_profiles[root].best_fused_gallery) if root in cluster_profiles else 1,
                    vault_max=16,
                    keypoints=p.keypoints,
                    has_face_lock=len(cluster_profiles[root].face_gallery) > 0 if root in cluster_profiles else False,
                    is_backfilled=True,
                    is_global_recovered=is_global,
                )
                existing = by_root.get(root)
                if existing is None or existing.time_since_update > 0:
                    by_root[root] = bf_entry
            else:
                rem_prob.append(p)

        for root in by_root.keys():
            cluster_obs_frames.setdefault(root, []).append(snap.frame_idx)

        prelim_timeline.append((snap, by_root, rem_prob))

    kept_roots: List[int] = []
    filtered_blips_count = 0
    for root, f_list in cluster_obs_frames.items():
        if (not show_verifying) and len(f_list) < min_track_frames:
            filtered_blips_count += 1
            continue
        kept_roots.append(root)

    kept_roots.sort(key=lambda r: (min(cluster_obs_frames[r]), r))
    root_to_seq: Dict[int, int] = {root: idx for idx, root in enumerate(kept_roots, start=1)}

    final_id_map: Dict[int, int] = {
        orig_tid: root_to_seq[root]
        for orig_tid, root in cluster_remap.items()
        if root in root_to_seq
    }

    global_events_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    global_recoveries: List[Dict[str, Any]] = []

    for m in raw_global_merges:
        root_final = cluster_remap.get(m["early_root"], m["early_root"])
        if root_final not in root_to_seq:
            continue
        seq_id = root_to_seq[root_final]
        tag = "mutual-best" if m["is_mutual"] else f"m={m['margin']:.3f}"
        summary = (
            f"{m['event_type']}: P1-Track #{m['late_root']} -> #{seq_id} after {m['gap_frames']}f "
            f"[{m['biometric_cue']}] (sim={m['best_sim']:.3f}, {tag})"
        )
        rec = {
            "Frame": m["frame"],
            "Recovery Type": m["recovery_type"],
            "Recovered ID": f"#{seq_id}",
            "Biometric Cue": m["biometric_cue"],
            "Source": f"Pass-1 Track #{m['late_root']}",
            "Occlusion Frames": m["gap_frames"],
            "Occlusion Sec": round(m["gap_frames"] / max(1.0, src_fps), 2),
            "Best Similarity": round(m["best_sim"], 3),
            "2nd Best Sim": round(m["second_best_sim"], 3),
            "Margin": round(m["margin"], 3),
            "Summary": summary,
        }
        global_recoveries.append(rec)
        global_events_by_frame.setdefault(m["frame"], []).append(
            {
                "frame": m["frame"],
                "event_type": m["event_type"],
                "track_id": seq_id,
                "message": summary,
                "details": {
                    "merged_from_pass1_id": m["late_root"],
                    "biometric_cue": m["biometric_cue"],
                    "gap_frames": m["gap_frames"],
                    "best_sim": m["best_sim"],
                    "second_best_sim": m["second_best_sim"],
                    "margin": m["margin"],
                    **m.get("details", {}),
                },
            }
        )

    consolidated_timeline: List[FrameSnapshot] = []
    final_confirmed_ids: Set[int] = set()

    for snap, by_root, rem_prob in prelim_timeline:
        seq_active: List[TrackBoxSnapshot] = []
        for root, entry in by_root.items():
            if root not in root_to_seq:
                continue
            seq_id = root_to_seq[root]
            seq_occ = root_to_seq.get(entry.occluder_id) if entry.occluder_id is not None else None
            seq_active.append(
                TrackBoxSnapshot(
                    track_id=seq_id,
                    box=entry.box,
                    score=entry.score,
                    depth_ema=entry.depth_ema,
                    time_since_update=entry.time_since_update,
                    is_occluded=entry.is_occluded,
                    is_depth_occluded=entry.is_depth_occluded,
                    occluder_id=seq_occ,
                    last_clean_reason=entry.last_clean_reason,
                    stitched_count=entry.stitched_count,
                    vault_len=entry.vault_len,
                    vault_max=entry.vault_max,
                    keypoints=entry.keypoints,
                    has_face_lock=entry.has_face_lock,
                    is_backfilled=entry.is_backfilled,
                    is_global_recovered=entry.is_global_recovered,
                )
            )
            final_confirmed_ids.add(seq_id)

        seq_active.sort(key=lambda item: item.track_id)
        frame_evs = list(snap.events)
        if snap.frame_idx in global_events_by_frame:
            frame_evs.extend(global_events_by_frame[snap.frame_idx])

        consolidated_timeline.append(
            FrameSnapshot(
                frame_idx=snap.frame_idx,
                detections_count=snap.detections_count,
                cmc_dx=snap.cmc_dx,
                cmc_dy=snap.cmc_dy,
                cmc_dtheta=snap.cmc_dtheta,
                cmc_inlier_ratio=snap.cmc_inlier_ratio,
                cmc_fallback=snap.cmc_fallback,
                active=seq_active,
                probationary=rem_prob,
                ghost_count=snap.ghost_count,
                events=frame_evs,
            )
        )

    # Combine Pass-1 STITCH_COMMIT + Pass-2 Biometric/Global Recoveries
    all_recoveries: List[Dict[str, Any]] = []
    for rec in pass1_recoveries:
        raw_id = int(rec["Recovered ID"].lstrip("#"))
        seq_id = final_id_map.get(raw_id)
        if seq_id is None:
            continue
        updated = dict(rec)
        updated["Recovered ID"] = f"#{seq_id}"
        all_recoveries.append(updated)
    all_recoveries.extend(global_recoveries)
    all_recoveries.sort(key=lambda r: (r["Frame"], r["Recovery Type"]))

    return (
        consolidated_timeline,
        final_id_map,
        all_recoveries,
        backfilled_frame_boxes,
        filtered_blips_count,
        final_confirmed_ids,
    )


# ---------------------------------------------------------------------------
# Pass 2 Consolidated Video Renderer (with Optional 17-Keypoint Skeleton Overlay)
# ---------------------------------------------------------------------------


def _draw_coco_skeleton(
    canvas: np.ndarray,
    kpts: Optional[np.ndarray],
    color: Tuple[int, int, int],
    conf_thresh: float = 0.40,
) -> None:
    """Draw 17-keypoint COCO skeleton limbs and joints."""
    if kpts is None or kpts.shape[0] < 17:
        return
    h, w = canvas.shape[:2]
    for i, j in COCO_SKELETON_EDGES:
        if float(kpts[i, 2]) >= conf_thresh and float(kpts[j, 2]) >= conf_thresh:
            p1 = (int(np.clip(round(float(kpts[i, 0])), 0, w - 1)), int(np.clip(round(float(kpts[i, 1])), 0, h - 1)))
            p2 = (int(np.clip(round(float(kpts[j, 0])), 0, w - 1)), int(np.clip(round(float(kpts[j, 1])), 0, h - 1)))
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
    for idx in range(17):
        if float(kpts[idx, 2]) >= conf_thresh:
            pt = (int(np.clip(round(float(kpts[idx, 0])), 0, w - 1)), int(np.clip(round(float(kpts[idx, 1])), 0, h - 1)))
            dot_col = (255, 255, 255) if idx < 5 else color
            cv2.circle(canvas, pt, 3, dot_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 3, (20, 20, 20), 1, cv2.LINE_AA)


class Pass2OverlayRenderer:
    """Clean renderer for Pass-2 sequential identities with optional 17-keypoint skeleton overlay."""

    def __init__(
        self,
        probation_frames: int = 15,
        show_verifying: bool = False,
        show_skeleton: bool = False,
        trail_length: int = 32,
        max_ticker_events: int = 6,
    ) -> None:
        self.probation_frames = max(15, int(probation_frames))
        self.show_verifying = show_verifying
        self.show_skeleton = show_skeleton
        self.trail_length = trail_length
        self.trails: Dict[int, Deque[Tuple[float, float]]] = {}
        self.ticker: Deque[Dict[str, Any]] = deque(maxlen=max_ticker_events)

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        snap: FrameSnapshot,
        pass1_stitches: int,
        pass2_face_locks: int,
        pass2_bio_merges: int,
        backend_info: str,
    ) -> np.ndarray:
        canvas = frame_bgr

        for ev in snap.events:
            if ev["event_type"] in (
                "STITCH_COMMIT",
                "HANDOFF_OVERLAP_BRIDGE",
                "ROTATION_SPATIAL_BRIDGE",
                "BIOMETRIC_FACE_LOCK",
                "BIOMETRIC_HEAD_BODY_SKIN",
                "GLOBAL_AI_RECOVERY",
                "MARGIN_VETO",
                "DEPTH_OCCLUDED",
                "GHOST_SPAWN",
            ):
                self.ticker.append(ev)

        for tr in snap.active:
            color = track_color_bgr(tr.track_id)
            x1, y1, x2, y2 = [int(round(v)) for v in tr.box]
            cx = 0.5 * (float(tr.box[0]) + float(tr.box[2]))
            cy = 0.5 * (float(tr.box[1]) + float(tr.box[3]))

            trail = self.trails.setdefault(tr.track_id, deque(maxlen=self.trail_length))
            trail.append((cx, cy))
            if len(trail) >= 2:
                pts = list(trail)
                for i in range(1, len(pts)):
                    p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                    p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                    thick = 1 if i < len(pts) // 2 else 2
                    cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)

            if self.show_skeleton and tr.keypoints is not None:
                _draw_coco_skeleton(canvas, tr.keypoints, color, conf_thresh=0.40)

            box_color = (0, 165, 255) if tr.is_occluded else color
            if tr.time_since_update > 0:
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), box_color, thickness=2)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            tag = ""
            if tr.is_global_recovered:
                tag = " [BIO-LOCK]" if tr.has_face_lock else " [BIO-STITCH]"
            elif tr.is_backfilled:
                tag = " [BACK-FILL]"
            elif tr.stitched_count > 0:
                tag = f" [S{tr.stitched_count}]"

            face_icon = " [FACE]" if tr.has_face_lock and not tr.is_global_recovered else ""
            header = f"#{tr.track_id}{tag}{face_icon} | d:{tr.depth_ema:.2f} | V:{tr.vault_len}/{tr.vault_max}"
            draw_label_pill(canvas, header, (x1, max(20, y1)), box_color, (15, 15, 15), font_scale=0.46, thickness=1)

            if tr.is_depth_occluded:
                badge = (
                    f"DEPTH-OCCLUDED (by #{tr.occluder_id}) | VAULT FROZEN"
                    if tr.occluder_id is not None
                    else "DEPTH-OCCLUDED | VAULT FROZEN"
                )
                draw_label_pill(canvas, badge, (x1, y2 + 16), (0, 75, 220), (255, 255, 255), font_scale=0.38)
            elif tr.is_occluded:
                draw_label_pill(canvas, "OVERLAP | VAULT LOCKED", (x1, y2 + 16), (0, 110, 200), (255, 255, 255), font_scale=0.38)
            elif tr.is_backfilled:
                draw_label_pill(canvas, "PASS-2 BACK-FILLED ID", (x1, y2 + 16), (170, 115, 20), (255, 255, 255), font_scale=0.38)

        if self.show_verifying:
            amber = (40, 215, 255)
            veto_col = (180, 40, 255)
            for pr in snap.probationary:
                if self.show_skeleton and pr.keypoints is not None:
                    _draw_coco_skeleton(canvas, pr.keypoints, amber, conf_thresh=0.40)
                x1, y1, x2, y2 = [int(round(v)) for v in pr.box]
                col = veto_col if pr.last_veto_reason else amber
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), col, thickness=2, dash_len=7, gap_len=5)
                label = f"VERIFYING ({pr.hits}/{self.probation_frames})"
                draw_label_pill(canvas, label, (x1, max(20, y1)), (30, 30, 30), col, font_scale=0.42)
                if pr.last_veto_reason:
                    draw_label_pill(canvas, pr.last_veto_reason, (x1, y2 + 16), veto_col, (255, 255, 255), font_scale=0.38)

        self._draw_hud(canvas, snap, pass1_stitches, pass2_face_locks, pass2_bio_merges, backend_info)
        self._draw_ticker(canvas)
        return canvas

    def _draw_hud(
        self,
        canvas: np.ndarray,
        snap: FrameSnapshot,
        pass1_stitches: int,
        pass2_face_locks: int,
        pass2_bio_merges: int,
        backend_info: str,
    ) -> None:
        h, w = canvas.shape[:2]
        panel_w, panel_h = min(w - 24, 620), 104
        x0, y0 = 12, 12
        x1, y1 = x0 + panel_w, min(h - 1, y0 + panel_h)
        roi = canvas[y0:y1, x0:x1]
        bg = np.full_like(roi, (16, 18, 24), dtype=np.uint8)
        cv2.addWeighted(bg, 0.78, roi, 0.22, 0.0, roi)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (85, 100, 125), 1, cv2.LINE_AA)

        cmc_txt = (
            "CMC: FALLBACK (identity)"
            if snap.cmc_fallback
            else f"CMC: dx={snap.cmc_dx:+.1f} dy={snap.cmc_dy:+.1f} rot={snap.cmc_dtheta:+.2f}d inl={snap.cmc_inlier_ratio*100:.0f}%"
        )
        lines = [
            (f"BIOMETRIC 2-PASS TRACKER // {backend_info}", (255, 225, 110), 0.42),
            (f"Frame: {snap.frame_idx:04d}  |  Confirmed Active: {len(snap.active)}  |  Dets: {snap.detections_count}", (235, 240, 245), 0.42),
            (cmc_txt, (120, 200, 255), 0.40),
            (
                f"P1 Stitches: {pass1_stitches}  |  Face Locks: {pass2_face_locks}  |  Head/Body/Skin Merges: {pass2_bio_merges}  |  Ghosts: {snap.ghost_count}",
                (170, 250, 195),
                0.39,
            ),
        ]
        y_cursor = y0 + 22
        for text, col, scale in lines:
            cv2.putText(canvas, text, (x0 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
            y_cursor += 22

    def _draw_ticker(self, canvas: np.ndarray) -> None:
        if not self.ticker:
            return
        h, w = canvas.shape[:2]
        line_h = 19
        pad = 8
        panel_h = pad * 2 + line_h * len(self.ticker) + 16
        panel_w = min(w - 24, 680)
        x0 = 12
        y0 = max(130, h - panel_h - 12)
        x1 = x0 + panel_w
        y1 = min(h - 1, y0 + panel_h)

        roi = canvas[y0:y1, x0:x1]
        bg = np.full_like(roi, (14, 16, 22), dtype=np.uint8)
        cv2.addWeighted(bg, 0.75, roi, 0.25, 0.0, roi)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (75, 85, 105), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "BIOMETRIC & OCCLUSION DECISION LOG",
            (x0 + 8, y0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (180, 195, 215),
            1,
            cv2.LINE_AA,
        )
        y_cursor = y0 + 33
        for ev in self.ticker:
            etype = ev["event_type"]
            if etype == "BIOMETRIC_FACE_LOCK":
                col = (90, 255, 120)
            elif etype in ("HANDOFF_OVERLAP_BRIDGE", "ROTATION_SPATIAL_BRIDGE"):
                col = (80, 225, 255)
            elif etype in ("BIOMETRIC_HEAD_BODY_SKIN", "GLOBAL_AI_RECOVERY"):
                col = (90, 240, 255)
            else:
                col = EVENT_COLORS.get(etype, (220, 220, 220))
            txt = f"[F{ev['frame']:04d} {etype}] {ev['message']}"
            if len(txt) > 96:
                txt = txt[:93] + "..."
            cv2.putText(canvas, txt, (x0 + 8, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1, cv2.LINE_AA)
            y_cursor += line_h


# ---------------------------------------------------------------------------
# End-to-End 2-Pass Biometric Tracking Job
# ---------------------------------------------------------------------------


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:48] or "uploaded_video"


def run_two_pass_biometric_job(
    input_path: Path,
    output_path: Path,
    events_path: Path,
    detector_weights: str,
    people_only: bool,
    biometric_mode: bool,
    reid_variant: str,
    depth_backend: str,
    max_ghost_age: int,
    show_verifying: bool,
    show_skeleton: bool,
    depth_stride: int,
    face_stride: int,
    conf_thresh: float,
    progress_bar: Any,
    status_placeholder: Any,
) -> Dict[str, Any]:
    """Execute Pass 1 (Pose + ArcFace + Head/Shape/Skin + LongHorizonTracker) + Pass 2 (Hierarchical Biometric Consolidation + H.264)."""
    device = resolve_device("auto")

    status_placeholder.markdown(f"**Loading cached Pose/Person Detector (`{detector_weights}` on `{device.upper()}`)...**")
    progress_bar.progress(0.01)
    detector = load_pose_detector(detector_weights=detector_weights, device=device)

    status_placeholder.markdown(f"**Loading cached DINOv2 Backbone (`{reid_variant}` on `{device.upper()}`)...**")
    progress_bar.progress(0.02)
    encoder = load_reid_encoder(reid_variant=reid_variant, device=device)

    status_placeholder.markdown(f"**Loading cached Depth Guard (`{depth_backend}` on `{device.upper()}`)...**")
    progress_bar.progress(0.03)
    depth_est = load_depth_estimator(depth_backend=depth_backend, device=device)

    face_analyzer: Optional[ArcFaceBiometricAnalyzer] = None
    if biometric_mode:
        status_placeholder.markdown(f"**Loading cached Facial Biometrics (`buffalo_l` ArcFace on `{device.upper()}`)...**")
        progress_bar.progress(0.04)
        face_analyzer = load_face_analyzer(device=device)

    detector.classes = [0] if (people_only or biometric_mode) else None
    detector.conf = conf_thresh
    detector.imgsz = 960

    # Pass-1 LongHorizonTracker honors all AGENTS.md invariants (probation >= 15f, sim >= 0.72, margin >= 0.08)
    ghost_cfg = GhostConfig(
        max_ghost_age=max_ghost_age,
        corridor_expand_px_per_frame=14.0,
    )
    tracker_cfg = TrackerConfig(
        det_thresh=conf_thresh,
        max_active_miss=35,
        probation_frames=15,
        stitch_min_sim=0.72,
        margin_min_diff=0.08,
        ghost_config=ghost_cfg,
    )
    tracker = LongHorizonTracker(config=tracker_cfg)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    use_gpu_frame = device == "cuda" and (encoder._model is not None or depth_est._model is not None)
    last_depth_map: Optional[np.ndarray] = None

    timeline: List[FrameSnapshot] = []
    raw_detections_per_frame: List[List[Tuple[np.ndarray, float, int]]] = []
    track_profiles: Dict[int, TrackBiometricProfile] = {}
    prob_profiles: Dict[int, TrackBiometricProfile] = {}
    track_class: Dict[int, int] = {}
    track_frames: Dict[int, Set[int]] = {}
    prob_frames: Dict[int, Set[int]] = {}
    stitch_prob_to_target: Dict[int, int] = {}
    new_track_prob_to_id: Dict[int, int] = {}
    pass1_recoveries: List[Dict[str, Any]] = []

    faces_extracted_total = 0
    frame_count = 0
    t_start = time.perf_counter()
    last_ui_tick = t_start
    fps_ema = src_fps

    # =======================================================================
    # PASS 1: Pose + Biometric Feature Extraction + LongHorizonTracker
    # =======================================================================
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        t0 = time.perf_counter()

        frame_gpu = upload_frame_gpu(frame, device) if use_gpu_frame else None
        boxes, scores, cls_ids, kpts = detector.detect(frame)
        raw_detections_per_frame.append(
            [(boxes[i].copy(), float(scores[i]), int(cls_ids[i])) for i in range(len(boxes))]
        )

        run_depth_now = last_depth_map is None or (frame_count - 1) % max(1, depth_stride) == 0
        cmc_res = None
        if run_depth_now:
            last_depth_map = depth_est.predict(frame, frame_gpu=frame_gpu)
        else:
            cmc_res = tracker.cmc.update(frame, boxes)
            last_depth_map = warp_depth_map(last_depth_map, cmc_res.matrix, frame.shape[:2])

        n_det = len(boxes)
        det_head_embs: List[Optional[np.ndarray]] = [None] * n_det
        det_gray_embs: List[Optional[np.ndarray]] = [None] * n_det
        det_face_obs: List[Optional[Tuple[np.ndarray, float]]] = [None] * n_det
        det_body_ratios: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * n_det
        det_skin_obs: List[Optional[Tuple[np.ndarray, np.ndarray, bool]]] = [None] * n_det
        det_viewpoints: List[str] = ["SIDE"] * n_det
        det_head_clean: List[bool] = [True] * n_det
        det_torso_colors: List[Optional[np.ndarray]] = [
            extract_torso_color_recent(frame, boxes[i]) for i in range(n_det)
        ]

        if biometric_mode and n_det > 0:
            head_boxes = np.stack(
                [compute_head_box_above_shoulders(boxes[i], kpts[i], frame.shape[:2]) for i in range(n_det)],
                axis=0,
            ).astype(np.float32)

            # Check head-only overlap independently of lower-body counter/crowd occlusion:
            # even when lower body is DEPTH-OCCLUDED behind a counter, a clear head (head_iou < 0.25)
            # still feeds head/hair, skin tone, and ArcFace into Pass-2 biometric profiles!
            for i in range(n_det):
                max_h_iou = 0.0
                for j in range(n_det):
                    if i == j:
                        continue
                    h_ov = _box_iou_single(head_boxes[i], head_boxes[j])
                    if h_ov > max_h_iou:
                        max_h_iou = h_ov
                det_head_clean[i] = max_h_iou < 0.25

            head_embs, gray_embs = encoder.extract_biometric_dino(frame, boxes, head_boxes, frame_gpu=frame_gpu)
            run_face_now = (
                face_analyzer is not None
                and face_analyzer.available
                and ((frame_count - 1) % max(1, face_stride) == 0)
            )

            fused_list: List[np.ndarray] = []
            for i in range(n_det):
                raw_r, valid_r, body_vec = extract_body_shape_vector(boxes[i], kpts[i], conf_thresh=0.40)
                skin_vec, med_ab, valid_skin = extract_skin_chroma_vector(frame, boxes[i], kpts[i], conf_thresh=0.40)
                bio_emb = build_clothes_invariant_embedding(head_embs[i], gray_embs[i], body_vec, skin_vec)
                fused_list.append(bio_emb)

                det_head_embs[i] = head_embs[i]
                det_gray_embs[i] = gray_embs[i]
                det_body_ratios[i] = (raw_r, valid_r)
                det_skin_obs[i] = (skin_vec, med_ab, valid_skin)

                face_found = False
                if run_face_now and face_analyzer is not None:
                    f_res = face_analyzer.extract_from_head_box(frame, head_boxes[i])
                    if f_res is not None:
                        det_face_obs[i] = f_res
                        face_found = True
                        faces_extracted_total += 1

                det_viewpoints[i] = classify_viewpoint(kpts[i], face_detected=face_found)

            embeddings = np.stack(fused_list, axis=0).astype(np.float32)
        else:
            embeddings = encoder.extract_standard(frame, boxes, frame_gpu=frame_gpu)
            for i in range(n_det):
                det_viewpoints[i] = classify_viewpoint(kpts[i], face_detected=False)

        detections: List[Detection] = []
        for i in range(n_det):
            d_val = extract_box_depth(last_depth_map, boxes[i], frame.shape[:2])
            detections.append(
                Detection(
                    box=boxes[i],
                    score=float(scores[i]),
                    class_id=int(cls_ids[i]),
                    embedding=embeddings[i],
                    depth=d_val,
                )
            )

        result = tracker.update(frame, detections, depth_map=last_depth_map, cmc_result=cmc_res)
        dt = max(1e-4, time.perf_counter() - t0)
        fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt)

        # Snapshot active tracks + update per-track biometric profiles
        active_snaps: List[TrackBoxSnapshot] = []
        for tr in result.active_tracks:
            track_frames.setdefault(tr.track_id, set()).add(frame_count)
            track_class[tr.track_id] = tr.class_id
            prof = track_profiles.setdefault(tr.track_id, TrackBiometricProfile())

            tr_kpts: Optional[np.ndarray] = None
            if tr.time_since_update == 0 and n_det > 0:
                mi = _match_box_to_detection(tr.current_box(), boxes)
                if mi is not None:
                    tr_kpts = kpts[mi].copy()
                    # Prevent head biometric starvation behind counters: store head/hair, skin, and
                    # ArcFace whenever the head box itself is unoccluded (head_iou < 0.25), even if
                    # lower body is DEPTH-OCCLUDED behind a counter.
                    head_ok = det_head_clean[mi]
                    prof.add_observation(
                        frame_idx=frame_count,
                        conf=float(scores[mi]),
                        box=boxes[mi],
                        fused_emb=embeddings[mi],
                        head_hair_emb=det_head_embs[mi] if head_ok else None,
                        gray_contour_emb=det_gray_embs[mi] if (not tr.is_depth_occluded) else None,
                        face_obs=det_face_obs[mi] if head_ok else None,
                        body_ratios=det_body_ratios[mi],
                        skin_obs=det_skin_obs[mi] if head_ok else None,
                        viewpoint=det_viewpoints[mi],
                        torso_color=det_torso_colors[mi],
                    )

            active_snaps.append(
                TrackBoxSnapshot(
                    track_id=tr.track_id,
                    box=tr.current_box().copy(),
                    score=float(tr.score),
                    depth_ema=float(tr.depth_ema),
                    time_since_update=int(tr.time_since_update),
                    is_occluded=bool(tr.is_occluded),
                    is_depth_occluded=bool(tr.is_depth_occluded or tr.status == "DEPTH-OCCLUDED"),
                    occluder_id=tr.occluder_id,
                    last_clean_reason=str(tr.last_clean_reason),
                    stitched_count=int(tr.stitched_count),
                    vault_len=len(tr.vault),
                    vault_max=int(tr.vault.max_entries),
                    keypoints=tr_kpts,
                    has_face_lock=len(prof.face_gallery) > 0,
                )
            )

        # Snapshot probationary tracks + capture their biometric profiles
        prob_snaps: List[ProbBoxSnapshot] = []
        for pr in result.probationary_tracks:
            prob_frames.setdefault(pr.prob_id, set()).add(frame_count)
            pprof = prob_profiles.setdefault(pr.prob_id, TrackBiometricProfile())
            pr_kpts: Optional[np.ndarray] = None
            if pr.time_since_update == 0 and n_det > 0:
                mi = _match_box_to_detection(pr.current_box(), boxes)
                if mi is not None:
                    pr_kpts = kpts[mi].copy()
                    head_ok = det_head_clean[mi]
                    pprof.add_observation(
                        frame_idx=frame_count,
                        conf=float(scores[mi]),
                        box=boxes[mi],
                        fused_emb=embeddings[mi],
                        head_hair_emb=det_head_embs[mi] if head_ok else None,
                        gray_contour_emb=det_gray_embs[mi],
                        face_obs=det_face_obs[mi] if head_ok else None,
                        body_ratios=det_body_ratios[mi],
                        skin_obs=det_skin_obs[mi] if head_ok else None,
                        viewpoint=det_viewpoints[mi],
                        torso_color=det_torso_colors[mi],
                    )

            prob_snaps.append(
                ProbBoxSnapshot(
                    prob_id=pr.prob_id,
                    box=pr.current_box().copy(),
                    hits=int(pr.hits),
                    depth_ema=float(pr.depth_ema),
                    last_veto_reason=pr.last_veto_reason,
                    keypoints=pr_kpts,
                )
            )

        frame_events: List[Dict[str, Any]] = []
        for ev in result.events:
            rec = {
                "frame": ev.frame_idx,
                "event_type": ev.event_type,
                "track_id": ev.track_id,
                "message": ev.message,
                "details": ev.details,
            }
            frame_events.append(rec)

            if ev.event_type == "STITCH_COMMIT":
                pid = ev.details.get("prob_id")
                if pid is not None:
                    pid_int = int(pid)
                    stitch_prob_to_target[pid_int] = int(ev.track_id)
                    if pid_int in prob_profiles:
                        track_profiles.setdefault(int(ev.track_id), TrackBiometricProfile()).merge_from(
                            prob_profiles[pid_int]
                        )
                ghost_age = int(ev.details.get("ghost_age", 0))
                pass1_recoveries.append(
                    {
                        "Frame": ev.frame_idx,
                        "Recovery Type": "Pass 1: STITCH_COMMIT",
                        "Recovered ID": f"#{ev.track_id}",
                        "Biometric Cue": (
                            "Head/Hair + Body Shape + Skin Tone" if biometric_mode else "Appearance + Clothing"
                        ),
                        "Source": f"Prob P{pid}" if pid is not None else "Ghost",
                        "Occlusion Frames": ghost_age,
                        "Occlusion Sec": round(ghost_age / max(1.0, src_fps), 2),
                        "Best Similarity": round(float(ev.details.get("best_sim", 0.0)), 3),
                        "2nd Best Sim": round(float(ev.details.get("second_best_sim", 0.0)), 3),
                        "Margin": round(float(ev.details.get("margin", 0.0)), 3),
                        "Summary": ev.message,
                    }
                )
            elif ev.event_type in ("NEW_TRACK", "NEW_ID_AFTER_VETO"):
                pid = ev.details.get("prob_id")
                if pid is not None:
                    pid_int = int(pid)
                    new_track_prob_to_id[pid_int] = int(ev.track_id)
                    if pid_int in prob_profiles:
                        track_profiles.setdefault(int(ev.track_id), TrackBiometricProfile()).merge_from(
                            prob_profiles[pid_int]
                        )

        timeline.append(
            FrameSnapshot(
                frame_idx=frame_count,
                detections_count=len(detections),
                cmc_dx=float(result.cmc.dx),
                cmc_dy=float(result.cmc.dy),
                cmc_dtheta=float(result.cmc.dtheta_deg),
                cmc_inlier_ratio=float(result.cmc.inlier_ratio),
                cmc_fallback=bool(result.cmc.used_fallback),
                active=active_snaps,
                probationary=prob_snaps,
                ghost_count=len(result.ghosts),
                events=frame_events,
            )
        )

        now_tick = time.perf_counter()
        if frame_count % 5 == 0 or frame_count == 1 or frame_count >= total_frames or (now_tick - last_ui_tick) >= 1.5:
            last_ui_tick = now_tick
            frac = min(0.72, 0.04 + 0.68 * (frame_count / float(total_frames)))
            progress_bar.progress(frac)
            st_counts = result.stats
            face_stat = f" &nbsp;|&nbsp; **Faces Captured:** `{faces_extracted_total}`" if biometric_mode else ""
            status_placeholder.markdown(
                f"**[Pass 1 / 2 — Biometric Tracking] Frame `{frame_count}` / `{total_frames}`** &nbsp;|&nbsp; "
                f"**Live FPS:** `{fps_ema:.1f}` &nbsp;|&nbsp; "
                f"**Active:** `{st_counts['active_count']}` &nbsp;|&nbsp; "
                f"**Ghosts:** `{st_counts['ghost_count']}` &nbsp;|&nbsp; "
                f"**Pass-1 Stitches:** `{st_counts['stitches']}`{face_stat}"
            )

    cap.release()
    t_pass1 = max(1e-3, time.perf_counter() - t_start)

    # =======================================================================
    # ZERO-OVERHEAD STANDARD BASELINE TRACKER (Pure IoU + 15f Coasting)
    # =======================================================================
    progress_bar.progress(0.73)
    status_placeholder.markdown(
        "**[Baseline Benchmark]** Running zero-overhead Standard IoU Baseline tracker on Pass-1 detections..."
    )
    baseline_timeline, baseline_unique_ids = run_standard_iou_baseline(
        raw_detections_per_frame=raw_detections_per_frame,
        max_coast_frames=15,
        iou_thresh=0.25,
    )

    # =======================================================================
    # PASS 2: Hierarchical Biometric Consolidation (Tier 0 -> Tier 1 -> Tier 2)
    # =======================================================================
    progress_bar.progress(0.74)
    status_placeholder.markdown(
        "**[Pass 2 / 2 — 3-Tier Biometric Consolidation]** Running overlap handoff bridge, "
        "Tier-1 ArcFace Face Lock, Tier-2 angle-aware Head/Body/Skin merging, and sequential re-numbering..."
    )

    (
        consolidated_timeline,
        final_id_map,
        all_recoveries,
        backfilled_frame_boxes,
        filtered_blips_count,
        final_confirmed_ids,
    ) = run_pass2_biometric_consolidation(
        timeline=timeline,
        track_profiles=track_profiles,
        track_class=track_class,
        track_frames=track_frames,
        prob_frames=prob_frames,
        stitch_prob_to_target=stitch_prob_to_target,
        new_track_prob_to_id=new_track_prob_to_id,
        pass1_recoveries=pass1_recoveries,
        src_fps=src_fps,
        biometric_mode=biometric_mode,
        show_verifying=show_verifying,
        min_track_frames=6,
        turn_bridge_max_gap=45,
        face_lock_thresh=0.40,
        face_margin_thresh=0.05,
        tier2_bio_thresh=0.66,
        tier2_margin_thresh=0.05,
        std_merge_thresh=0.65,
        std_margin_thresh=0.05,
    )

    pass1_stitch_count = int(tracker.counters.get("stitches", 0))
    pass2_turn_bridges = sum(
        1
        for r in all_recoveries
        if "ROTATION_SPATIAL_BRIDGE" in r["Recovery Type"] or "HANDOFF_OVERLAP_BRIDGE" in r["Recovery Type"]
    )
    pass2_face_locks = sum(1 for r in all_recoveries if "BIOMETRIC_FACE_LOCK" in r["Recovery Type"])
    pass2_bio_merges = sum(
        1
        for r in all_recoveries
        if "BIOMETRIC_HEAD_BODY_SKIN" in r["Recovery Type"] or "Global AI Recovery" in r["Recovery Type"]
    )

    # Aggregate merged profiles per final sequential ID to build rich per-ID stats & baseline overlap map
    seq_profiles: Dict[int, TrackBiometricProfile] = {}
    for orig_tid, seq_id in final_id_map.items():
        if orig_tid in track_profiles:
            seq_profiles.setdefault(seq_id, TrackBiometricProfile()).merge_from(track_profiles[orig_tid])

    person_stats = _build_person_stats_and_fragments(
        consolidated_timeline=consolidated_timeline,
        baseline_timeline=baseline_timeline,
        seq_profiles=seq_profiles,
        all_recoveries=all_recoveries,
        src_fps=src_fps,
    )

    with open(events_path, "w", encoding="utf-8") as log_file:
        for snap in consolidated_timeline:
            for ev in snap.events:
                log_file.write(json.dumps(ev) + "\n")

    # =======================================================================
    # PASS 2 RENDER: Stream Consolidated Sequential IDs to Even-Dim H.264 + Capture Thumbnails
    # =======================================================================
    cap2 = cv2.VideoCapture(str(input_path))
    if not cap2.isOpened():
        raise RuntimeError(f"Could not reopen uploaded video for Pass 2 render: {input_path}")

    even_w, even_h = _even_dims(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(
        str(output_path),
        fourcc,
        src_fps,
        (even_w, even_h),
        max_queue=32,
        transcode_h264=True,
    )
    renderer = Pass2OverlayRenderer(
        probation_frames=tracker_cfg.probation_frames,
        show_verifying=show_verifying,
        show_skeleton=show_skeleton,
    )
    mode_tag = "BIOMETRIC (0% Clothing)" if biometric_mode else "STANDARD"
    face_tag = "ArcFace" if (face_analyzer is not None and face_analyzer.available) else "Head/Body/Skin"
    backend_info = f"{mode_tag} | {Path(detector_weights).stem} | {encoder.backbone_name} + {face_tag}"

    person_thumbnails: Dict[int, bytes] = {}
    best_thumb_rank: Dict[int, float] = {}

    rendered_frames = 0
    last_render_tick = time.perf_counter()
    for idx, snap in enumerate(consolidated_timeline, start=1):
        ok, frame = cap2.read()
        if not ok:
            break

        # Capture best unoccluded high-confidence crop per sequential ID BEFORE drawing boxes
        for tr in snap.active:
            bw = max(1.0, float(tr.box[2] - tr.box[0]))
            bh = max(1.0, float(tr.box[3] - tr.box[1]))
            obs_bonus = 1.0 if (tr.time_since_update == 0 and not tr.is_depth_occluded) else 0.15
            rank = obs_bonus * max(0.1, float(tr.score)) * math.sqrt(bw * bh)
            if rank > best_thumb_rank.get(tr.track_id, -1.0):
                jpg = _encode_thumbnail_jpeg(frame, tr.box)
                if jpg is not None:
                    best_thumb_rank[tr.track_id] = rank
                    person_thumbnails[tr.track_id] = jpg

        vis = renderer.render_frame(
            frame,
            snap,
            pass1_stitches=pass1_stitch_count,
            pass2_face_locks=pass2_face_locks,
            pass2_bio_merges=pass2_bio_merges,
            backend_info=backend_info,
        )
        writer.write(_ensure_even_frame(vis, even_w, even_h))
        rendered_frames = idx

        now_r = time.perf_counter()
        if idx % 5 == 0 or idx == 1 or idx >= len(consolidated_timeline) or (now_r - last_render_tick) >= 1.5:
            last_render_tick = now_r
            frac = 0.76 + 0.20 * (idx / float(max(1, len(consolidated_timeline))))
            progress_bar.progress(min(0.96, frac))
            status_placeholder.markdown(
                f"**[Pass 2 / 2 — Rendering Consolidated Video] Frame `{idx}` / `{len(consolidated_timeline)}`** &nbsp;|&nbsp; "
                f"**Sequential IDs:** `{len(final_confirmed_ids)}` &nbsp;|&nbsp; "
                f"**Face Locks:** `{pass2_face_locks}` &nbsp;|&nbsp; "
                f"**Handoffs/Merges:** `{pass2_turn_bridges + pass2_bio_merges}`"
            )

    cap2.release()
    status_placeholder.markdown(
        "**Finalizing browser-playable H.264 (`avc1` / `yuv420p`) video via `imageio-ffmpeg`...**"
    )
    _close_writer_with_keepalive(writer, progress_bar, status_placeholder)
    progress_bar.progress(1.0)

    total_elapsed = max(1e-3, time.perf_counter() - t_start)
    pass1_fps = frame_count / t_pass1
    overall_fps = frame_count / total_elapsed
    max_occ_frames = max((int(r["Occlusion Frames"]) for r in all_recoveries), default=0)
    merged_ids_count = max(0, len(track_frames) - len(final_confirmed_ids) - filtered_blips_count)

    return {
        "source_video_path": str(input_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "events_path": str(events_path),
        "frames_processed": frame_count,
        "rendered_frames": rendered_frames,
        "src_fps": round(src_fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "elapsed_sec": round(total_elapsed, 2),
        "pass1_fps": round(pass1_fps, 2),
        "avg_fps": round(overall_fps, 2),
        "device": device,
        "biometric_mode": biometric_mode,
        "detector_weights": detector_weights,
        "imgsz": detector.imgsz,
        "reid_backend": encoder.backbone_name,
        "face_backend": face_analyzer.provider_label if face_analyzer is not None else "Disabled",
        "depth_backend": depth_est.backend,
        "people_only": people_only or biometric_mode,
        "faces_extracted_total": faces_extracted_total,
        "confirmed_unique_ids": len(final_confirmed_ids),
        "baseline_unique_ids": baseline_unique_ids,
        "pass1_unique_ids": len(track_frames),
        "pass1_stitches": pass1_stitch_count,
        "pass2_turn_bridges": pass2_turn_bridges,
        "pass2_face_locks": pass2_face_locks,
        "pass2_bio_merges": pass2_bio_merges,
        "total_recoveries": len(all_recoveries),
        "backfilled_frame_boxes": backfilled_frame_boxes,
        "remapped_ids_count": merged_ids_count,
        "filtered_blips_count": filtered_blips_count,
        "recoveries": all_recoveries,
        "max_occ_frames": max_occ_frames,
        "max_occ_sec": round(max_occ_frames / max(1.0, src_fps), 2),
        "counters": dict(tracker.counters),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Session-cached Pass-1/Pass-2 state for zero-rerun Tab 2 & Tab 3 rendering
        "consolidated_timeline": consolidated_timeline,
        "baseline_timeline": baseline_timeline,
        "person_thumbnails": person_thumbnails,
        "person_stats": person_stats,
        "solo_renders": {},
        "comparison_renders": {},
    }


# ---------------------------------------------------------------------------
# Zero-Overhead Baseline Tracker, Person Stats, & On-Demand Video Renderers
# ---------------------------------------------------------------------------


def run_standard_iou_baseline(
    raw_detections_per_frame: List[List[Tuple[np.ndarray, float, int]]],
    max_coast_frames: int = 15,
    iou_thresh: float = 0.25,
) -> Tuple[List[List[Tuple[int, np.ndarray, float]]], int]:
    """Pure-IoU + Hungarian baseline tracker with short 15-frame coasting.

    No Re-ID, no occlusion locking, no Pass-2 consolidation — runs in <1s on cached
    Pass-1 detections for fair apples-to-apples fragmentation benchmarking.
    """
    next_id = 1
    # active: tid -> {"box": xyxy, "miss": int, "score": float, "cls": int}
    active: Dict[int, Dict[str, Any]] = {}
    baseline_timeline: List[List[Tuple[int, np.ndarray, float]]] = []

    for dets in raw_detections_per_frame:
        tids = list(active.keys())
        n_tr = len(tids)
        n_det = len(dets)
        matched_tr: Dict[int, int] = {}
        unmatched_dets: Set[int] = set(range(n_det))

        if n_tr > 0 and n_det > 0:
            cost = np.full((n_tr, n_det), 1e5, dtype=np.float32)
            for r, tid in enumerate(tids):
                tb = active[tid]["box"]
                tcls = active[tid]["cls"]
                for c, (dbox, _, dcls) in enumerate(dets):
                    if dcls != tcls:
                        continue
                    ov = _box_iou_single(tb, dbox)
                    if ov >= iou_thresh:
                        cost[r, c] = 1.0 - ov

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.0 - iou_thresh + 1e-5:
                    tid = tids[r]
                    matched_tr[tid] = c
                    unmatched_dets.discard(c)

        # Update matched & coast unmatched tracks
        for tid in list(active.keys()):
            if tid in matched_tr:
                dbox, dscore, _ = dets[matched_tr[tid]]
                active[tid]["box"] = dbox.copy()
                active[tid]["score"] = float(dscore)
                active[tid]["miss"] = 0
            else:
                active[tid]["miss"] += 1
                if active[tid]["miss"] > max_coast_frames:
                    del active[tid]

        # Spawn new baseline ID for every unmatched detection
        for c in sorted(unmatched_dets):
            dbox, dscore, dcls = dets[c]
            active[next_id] = {
                "box": dbox.copy(),
                "score": float(dscore),
                "cls": int(dcls),
                "miss": 0,
            }
            next_id += 1

        # Record currently observed baseline tracks for rendering
        frame_out = [
            (tid, st_d["box"].copy(), float(st_d["score"]))
            for tid, st_d in sorted(active.items())
            if st_d["miss"] == 0
        ]
        baseline_timeline.append(frame_out)

    total_unique_ids = next_id - 1
    return baseline_timeline, total_unique_ids


def _encode_thumbnail_jpeg(frame_bgr: np.ndarray, box: np.ndarray, target_h: int = 180) -> Optional[bytes]:
    """Crop a padded, clean person portrait and encode as compact JPEG bytes."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = 0.12 * bw, 0.08 * bh
    cx1 = max(0, int(math.floor(x1 - px)))
    cy1 = max(0, int(math.floor(y1 - py)))
    cx2 = min(w, int(math.ceil(x2 + px)))
    cy2 = min(h, int(math.ceil(y2 + py)))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    scale = float(target_h) / float(max(1, ch))
    nw = max(48, min(220, int(round(cw * scale))))
    resized = cv2.resize(crop, (nw, target_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return bytes(buf.tobytes()) if ok else None


def _compute_frame_intervals(sorted_frames: List[int]) -> List[Tuple[int, int]]:
    if not sorted_frames:
        return []
    intervals: List[Tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f > prev + 1:
            intervals.append((start, prev))
            start = f
        prev = f
    intervals.append((start, prev))
    return intervals


def _build_person_stats_and_fragments(
    consolidated_timeline: List[FrameSnapshot],
    baseline_timeline: List[List[Tuple[int, np.ndarray, float]]],
    seq_profiles: Dict[int, TrackBiometricProfile],
    all_recoveries: List[Dict[str, Any]],
    src_fps: float,
) -> Dict[int, Dict[str, Any]]:
    """Build per-ID stats, appearance intervals, biometric modalities, and overlapping baseline IDs."""
    frames_by_id: Dict[int, List[int]] = {}
    base_overlap_counts: Dict[int, Dict[int, int]] = {}
    face_lock_flag: Dict[int, bool] = {}

    for idx, snap in enumerate(consolidated_timeline):
        base_frame = baseline_timeline[idx] if idx < len(baseline_timeline) else []
        for tr in snap.active:
            tid = tr.track_id
            frames_by_id.setdefault(tid, []).append(snap.frame_idx)
            if tr.has_face_lock:
                face_lock_flag[tid] = True
            if base_frame:
                best_bid, best_iou = None, 0.35
                for bid, bbox, _ in base_frame:
                    ov = _box_iou_single(tr.box, bbox)
                    if ov > best_iou:
                        best_iou = ov
                        best_bid = bid
                if best_bid is not None:
                    d_map = base_overlap_counts.setdefault(tid, {})
                    d_map[best_bid] = d_map.get(best_bid, 0) + 1

    rec_count_by_id: Dict[int, int] = {}
    for rec in all_recoveries:
        try:
            rid = int(str(rec.get("Recovered ID", "#0")).lstrip("#"))
            rec_count_by_id[rid] = rec_count_by_id.get(rid, 0) + 1
        except ValueError:
            pass

    stats: Dict[int, Dict[str, Any]] = {}
    fps_safe = max(1.0, float(src_fps))
    for tid in sorted(frames_by_id.keys()):
        f_list = sorted(set(frames_by_id[tid]))
        intervals = _compute_frame_intervals(f_list)
        gaps = [intervals[i][0] - intervals[i - 1][1] - 1 for i in range(1, len(intervals))]
        max_gap = max(gaps) if gaps else 0

        prof = seq_profiles.get(tid)
        face_n = len(prof.face_gallery) if prof is not None else 0
        angles = [ang for ang in VIEWPOINT_BUCKETS if prof is not None and len(prof.head_gallery[ang]) > 0]
        has_body = prof is not None and len(prof.ratio_samples) > 0
        has_skin = prof is not None and len(prof.skin_chroma_gallery) > 0

        mods: List[str] = []
        if face_n > 0:
            mods.append(f"ArcFace ({face_n})")
        if angles:
            mods.append(f"Head ({'/'.join(angles)})")
        if has_body:
            mods.append("Body Shape")
        if has_skin:
            mods.append("Skin Chroma")
        if not mods:
            mods.append("Appearance")

        # Keep baseline IDs that genuinely tracked this person (>= 3 frames overlap)
        b_counts = base_overlap_counts.get(tid, {})
        frag_ids = sorted(bid for bid, cnt in b_counts.items() if cnt >= 3)
        if not frag_ids and b_counts:
            frag_ids = sorted(b_counts.keys())

        stats[tid] = {
            "track_id": tid,
            "first_frame": f_list[0],
            "last_frame": f_list[-1],
            "frames_visible": len(f_list),
            "duration_sec": round(len(f_list) / fps_safe, 2),
            "intervals": intervals,
            "intervals_text": ", ".join(
                f"F{a}–F{b} ({(b - a + 1) / fps_safe:.1f}s)" for a, b in intervals[:6]
            )
            + (" ..." if len(intervals) > 6 else ""),
            "occlusion_gaps_count": len(gaps),
            "max_gap_frames": max_gap,
            "max_gap_sec": round(max_gap / fps_safe, 2),
            "has_face_lock": face_n > 0 or face_lock_flag.get(tid, False),
            "face_samples": face_n,
            "recovery_count": rec_count_by_id.get(tid, 0),
            "modalities": ", ".join(mods),
            "baseline_fragment_ids": frag_ids,
        }
    return stats


def render_single_person_video(
    source_video_path: Path,
    output_path: Path,
    consolidated_timeline: List[FrameSnapshot],
    target_id: int,
    spotlight_mode: str,
    draw_trail: bool,
    show_occluded_hud: bool,
    show_skeleton: bool,
    src_fps: float,
    progress_bar: Any,
    status_placeholder: Any,
) -> Path:
    """Render an on-demand H.264 solo focus video highlighting ONLY `target_id` (zero AI re-run)."""
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for Solo render: {source_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    even_w, even_h = _even_dims(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(
        str(output_path),
        fourcc,
        src_fps,
        (even_w, even_h),
        max_queue=32,
        transcode_h264=True,
    )

    # Precompute target lifespan so OCCLUDED / RE-ACQUIRING HUD only appears during active gaps
    target_frames = [
        snap.frame_idx for snap in consolidated_timeline if any(a.track_id == target_id for a in snap.active)
    ]
    first_f = min(target_frames) if target_frames else -1
    last_f = max(target_frames) if target_frames else -1

    color = track_color_bgr(target_id)
    dim_bg = spotlight_mode.startswith("Dim Background")
    trail_enabled = draw_trail or ("Motion Trail" in spotlight_mode)
    trail: Deque[Tuple[float, float]] = deque(maxlen=48)
    last_seen_frame: Optional[int] = None
    last_seen_box: Optional[np.ndarray] = None

    total = max(1, len(consolidated_timeline))
    last_tick = time.perf_counter()

    for idx, snap in enumerate(consolidated_timeline, start=1):
        ok, frame = cap.read()
        if not ok:
            break

        target_snap = next((a for a in snap.active if a.track_id == target_id), None)

        if dim_bg:
            # Darken everything outside the selected person's bounding box by 55%
            canvas = (frame.astype(np.float32) * 0.45).astype(np.uint8)
            if target_snap is not None:
                fh, fw = frame.shape[:2]
                bw = max(1.0, float(target_snap.box[2] - target_snap.box[0]))
                bh = max(1.0, float(target_snap.box[3] - target_snap.box[1]))
                pad_x, pad_y = int(round(0.08 * bw)), int(round(0.06 * bh))
                rx1 = max(0, int(round(float(target_snap.box[0]))) - pad_x)
                ry1 = max(0, int(round(float(target_snap.box[1]))) - pad_y)
                rx2 = min(fw, int(round(float(target_snap.box[2]))) + pad_x)
                ry2 = min(fh, int(round(float(target_snap.box[3]))) + pad_y)
                if rx2 > rx1 and ry2 > ry1:
                    canvas[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]
        else:
            canvas = frame

        if target_snap is not None:
            last_seen_frame = snap.frame_idx
            last_seen_box = target_snap.box.copy()
            x1, y1, x2, y2 = [int(round(v)) for v in target_snap.box]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            trail.append((cx, cy))

            if trail_enabled and len(trail) >= 2:
                pts = list(trail)
                for i in range(1, len(pts)):
                    p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                    p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                    thick = 2 if i < len(pts) // 2 else 3
                    cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)

            if show_skeleton and target_snap.keypoints is not None:
                _draw_coco_skeleton(canvas, target_snap.keypoints, color, conf_thresh=0.40)

            if target_snap.time_since_update > 0:
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), (0, 180, 255), thickness=3)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            face_tag = " | FACE LOCKED" if target_snap.has_face_lock else ""
            header = f"TARGET PERSON #{target_id}{face_tag}"
            draw_label_pill(canvas, header, (x1, max(24, y1 - 4)), color, (12, 16, 24), font_scale=0.56, thickness=2)

            if target_snap.is_depth_occluded:
                draw_label_pill(canvas, "DEPTH-OCCLUDED (VAULT FROZEN)", (x1, y2 + 18), (0, 75, 220), (255, 255, 255), font_scale=0.42)
        else:
            # Target not visible on this frame
            if trail_enabled and len(trail) >= 2:
                pts = list(trail)
                for i in range(1, len(pts)):
                    p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                    p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                    cv2.line(canvas, p0, p1, (140, 140, 140), 1, cv2.LINE_AA)

            if show_occluded_hud and first_f <= snap.frame_idx <= last_f:
                gap_f = (snap.frame_idx - last_seen_frame) if last_seen_frame is not None else 0
                gap_s = gap_f / max(1.0, src_fps)
                if last_seen_box is not None:
                    lx1, ly1, lx2, ly2 = [int(round(v)) for v in last_seen_box]
                    draw_dashed_rect(canvas, (lx1, ly1), (lx2, ly2), (0, 180, 255), thickness=2, dash_len=8, gap_len=6)
                banner = f"TARGET #{target_id} OCCLUDED / RE-ACQUIRING  |  Hidden {gap_f}f ({gap_s:.1f}s)"
                draw_label_pill(canvas, banner, (16, 44), (20, 85, 215), (255, 255, 255), font_scale=0.60, thickness=2)

        # Clean top-left solo focus pill
        status_txt = "LOCKED ON TARGET" if target_snap is not None else ("OCCLUDED / RE-ACQUIRING" if first_f <= snap.frame_idx <= last_f else "NOT IN FRAME")
        pill_col = (24, 130, 85) if target_snap is not None else (45, 55, 72)
        draw_label_pill(
            canvas,
            f"SOLO FOCUS: PERSON #{target_id}  |  F{snap.frame_idx:04d}  |  {status_txt}",
            (14, max(24, even_h - 14)),
            pill_col,
            (245, 248, 255),
            font_scale=0.48,
            thickness=1,
        )

        writer.write(_ensure_even_frame(canvas, even_w, even_h))

        now_t = time.perf_counter()
        if idx % 5 == 0 or idx == 1 or idx >= total or (now_t - last_tick) >= 1.5:
            last_tick = now_t
            progress_bar.progress(min(0.95, 0.95 * (idx / float(total))))
            status_placeholder.markdown(
                f"**Rendering Person `#{target_id}` Spotlight Video:** Frame `{idx}` / `{total}`"
            )

    cap.release()
    status_placeholder.markdown(f"**Transcoding Person `#{target_id}` H.264 (`yuv420p`) video...**")
    _close_writer_with_keepalive(writer, progress_bar, status_placeholder)
    progress_bar.progress(1.0)
    return output_path


def _draw_comparison_header(
    panel: np.ndarray,
    title: str,
    subtitle: str,
    accent_bgr: Tuple[int, int, int],
) -> None:
    h, w = panel.shape[:2]
    bar_h = 54
    roi = panel[0:bar_h, 0:w]
    bg = np.full_like(roi, (14, 18, 28), dtype=np.uint8)
    cv2.addWeighted(bg, 0.86, roi, 0.14, 0.0, roi)
    cv2.line(panel, (0, bar_h - 1), (w, bar_h - 1), accent_bgr, 2, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (6, bar_h - 1), accent_bgr, -1)
    cv2.putText(panel, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 248, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43, accent_bgr, 1, cv2.LINE_AA)


def render_side_by_side_comparison_video(
    source_video_path: Path,
    output_path: Path,
    baseline_timeline: List[List[Tuple[int, np.ndarray, float]]],
    consolidated_timeline: List[FrameSnapshot],
    src_fps: float,
    focus_person_id: Optional[int],
    focus_baseline_ids: Optional[Set[int]],
    progress_bar: Any,
    status_placeholder: Any,
) -> Path:
    """Render horizontally stacked Standard Baseline (Left) vs Long-Horizon Biometric AI (Right) H.264 video.

    Enforces total stacked width <= 1600px and strictly even width/height for yuv420p compatibility.
    """
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for Comparison render: {source_video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Each half-panel <= 800px wide so stacked total width <= 1600px, strictly even dimensions
    max_panel_w = 800
    scale = min(1.0, float(max_panel_w) / float(max(1, src_w)))
    panel_w = max(2, int(round(src_w * scale)))
    panel_h = max(2, int(round(src_h * scale)))
    panel_w, panel_h = _even_dims(panel_w, panel_h)
    sx = float(panel_w) / float(max(1, src_w))
    sy = float(panel_h) / float(max(1, src_h))

    out_w, out_h = panel_w * 2, panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(
        str(output_path),
        fourcc,
        src_fps,
        (out_w, out_h),
        max_queue=32,
        transcode_h264=True,
    )

    seen_base_ids: Set[int] = set()
    seen_ours_ids: Set[int] = set()
    base_trails: Dict[int, Deque[Tuple[float, float]]] = {}
    ours_trails: Dict[int, Deque[Tuple[float, float]]] = {}

    total = max(1, len(consolidated_timeline))
    last_tick = time.perf_counter()

    for idx, snap in enumerate(consolidated_timeline, start=1):
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        left_panel = small.copy()
        right_panel = small.copy()

        # Filter tracks if comparing Selected Person Only
        base_frame = baseline_timeline[idx - 1] if (idx - 1) < len(baseline_timeline) else []
        if focus_person_id is not None:
            allowed_base = focus_baseline_ids or set()
            base_draw = [item for item in base_frame if item[0] in allowed_base]
            ours_draw = [tr for tr in snap.active if tr.track_id == focus_person_id]
        else:
            base_draw = base_frame
            ours_draw = snap.active

        # --- LEFT PANEL: STANDARD BASELINE TRACKER ---
        for bid, bbox, _ in base_draw:
            seen_base_ids.add(bid)
            col = track_color_bgr(bid)
            x1 = int(round(float(bbox[0]) * sx))
            y1 = int(round(float(bbox[1]) * sy))
            x2 = int(round(float(bbox[2]) * sx))
            y2 = int(round(float(bbox[3]) * sy))
            tr_q = base_trails.setdefault(bid, deque(maxlen=24))
            tr_q.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            if len(tr_q) >= 2:
                pts = list(tr_q)
                for k in range(1, len(pts)):
                    cv2.line(left_panel, (int(pts[k - 1][0]), int(pts[k - 1][1])), (int(pts[k][0]), int(pts[k][1])), col, 1 if k < len(pts) // 2 else 2, cv2.LINE_AA)
            cv2.rectangle(left_panel, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            draw_label_pill(left_panel, f"STD #{bid}", (x1, max(70, y1)), col, (15, 15, 15), font_scale=0.44, thickness=1)

        left_title = "STANDARD TRACKER (Pure IoU)" if focus_person_id is None else f"STANDARD TRACKER (Person #{focus_person_id} Area)"
        left_sub = f"Active: {len(base_draw)}  |  Unique IDs Created So Far: {len(seen_base_ids)}"
        _draw_comparison_header(left_panel, left_title, left_sub, accent_bgr=(90, 110, 255))

        # --- RIGHT PANEL: LONG-HORIZON BIOMETRIC AI ---
        for tr in ours_draw:
            seen_ours_ids.add(tr.track_id)
            col = track_color_bgr(tr.track_id)
            x1 = int(round(float(tr.box[0]) * sx))
            y1 = int(round(float(tr.box[1]) * sy))
            x2 = int(round(float(tr.box[2]) * sx))
            y2 = int(round(float(tr.box[3]) * sy))
            tr_q = ours_trails.setdefault(tr.track_id, deque(maxlen=24))
            tr_q.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            if len(tr_q) >= 2:
                pts = list(tr_q)
                for k in range(1, len(pts)):
                    cv2.line(right_panel, (int(pts[k - 1][0]), int(pts[k - 1][1])), (int(pts[k][0]), int(pts[k][1])), col, 1 if k < len(pts) // 2 else 2, cv2.LINE_AA)
            bcol = (0, 165, 255) if tr.is_occluded else col
            if tr.time_since_update > 0:
                draw_dashed_rect(right_panel, (x1, y1), (x2, y2), bcol, thickness=2)
            else:
                cv2.rectangle(right_panel, (x1, y1), (x2, y2), bcol, 2, cv2.LINE_AA)
            tag = " [FACE]" if tr.has_face_lock else ""
            draw_label_pill(right_panel, f"BIO #{tr.track_id}{tag}", (x1, max(70, y1)), bcol, (15, 15, 15), font_scale=0.44, thickness=1)

        right_title = "LONG-HORIZON BIOMETRIC AI" if focus_person_id is None else f"LONG-HORIZON BIOMETRIC AI (Person #{focus_person_id})"
        right_sub = f"Active: {len(ours_draw)}  |  Consistent IDs So Far: {len(seen_ours_ids)}"
        _draw_comparison_header(right_panel, right_title, right_sub, accent_bgr=(248, 189, 56))

        stacked = cv2.hconcat([left_panel, right_panel])
        # Subtle crisp center divider line
        cv2.line(stacked, (panel_w, 0), (panel_w, panel_h - 1), (55, 65, 81), 2)
        writer.write(stacked)

        now_t = time.perf_counter()
        if idx % 5 == 0 or idx == 1 or idx >= total or (now_t - last_tick) >= 1.5:
            last_tick = now_t
            progress_bar.progress(min(0.95, 0.95 * (idx / float(total))))
            status_placeholder.markdown(
                f"**Rendering Side-by-Side Comparison (`{out_w}x{out_h}`):** Frame `{idx}` / `{total}` &nbsp;|&nbsp; "
                f"Standard IDs: `{len(seen_base_ids)}` vs Biometric AI IDs: `{len(seen_ours_ids)}`"
            )

    cap.release()
    status_placeholder.markdown("**Transcoding Side-by-Side Comparison H.264 (`yuv420p`) video...**")
    _close_writer_with_keepalive(writer, progress_bar, status_placeholder)
    progress_bar.progress(1.0)
    return output_path


# ---------------------------------------------------------------------------
# Dark-Theme Styling & Multi-Tab Streamlit Web App UI
# ---------------------------------------------------------------------------


def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Keep sidebar toggle accessible while hiding Streamlit deploy/footer clutter */
        .stAppDeployButton, footer, #MainMenu {visibility: hidden !important; display: none !important;}
        .block-container {padding-top: 1.4rem !important; padding-bottom: 2.5rem !important;}

        /* Sleek top hero header bar */
        .hero-banner {
            background: linear-gradient(135deg, #111827 0%, #0F172A 60%, #0B192C 100%);
            border: 1px solid #1F2937;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .hero-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0 0 0.25rem 0;
            letter-spacing: -0.01em;
        }
        .hero-subtitle {
            font-size: 0.92rem;
            color: #94A3B8;
            margin: 0 0 0.85rem 0;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.76rem;
            font-weight: 600;
            color: #E2E8F0;
            background: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.30);
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
        }
        .status-pill.gpu {
            color: #34D399;
            background: rgba(52, 211, 153, 0.12);
            border-color: rgba(52, 211, 153, 0.35);
        }

        /* Elevated dark glass KPI metric cards */
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border: 1px solid #1F2937;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.28);
        }
        [data-testid="stMetricLabel"] p {
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 700 !important;
        }

        /* Clean workspace tab bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: #111827;
            padding: 0.35rem;
            border-radius: 12px;
            border: 1px solid #1F2937;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.55rem 1.05rem;
            font-weight: 600;
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.16) !important;
            color: #38BDF8 !important;
            border: 1px solid rgba(56, 189, 248, 0.40) !important;
        }

        /* Roster card caption */
        .roster-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 0.55rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_header(device: str, detector_weights: str, reid_variant: str, biometric_mode: bool) -> None:
    dev_label = "GPU / CUDA FP16" if device == "cuda" else "CPU Mode"
    dev_cls = "status-pill gpu" if device == "cuda" else "status-pill"
    det_label = Path(detector_weights).stem.upper()
    reid_label = "DINOv2 ViT-B/14" if "vitb14" in reid_variant else "DINOv2 ViT-S/14"
    face_label = "ArcFace buffalo_l" if biometric_mode else "Standard Mode"
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">🎯 Long-Horizon Biometric Multi-Object Tracker</div>
            <div class="hero-subtitle">
                Clothes-invariant 2-Pass AI tracking with Pose Viewpoint Bucketing, ArcFace Facial Lock,
                Depth Occlusion Protection, Single-Person Spotlight, and Zero-Overhead Baseline Benchmarking.
            </div>
            <div class="pill-row">
                <span class="{dev_cls}">● {dev_label}</span>
                <span class="status-pill">{det_label}</span>
                <span class="status-pill">{reid_label}</span>
                <span class="status-pill">{face_label}</span>
                <span class="status-pill">Depth Anything V2</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_custom_css()
    device_now = resolve_device("auto")

    with st.sidebar:
        st.header("Biometric 2-Pass Configuration")

        biometric_mode = st.toggle(
            "Clothes-Invariant Biometric Mode",
            value=True,
            help=(
                "ON: 0% weight on clothing color. Uses ArcFace + Head/Hair above shoulders + "
                "Grayscale Build + Skeletal Body Ratios + Lab (a*, b*) Skin Chroma.\n"
                "OFF: Standard appearance + 2-zone clothing fusion."
            ),
        )

        if biometric_mode:
            detector_choice = st.selectbox(
                "Pose & Person Detector (imgsz=960)",
                options=[
                    ("yolo11l-pose.pt — YOLO11-Large Pose (Default, 17 Keypoints)", "yolo11l-pose.pt"),
                    ("yolo11x-pose.pt — YOLO11-XLarge Pose (Ultra-Accurate)", "yolo11x-pose.pt"),
                ],
                index=0,
                format_func=lambda item: item[0],
            )
            detector_weights = detector_choice[1]
            people_only = True
        else:
            detector_choice = st.selectbox(
                "Detector Model (imgsz=960)",
                options=[
                    ("yolo11l.pt — YOLO11-Large (Default)", "yolo11l.pt"),
                    ("yolo11x.pt — YOLO11-XLarge (Ultra-Accurate)", "yolo11x.pt"),
                    ("yolov8n.pt — YOLOv8-Nano (Fast)", "yolov8n.pt"),
                ],
                index=0,
                format_func=lambda item: item[0],
            )
            detector_weights = detector_choice[1]
            target_mode = st.radio(
                "Tracking Target",
                options=["People Only (Most Accurate)", "All Objects"],
                index=0,
            )
            people_only = target_mode == "People Only (Most Accurate)"

        reid_choice = st.selectbox(
            "Head / Shape DINOv2 Backbone",
            options=[
                ("dinov2_vitb14 — DINOv2 ViT-B/14 (Default, Max Accuracy)", "dinov2_vitb14"),
                ("dinov2_vits14 — DINOv2 ViT-S/14 (Faster Fallback)", "dinov2_vits14"),
            ],
            index=0,
            format_func=lambda item: item[0],
        )
        reid_variant = reid_choice[1]

        depth_backend = st.selectbox(
            "Depth Occlusion Model",
            options=["depth_anything", "midas_small", "y_prior"],
            index=0,
            help="depth_anything loads depth-anything/Depth-Anything-V2-Small-hf.",
        )

        ghost_preset = st.selectbox(
            "Ghost Memory Duration",
            options=[
                ("60s / 1800 frames (Default)", 1800),
                ("40s / 1200 frames", 1200),
                ("30s / 900 frames", 900),
                ("90s / 2700 frames", 2700),
            ],
            index=0,
            format_func=lambda item: item[0],
        )
        max_ghost_age = ghost_preset[1]

        show_skeleton = st.toggle(
            "Draw 17-keypoint skeleton overlay",
            value=False,
            help="Draws COCO 17-keypoint skeletal joints and limbs color-coded by persistent #ID.",
        )

        show_verifying = st.toggle(
            "Show unconfirmed verifying boxes & <6f blips",
            value=False,
            help="Off by default so only stable sequential IDs (#1, #2, #3...) with >=6 frames appear.",
        )

        with st.expander("Advanced Speed / Detection", expanded=False):
            depth_stride = st.slider("Depth Model Stride (frames)", min_value=1, max_value=4, value=2)
            face_stride = st.slider(
                "ArcFace Extraction Stride (frames)",
                min_value=1,
                max_value=4,
                value=1,
                help="1 = check visible head crops every frame for maximum face capture.",
            )
            conf_thresh = st.slider(
                "Detection Confidence Threshold",
                min_value=0.20,
                max_value=0.70,
                value=0.35,
                step=0.05,
            )

        st.divider()
        st.caption(
            "**Locked AGENTS.md & 3-Tier Cascade**\n"
            "- Pass-1 Clean-View Gate & Depth Lock (`IoU >= 0.25`)\n"
            "- Pass-1 Probation & Margin (`>= 15f`, `sim >= 0.72`, `margin >= 0.08`)\n"
            "- Pass-2 Tier 0: `HANDOFF_OVERLAP_BRIDGE` + `ROTATION_SPATIAL_BRIDGE`\n"
            "- Pass-2 Tier 1: `BIOMETRIC_FACE_LOCK` (ArcFace `>= 0.40`, 0% clothing)\n"
            "- Pass-2 Tier 2: Angle-aware `BIOMETRIC_HEAD_BODY_SKIN` (`>= 0.66`, 0% clothing)"
        )

    _render_hero_header(device_now, detector_weights, reid_variant, biometric_mode)

    # -----------------------------------------------------------------------
    # Video Upload & Trigger Bar
    # -----------------------------------------------------------------------
    up_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload video (.mp4, .mov, .avi, .mkv)",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=False,
        )
    with btn_col:
        run_label = (
            "🚀 Run Biometric Tracking"
            if biometric_mode
            else "🚀 Run Standard Tracking"
        )
        run_clicked = st.button(
            run_label,
            type="primary",
            disabled=(uploaded_file is None),
            width="stretch",
        )

    if uploaded_file is not None:
        up_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"Ready: `{uploaded_file.name}` ({up_mb:.2f} MB)")

    if run_clicked and uploaded_file is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
        stem = _safe_stem(uploaded_file.name)

        saved_input = UPLOADS_DIR / f"{stem}_{ts}{ext}"
        saved_output = UPLOADS_DIR / f"{stem}_{ts}_tracked_biometric.mp4"
        saved_events = UPLOADS_DIR / f"{stem}_{ts}_events_biometric.jsonl"
        saved_input.write_bytes(uploaded_file.getvalue())

        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        status_placeholder.markdown(f"Saved upload to `{saved_input.relative_to(ROOT)}`. Preparing cached AI models...")

        try:
            job_result = run_two_pass_biometric_job(
                input_path=saved_input,
                output_path=saved_output,
                events_path=saved_events,
                detector_weights=detector_weights,
                people_only=people_only,
                biometric_mode=biometric_mode,
                reid_variant=reid_variant,
                depth_backend=depth_backend,
                max_ghost_age=max_ghost_age,
                show_verifying=show_verifying,
                show_skeleton=show_skeleton,
                depth_stride=depth_stride,
                face_stride=face_stride,
                conf_thresh=conf_thresh,
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
            )
            st.session_state["last_upload_run_v6"] = job_result
            status_placeholder.success(
                f"Tracking complete! Processed **{job_result['frames_processed']}** frames in "
                f"**{job_result['elapsed_sec']}s** (Pass-1: **{job_result['pass1_fps']:.1f} FPS**, "
                f"End-to-End: **{job_result['avg_fps']:.1f} FPS**) — "
                f"**{job_result['confirmed_unique_ids']}** Confirmed People vs "
                f"**{job_result['baseline_unique_ids']}** Standard Baseline IDs."
            )
        except Exception as exc:
            status_placeholder.error(f"Biometric 2-Pass Tracking failed: {exc}")
            raise

    # -----------------------------------------------------------------------
    # 4-Tab Professional Workspace (persisted in session_state — zero Pass-1 reruns!)
    # -----------------------------------------------------------------------
    res: Optional[Dict[str, Any]] = st.session_state.get("last_upload_run_v6")
    if res is None:
        st.info(
            "Upload a `.mp4`, `.mov`, `.avi`, or `.mkv` video above and click **Run Biometric Tracking** to unlock "
            "Full Scene Tracking, Single-Person Spotlight Focus, Standard vs. Long-Horizon Comparison, and the Biometric Recovery Log."
        )
        return

    counters = res.get("counters", {})
    person_stats: Dict[int, Dict[str, Any]] = res.get("person_stats", {})
    person_thumbs: Dict[int, bytes] = res.get("person_thumbnails", {})
    consolidated_timeline: List[FrameSnapshot] = res.get("consolidated_timeline", [])
    baseline_timeline: List[List[Tuple[int, np.ndarray, float]]] = res.get("baseline_timeline", [])
    recoveries: List[Dict[str, Any]] = res.get("recoveries", [])
    source_video_path = Path(res.get("source_video_path", res.get("input_path", "")))
    src_fps = float(res.get("src_fps", 30.0))

    confirmed_ids_count = int(res.get("confirmed_unique_ids", len(person_stats)))
    baseline_ids_count = int(res.get("baseline_unique_ids", confirmed_ids_count))
    pass1_stitches = int(res.get("pass1_stitches", counters.get("stitches", 0)))
    pass2_turns = int(res.get("pass2_turn_bridges", 0))
    pass2_face = int(res.get("pass2_face_locks", 0))
    pass2_bio = int(res.get("pass2_bio_merges", 0))
    total_recoveries = int(res.get("total_recoveries", pass1_stitches + pass2_turns + pass2_face + pass2_bio))
    max_occ_frames = int(res.get("max_occ_frames", 0))
    max_occ_sec = float(res.get("max_occ_sec", 0.0))
    pass1_fps = float(res.get("pass1_fps", res.get("avg_fps", 0.0)))
    avg_fps = float(res.get("avg_fps", 0.0))

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎬 Full Scene Tracking",
            "🔍 Single-Person Focus (Select by ID)",
            "⚖️ Standard vs. Long-Horizon Comparison",
            "🧬 Biometric Recovery Log & Timeline",
        ]
    )

    # =======================================================================
    # TAB 1 — Full Scene Tracking
    # =======================================================================
    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Confirmed People" if res.get("people_only", True) else "Confirmed Objects",
            confirmed_ids_count,
            delta=f"Sequential #1..#{confirmed_ids_count}" if confirmed_ids_count > 0 else "None",
            delta_color="off",
        )
        k2.metric(
            "Total Biometric Recoveries",
            total_recoveries,
            delta=f"Handoff/Turn: {pass2_turns} | Face: {pass2_face} | Bio: {pass2_bio} | P1: {pass1_stitches}",
            delta_color="off",
        )
        k3.metric(
            "Max Occlusion Survived",
            f"{max_occ_frames}f" if max_occ_frames > 0 else "0f",
            delta=f"{max_occ_sec:.2f}s | {res.get('backfilled_frame_boxes', 0)}f back-filled",
            delta_color="off",
        )
        k4.metric(
            "Processing FPS",
            f"{pass1_fps:.1f} / {avg_fps:.1f}",
            delta=f"Pass-1 / Overall ({res.get('frames_processed', 0)}f @ {res.get('resolution', 'N/A')})",
            delta_color="off",
        )

        out_path = Path(res["output_path"])
        if out_path.exists():
            video_bytes = out_path.read_bytes()
            size_mb = len(video_bytes) / (1024 * 1024)
            hdr_l, hdr_r = st.columns([3, 1], vertical_alignment="center")
            hdr_l.subheader("Full-Scene Consolidated Output (H.264 / `yuv420p`)")
            hdr_r.download_button(
                label=f"⬇️ Download Full Tracked MP4 ({size_mb:.1f} MB)",
                data=video_bytes,
                file_name=out_path.name,
                mime="video/mp4",
                width="stretch",
            )
            st.caption(
                f"Stack: `{res.get('detector_weights', 'yolo11l-pose.pt')}@{res.get('imgsz', 960)}` · "
                f"`{res.get('reid_backend', 'dinov2_vitb14')}` · `{res.get('face_backend', 'ArcFace')}` · "
                f"`{res.get('depth_backend', 'depth_anything')}` ({str(res.get('device', 'cpu')).upper()}) · "
                f"Faces Captured: `{res.get('faces_extracted_total', 0)}` · Depth Locks: `{counters.get('depth_occlusions', 0)}`"
            )
            st.video(video_bytes, format="video/mp4")
        else:
            st.warning(f"Tracked output video not found at `{out_path}`.")

        st.subheader("Detected People Roster")
        if person_stats:
            ids_sorted = sorted(person_stats.keys())
            cols_per_row = min(6, max(1, len(ids_sorted)))
            for row_start in range(0, len(ids_sorted), cols_per_row):
                row_ids = ids_sorted[row_start : row_start + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for col_obj, pid in zip(row_cols, row_ids):
                    pinfo = person_stats[pid]
                    with col_obj:
                        if pid in person_thumbs:
                            st.image(person_thumbs[pid], width="stretch")
                        face_badge = "🟢 Face Locked" if pinfo["has_face_lock"] else "🔵 Head+Body"
                        rec_txt = f" · {pinfo['recovery_count']} rec" if pinfo["recovery_count"] > 0 else ""
                        st.markdown(
                            f"**Person #{pid}**  \n"
                            f"`{pinfo['duration_sec']:.1f}s` ({pinfo['frames_visible']}f)  \n"
                            f"{face_badge}{rec_txt}"
                        )
        else:
            st.info("No confirmed people were detected in this clip.")

    # =======================================================================
    # TAB 2 — Single-Person Focus (Select by ID)
    # =======================================================================
    with tab2:
        if not person_stats:
            st.info("No confirmed identities available to isolate.")
        else:
            ids_sorted = sorted(person_stats.keys())

            def _fmt_person(pid: int) -> str:
                ps = person_stats[pid]
                fb = "Face Locked" if ps["has_face_lock"] else "Head+Body+Skin"
                rc = ps["recovery_count"]
                rlabel = f"{rc} Recovery" if rc == 1 else f"{rc} Recoveries"
                return f"Person #{pid} — Seen {ps['duration_sec']:.1f}s | {fb} | {rlabel}"

            sel_col, prev_col = st.columns([3, 1], vertical_alignment="top")
            with sel_col:
                selected_pid = st.selectbox(
                    "Select Target Person by Persistent ID",
                    options=ids_sorted,
                    index=0,
                    format_func=_fmt_person,
                    key="solo_target_pid",
                )
                spotlight_mode = st.selectbox(
                    "Spotlight Mode",
                    options=[
                        "Dim Background (Focus Only Selected ID)",
                        "Clean Solo Box (Normal Background)",
                        "Solo Box + Motion Trail",
                    ],
                    index=0,
                    key="solo_spotlight_mode",
                )
                opt_c1, opt_c2, opt_c3 = st.columns(3)
                draw_trail_opt = opt_c1.toggle("Draw Movement Trajectory Trail", value=True, key="solo_trail")
                show_occ_hud_opt = opt_c2.toggle(
                    'Show "TARGET OCCLUDED / RE-ACQUIRING" HUD',
                    value=True,
                    key="solo_occ_hud",
                )
                solo_skel_opt = opt_c3.toggle("Draw 17-Keypoint Skeleton", value=show_skeleton, key="solo_skel")

                render_solo_clicked = st.button(
                    f"🎯 Render Selected Person Video (Person #{selected_pid})",
                    type="primary",
                    width="stretch",
                )
            with prev_col:
                if selected_pid in person_thumbs:
                    st.image(person_thumbs[selected_pid], caption=f"Person #{selected_pid} Portrait", width="stretch")

            solo_cache_key = f"p{selected_pid}_{spotlight_mode[:3]}_{int(draw_trail_opt)}_{int(show_occ_hud_opt)}_{int(solo_skel_opt)}"
            if render_solo_clicked:
                if not source_video_path.exists():
                    st.error(f"Source video missing at `{source_video_path}`.")
                else:
                    solo_out = UPLOADS_DIR / f"{source_video_path.stem}_solo_person_{selected_pid}_{datetime.now().strftime('%H%M%S')}.mp4"
                    s_prog = st.progress(0.0)
                    s_stat = st.empty()
                    render_single_person_video(
                        source_video_path=source_video_path,
                        output_path=solo_out,
                        consolidated_timeline=consolidated_timeline,
                        target_id=selected_pid,
                        spotlight_mode=spotlight_mode,
                        draw_trail=draw_trail_opt,
                        show_occluded_hud=show_occ_hud_opt,
                        show_skeleton=solo_skel_opt,
                        src_fps=src_fps,
                        progress_bar=s_prog,
                        status_placeholder=s_stat,
                    )
                    res.setdefault("solo_renders", {})[solo_cache_key] = str(solo_out)
                    s_stat.success(f"Person #{selected_pid} spotlight video ready!")

            # Display selected person's individual stats & intervals
            pinfo = person_stats[selected_pid]
            sk1, sk2, sk3, sk4 = st.columns(4)
            sk1.metric("Duration Visible", f"{pinfo['duration_sec']:.1f}s", delta=f"{pinfo['frames_visible']} frames", delta_color="off")
            sk2.metric("Biometric Recoveries", pinfo["recovery_count"], delta=f"{pinfo['occlusion_gaps_count']} occlusion gap(s)", delta_color="off")
            sk3.metric("ArcFace Samples", pinfo["face_samples"], delta="Face Locked" if pinfo["has_face_lock"] else "Head+Body+Skin", delta_color="off")
            bfrags = pinfo["baseline_fragment_ids"]
            sk4.metric("Baseline IDs on Target", len(bfrags), delta=", ".join(f"#{b}" for b in bfrags[:5]) or "N/A", delta_color="off")
            st.caption(f"**Appearance Intervals:** {pinfo['intervals_text']}  |  **Modalities:** {pinfo['modalities']}")

            solo_path_str = res.get("solo_renders", {}).get(solo_cache_key)
            if solo_path_str and Path(solo_path_str).exists():
                sp = Path(solo_path_str)
                s_bytes = sp.read_bytes()
                s_mb = len(s_bytes) / (1024 * 1024)
                sh_l, sh_r = st.columns([3, 1], vertical_alignment="center")
                sh_l.subheader(f"Person #{selected_pid} Spotlight Output (H.264 / `yuv420p`)")
                sh_r.download_button(
                    label=f"⬇️ Download Person #{selected_pid} MP4 ({s_mb:.1f} MB)",
                    data=s_bytes,
                    file_name=sp.name,
                    mime="video/mp4",
                    width="stretch",
                )
                st.video(s_bytes, format="video/mp4")

    # =======================================================================
    # TAB 3 — Standard vs. Long-Horizon Comparison
    # =======================================================================
    with tab3:
        splits_avoided = max(0, baseline_ids_count - confirmed_ids_count)
        frag_red_pct = (
            100.0 * (baseline_ids_count - confirmed_ids_count) / float(max(1, baseline_ids_count))
            if baseline_ids_count > 0
            else 0.0
        )

        ck1, ck2, ck3, ck4, ck5 = st.columns(5)
        ck1.metric("Standard Baseline IDs", baseline_ids_count, delta="Pure IoU (15f coast)", delta_color="off")
        ck2.metric("Long-Horizon Biometric IDs", confirmed_ids_count, delta="3-Tier Biometric AI", delta_color="off")
        ck3.metric("ID Fragmentation Reduction", f"{frag_red_pct:.1f}%", delta="Lower IDs = less splitting", delta_color="off")
        ck4.metric("Baseline Splits Avoided", splits_avoided, delta=f"{baseline_ids_count} → {confirmed_ids_count}", delta_color="off")
        ck5.metric("Longest Occlusion Bridged", f"{max_occ_frames}f", delta=f"{max_occ_sec:.2f}s", delta_color="off")

        st.divider()
        cmp_c1, cmp_c2 = st.columns([2, 2], vertical_alignment="bottom")
        with cmp_c1:
            cmp_scope = st.radio(
                "Comparison Scope",
                options=["Full Scene (All People)", "Selected Person Only"],
                index=0,
                horizontal=True,
                key="cmp_scope_radio",
            )
        focus_cmp_pid: Optional[int] = None
        focus_cmp_base_ids: Optional[Set[int]] = None
        with cmp_c2:
            if cmp_scope == "Selected Person Only" and person_stats:
                focus_cmp_pid = st.selectbox(
                    "Person to Compare (Standard Fragmentation vs Biometric Continuity)",
                    options=sorted(person_stats.keys()),
                    index=0,
                    format_func=lambda pid: (
                        f"Person #{pid} — Standard used {len(person_stats[pid]['baseline_fragment_ids'])} ID(s) vs 1 Biometric ID"
                    ),
                    key="cmp_focus_pid",
                )
                focus_cmp_base_ids = set(person_stats[focus_cmp_pid]["baseline_fragment_ids"])

        render_cmp_clicked = st.button(
            "⚖️ Render Side-by-Side Comparison Video",
            type="primary",
            width="stretch",
        )

        cmp_cache_key = "full_scene" if focus_cmp_pid is None else f"person_{focus_cmp_pid}"
        if render_cmp_clicked:
            if not source_video_path.exists():
                st.error(f"Source video missing at `{source_video_path}`.")
            else:
                cmp_out = UPLOADS_DIR / f"{source_video_path.stem}_comparison_{cmp_cache_key}_{datetime.now().strftime('%H%M%S')}.mp4"
                c_prog = st.progress(0.0)
                c_stat = st.empty()
                render_side_by_side_comparison_video(
                    source_video_path=source_video_path,
                    output_path=cmp_out,
                    baseline_timeline=baseline_timeline,
                    consolidated_timeline=consolidated_timeline,
                    src_fps=src_fps,
                    focus_person_id=focus_cmp_pid,
                    focus_baseline_ids=focus_cmp_base_ids,
                    progress_bar=c_prog,
                    status_placeholder=c_stat,
                )
                res.setdefault("comparison_renders", {})[cmp_cache_key] = str(cmp_out)
                c_stat.success("Side-by-Side Comparison H.264 video ready!")

        cmp_path_str = res.get("comparison_renders", {}).get(cmp_cache_key)
        if cmp_path_str and Path(cmp_path_str).exists():
            cp = Path(cmp_path_str)
            c_bytes = cp.read_bytes()
            c_mb = len(c_bytes) / (1024 * 1024)
            ch_l, ch_r = st.columns([3, 1], vertical_alignment="center")
            ch_l.subheader("Side-by-Side Comparison (Left: Standard Baseline | Right: Long-Horizon Biometric AI)")
            ch_r.download_button(
                label=f"⬇️ Download Comparison MP4 ({c_mb:.1f} MB)",
                data=c_bytes,
                file_name=cp.name,
                mime="video/mp4",
                width="stretch",
            )
            st.video(c_bytes, format="video/mp4")

    # =======================================================================
    # TAB 4 — Biometric Recovery Log & Timeline
    # =======================================================================
    with tab4:
        st.subheader("Biometric & Occlusion Recovery Events")
        if recoveries:
            rec_df = pd.DataFrame(recoveries)
            st.dataframe(rec_df, width="stretch", hide_index=True)
        else:
            st.info(
                "No disappearance, turn-bridge, or outfit-change recoveries were required in this clip "
                "(all confirmed identities stayed continuously tracked or remained ambiguous)."
            )

        st.subheader("Per-ID Presence Timeline & Biometric Modalities")
        if person_stats:
            tl_rows = []
            for pid in sorted(person_stats.keys()):
                ps = person_stats[pid]
                tl_rows.append(
                    {
                        "Person ID": f"#{pid}",
                        "First Frame": ps["first_frame"],
                        "Last Frame": ps["last_frame"],
                        "Frames Tracked": ps["frames_visible"],
                        "Duration (s)": ps["duration_sec"],
                        "Occlusion Gaps": ps["occlusion_gaps_count"],
                        "Max Gap (f)": ps["max_gap_frames"],
                        "Recoveries": ps["recovery_count"],
                        "Baseline IDs Used": len(ps["baseline_fragment_ids"]),
                        "Biometric Modalities": ps["modalities"],
                        "Appearance Intervals": ps["intervals_text"],
                    }
                )
            st.dataframe(pd.DataFrame(tl_rows), width="stretch", hide_index=True)
        else:
            st.info("No confirmed tracks to display.")


if __name__ == "__main__":
    main()
