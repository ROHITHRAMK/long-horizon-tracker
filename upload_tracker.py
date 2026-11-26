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
import base64
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
import html as _html
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import streamlit as st

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    go = None  # type: ignore[assignment]
    PLOTLY_AVAILABLE = False

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
from modules import (
    BiometricVault,
    CLIPTargetSearcher,
    DEPTH_CATALOG,
    DETECTOR_CATALOG,
    HeatmapGenerator,
    MaskCleaner,
    NextGenDepthEstimator,
    NextGenPoseDetector,
    NextGenReIDEncoder,
    PolygonZone,
    PoseBehaviorAnalyzer,
    REID_CATALOG,
    ZONE_PRESETS,
    ZoneAnalyzer,
    apply_privacy_blur,
    draw_zones_overlay,
    export_to_tensorrt,
    format_polygon_coords,
    generate_forensic_csv,
    generate_forensic_pdf,
    load_clip_searcher,
    load_mask_cleaner,
    load_nextgen_depth_estimator,
    load_nextgen_detector,
    load_nextgen_reid_encoder,
    parse_polygon_coords,
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
        self._use_half = self.device == "cuda"
        if self._use_half:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes_xyxy [N,4], scores [N], class_ids [N], keypoints_xyc [N,17,3])."""
        try:
            results = self.model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=self.classes,
                device=self.device,
                half=self._use_half,
                verbose=False,
            )
        except Exception:
            if self._use_half:
                self._use_half = False
                results = self.model.predict(
                    frame_bgr,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    classes=self.classes,
                    device=self.device,
                    half=False,
                    verbose=False,
                )
            else:
                raise
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

    def _forward_chunked(self, batch: Any, chunk_size: int = 128) -> Any:
        """Run a single batched FP16 forward pass on GPU (chunking only when batch > chunk_size)."""
        torch = self._torch
        use_cuda = self.device == "cuda"
        n = batch.shape[0]
        if n <= chunk_size:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.float().reshape(feats.shape[0], -1)
            return torch.nn.functional.normalize(feats, p=2, dim=1)

        parts = []
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
            all_feats = self._forward_chunked(all_crops, chunk_size=128)

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
                    torch.cat([full_crops, flip_crops, torso_crops], dim=0), chunk_size=128
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


@st.cache_resource(show_spinner=False)
def get_biometric_vault(db_path: str = "data/biometric_vault.db") -> BiometricVault:
    """Return a cached SQLite BiometricVault instance."""
    return BiometricVault(db_path=db_path)


def _resolve_tensorrt_or_pt(weights_name: str) -> Tuple[str, bool]:
    """Auto-select `<stem>.engine` if a TensorRT engine already exists next to `.pt` weights."""
    w = str(weights_name)
    if w.lower().endswith(".pt"):
        eng = Path(w).with_suffix(".engine")
        if eng.exists():
            return str(eng), True
        eng_root = ROOT / eng.name
        if eng_root.exists():
            return str(eng_root), True
    return w, w.lower().endswith(".engine")


FAST_H264_OUTPUT_PARAMS: List[str] = [
    "-preset",
    "veryfast",
    "-crf",
    "22",
    "-movflags",
    "+faststart",
    "-pix_fmt",
    "yuv420p",
]


class FastH264VideoWriter:
    """Single-pass asynchronous H.264 writer piping BGR24 frames straight into `imageio-ffmpeg`.

    Eliminates the slow two-stage `mp4v` intermediate file + post-hoc ffmpeg transcode and uses
    `-preset veryfast -crf 22 -movflags +faststart -pix_fmt yuv420p` for 2x-3x faster browser-ready
    MP4 export. Falls back seamlessly to `AsyncVideoWriter` if `imageio-ffmpeg` pipe init fails.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        max_queue: int = 32,
    ) -> None:
        self.output_path = str(output_path)
        self.fps = max(1.0, float(fps))
        self.width = int(frame_size[0])
        self.height = int(frame_size[1])
        self._fallback: Optional[AsyncVideoWriter] = None
        self._queue: Optional[queue.Queue[Optional[np.ndarray]]] = None
        self._worker: Optional[threading.Thread] = None
        self._error: List[BaseException] = []
        self._closed = False

        try:
            import imageio_ffmpeg

            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            gen = imageio_ffmpeg.write_frames(
                self.output_path,
                size=(self.width, self.height),
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                fps=self.fps,
                codec="libx264",
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=list(FAST_H264_OUTPUT_PARAMS),
            )
            gen.send(None)
            self._gen = gen
            self._queue = queue.Queue(maxsize=max(4, int(max_queue)))
            self._worker = threading.Thread(target=self._run, name="FastH264Writer", daemon=True)
            self._worker.start()
        except Exception:
            self._gen = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._fallback = AsyncVideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.width, self.height),
                max_queue=max_queue,
                transcode_h264=True,
            )

    def _run(self) -> None:
        assert self._queue is not None and self._gen is not None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._gen.send(item)
        except BaseException as exc:  # pragma: no cover
            self._error.append(exc)
        finally:
            try:
                self._gen.close()
            except BaseException as exc:  # pragma: no cover
                if not self._error:
                    self._error.append(exc)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._fallback is not None:
            self._fallback.write(frame_bgr)
            return
        if self._error:
            raise self._error[0]
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        assert self._queue is not None
        self._queue.put(frame_bgr)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fallback is not None:
            self._fallback.close()
            return
        if self._queue is not None:
            self._queue.put(None)
        if self._worker is not None:
            self._worker.join()
        if self._error:
            raise self._error[0]


def _compute_pairwise_max_ious(boxes: np.ndarray) -> Tuple[np.ndarray, float]:
    """Vectorized per-box max IoU with any OTHER detection on the frame + global max pair IoU."""
    n = len(boxes)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32), 0.0
    b = np.asarray(boxes, dtype=np.float32).reshape(n, 4)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    areas = np.maximum(1.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    union = np.maximum(1e-6, areas[:, None] + areas[None, :] - inter)
    iou_mat = inter / union
    np.fill_diagonal(iou_mat, 0.0)
    per_box_max = iou_mat.max(axis=1).astype(np.float32)
    return per_box_max, float(per_box_max.max())


def _should_run_depth_frame(
    frame_idx: int,
    last_depth_map: Optional[np.ndarray],
    max_pair_iou: float,
    depth_stride: int,
    turbo_gpu: bool,
) -> bool:
    """Smart Depth Gating: run Depth Anything V2 on overlaps (IoU >= 0.15) or every 4th frame in Turbo mode."""
    if last_depth_map is None:
        return True
    if turbo_gpu:
        return (max_pair_iou >= 0.15) or ((frame_idx - 1) % 4 == 0)
    return (frame_idx - 1) % max(1, int(depth_stride)) == 0


def _should_run_arcface_for_det(
    frame_idx: int,
    pre_viewpoint: str,
    matched_face_count: int,
    face_stride: int,
    turbo_gpu: bool,
) -> bool:
    """Pose-Gated ArcFace: skip BACK views; run every frame until >=4 face samples, then every 5th frame."""
    if not turbo_gpu:
        return (frame_idx - 1) % max(1, int(face_stride)) == 0
    if pre_viewpoint == "BACK":
        return False
    if matched_face_count >= 4:
        return (frame_idx - 1) % 5 == 0
    return True


def _should_run_dino_for_det(
    frame_idx: int,
    max_det_iou: float,
    matched_prev_iou: float,
    matched_is_occluded: bool,
    matched_gallery_count: int,
    turbo_gpu: bool,
) -> bool:
    """Adaptive DINOv2 Stride: skip 2/3 frames for clean, non-occluded, isolated tracks with >=5 gallery samples."""
    if not turbo_gpu:
        return True
    if (
        max_det_iou < 0.15
        and matched_prev_iou >= 0.60
        and (not matched_is_occluded)
        and matched_gallery_count >= 5
    ):
        return (frame_idx - 1) % 3 == 0
    return True


def _match_det_to_prev_cache(
    box: np.ndarray,
    prev_active_cache: Dict[int, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Find best IoU match for a current detection against previous frame's active track cache."""
    best_entry: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for entry in prev_active_cache.values():
        ov = _box_iou_single(box, entry["box"])
        if ov > best_iou:
            best_iou = ov
            best_entry = entry
    return best_entry, best_iou


def _close_writer_with_keepalive(
    writer: Any,
    progress_bar: Any,
    status_placeholder: Any,
) -> None:
    """Finalize H.264 video writer while keeping the Streamlit WebSocket alive."""
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
        vault_names: Optional[Dict[int, str]] = None,
        behavior_labels: Optional[Dict[int, str]] = None,
        privacy_mode: str = "Off",
        privacy_style: str = "Gaussian Blur",
    ) -> np.ndarray:
        if (privacy_mode or "Off").strip().lower() == "anonymize all faces":
            canvas = apply_privacy_blur(
                frame_bgr,
                snap.active,
                mode="Anonymize All Faces",
                style=privacy_style,
            )
        else:
            canvas = frame_bgr

        vnames = vault_names or {}
        blabels = behavior_labels or {}

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
            v_lbl = f" {vnames[tr.track_id]}" if tr.track_id in vnames else ""
            beh_state = blabels.get(tr.track_id, "")
            beh_tag = f" [{beh_state}]" if beh_state in ("FALL_ALERT", "HANDS_RAISED", "RUNNING") else ""
            header = f"#{tr.track_id}{v_lbl}{tag}{face_icon}{beh_tag} | d:{tr.depth_ema:.2f} | V:{tr.vault_len}/{tr.vault_max}"
            draw_label_pill(canvas, header, (x1, max(20, y1)), box_color, (15, 15, 15), font_scale=0.46, thickness=1)

            if beh_state == "FALL_ALERT":
                draw_label_pill(canvas, "FALL ALERT!", (x1, max(38, y1 - 22)), (35, 35, 235), (255, 255, 255), font_scale=0.50, thickness=2)
            elif beh_state in ("HANDS_RAISED", "RUNNING"):
                draw_label_pill(canvas, beh_state.replace("_", " "), (x1, max(38, y1 - 22)), (20, 130, 240), (255, 255, 255), font_scale=0.44, thickness=1)

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
    detector_imgsz: int = 768,
    stepper_placeholder: Optional[Any] = None,
    turbo_gpu: bool = True,
    mask_clean_biometrics: bool = False,
    match_vault: bool = True,
    enable_pose_behavior: bool = True,
    privacy_mode: str = "Off",
    privacy_style: str = "Gaussian Blur",
    camera_label: str = "Uploaded Video",
) -> Dict[str, Any]:
    """Execute Pass 1 (Pose + ArcFace + Head/Shape/Skin + LongHorizonTracker) + Pass 2 (Hierarchical Biometric Consolidation + H.264)."""
    device = resolve_device("auto")
    torch_mod: Optional[Any] = None
    if device == "cuda":
        try:
            import torch as _torch_mod

            torch_mod = _torch_mod
            if _torch_mod.cuda.is_available():
                _torch_mod.backends.cudnn.benchmark = True
        except Exception:
            torch_mod = None
    else:
        try:
            import torch as _torch_mod

            torch_mod = _torch_mod
        except Exception:
            torch_mod = None

    if stepper_placeholder is not None:
        stepper_placeholder.markdown(_render_pipeline_stepper(1), unsafe_allow_html=True)

    resolved_det_weights, using_trt = _resolve_tensorrt_or_pt(detector_weights)
    model_notes: List[str] = []

    status_placeholder.markdown(f"**Loading cached Pose/Person Detector (`{resolved_det_weights}` on `{device.upper()}`)...**")
    progress_bar.progress(0.01)
    use_nextgen_det = (
        "yolo26" in resolved_det_weights.lower()
        or "rfdetr" in resolved_det_weights.lower()
        or using_trt
    )
    if use_nextgen_det:
        detector: Any = load_nextgen_detector(model_name=resolved_det_weights, device=device)
        if getattr(detector, "status_note", ""):
            model_notes.append(detector.status_note)
    else:
        detector = load_pose_detector(detector_weights=resolved_det_weights, device=device)

    status_placeholder.markdown(f"**Loading cached Re-ID Backbone (`{reid_variant}` on `{device.upper()}`)...**")
    progress_bar.progress(0.02)
    use_nextgen_reid = ("dinov3" in reid_variant.lower()) or ("/" in reid_variant)
    if use_nextgen_reid:
        encoder: Any = load_nextgen_reid_encoder(variant=reid_variant, device=device)
        if getattr(encoder, "status_note", ""):
            model_notes.append(encoder.status_note)
    else:
        encoder = load_reid_encoder(reid_variant=reid_variant, device=device)

    status_placeholder.markdown(f"**Loading cached Depth Guard (`{depth_backend}` on `{device.upper()}`)...**")
    progress_bar.progress(0.03)
    use_nextgen_depth = depth_backend not in ("depth_anything", "midas_small", "y_prior")
    if use_nextgen_depth:
        depth_est: Any = load_nextgen_depth_estimator(backend=depth_backend, device=device)
        if getattr(depth_est, "status_note", ""):
            model_notes.append(depth_est.status_note)
    else:
        depth_est = load_depth_estimator(depth_backend=depth_backend, device=device)

    mask_cleaner: Optional[MaskCleaner] = None
    if mask_clean_biometrics and biometric_mode:
        status_placeholder.markdown(f"**Loading cached Mask-Clean Segmentation (`yolo26m-seg` on `{device.upper()}`)...**")
        mask_cleaner = load_mask_cleaner(model_name="yolo26m-seg.pt", device=device)
        if getattr(mask_cleaner, "status_note", ""):
            model_notes.append(mask_cleaner.status_note)

    face_analyzer: Optional[ArcFaceBiometricAnalyzer] = None
    if biometric_mode:
        status_placeholder.markdown(f"**Loading cached Facial Biometrics (`buffalo_l` ArcFace on `{device.upper()}`)...**")
        progress_bar.progress(0.04)
        face_analyzer = load_face_analyzer(device=device)

    detector.classes = [0] if (people_only or biometric_mode) else None
    detector.conf = conf_thresh
    detector.imgsz = int(detector_imgsz)

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
    depth_runs_total = 0
    depth_skips_total = 0
    dino_runs_total = 0
    dino_skips_total = 0
    face_skips_total = 0
    prev_active_cache: Dict[int, Dict[str, Any]] = {}
    frame_count = 0
    t_start = time.perf_counter()
    last_ui_tick = t_start
    fps_ema = src_fps

    # =======================================================================
    # PASS 1: Pose + Biometric Feature Extraction + LongHorizonTracker
    # =======================================================================
    infer_ctx = torch_mod.inference_mode() if torch_mod is not None else contextlib.nullcontext()
    use_cuda_amp = device == "cuda" and torch_mod is not None

    with infer_ctx:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            t0 = time.perf_counter()

            boxes, scores, cls_ids, kpts = detector.detect(frame)
            n_det = len(boxes)
            raw_detections_per_frame.append(
                [(boxes[i].copy(), float(scores[i]), int(cls_ids[i])) for i in range(n_det)]
            )

            per_det_max_iou, max_pair_iou = _compute_pairwise_max_ious(boxes)
            run_depth_now = _should_run_depth_frame(
                frame_idx=frame_count,
                last_depth_map=last_depth_map,
                max_pair_iou=max_pair_iou,
                depth_stride=depth_stride,
                turbo_gpu=turbo_gpu,
            )

            # Match current detections to previous frame's active tracks for Smart Keyframe Gating
            det_prev_entry: List[Optional[Dict[str, Any]]] = [None] * n_det
            det_prev_iou: List[float] = [0.0] * n_det
            if turbo_gpu and n_det > 0 and prev_active_cache:
                for i in range(n_det):
                    ent, ov = _match_det_to_prev_cache(boxes[i], prev_active_cache)
                    det_prev_entry[i] = ent
                    det_prev_iou[i] = ov

            # Determine which detections need a fresh DINOv2 forward pass on this frame
            need_dino_indices: List[int] = []
            for i in range(n_det):
                ent = det_prev_entry[i]
                can_reuse = False
                if ent is not None:
                    has_cached = (
                        (ent.get("head_emb") is not None and ent.get("gray_emb") is not None)
                        if biometric_mode
                        else (ent.get("std_emb") is not None)
                    )
                    if has_cached and not _should_run_dino_for_det(
                        frame_idx=frame_count,
                        max_det_iou=float(per_det_max_iou[i]),
                        matched_prev_iou=det_prev_iou[i],
                        matched_is_occluded=bool(ent.get("is_occluded", False)),
                        matched_gallery_count=int(ent.get("gallery_count", 0)),
                        turbo_gpu=turbo_gpu,
                    ):
                        can_reuse = True
                if can_reuse:
                    dino_skips_total += 1
                else:
                    need_dino_indices.append(i)
                    dino_runs_total += 1

            # Upload full frame to GPU only if Depth or at least one DINOv2 crop needs it
            need_gpu_upload = use_gpu_frame and (run_depth_now or len(need_dino_indices) > 0)
            frame_gpu = upload_frame_gpu(frame, device) if need_gpu_upload else None

            cmc_res = None
            if run_depth_now:
                depth_runs_total += 1
                if use_cuda_amp:
                    with torch_mod.amp.autocast("cuda", dtype=torch_mod.float16):
                        last_depth_map = depth_est.predict(frame, frame_gpu=frame_gpu)
                else:
                    last_depth_map = depth_est.predict(frame, frame_gpu=frame_gpu)
            else:
                depth_skips_total += 1
                cmc_res = tracker.cmc.update(frame, boxes)
                last_depth_map = warp_depth_map(last_depth_map, cmc_res.matrix, frame.shape[:2])

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

                # Optional Mask-Clean background neutralization before DINO & skin chroma extraction
                bio_frame = frame
                bio_frame_gpu = frame_gpu
                if mask_cleaner is not None and mask_cleaner.available and need_dino_indices:
                    cleaned_fr, matched_cnt = mask_cleaner.build_mask_cleaned_frame(frame, boxes)
                    if matched_cnt > 0:
                        bio_frame = cleaned_fr
                        bio_frame_gpu = upload_frame_gpu(bio_frame, device) if use_gpu_frame else None

                # Single-Batch DINOv2/DINOv3 forward pass ONLY for detections requiring fresh embeddings
                head_embs_list: List[Optional[np.ndarray]] = [None] * n_det
                gray_embs_list: List[Optional[np.ndarray]] = [None] * n_det
                if need_dino_indices:
                    sub_idx = np.asarray(need_dino_indices, dtype=np.int64)
                    sub_head, sub_gray = encoder.extract_biometric_dino(
                        bio_frame, boxes[sub_idx], head_boxes[sub_idx], frame_gpu=bio_frame_gpu
                    )
                    for k_pos, det_i in enumerate(need_dino_indices):
                        head_embs_list[det_i] = sub_head[k_pos]
                        gray_embs_list[det_i] = sub_gray[k_pos]
                for i in range(n_det):
                    if head_embs_list[i] is None:
                        ent = det_prev_entry[i]
                        assert ent is not None
                        head_embs_list[i] = ent["head_emb"]
                        gray_embs_list[i] = ent["gray_emb"]

                face_ready = face_analyzer is not None and face_analyzer.available
                fused_list: List[np.ndarray] = []
                for i in range(n_det):
                    h_emb_i = head_embs_list[i]
                    g_emb_i = gray_embs_list[i]
                    assert h_emb_i is not None and g_emb_i is not None
                    raw_r, valid_r, body_vec = extract_body_shape_vector(boxes[i], kpts[i], conf_thresh=0.40)
                    skin_vec, med_ab, valid_skin = extract_skin_chroma_vector(bio_frame, boxes[i], kpts[i], conf_thresh=0.40)
                    bio_emb = build_clothes_invariant_embedding(h_emb_i, g_emb_i, body_vec, skin_vec)
                    fused_list.append(bio_emb)

                    det_head_embs[i] = h_emb_i
                    det_gray_embs[i] = g_emb_i
                    det_body_ratios[i] = (raw_r, valid_r)
                    det_skin_obs[i] = (skin_vec, med_ab, valid_skin)

                    pre_vp = classify_viewpoint(kpts[i], face_detected=False)
                    ent_i = det_prev_entry[i]
                    matched_faces = (
                        int(ent_i.get("face_count", 0))
                        if (ent_i is not None and det_prev_iou[i] >= 0.50)
                        else 0
                    )
                    face_found = False
                    if face_ready and face_analyzer is not None:
                        if _should_run_arcface_for_det(
                            frame_idx=frame_count,
                            pre_viewpoint=pre_vp,
                            matched_face_count=matched_faces,
                            face_stride=face_stride,
                            turbo_gpu=turbo_gpu,
                        ):
                            f_res = face_analyzer.extract_from_head_box(frame, head_boxes[i])
                            if f_res is not None:
                                det_face_obs[i] = f_res
                                face_found = True
                                faces_extracted_total += 1
                        else:
                            face_skips_total += 1

                    det_viewpoints[i] = "FRONT" if face_found else pre_vp

                embeddings = np.stack(fused_list, axis=0).astype(np.float32)
            elif n_det > 0:
                if len(need_dino_indices) == n_det:
                    embeddings = encoder.extract_standard(frame, boxes, frame_gpu=frame_gpu)
                else:
                    std_list: List[Optional[np.ndarray]] = [None] * n_det
                    if need_dino_indices:
                        sub_idx = np.asarray(need_dino_indices, dtype=np.int64)
                        sub_embs = encoder.extract_standard(frame, boxes[sub_idx], frame_gpu=frame_gpu)
                        for k_pos, det_i in enumerate(need_dino_indices):
                            std_list[det_i] = sub_embs[k_pos]
                    for i in range(n_det):
                        if std_list[i] is None:
                            ent = det_prev_entry[i]
                            assert ent is not None
                            std_list[i] = ent["std_emb"]
                    embeddings = np.stack([e for e in std_list if e is not None], axis=0).astype(np.float32)
                for i in range(n_det):
                    det_viewpoints[i] = classify_viewpoint(kpts[i], face_detected=False)
            else:
                embeddings = np.zeros((0, 1), dtype=np.float32)

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

            # Snapshot active tracks + update per-track biometric profiles & Smart Gating cache
            active_snaps: List[TrackBoxSnapshot] = []
            next_active_cache: Dict[int, Dict[str, Any]] = {}
            for tr in result.active_tracks:
                track_frames.setdefault(tr.track_id, set()).add(frame_count)
                track_class[tr.track_id] = tr.class_id
                prof = track_profiles.setdefault(tr.track_id, TrackBiometricProfile())

                tr_kpts: Optional[np.ndarray] = None
                cached_head: Optional[np.ndarray] = None
                cached_gray: Optional[np.ndarray] = None
                cached_std: Optional[np.ndarray] = None
                if tr.time_since_update == 0 and n_det > 0:
                    mi = _match_box_to_detection(tr.current_box(), boxes)
                    if mi is not None:
                        tr_kpts = kpts[mi].copy()
                        cached_head = det_head_embs[mi]
                        cached_gray = det_gray_embs[mi]
                        cached_std = embeddings[mi] if not biometric_mode else None
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

                prev_tr_ent = prev_active_cache.get(tr.track_id)
                next_active_cache[tr.track_id] = {
                    "box": tr.current_box().copy(),
                    "head_emb": cached_head if cached_head is not None else (prev_tr_ent["head_emb"] if prev_tr_ent else None),
                    "gray_emb": cached_gray if cached_gray is not None else (prev_tr_ent["gray_emb"] if prev_tr_ent else None),
                    "std_emb": cached_std if cached_std is not None else (prev_tr_ent["std_emb"] if prev_tr_ent else None),
                    "is_occluded": bool(tr.is_occluded or tr.is_depth_occluded or tr.status == "DEPTH-OCCLUDED"),
                    "gallery_count": len(prof.best_fused_gallery),
                    "face_count": len(prof.face_gallery),
                }

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
            prev_active_cache = next_active_cache

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
            if frame_count == 1 and stepper_placeholder is not None:
                stepper_placeholder.markdown(_render_pipeline_stepper(2), unsafe_allow_html=True)
            if frame_count % 5 == 0 or frame_count == 1 or frame_count >= total_frames or (now_tick - last_ui_tick) >= 1.5:
                last_ui_tick = now_tick
                frac = min(0.72, 0.04 + 0.68 * (frame_count / float(total_frames)))
                progress_bar.progress(frac)
                st_counts = result.stats
                face_stat = f" &nbsp;|&nbsp; **Faces Captured:** `{faces_extracted_total}`" if biometric_mode else ""
                turbo_stat = " &nbsp;|&nbsp; **Mode:** `TURBO FP16`" if turbo_gpu else ""
                status_placeholder.markdown(
                    f"**[Pass 1 / 2 — Biometric Tracking] Frame `{frame_count}` / `{total_frames}`** &nbsp;|&nbsp; "
                    f"**Live FPS:** `{fps_ema:.1f}` &nbsp;|&nbsp; "
                    f"**Active:** `{st_counts['active_count']}` &nbsp;|&nbsp; "
                    f"**Ghosts:** `{st_counts['ghost_count']}` &nbsp;|&nbsp; "
                    f"**Pass-1 Stitches:** `{st_counts['stitches']}`{face_stat}{turbo_stat}"
                )

    cap.release()
    t_pass1 = max(1e-3, time.perf_counter() - t_start)

    # =======================================================================
    # ZERO-OVERHEAD STANDARD BASELINE TRACKER (Pure IoU + 15f Coasting)
    # =======================================================================
    if stepper_placeholder is not None:
        stepper_placeholder.markdown(_render_pipeline_stepper(3), unsafe_allow_html=True)
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

    # Zero-cost Pose Behavior & Fall Detection over consolidated timeline
    pose_result: Optional[Dict[str, Any]] = None
    if enable_pose_behavior and consolidated_timeline:
        pose_result = PoseBehaviorAnalyzer(smooth_window=5).analyze_timeline(
            consolidated_timeline, src_fps=src_fps
        )
    pose_frame_labels: Dict[int, Dict[int, str]] = (
        pose_result.get("frame_labels", {}) if pose_result is not None else {}
    )

    # Cross-video Biometric Vault matching (lookup before rendering so named labels appear in Pass-2 video)
    vault_matches: Dict[int, Dict[str, Any]] = {}
    vault_names_for_hud: Dict[int, str] = {}
    if match_vault and biometric_mode and seq_profiles:
        try:
            vault = get_biometric_vault()
            vault_all = {item["vault_id"]: item for item in vault.list_identities(include_sightings=False)}
            for seq_id in sorted(seq_profiles.keys()):
                matched = vault.match_track_to_vault(seq_profiles[seq_id], face_thresh=0.42, bio_thresh=0.70)
                if matched is not None:
                    v_id, v_name, v_conf, v_cue = matched
                    v_rec = vault_all.get(v_id, {})
                    is_wl = bool(v_rec.get("is_watchlist", False))
                    vault_matches[seq_id] = {
                        "vault_id": v_id,
                        "display_name": v_name,
                        "is_watchlist": is_wl,
                        "confidence": v_conf,
                        "cue": v_cue,
                    }
                    if v_name:
                        vault_names_for_hud[seq_id] = f"[{v_name}]"
        except Exception:
            pass

    with open(events_path, "w", encoding="utf-8") as log_file:
        for snap in consolidated_timeline:
            for ev in snap.events:
                log_file.write(json.dumps(ev) + "\n")

    # =======================================================================
    # PASS 2 RENDER: Stream Consolidated Sequential IDs to Even-Dim H.264 + Capture Thumbnails
    # =======================================================================
    if stepper_placeholder is not None:
        stepper_placeholder.markdown(_render_pipeline_stepper(4), unsafe_allow_html=True)
    cap2 = cv2.VideoCapture(str(input_path))
    if not cap2.isOpened():
        raise RuntimeError(f"Could not reopen uploaded video for Pass 2 render: {input_path}")

    even_w, even_h = _even_dims(width, height)
    writer = FastH264VideoWriter(
        str(output_path),
        fps=src_fps,
        frame_size=(even_w, even_h),
        max_queue=32,
    )
    renderer = Pass2OverlayRenderer(
        probation_frames=tracker_cfg.probation_frames,
        show_verifying=show_verifying,
        show_skeleton=show_skeleton,
    )
    mode_tag = "BIOMETRIC (0% Clothing)" if biometric_mode else "STANDARD"
    face_tag = "ArcFace" if (face_analyzer is not None and face_analyzer.available) else "Head/Body/Skin"
    backend_info = f"{mode_tag} | {Path(getattr(detector, 'model_name', detector_weights)).stem} | {encoder.backbone_name} + {face_tag}"

    person_thumbnails: Dict[int, bytes] = {}
    best_thumb_rank: Dict[int, float] = {}

    rendered_frames = 0
    last_render_tick = time.perf_counter()
    for idx, snap in enumerate(consolidated_timeline, start=1):
        ok, frame = cap2.read()
        if not ok:
            break

        # Capture best unoccluded high-confidence crop per sequential ID BEFORE drawing boxes/blur
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
            vault_names=vault_names_for_hud,
            behavior_labels=pose_frame_labels.get(snap.frame_idx, {}),
            privacy_mode=privacy_mode,
            privacy_style=privacy_style,
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

    # Persist matched/new confirmed sightings into the SQLite Biometric Vault (with captured portrait thumbnails)
    if match_vault and biometric_mode and seq_profiles:
        try:
            vault = get_biometric_vault()
            vid_label = input_path.name
            for seq_id in sorted(seq_profiles.keys()):
                ps = person_stats.get(seq_id, {})
                f_sec = round(max(0, int(ps.get("first_frame", 1)) - 1) / max(1.0, src_fps), 2)
                l_sec = round(max(0, int(ps.get("last_frame", 1)) - 1) / max(1.0, src_fps), 2)
                existing_m = vault_matches.get(seq_id)
                vid_out = vault.upsert_track_identity(
                    track_biometrics=seq_profiles[seq_id],
                    vault_id=existing_m["vault_id"] if existing_m else None,
                    video_name=vid_label,
                    camera_label=camera_label,
                    first_sec=f_sec,
                    last_sec=l_sec,
                    thumbnail_jpeg=person_thumbnails.get(seq_id),
                )
                if existing_m is None:
                    vault_matches[seq_id] = {
                        "vault_id": vid_out,
                        "display_name": None,
                        "is_watchlist": False,
                        "confidence": 1.0,
                        "cue": "NEW_VAULT_PROFILE",
                    }
        except Exception:
            pass

    progress_bar.progress(1.0)
    if stepper_placeholder is not None:
        stepper_placeholder.markdown(_render_pipeline_stepper(5), unsafe_allow_html=True)

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
        "detector_weights": getattr(detector, "model_name", detector_weights),
        "imgsz": detector.imgsz,
        "reid_backend": encoder.backbone_name,
        "face_backend": face_analyzer.provider_label if face_analyzer is not None else "Disabled",
        "depth_backend": depth_est.backend,
        "model_notes": model_notes,
        "people_only": people_only or biometric_mode,
        "faces_extracted_total": faces_extracted_total,
        "turbo_gpu": bool(turbo_gpu),
        "depth_runs": depth_runs_total,
        "depth_skips": depth_skips_total,
        "dino_runs": dino_runs_total,
        "dino_skips": dino_skips_total,
        "face_skips": face_skips_total,
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
        # Session-cached Pass-1/Pass-2 state for zero-rerun Tabs 1-6
        "consolidated_timeline": consolidated_timeline,
        "baseline_timeline": baseline_timeline,
        "person_thumbnails": person_thumbnails,
        "person_stats": person_stats,
        "seq_profiles": seq_profiles,
        "vault_matches": vault_matches,
        "pose_result": pose_result,
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
            "angles": angles,
            "angle_code": "/".join(a[0] for a in angles) if angles else "—",
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
    privacy_mode: str = "Off",
    privacy_style: str = "Gaussian Blur",
    target_display_name: Optional[str] = None,
) -> Path:
    """Render an on-demand H.264 solo focus video highlighting ONLY `target_id` (zero AI re-run)."""
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for Solo render: {source_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    even_w, even_h = _even_dims(width, height)

    writer = FastH264VideoWriter(
        str(output_path),
        fps=src_fps,
        frame_size=(even_w, even_h),
        max_queue=32,
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

        # Apply optional selective privacy anonymization before dimming/HUD overlays
        base_frame = (
            apply_privacy_blur(
                frame,
                snap.active,
                mode=privacy_mode,
                style=privacy_style,
                target_id=target_id,
            )
            if (privacy_mode or "Off") != "Off"
            else frame
        )

        if dim_bg:
            # Darken everything outside the selected person's bounding box by 55%
            canvas = (base_frame.astype(np.float32) * 0.45).astype(np.uint8)
            if target_snap is not None:
                fh, fw = base_frame.shape[:2]
                bw = max(1.0, float(target_snap.box[2] - target_snap.box[0]))
                bh = max(1.0, float(target_snap.box[3] - target_snap.box[1]))
                pad_x, pad_y = int(round(0.08 * bw)), int(round(0.06 * bh))
                rx1 = max(0, int(round(float(target_snap.box[0]))) - pad_x)
                ry1 = max(0, int(round(float(target_snap.box[1]))) - pad_y)
                rx2 = min(fw, int(round(float(target_snap.box[2]))) + pad_x)
                ry2 = min(fh, int(round(float(target_snap.box[3]))) + pad_y)
                if rx2 > rx1 and ry2 > ry1:
                    canvas[ry1:ry2, rx1:rx2] = base_frame[ry1:ry2, rx1:rx2]
        else:
            canvas = base_frame.copy()

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
            name_tag = f" ({target_display_name})" if target_display_name else ""
            header = f"TARGET PERSON #{target_id}{name_tag}{face_tag}"
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
    writer = FastH264VideoWriter(
        str(output_path),
        fps=src_fps,
        frame_size=(out_w, out_h),
        max_queue=32,
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
# Enterprise Command-Center UI, Stepper, Dossier Cards & Interactive Plotly Charts
# ---------------------------------------------------------------------------


def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Preserve sidebar toggle while hiding deploy/footer clutter */
        .stAppDeployButton, footer, #MainMenu {visibility: hidden !important; display: none !important;}

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Deep command-center gradient mesh background */
        .stApp {
            background-color: #060913 !important;
            background-image:
                radial-gradient(ellipse 75% 42% at 22% -6%, rgba(6, 182, 212, 0.14), transparent 62%),
                radial-gradient(ellipse 65% 38% at 84% -8%, rgba(99, 102, 241, 0.12), transparent 60%);
            background-attachment: fixed;
        }
        section[data-testid="stSidebar"] {
            background-color: #0D1323 !important;
            border-right: 1px solid rgba(56, 189, 248, 0.14);
        }
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
        }

        /* Pulsing online dot animation */
        @keyframes pulseDot {
            0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.75); }
            70%  { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0.0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.0); }
        }
        @keyframes pulseStep {
            0%, 100% { box-shadow: 0 0 12px rgba(6, 182, 212, 0.35); }
            50%      { box-shadow: 0 0 22px rgba(6, 182, 212, 0.65); }
        }

        /* Top Enterprise Command Header */
        .cmd-header {
            background: linear-gradient(135deg, rgba(13, 19, 35, 0.92) 0%, rgba(15, 23, 42, 0.84) 60%, rgba(8, 24, 40, 0.88) 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-top: 2px solid #06B6D4;
            border-radius: 16px;
            padding: 1.25rem 1.55rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.45);
        }
        .cmd-top-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.4rem;
        }
        .cmd-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .cmd-subtitle {
            font-size: 0.88rem;
            color: #94A3B8;
            margin: 0 0 0.85rem 0;
        }
        .online-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 600;
            color: #34D399;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.40);
            padding: 0.30rem 0.80rem;
            border-radius: 999px;
            letter-spacing: 0.04em;
        }
        .online-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10B981;
            animation: pulseDot 1.8s infinite;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .model-badge {
            display: inline-flex;
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            font-weight: 500;
            color: #CBD5E1;
            background: rgba(6, 182, 212, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.28);
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
        }

        /* Glassmorphic KPI metric cards with 2px gradient top accent */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72) !important;
            border: 1px solid rgba(56, 189, 248, 0.18) !important;
            border-top: 2px solid #06B6D4 !important;
            border-radius: 14px !important;
            padding: 0.95rem 1.15rem !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        }
        [data-testid="stMetricLabel"] p {
            color: #94A3B8 !important;
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: #F8FAFC !important;
            font-weight: 700 !important;
            font-size: 1.65rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem !important;
        }

        /* Sleek Upload Dropzone */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(13, 19, 35, 0.72) !important;
            border: 1.5px dashed rgba(6, 182, 212, 0.42) !important;
            border-radius: 14px !important;
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #06B6D4 !important;
            background: rgba(6, 182, 212, 0.07) !important;
        }

        /* Navigation Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(13, 19, 35, 0.85);
            padding: 0.35rem;
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.16);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.55rem 1.05rem;
            font-weight: 600;
            font-size: 0.86rem;
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.22), rgba(99, 102, 241, 0.18)) !important;
            color: #22D3EE !important;
            border: 1px solid rgba(6, 182, 212, 0.45) !important;
        }

        /* 4-Stage Pipeline Stepper */
        .stepper-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin: 0.75rem 0 0.65rem 0;
        }
        .step-box {
            border-radius: 11px;
            padding: 0.60rem 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.65);
            color: #64748B;
        }
        .step-box.done {
            border-color: rgba(16, 185, 129, 0.55);
            background: rgba(16, 185, 129, 0.12);
            color: #34D399;
        }
        .step-box.active {
            border-color: #06B6D4;
            background: rgba(6, 182, 212, 0.16);
            color: #22D3EE;
            animation: pulseStep 1.6s infinite;
        }

        /* Dark Cinema Video Bezel Header */
        .cinema-bar {
            background: linear-gradient(90deg, #0D1323 0%, #111C33 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-bottom: none;
            border-radius: 12px 12px 0 0;
            padding: 0.55rem 0.95rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 0.35rem;
        }
        .cinema-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.80rem;
            font-weight: 600;
            color: #E2E8F0;
            letter-spacing: 0.03em;
        }
        .cinema-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #38BDF8;
        }

        /* Biometric Dossier Card + Targeting Reticle Frame */
        .dossier-card {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(56, 189, 248, 0.20);
            border-top: 2px solid #06B6D4;
            border-radius: 14px;
            padding: 0.75rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.30);
        }
        .reticle-wrap {
            position: relative;
            width: 100%;
            height: 165px;
            border-radius: 10px;
            overflow: hidden;
            background: #060913;
            border: 1px solid rgba(6, 182, 212, 0.30);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.60rem;
        }
        .reticle-wrap img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .reticle-corner {
            position: absolute;
            width: 12px;
            height: 12px;
            border-color: #06B6D4;
            border-style: solid;
            pointer-events: none;
        }
        .rc-tl { top: 6px; left: 6px; border-width: 2px 0 0 2px; }
        .rc-tr { top: 6px; right: 6px; border-width: 2px 2px 0 0; }
        .rc-bl { bottom: 6px; left: 6px; border-width: 0 0 2px 2px; }
        .rc-br { bottom: 6px; right: 6px; border-width: 0 2px 2px 0; }

        .dossier-id {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.02rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 0.15rem;
        }
        .dossier-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: #94A3B8;
            margin-bottom: 0.45rem;
        }
        .mini-badge {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.66rem;
            font-weight: 600;
            padding: 0.15rem 0.45rem;
            border-radius: 6px;
            margin-right: 0.25rem;
            margin-bottom: 0.20rem;
        }
        .mb-face { color: #34D399; background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.40); }
        .mb-angle { color: #38BDF8; background: rgba(6, 182, 212, 0.14); border: 1px solid rgba(6, 182, 212, 0.38); }
        .mb-rec { color: #FBBF24; background: rgba(245, 158, 11, 0.14); border: 1px solid rgba(245, 158, 11, 0.40); }

        /* Enlarged Solo Dossier Header Card */
        .solo-dossier {
            display: flex;
            gap: 1rem;
            align-items: center;
            background: rgba(15, 23, 42, 0.80);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-left: 3px solid #06B6D4;
            border-radius: 14px;
            padding: 0.85rem 1.05rem;
            margin-bottom: 0.85rem;
        }
        .solo-dossier-img {
            width: 96px;
            height: 120px;
            border-radius: 10px;
            object-fit: cover;
            border: 1px solid rgba(6, 182, 212, 0.45);
            background: #060913;
            flex-shrink: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_command_header(
    device: str,
    detector_weights: str,
    reid_variant: str,
    biometric_mode: bool,
    detector_imgsz: int,
    depth_backend: str = "depth_anything",
) -> None:
    dev_str = "SYSTEM ONLINE · CUDA" if device == "cuda" else "SYSTEM ONLINE · CPU"
    resolved_det, is_trt = _resolve_tensorrt_or_pt(detector_weights)
    det_label = f"{Path(resolved_det).stem.upper()}{' [TRT]' if is_trt else ''} @{detector_imgsz}"
    if "dinov3" in reid_variant.lower():
        reid_label = "DINOv3 ConvNeXt" if "convnext" in reid_variant.lower() else "DINOv3 ViT-B/16"
    elif "vits14" in reid_variant.lower():
        reid_label = "DINOv2 ViT-S/14"
    else:
        reid_label = "DINOv2 ViT-B/14"
    face_label = "ArcFace buffalo_l" if biometric_mode else "Standard Re-ID"
    if "da3" in depth_backend.lower():
        depth_label = "Depth V3"
    elif "yolo26" in depth_backend.lower():
        depth_label = "YOLO26-Depth"
    elif depth_backend == "midas_small":
        depth_label = "MiDaS"
    elif depth_backend == "y_prior":
        depth_label = "Y-Prior"
    else:
        depth_label = "Depth V2"

    st.markdown(
        f"""
        <div class="cmd-header">
            <div class="cmd-top-row">
                <div class="cmd-title">🎯 LONG-HORIZON BIOMETRIC MOT // COMMAND CENTER</div>
                <div class="online-pill"><span class="online-dot"></span>{dev_str}</div>
            </div>
            <div class="cmd-subtitle">
                Enterprise 2-Pass Clothes-Invariant Tracking · Persistent Biometric Vault ·
                Spatial Zones & Heatmaps · Pose Anomaly Alerts · CLIP Search · Live CCTV & Forensics
            </div>
            <div class="badge-row">
                <span class="model-badge">{_html.escape(det_label)}</span>
                <span class="model-badge">{_html.escape(reid_label)}</span>
                <span class="model-badge">{_html.escape(face_label)}</span>
                <span class="model-badge">{_html.escape(depth_label)}</span>
                <span class="model-badge">3-Tier Pass-2 Cascade</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_stepper(active_stage: int) -> str:
    """Return HTML for the 4-stage live execution stepper (1..4 active, 5=all done)."""
    stages = [
        ("[01]", "YOLO11-Pose Detection"),
        ("[02]", "Multi-Angle Biometrics"),
        ("[03]", "Pass-2 Consolidation"),
        ("[04]", "H.264 Export"),
    ]
    blocks = []
    for idx, (code, label) in enumerate(stages, start=1):
        if active_stage > idx:
            cls, icon = "step-box done", "✓"
        elif active_stage == idx:
            cls, icon = "step-box active", "◉"
        else:
            cls, icon = "step-box", "○"
        blocks.append(f'<div class="{cls}"><span>{code} {label}</span><span>{icon}</span></div>')
    return f'<div class="stepper-grid">{"".join(blocks)}</div>'


def _render_cinema_bar(title: str, meta: str) -> None:
    st.markdown(
        f'<div class="cinema-bar"><span class="cinema-title">{_html.escape(title)}</span>'
        f'<span class="cinema-meta">{_html.escape(meta)}</span></div>',
        unsafe_allow_html=True,
    )


def _thumb_to_data_uri(jpg_bytes: Optional[bytes]) -> str:
    if not jpg_bytes:
        return ""
    b64 = base64.b64encode(jpg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _render_dossier_card_html(
    pid: int,
    pinfo: Dict[str, Any],
    thumb_bytes: Optional[bytes],
    vault_info: Optional[Dict[str, Any]] = None,
    clip_score_pct: Optional[float] = None,
    behavior_summary: Optional[Dict[str, Any]] = None,
) -> str:
    uri = _thumb_to_data_uri(thumb_bytes)
    img_tag = f'<img src="{uri}" alt="Person #{pid}" />' if uri else '<span class="mono" style="color:#475569;">NO CROP</span>'
    badges = []
    if clip_score_pct is not None:
        badges.append(f'<span class="mini-badge" style="background:rgba(16,185,129,0.22);color:#34D399;border:1px solid rgba(52,211,153,0.45);">CLIP {clip_score_pct:.0f}%</span>')
    if vault_info and vault_info.get("is_watchlist"):
        badges.append('<span class="mini-badge" style="background:rgba(244,63,94,0.22);color:#FB7185;border:1px solid rgba(251,113,133,0.45);">WATCHLIST</span>')
    if pinfo.get("has_face_lock"):
        badges.append('<span class="mini-badge mb-face">FACE</span>')
    ang_code = pinfo.get("angle_code", "—")
    if ang_code and ang_code != "—":
        badges.append(f'<span class="mini-badge mb-angle">{_html.escape(ang_code)}</span>')
    if int(pinfo.get("recovery_count", 0)) > 0:
        badges.append(f'<span class="mini-badge mb-rec">RECOVERED ×{int(pinfo["recovery_count"])}</span>')
    if behavior_summary and behavior_summary.get("anomalies"):
        for an in behavior_summary["anomalies"][:2]:
            badges.append(f'<span class="mini-badge" style="background:rgba(245,158,11,0.20);color:#FBBF24;border:1px solid rgba(251,191,36,0.40);">{_html.escape(an)}</span>')
    if not badges:
        badges.append('<span class="mini-badge mb-angle">TRACKED</span>')

    vname = vault_info.get("display_name") if vault_info else None
    id_heading = f"ID #{pid:02d} · {_html.escape(vname)}" if vname else f"ID #{pid:02d}"
    hl_style = "border:1.5px solid #34D399;box-shadow:0 0 18px rgba(16,185,129,0.24);" if clip_score_pct is not None else ""

    return (
        f'<div class="dossier-card" style="{hl_style}">'
        f'  <div class="reticle-wrap">'
        f'    <span class="reticle-corner rc-tl"></span><span class="reticle-corner rc-tr"></span>'
        f'    <span class="reticle-corner rc-bl"></span><span class="reticle-corner rc-br"></span>'
        f'    {img_tag}'
        f'  </div>'
        f'  <div class="dossier-id">{id_heading}</div>'
        f'  <div class="dossier-sub">{pinfo["duration_sec"]:.1f}s · {pinfo["frames_visible"]}f · {pinfo["recovery_count"]} rec</div>'
        f'  <div>{"".join(badges)}</div>'
        f'</div>'
    )


def build_cumulative_id_chart(
    baseline_timeline: List[List[Tuple[int, np.ndarray, float]]],
    consolidated_timeline: List[FrameSnapshot],
    src_fps: float,
) -> Optional[Any]:
    """Interactive dark-theme Plotly chart: Cumulative Unique IDs Over Time (Baseline vs Long-Horizon)."""
    if not PLOTLY_AVAILABLE or not consolidated_timeline:
        return None
    fps_safe = max(1.0, float(src_fps))
    n = len(consolidated_timeline)
    step = max(1, n // 300)

    seen_base: Set[int] = set()
    seen_ours: Set[int] = set()
    ts_list: List[float] = []
    base_curve: List[int] = []
    ours_curve: List[int] = []

    for idx, snap in enumerate(consolidated_timeline):
        if idx < len(baseline_timeline):
            for bid, _, _ in baseline_timeline[idx]:
                seen_base.add(bid)
        for tr in snap.active:
            seen_ours.add(tr.track_id)
        if idx % step == 0 or idx == n - 1:
            ts_list.append(round(snap.frame_idx / fps_safe, 2))
            base_curve.append(len(seen_base))
            ours_curve.append(len(seen_ours))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_list,
            y=base_curve,
            mode="lines",
            name="Standard Baseline (Pure IoU)",
            line=dict(color="#F43F5E", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.12)",
            hovertemplate="Time: %{x:.1f}s<br>Standard IDs: <b>%{y}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_list,
            y=ours_curve,
            mode="lines",
            name="Long-Horizon Biometric AI",
            line=dict(color="#06B6D4", width=3.0),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.14)",
            hovertemplate="Time: %{x:.1f}s<br>Biometric AI IDs: <b>%{y}</b><extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Cumulative Unique IDs Over Time (Fragmentation Curve)", font=dict(family="Inter", size=14, color="#E2E8F0")),
        height=310,
        margin=dict(l=46, r=20, t=44, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#CBD5E1", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(family="Inter", size=12)),
        xaxis=dict(title="Time (seconds)", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
        yaxis=dict(title="Cumulative Unique IDs", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
    )
    return fig


def build_fragmentation_gauge(frag_red_pct: float) -> Optional[Any]:
    """Compact dark-theme Plotly gauge showing ID Fragmentation Reduction %."""
    if not PLOTLY_AVAILABLE:
        return None
    val = float(np.clip(frag_red_pct, 0.0, 100.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(suffix="%", font=dict(family="JetBrains Mono", size=34, color="#22D3EE")),
            title=dict(text="ID Fragmentation Reduction", font=dict(family="Inter", size=13, color="#CBD5E1")),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(family="JetBrains Mono", size=10, color="#94A3B8")),
                bar=dict(color="#06B6D4"),
                bgcolor="rgba(15, 23, 42, 0.85)",
                borderwidth=1,
                bordercolor="rgba(56, 189, 248, 0.25)",
                steps=[
                    dict(range=[0, 35], color="rgba(244, 63, 94, 0.16)"),
                    dict(range=[35, 70], color="rgba(245, 158, 11, 0.16)"),
                    dict(range=[70, 100], color="rgba(16, 185, 129, 0.20)"),
                ],
            ),
        )
    )
    fig.update_layout(
        height=310,
        margin=dict(l=20, r=20, t=44, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_gantt_presence_chart(
    person_stats: Dict[int, Dict[str, Any]],
    recoveries: List[Dict[str, Any]],
    src_fps: float,
) -> Optional[Any]:
    """Interactive Plotly Gantt Identity Presence & Occlusion Recovery Timeline."""
    if not PLOTLY_AVAILABLE or not person_stats:
        return None
    fps_safe = max(1.0, float(src_fps))
    ids_sorted = sorted(person_stats.keys())

    fig = go.Figure()
    seg_x: List[Optional[float]] = []
    seg_y: List[Optional[str]] = []
    gap_x: List[Optional[float]] = []
    gap_y: List[Optional[str]] = []

    for pid in ids_sorted:
        label = f"Person #{pid:02d}"
        intervals = person_stats[pid].get("intervals", [])
        for k, (sf, ef) in enumerate(intervals):
            t0 = sf / fps_safe
            t1 = max(t0 + 0.08, ef / fps_safe)
            seg_x.extend([t0, t1, None])
            seg_y.extend([label, label, None])
            if k > 0:
                prev_end = intervals[k - 1][1] / fps_safe
                gap_x.extend([prev_end, t0, None])
                gap_y.extend([label, label, None])

    if seg_x:
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines",
                name="Active Visibility",
                line=dict(color="#06B6D4", width=9),
                hovertemplate="%{y}<br>Visible @ %{x:.2f}s<extra></extra>",
            )
        )
    if gap_x:
        fig.add_trace(
            go.Scatter(
                x=gap_x,
                y=gap_y,
                mode="lines",
                name="Bridged Occlusion Gap",
                line=dict(color="#F59E0B", width=2.5, dash="dot"),
                hovertemplate="%{y}<br>Occluded Gap @ %{x:.2f}s<extra></extra>",
            )
        )

    # Recovery event emerald diamonds
    rec_x: List[float] = []
    rec_y: List[str] = []
    rec_txt: List[str] = []
    for rec in recoveries:
        try:
            rid = int(str(rec.get("Recovered ID", "#0")).lstrip("#"))
        except ValueError:
            continue
        if rid not in person_stats:
            continue
        t_ev = float(rec.get("Frame", 0)) / fps_safe
        rec_x.append(t_ev)
        rec_y.append(f"Person #{rid:02d}")
        rec_txt.append(
            f"<b>{rec.get('Recovery Type', 'Recovery')}</b><br>"
            f"Person #{rid:02d} @ {t_ev:.2f}s (F{rec.get('Frame', 0)})<br>"
            f"Cue: {rec.get('Biometric Cue', 'N/A')}<br>"
            f"Sim: {rec.get('Best Similarity', 0)} | Gap: {rec.get('Occlusion Frames', 0)}f"
        )

    if rec_x:
        fig.add_trace(
            go.Scatter(
                x=rec_x,
                y=rec_y,
                mode="markers",
                name="Biometric Recovery Event",
                marker=dict(color="#10B981", size=12, symbol="diamond", line=dict(color="#ECFDF5", width=1.2)),
                text=rec_txt,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    chart_h = max(280, min(680, 95 + 38 * len(ids_sorted)))
    fig.update_layout(
        title=dict(text="Identity Presence, Occlusion Gaps & Biometric Recovery Events", font=dict(family="Inter", size=14, color="#E2E8F0")),
        height=chart_h,
        margin=dict(l=92, r=24, t=48, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#CBD5E1", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(family="Inter", size=12)),
        xaxis=dict(title="Timeline (seconds)", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(148,163,184,0.10)",
            categoryorder="array",
            categoryarray=[f"Person #{pid:02d}" for pid in ids_sorted],
        ),
    )
    return fig


def main() -> None:
    _inject_custom_css()
    device_now = resolve_device("auto")

    # -----------------------------------------------------------------------
    # Sidebar Control Deck with Quick Presets & Organized Expanders
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Command Control Deck")

        preset = st.radio(
            "Operating Preset",
            options=[
                "Ultra Biometric (Best Accuracy)",
                "Balanced (Recommended)",
                "Fast Preview",
            ],
            index=1,
            help="Automatically sets optimal detector, resolution, and stride defaults; override any knob below.",
        )

        preset_defaults = {
            "Ultra Biometric (Best Accuracy)": {
                "pose_idx": 3,   # yolo11l-pose.pt
                "imgsz": 960,
                "reid_idx": 0,   # dinov2_vitb14
                "depth_stride": 1,
                "face_stride": 1,
                "biometric": True,
                "turbo_gpu": True,
            },
            "Balanced (Recommended)": {
                "pose_idx": 4,   # yolo11m-pose.pt
                "imgsz": 768,
                "reid_idx": 0,   # dinov2_vitb14
                "depth_stride": 2,
                "face_stride": 1,
                "biometric": True,
                "turbo_gpu": True,
            },
            "Fast Preview": {
                "pose_idx": 6,   # yolo11n-pose.pt
                "imgsz": 640,
                "reid_idx": 1,   # dinov2_vits14
                "depth_stride": 3,
                "face_stride": 2,
                "biometric": True,
                "turbo_gpu": True,
            },
        }[preset]

        with st.expander("🧠 AI Model Stack (2026 Ready)", expanded=True):
            turbo_gpu = st.toggle(
                "Turbo GPU Optimization (Batched FP16 + Smart Gating)",
                value=preset_defaults["turbo_gpu"],
                key=f"turbo_gpu_{preset}",
                help=(
                    "ON: FP16 Pose + FP16 single-batch DINO + Depth gating (overlap IoU>=0.15 or /4f) "
                    "+ Pose-gated ArcFace (skip BACK, throttle >=4 samples) + Adaptive DINO stride."
                ),
            )
            biometric_mode = st.toggle(
                "Clothes-Invariant Biometric Mode",
                value=preset_defaults["biometric"],
                key=f"bio_mode_{preset}",
                help="ON: 0% clothing weight (ArcFace + Head/Hair + Grayscale Build + Body Ratios + Lab Skin Chroma).",
            )
            if biometric_mode:
                pose_options = [
                    ("yolo26l-pose.pt — YOLO26-Large Pose (2026)", "yolo26l-pose.pt"),
                    ("yolo26m-pose.pt — YOLO26-Medium Pose (2026)", "yolo26m-pose.pt"),
                    ("yolo26n-pose.pt — YOLO26-Nano Pose (2026 Fast)", "yolo26n-pose.pt"),
                    ("yolo11l-pose.pt — YOLO11-Large Pose (Ultra)", "yolo11l-pose.pt"),
                    ("yolo11m-pose.pt — YOLO11-Medium Pose (Balanced)", "yolo11m-pose.pt"),
                    ("yolo11x-pose.pt — YOLO11-XLarge Pose (Max)", "yolo11x-pose.pt"),
                    ("yolo11n-pose.pt — YOLO11-Nano Pose (Fast)", "yolo11n-pose.pt"),
                    ("rfdetr — RF-DETR Transformer (Optional)", "rfdetr"),
                ]
                detector_choice = st.selectbox(
                    "Pose & Person Detector",
                    options=pose_options,
                    index=preset_defaults["pose_idx"],
                    format_func=lambda item: item[0],
                    key=f"pose_det_{preset}",
                )
                detector_weights = detector_choice[1]
                people_only = True
            else:
                std_options = [
                    ("yolo11m.pt — YOLO11-Medium (Balanced)", "yolo11m.pt"),
                    ("yolo11l.pt — YOLO11-Large (Ultra)", "yolo11l.pt"),
                    ("yolo11x.pt — YOLO11-XLarge (Max)", "yolo11x.pt"),
                    ("yolov8n.pt — YOLOv8-Nano (Fast)", "yolov8n.pt"),
                    ("rfdetr — RF-DETR Transformer (Optional)", "rfdetr"),
                ]
                detector_choice = st.selectbox(
                    "Object Detector",
                    options=std_options,
                    index=0,
                    format_func=lambda item: item[0],
                    key=f"std_det_{preset}",
                )
                detector_weights = detector_choice[1]
                target_mode = st.radio(
                    "Tracking Target",
                    options=["People Only (Most Accurate)", "All Objects"],
                    index=0,
                )
                people_only = target_mode == "People Only (Most Accurate)"

            resolved_det_preview, trt_detected = _resolve_tensorrt_or_pt(detector_weights)
            if trt_detected:
                st.caption(f"⚡ TensorRT engine detected: `{Path(resolved_det_preview).name}` (auto-selected)")

            imgsz_options = [640, 768, 960, 1024, 1280]
            detector_imgsz = st.select_slider(
                "Detector Resolution (imgsz)",
                options=imgsz_options,
                value=preset_defaults["imgsz"],
                key=f"imgsz_{preset}",
            )

            reid_choice = st.selectbox(
                "Re-ID Backbone (DINOv2 / DINOv3)",
                options=REID_CATALOG,
                index=preset_defaults["reid_idx"],
                format_func=lambda item: item[0],
                key=f"reid_{preset}",
            )
            reid_variant = reid_choice[1]

            depth_choice = st.selectbox(
                "Depth Occlusion Guard",
                options=DEPTH_CATALOG,
                index=0,
                format_func=lambda item: item[0],
            )
            depth_backend = depth_choice[1]

            if st.button(
                "⚡ Export Current YOLO Model to TensorRT (.engine)",
                disabled=(not detector_weights.lower().endswith(".pt")),
                width="stretch",
            ):
                with st.spinner(f"Exporting `{detector_weights}` to FP16 TensorRT (imgsz={detector_imgsz})..."):
                    trt_out = export_to_tensorrt(detector_weights, imgsz=detector_imgsz, half=True, device=0)
                if trt_out.get("ok"):
                    st.success(trt_out["message"])
                else:
                    st.warning(trt_out["message"])

        with st.expander("🛡️ Vault, Mask-Clean, Behavior & Privacy", expanded=True):
            mask_clean_biometrics = st.toggle(
                "Mask-Clean Biometrics (YOLO-Seg Background Strip)",
                value=False,
                help="Runs YOLO-Seg on keyframes to replace background pixels with ImageNet mean gray before DINO/skin extraction.",
            )
            match_vault = st.toggle(
                "Match Against Persistent Biometric Vault",
                value=True,
                help="Matches confirmed people against SQLite cross-video memory & watchlist (`data/biometric_vault.db`).",
            )
            enable_pose_behavior = st.toggle(
                "Pose Behavior & Fall Detection",
                value=True,
                help="Zero-cost 17-keypoint classifier: FALL_ALERT, RUNNING, HANDS_RAISED, SITTING, STANDING.",
            )
            privacy_mode = st.selectbox(
                "Privacy Blur Mode",
                options=["Off", "Blur Non-Targets in Spotlight", "Anonymize All Faces"],
                index=0,
                help="Selectively anonymizes non-target faces in Solo Spotlight or all faces across Full Scene renders.",
            )
            privacy_style = st.radio(
                "Privacy Blur Style",
                options=["Gaussian Blur", "Pixelate"],
                index=0,
                horizontal=True,
            )

        with st.expander("🧬 Biometric & Re-ID Engine", expanded=False):
            ghost_preset = st.selectbox(
                "Ghost Particle Memory",
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

            conf_thresh = st.slider(
                "Detection Confidence",
                min_value=0.20,
                max_value=0.70,
                value=0.35,
                step=0.05,
            )
            depth_stride = st.slider(
                "Depth Stride (frames)",
                min_value=1,
                max_value=4,
                value=preset_defaults["depth_stride"],
                key=f"dstride_{preset}",
            )
            face_stride = st.slider(
                "ArcFace Stride (frames)",
                min_value=1,
                max_value=4,
                value=preset_defaults["face_stride"],
                key=f"fstride_{preset}",
            )

        with st.expander("🎥 HUD & Visual Overlays", expanded=False):
            show_skeleton = st.toggle(
                "Draw 17-keypoint skeleton overlay",
                value=False,
                help="Renders COCO 17-keypoint limbs and joints color-coded by persistent #ID.",
            )
            show_verifying = st.toggle(
                "Show unconfirmed verifying boxes & <6f blips",
                value=False,
                help="Off by default so only stable sequential IDs (#1, #2, #3...) appear.",
            )

        st.divider()
        st.caption(
            "**AGENTS.md Core Invariants**\n"
            "- Clean-View Gate & Depth Lock (`IoU >= 0.25`)\n"
            "- Pass-1 Probation (`>= 15f`, `sim >= 0.72`, `margin >= 0.08`)\n"
            "- Pass-2 Tier 0: Overlap Handoff & Turn Bridge\n"
            "- Pass-2 Tier 1: ArcFace Lock (`>= 0.40`, 0% clothing)\n"
            "- Pass-2 Tier 2: Angle-Aware Biometrics (`>= 0.66`)"
        )

    _render_command_header(device_now, detector_weights, reid_variant, biometric_mode, detector_imgsz, depth_backend)

    # -----------------------------------------------------------------------
    # Video Upload Dropzone & Execution Trigger
    # -----------------------------------------------------------------------
    up_col, cam_col, btn_col = st.columns([2.6, 1.0, 1.0], vertical_alignment="bottom")
    with up_col:
        uploaded_file = st.file_uploader(
            "Drop surveillance or multi-person video (.mp4, .mov, .avi, .mkv)",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=False,
        )
    with cam_col:
        camera_label_input = st.text_input(
            "Camera / Location Label",
            value="Camera 1",
            help="Stored in Biometric Vault sightings for cross-camera audit trails.",
        )
    with btn_col:
        run_label = "🚀 Launch Biometric Pipeline" if biometric_mode else "🚀 Launch Standard Pipeline"
        run_clicked = st.button(
            run_label,
            type="primary",
            disabled=(uploaded_file is None),
            width="stretch",
        )

    if uploaded_file is not None:
        up_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"Staged input: `{uploaded_file.name}` · `{up_mb:.2f} MB` · Preset: `{preset}` · Camera: `{camera_label_input}`")

    if run_clicked and uploaded_file is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
        stem = _safe_stem(uploaded_file.name)

        saved_input = UPLOADS_DIR / f"{stem}_{ts}{ext}"
        saved_output = UPLOADS_DIR / f"{stem}_{ts}_tracked_biometric.mp4"
        saved_events = UPLOADS_DIR / f"{stem}_{ts}_events_biometric.jsonl"
        saved_input.write_bytes(uploaded_file.getvalue())

        stepper_placeholder = st.empty()
        stepper_placeholder.markdown(_render_pipeline_stepper(1), unsafe_allow_html=True)
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
                detector_imgsz=detector_imgsz,
                stepper_placeholder=stepper_placeholder,
                turbo_gpu=turbo_gpu,
                mask_clean_biometrics=mask_clean_biometrics,
                match_vault=match_vault,
                enable_pose_behavior=enable_pose_behavior,
                privacy_mode=privacy_mode,
                privacy_style=privacy_style,
                camera_label=camera_label_input or "Camera 1",
            )
            st.session_state["last_upload_run_v6"] = job_result
            st.session_state.pop("clip_search_results_v6", None)
            st.session_state.pop("last_zone_result_v6", None)
            st.session_state.pop("last_heatmap_result_v6", None)
            status_placeholder.success(
                f"Pipeline complete! Processed **{job_result['frames_processed']}** frames in "
                f"**{job_result['elapsed_sec']}s** (Pass-1: **{job_result['pass1_fps']:.1f} FPS**, "
                f"End-to-End: **{job_result['avg_fps']:.1f} FPS**) — "
                f"**{job_result['confirmed_unique_ids']}** Confirmed People vs "
                f"**{job_result['baseline_unique_ids']}** Standard Baseline IDs."
            )
            for note in job_result.get("model_notes", []):
                if "fell back" in note.lower() or "unavailable" in note.lower():
                    st.info(f"Model Fallback Note: {note}")
        except Exception as exc:
            status_placeholder.error(f"Biometric 2-Pass Tracking failed: {exc}")
            raise

    # -----------------------------------------------------------------------
    # 6-Tab Enterprise Command Workspace (Vault & Live CCTV always available!)
    # -----------------------------------------------------------------------
    res: Optional[Dict[str, Any]] = st.session_state.get("last_upload_run_v6")
    vault = get_biometric_vault()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🎬 Full Scene & AI Search",
            "🔍 Single-Person Focus & Privacy",
            "⚖️ Standard vs. Long-Horizon",
            "🗺️ Zones, Heatmap & Behavior",
            "🔐 Biometric Vault & Watchlist",
            "📡 Live CCTV & Forensic Export",
        ]
    )

    if res is not None:
        counters = res.get("counters", {})
        person_stats: Dict[int, Dict[str, Any]] = res.get("person_stats", {})
        person_thumbs: Dict[int, bytes] = res.get("person_thumbnails", {})
        consolidated_timeline: List[FrameSnapshot] = res.get("consolidated_timeline", [])
        baseline_timeline: List[List[Tuple[int, np.ndarray, float]]] = res.get("baseline_timeline", [])
        recoveries: List[Dict[str, Any]] = res.get("recoveries", [])
        seq_profiles: Dict[int, TrackBiometricProfile] = res.get("seq_profiles", {})
        vault_matches: Dict[int, Dict[str, Any]] = res.get("vault_matches", {})
        pose_result: Optional[Dict[str, Any]] = res.get("pose_result")
        pose_summaries: Dict[int, Dict[str, Any]] = (pose_result or {}).get("track_summaries", {})
        source_video_path = Path(res.get("source_video_path", res.get("input_path", "")))
        src_fps = float(res.get("src_fps", 30.0))
        total_video_frames = max(1, int(res.get("frames_processed", len(consolidated_timeline) or 1)))
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
    else:
        counters = {}
        person_stats = {}
        person_thumbs = {}
        consolidated_timeline = []
        baseline_timeline = []
        recoveries = []
        seq_profiles = {}
        vault_matches = {}
        pose_result = None
        pose_summaries = {}
        source_video_path = Path("")
        src_fps = 30.0
        total_video_frames = 1
        confirmed_ids_count = 0
        baseline_ids_count = 0
        pass1_stitches = 0
        pass2_turns = 0
        pass2_face = 0
        pass2_bio = 0
        total_recoveries = 0
        max_occ_frames = 0
        max_occ_sec = 0.0
        pass1_fps = 0.0
        avg_fps = 0.0

    # =======================================================================
    # TAB 1 — Full Scene Tracking, Natural-Language CLIP Search & Dossier Roster
    # =======================================================================
    with tab1:
        if res is None:
            st.info(
                "Upload a video above and click **Launch Biometric Pipeline** to view Full Scene Tracking, "
                "Natural-Language CLIP Target Search, and the Biometric Dossier Roster. "
                "(You can already enroll Watchlist photos in **Tab 5** or open Live CCTV in **Tab 6**!)"
            )
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Confirmed People" if res.get("people_only", True) else "Confirmed Objects",
                confirmed_ids_count,
                delta=f"Sequential #01..#{confirmed_ids_count:02d}" if confirmed_ids_count > 0 else "None",
                delta_color="off",
            )
            k2.metric(
                "Total Biometric Recoveries",
                total_recoveries,
                delta=f"T0:{pass2_turns} · Face:{pass2_face} · Bio:{pass2_bio} · P1:{pass1_stitches}",
                delta_color="off",
            )
            k3.metric(
                "Max Occlusion Survived",
                f"{max_occ_frames}f" if max_occ_frames > 0 else "0f",
                delta=f"{max_occ_sec:.2f}s · {res.get('backfilled_frame_boxes', 0)}f back-filled",
                delta_color="off",
            )
            k4.metric(
                "Processing FPS",
                f"{pass1_fps:.1f} / {avg_fps:.1f}",
                delta=f"Pass-1 / Total ({total_video_frames}f @ {res.get('resolution', 'N/A')})",
                delta_color="off",
            )

            out_path = Path(res["output_path"])
            if out_path.exists():
                video_bytes = out_path.read_bytes()
                size_mb = len(video_bytes) / (1024 * 1024)
                hdr_l, hdr_r = st.columns([3.2, 1.0], vertical_alignment="bottom")
                with hdr_l:
                    _render_cinema_bar(
                        "PRIMARY FEED // PASS-2 BIOMETRIC CONSOLIDATED OUTPUT",
                        f"{res.get('resolution', 'N/A')} · {src_fps:.1f} FPS · H.264 YUV420P · {size_mb:.1f} MB",
                    )
                with hdr_r:
                    st.download_button(
                        label=f"⬇️ Download Full MP4 ({size_mb:.1f} MB)",
                        data=video_bytes,
                        file_name=out_path.name,
                        mime="video/mp4",
                        width="stretch",
                    )
                st.video(video_bytes, format="video/mp4")
            else:
                st.warning(f"Tracked output video not found at `{out_path}`.")

            # --- Natural-Language CLIP Target Search Bar ---
            st.markdown("#### 🔎 Natural-Language CLIP Target Search")
            sq_col, sb_col, sc_col = st.columns([3.2, 0.9, 0.7], vertical_alignment="bottom")
            with sq_col:
                clip_query = st.text_input(
                    "Search people by description...",
                    placeholder="e.g. 'person wearing glasses near the counter', 'man carrying a backpack', 'red shirt'",
                    key="clip_nl_query_input",
                )
            with sb_col:
                clip_search_clicked = st.button(
                    "🔍 Search IDs",
                    type="primary",
                    disabled=(not clip_query.strip() or not person_stats),
                    width="stretch",
                )
            with sc_col:
                if st.button("Clear", width="stretch", key="clip_clear_btn"):
                    st.session_state.pop("clip_search_results_v6", None)

            if clip_search_clicked and clip_query.strip():
                with st.spinner("Loading FP16 CLIP ViT-B/32 & ranking confirmed people..."):
                    searcher = load_clip_searcher(device=device_now)
                    if not searcher.available:
                        st.warning(searcher.status_note)
                    else:
                        if res.get("clip_indexed_video") != str(source_video_path):
                            n_idx = searcher.index_from_timeline(source_video_path, consolidated_timeline, top_k=3)
                            res["clip_indexed_video"] = str(source_video_path)
                            res["clip_indexed_count"] = n_idx
                        ranked = searcher.search(clip_query.strip(), top_k=max(4, len(person_stats)))
                        st.session_state["clip_search_results_v6"] = {
                            "query": clip_query.strip(),
                            "ranked": ranked,
                        }

            clip_scores_map: Dict[int, float] = {}
            clip_state = st.session_state.get("clip_search_results_v6")
            if clip_state and clip_state.get("ranked"):
                ranked_items = clip_state["ranked"]
                st.success(f"CLIP Search for **\"{clip_state['query']}\"** — Top matches highlighted below:")
                top_cols = st.columns(min(4, len(ranked_items)))
                for col_i, (rtid, rpct, rcrop) in zip(top_cols, ranked_items[:4]):
                    clip_scores_map[int(rtid)] = float(rpct)
                    with col_i:
                        st.image(rcrop, caption=f"Person #{rtid:02d} · {rpct:.1f}% match", width="stretch")

            # --- Biometric Dossier Roster + Inline Name / Save to Vault ---
            st.markdown("#### 🗂️ Biometric Dossier Roster")
            if person_stats:
                if clip_scores_map:
                    ids_sorted = sorted(
                        person_stats.keys(),
                        key=lambda pid: (-clip_scores_map.get(pid, -1.0), pid),
                    )
                else:
                    ids_sorted = sorted(person_stats.keys())

                for row_start in range(0, len(ids_sorted), 4):
                    row_ids = ids_sorted[row_start : row_start + 4]
                    row_cols = st.columns(4)
                    for col_obj, pid in zip(row_cols, row_ids):
                        with col_obj:
                            v_inf = vault_matches.get(pid)
                            st.markdown(
                                _render_dossier_card_html(
                                    pid,
                                    person_stats[pid],
                                    person_thumbs.get(pid),
                                    vault_info=v_inf,
                                    clip_score_pct=clip_scores_map.get(pid),
                                    behavior_summary=pose_summaries.get(pid),
                                ),
                                unsafe_allow_html=True,
                            )
                            with st.expander(f"🏷️ Name / Save #{pid:02d} to Vault", expanded=False):
                                cur_vname = (v_inf or {}).get("display_name") or ""
                                cur_wl = bool((v_inf or {}).get("is_watchlist", False))
                                new_lbl = st.text_input(
                                    "Person Name",
                                    value=cur_vname,
                                    placeholder="e.g. Cashier - Arjun",
                                    key=f"inline_vname_{pid}",
                                )
                                new_wl = st.checkbox("Watchlist Alert", value=cur_wl, key=f"inline_vwl_{pid}")
                                if st.button("💾 Save to Vault", key=f"inline_vsave_{pid}", width="stretch"):
                                    ps = person_stats[pid]
                                    f_sec = round(max(0, int(ps.get("first_frame", 1)) - 1) / max(1.0, src_fps), 2)
                                    l_sec = round(max(0, int(ps.get("last_frame", 1)) - 1) / max(1.0, src_fps), 2)
                                    existing_vid = v_inf["vault_id"] if v_inf else None
                                    saved_vid = vault.upsert_track_identity(
                                        track_biometrics=seq_profiles.get(pid),
                                        vault_id=existing_vid,
                                        display_name=new_lbl.strip() or None,
                                        is_watchlist=new_wl,
                                        video_name=source_video_path.name if source_video_path.name else "video.mp4",
                                        camera_label=camera_label_input or "Camera 1",
                                        first_sec=f_sec,
                                        last_sec=l_sec,
                                        thumbnail_jpeg=person_thumbs.get(pid),
                                    )
                                    vault.rename_identity(saved_vid, new_lbl.strip() or None, is_watchlist=new_wl)
                                    vault_matches[pid] = {
                                        "vault_id": saved_vid,
                                        "display_name": new_lbl.strip() or None,
                                        "is_watchlist": new_wl,
                                        "confidence": 1.0,
                                        "cue": "USER_LABELED",
                                    }
                                    res["vault_matches"] = vault_matches
                                    st.success(f"Saved #{pid:02d} -> `{new_lbl.strip() or f'Vault #{saved_vid}'}`")
                                    st.rerun()
            else:
                st.info("No confirmed people were detected in this clip.")

    # =======================================================================
    # TAB 2 — Single-Person Focus & Privacy Mode
    # =======================================================================
    with tab2:
        if res is None or not person_stats:
            st.info("Run the pipeline on a video first to isolate a single person and render Privacy Spotlight exports.")
        else:
            ids_sorted = sorted(person_stats.keys())

            def _fmt_person(pid: int) -> str:
                ps = person_stats[pid]
                v_inf = vault_matches.get(pid) or {}
                v_name = f" ({v_inf['display_name']})" if v_inf.get("display_name") else ""
                fb = "Face Locked" if ps["has_face_lock"] else f"Angles {ps.get('angle_code', '—')}"
                rc = ps["recovery_count"]
                rlabel = f"{rc} Recovery" if rc == 1 else f"{rc} Recoveries"
                return f"Person #{pid:02d}{v_name} — Seen {ps['duration_sec']:.1f}s | {fb} | {rlabel}"

            ctrl_col, dos_col = st.columns([1.55, 1.45], vertical_alignment="top")
            with ctrl_col:
                selected_pid = st.selectbox(
                    "Select Target Identity",
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
                draw_trail_opt = opt_c1.toggle("Motion Trail", value=True, key="solo_trail")
                show_occ_hud_opt = opt_c2.toggle("Occluded HUD", value=True, key="solo_occ_hud")
                solo_skel_opt = opt_c3.toggle("17-Kpt Skeleton", value=show_skeleton, key="solo_skel")

                priv_c1, priv_c2 = st.columns([1.4, 1.0])
                solo_blur_others = priv_c1.toggle(
                    "Privacy Blur (Blur everyone else's faces)",
                    value=(privacy_mode != "Off"),
                    key="solo_privacy_blur",
                )
                solo_blur_style = priv_c2.selectbox(
                    "Blur Style",
                    options=["Gaussian Blur", "Pixelate"],
                    index=0 if privacy_style == "Gaussian Blur" else 1,
                    key="solo_privacy_style",
                )

                render_solo_clicked = st.button(
                    f"🎯 Render Selected Person Video (ID #{selected_pid:02d})",
                    type="primary",
                    width="stretch",
                )

            pinfo = person_stats[selected_pid]
            v_sel = vault_matches.get(selected_pid) or {}
            sel_name = v_sel.get("display_name")
            cov_pct = min(100.0, 100.0 * float(pinfo["frames_visible"]) / float(total_video_frames))
            bfrags = pinfo["baseline_fragment_ids"]
            with dos_col:
                uri = _thumb_to_data_uri(person_thumbs.get(selected_pid))
                img_html = (
                    f'<img class="solo-dossier-img" src="{uri}" alt="Person #{selected_pid}" />'
                    if uri
                    else '<div class="solo-dossier-img" style="display:flex;align-items:center;justify-content:center;color:#475569;font-family:JetBrains Mono;font-size:0.75rem;">NO CROP</div>'
                )
                face_b = '<span class="mini-badge mb-face">ARCFACE LOCKED</span>' if pinfo["has_face_lock"] else '<span class="mini-badge mb-angle">HEAD/BODY/SKIN</span>'
                ang_b = f'<span class="mini-badge mb-angle">ANGLES: {_html.escape(pinfo.get("angle_code", "—"))}</span>'
                name_hdr = f" · {_html.escape(sel_name)}" if sel_name else ""
                st.markdown(
                    f"""
                    <div class="solo-dossier">
                        {img_html}
                        <div>
                            <div class="dossier-id">TARGET DOSSIER // PERSON #{selected_pid:02d}{name_hdr}</div>
                            <div style="margin:0.25rem 0 0.45rem 0;">{face_b}{ang_b}</div>
                            <div class="dossier-sub">
                                Coverage: <b>{cov_pct:.1f}%</b> ({pinfo['duration_sec']:.1f}s / {pinfo['frames_visible']}f)<br/>
                                Max Gap Bridged: <b>{pinfo['max_gap_frames']}f</b> ({pinfo['max_gap_sec']:.2f}s) · Recoveries: <b>{pinfo['recovery_count']}</b><br/>
                                Baseline IDs on Target: <b>{len(bfrags)}</b> ({", ".join(f"#{b}" for b in bfrags[:6]) or "N/A"})<br/>
                                Modalities: {_html.escape(pinfo['modalities'])}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            eff_priv_mode = (
                "Anonymize All Faces"
                if privacy_mode == "Anonymize All Faces"
                else ("Blur Everyone Except Selected Target" if solo_blur_others else "Off")
            )
            solo_cache_key = (
                f"p{selected_pid}_{spotlight_mode[:3]}_{int(draw_trail_opt)}_"
                f"{int(show_occ_hud_opt)}_{int(solo_skel_opt)}_{eff_priv_mode[:4]}_{solo_blur_style[:3]}"
            )
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
                        privacy_mode=eff_priv_mode,
                        privacy_style=solo_blur_style,
                        target_display_name=sel_name,
                    )
                    res.setdefault("solo_renders", {})[solo_cache_key] = str(solo_out)
                    s_stat.success(f"Person #{selected_pid:02d} spotlight video ready!")

            st.caption(f"**Appearance Intervals:** `{pinfo['intervals_text']}`")

            solo_path_str = res.get("solo_renders", {}).get(solo_cache_key)
            if solo_path_str and Path(solo_path_str).exists():
                sp = Path(solo_path_str)
                s_bytes = sp.read_bytes()
                s_mb = len(s_bytes) / (1024 * 1024)
                sh_l, sh_r = st.columns([3.2, 1.0], vertical_alignment="bottom")
                with sh_l:
                    priv_lbl = f" · PRIVACY: {eff_priv_mode.upper()}" if eff_priv_mode != "Off" else ""
                    _render_cinema_bar(
                        f"SPOTLIGHT FEED // TARGET PERSON #{selected_pid:02d}",
                        f"{spotlight_mode.split('(')[0].strip().upper()}{priv_lbl} · H.264 YUV420P · {s_mb:.1f} MB",
                    )
                with sh_r:
                    st.download_button(
                        label=f"⬇️ Download Person #{selected_pid:02d} MP4 ({s_mb:.1f} MB)",
                        data=s_bytes,
                        file_name=sp.name,
                        mime="video/mp4",
                        width="stretch",
                    )
                st.video(s_bytes, format="video/mp4")

    # =======================================================================
    # TAB 3 — Standard vs. Long-Horizon Comparison + Plotly Fragmentation Curve
    # =======================================================================
    with tab3:
        if res is None:
            st.info("Run the pipeline on a video first to benchmark Standard IoU vs. Long-Horizon Biometric AI.")
        else:
            splits_avoided = max(0, baseline_ids_count - confirmed_ids_count)
            frag_red_pct = (
                100.0 * (baseline_ids_count - confirmed_ids_count) / float(max(1, baseline_ids_count))
                if baseline_ids_count > 0
                else 0.0
            )

            ck1, ck2, ck3, ck4, ck5 = st.columns(5)
            ck1.metric("Standard Baseline IDs", baseline_ids_count, delta="Pure IoU (15f coast)", delta_color="off")
            ck2.metric("Long-Horizon Biometric IDs", confirmed_ids_count, delta="3-Tier Biometric AI", delta_color="off")
            ck3.metric("Fragmentation Reduction", f"{frag_red_pct:.1f}%", delta="Lower IDs = less splitting", delta_color="off")
            ck4.metric("Baseline Splits Avoided", splits_avoided, delta=f"{baseline_ids_count} → {confirmed_ids_count}", delta_color="off")
            ck5.metric("Longest Occlusion Bridged", f"{max_occ_frames}f", delta=f"{max_occ_sec:.2f}s", delta_color="off")

            if PLOTLY_AVAILABLE:
                ch_left, ch_right = st.columns([2.4, 1.0])
                cum_fig = build_cumulative_id_chart(baseline_timeline, consolidated_timeline, src_fps)
                gauge_fig = build_fragmentation_gauge(frag_red_pct)
                if cum_fig is not None:
                    ch_left.plotly_chart(cum_fig, width="stretch")
                if gauge_fig is not None:
                    ch_right.plotly_chart(gauge_fig, width="stretch")

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
                            f"Person #{pid:02d} — Standard used {len(person_stats[pid]['baseline_fragment_ids'])} ID(s) vs 1 Biometric ID"
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
                ch_l, ch_r = st.columns([3.2, 1.0], vertical_alignment="bottom")
                with ch_l:
                    _render_cinema_bar(
                        "BENCHMARK FEED // LEFT: STANDARD BASELINE  |  RIGHT: LONG-HORIZON BIOMETRIC AI",
                        f"H.264 YUV420P · {c_mb:.1f} MB",
                    )
                with ch_r:
                    st.download_button(
                        label=f"⬇️ Download Comparison MP4 ({c_mb:.1f} MB)",
                        data=c_bytes,
                        file_name=cp.name,
                        mime="video/mp4",
                        width="stretch",
                    )
                st.video(c_bytes, format="video/mp4")

    # =======================================================================
    # TAB 4 — Spatial Zones, Movement Heatmap, Behavior Alerts & Telemetry
    # =======================================================================
    with tab4:
        if res is None or not consolidated_timeline:
            st.info("Run the pipeline on a video first to configure Spatial Zones, render the Traffic Heatmap, and inspect Behavior Alerts.")
        else:
            st.markdown("#### 🗺️ Polygon Zone & Loitering Configuration")
            zc1, zc2, zc3 = st.columns([1.4, 2.0, 1.0])
            with zc1:
                zone_preset_choice = st.selectbox(
                    "Zone Preset Template",
                    options=[
                        "Left Half / Counter + Right Half / Entrance",
                        "Full Frame",
                        "Left Half / Counter",
                        "Right Half / Entrance",
                        "Center Box",
                        "Custom Polygon",
                    ],
                    index=0,
                    key="zone_preset_select",
                )
            with zc2:
                default_coord_str = format_polygon_coords(
                    ZONE_PRESETS.get(zone_preset_choice, ZONE_PRESETS["Center Box"])
                )
                custom_coords_txt = st.text_input(
                    "Custom Normalized Polygon (`x1,y1; x2,y2; ...`)",
                    value=default_coord_str,
                    disabled=(zone_preset_choice != "Custom Polygon"),
                    key=f"zone_custom_coords_{zone_preset_choice}",
                )
            with zc3:
                loiter_sec_cfg = st.slider(
                    "Loiter Alert (sec)",
                    min_value=2.0,
                    max_value=60.0,
                    value=10.0,
                    step=1.0,
                    key="zone_loiter_slider",
                )

            if zone_preset_choice == "Left Half / Counter + Right Half / Entrance":
                active_zones = [
                    PolygonZone("Left Half / Counter", list(ZONE_PRESETS["Left Half / Counter"]), (212, 182, 6)),
                    PolygonZone("Right Half / Entrance", list(ZONE_PRESETS["Right Half / Entrance"]), (11, 158, 245)),
                ]
            elif zone_preset_choice == "Custom Polygon":
                parsed_pts = parse_polygon_coords(custom_coords_txt) or list(ZONE_PRESETS["Center Box"])
                active_zones = [PolygonZone("Custom Zone", parsed_pts, (129, 185, 16))]
            else:
                active_zones = [PolygonZone(zone_preset_choice, list(ZONE_PRESETS[zone_preset_choice]), (212, 182, 6))]

            fw = int(res.get("width", 1280))
            fh = int(res.get("height", 720))
            zone_res = ZoneAnalyzer(
                zones=active_zones,
                loiter_threshold_sec=loiter_sec_cfg,
                src_fps=src_fps,
            ).analyze_timeline(consolidated_timeline, frame_size=(fw, fh), src_fps=src_fps)
            st.session_state["last_zone_result_v6"] = zone_res

            # Generate & cache Movement Heatmap over clean median background
            hm_cache_key = f"{source_video_path.name}_{len(consolidated_timeline)}"
            hm_cached = st.session_state.get("last_heatmap_result_v6")
            if hm_cached is None or hm_cached.get("cache_key") != hm_cache_key:
                hm_gen = HeatmapGenerator(alpha=0.45)
                hm_data = hm_gen.generate(
                    consolidated_timeline,
                    video_path=source_video_path if source_video_path.exists() else None,
                    frame_size=(fw, fh),
                )
                hm_data["cache_key"] = hm_cache_key
                st.session_state["last_heatmap_result_v6"] = hm_data
                hm_cached = hm_data

            hm_col, tbl_col = st.columns([1.45, 1.55], vertical_alignment="top")
            with hm_col:
                st.markdown("#### 🔥 Traffic Movement Heatmap (`COLORMAP_TURBO`)")
                overlay_zones_on_hm = st.toggle("Overlay Polygon Zones on Heatmap", value=True, key="hm_overlay_zones")
                hm_bgr_vis = hm_cached["bgr_image"]
                if overlay_zones_on_hm:
                    peak_occ_map = {z["zone_name"]: z["peak_occupancy"] for z in zone_res["zone_summaries"]}
                    alert_zset = {z["zone_name"] for z in zone_res["zone_summaries"] if z["loitering_alerts"] > 0}
                    hm_bgr_vis = draw_zones_overlay(hm_bgr_vis, active_zones, active_counts=peak_occ_map, alert_zones=alert_zset)
                st.image(cv2.cvtColor(hm_bgr_vis, cv2.COLOR_BGR2RGB), width="stretch")
                st.download_button(
                    label="⬇️ Download Heatmap PNG",
                    data=hm_cached["png_bytes"],
                    file_name=f"{source_video_path.stem}_traffic_heatmap.png",
                    mime="image/png",
                    width="stretch",
                )

            with tbl_col:
                st.markdown("#### 📊 Zone Dwell-Time & Crossing Summary")
                if zone_res["zone_summaries"]:
                    st.dataframe(pd.DataFrame(zone_res["zone_summaries"]), width="stretch", hide_index=True)

                st.markdown("#### 🚨 Behavior & Spatial Anomaly Feed")
                combined_alerts: List[Dict[str, Any]] = []
                for ev in zone_res["events"]:
                    combined_alerts.append(
                        {
                            "Time (s)": ev["timestamp_sec"],
                            "Person": f"#{int(ev['track_id']):02d}",
                            "Event Type": ev["event_type"],
                            "Context": ev["zone_name"],
                            "Duration (s)": ev["duration_sec"],
                            "Severity": ev["severity"],
                        }
                    )
                if pose_result:
                    for ev in pose_result.get("events", []):
                        combined_alerts.append(
                            {
                                "Time (s)": ev["timestamp_sec"],
                                "Person": f"#{int(ev['track_id']):02d}",
                                "Event Type": ev["event_type"],
                                "Context": "Pose Kinetics",
                                "Duration (s)": ev["duration_sec"],
                                "Severity": ev["severity"],
                            }
                        )
                combined_alerts.sort(key=lambda r: float(r["Time (s)"]))
                if combined_alerts:
                    st.dataframe(pd.DataFrame(combined_alerts), width="stretch", hide_index=True)
                else:
                    st.success("Zero loitering, fall, running, or hands-raised anomalies detected.")

            if zone_res["track_zone_stats"]:
                with st.expander("📋 Per-Person Zone Dwell Breakdown", expanded=False):
                    st.dataframe(pd.DataFrame(zone_res["track_zone_stats"]), width="stretch", hide_index=True)

            st.divider()
            if PLOTLY_AVAILABLE and person_stats:
                gantt_fig = build_gantt_presence_chart(person_stats, recoveries, src_fps)
                if gantt_fig is not None:
                    st.plotly_chart(gantt_fig, width="stretch")

            with st.expander(f"🧬 Biometric & Occlusion Recovery Event Log ({len(recoveries)} events)", expanded=False):
                if recoveries:
                    rec_df = pd.DataFrame(recoveries)
                    ev_types = sorted(rec_df["Recovery Type"].dropna().unique().tolist())
                    selected_types = st.multiselect(
                        "Filter Recovery Event Types",
                        options=ev_types,
                        default=ev_types,
                        key="recovery_type_filter",
                    )
                    filtered_df = rec_df[rec_df["Recovery Type"].isin(selected_types)] if selected_types else rec_df
                    st.dataframe(filtered_df, width="stretch", hide_index=True)
                else:
                    st.info("No disappearance, turn-bridge, or outfit-change recoveries were required in this clip.")

    # =======================================================================
    # TAB 5 — Biometric Vault & Watchlist (Persistent Cross-Video Memory)
    # =======================================================================
    with tab5:
        st.markdown("#### 🔐 Enroll Reference Photo onto Persistent Watchlist")
        st.caption(
            "Enroll a named person (e.g. **Cashier - Arjun**, **VIP Guest**) BEFORE running a video so they are "
            "automatically recognized across cameras and outfit changes via Tier-1 ArcFace + Head/Skin biometrics."
        )
        en_c1, en_c2, en_c3 = st.columns([2.0, 1.4, 1.0], vertical_alignment="bottom")
        with en_c1:
            ref_photo = st.file_uploader(
                "Upload Reference Portrait (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"],
                key="vault_ref_photo_upload",
            )
        with en_c2:
            ref_name = st.text_input(
                "Display Name",
                placeholder="e.g. Cashier - Arjun",
                key="vault_ref_name_input",
            )
        with en_c3:
            ref_watchlist = st.toggle("Flag as Watchlist", value=True, key="vault_ref_wl_toggle")
            enroll_clicked = st.button(
                "➕ Enroll into Vault",
                type="primary",
                disabled=(ref_photo is None or not ref_name.strip()),
                width="stretch",
            )

        if enroll_clicked and ref_photo is not None and ref_name.strip():
            raw_bytes = np.frombuffer(ref_photo.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode uploaded image.")
            else:
                with st.spinner("Extracting ArcFace + DINOv2 head + CIE Lab skin embeddings..."):
                    f_app = load_face_analyzer(device=device_now)
                    d_enc = load_reid_encoder(reid_variant="dinov2_vitb14", device=device_now)
                    enr = vault.enroll_reference_photo(
                        image_bgr=img_bgr,
                        display_name=ref_name.strip(),
                        face_app=f_app,
                        dinov2_encoder=d_enc,
                        is_watchlist=ref_watchlist,
                    )
                st.success(
                    f"Enrolled **{enr['display_name']}** as `Vault #{enr['vault_id']}` · "
                    f"Face Lock: `{'YES' if enr['face_extracted'] else 'NO'}` · "
                    f"Modalities: `{', '.join(enr['modalities_stored']) or 'none'}`"
                )
                st.rerun()

        st.divider()
        vault_items = vault.list_identities(watchlist_only=False, include_sightings=True)
        vh_l, vh_r = st.columns([3.0, 1.0], vertical_alignment="center")
        with vh_l:
            st.markdown(f"#### 🗃️ Stored Vault Identities ({len(vault_items)} profiles in `data/biometric_vault.db`)")
        with vh_r:
            if vault_items and st.button("🗑️ Clear Entire Vault", key="vault_clear_all_btn", width="stretch"):
                n_rem = vault.clear_vault()
                st.warning(f"Cleared {n_rem} profiles from Biometric Vault.")
                st.rerun()

        if not vault_items:
            st.info("Biometric Vault is currently empty. Enroll a reference photo above or run a video with Vault Matching enabled.")
        else:
            for row_start in range(0, len(vault_items), 4):
                chunk = vault_items[row_start : row_start + 4]
                vcols = st.columns(4)
                for vcol, item in zip(vcols, chunk):
                    with vcol:
                        vid = item["vault_id"]
                        vname = item["display_name"] or f"Unnamed Vault #{vid}"
                        uri = _thumb_to_data_uri(item.get("thumbnail_jpeg"))
                        img_tag = f'<img src="{uri}" alt="{_html.escape(vname)}" />' if uri else '<span class="mono" style="color:#475569;">NO CROP</span>'
                        wl_b = '<span class="mini-badge" style="background:rgba(244,63,94,0.22);color:#FB7185;border:1px solid rgba(251,113,133,0.45);">WATCHLIST</span>' if item["is_watchlist"] else ""
                        f_b = '<span class="mini-badge mb-face">ARCFACE</span>' if item["has_face"] else '<span class="mini-badge mb-angle">HEAD/SKIN</span>'
                        st.markdown(
                            f"""
                            <div class="dossier-card">
                              <div class="reticle-wrap">
                                <span class="reticle-corner rc-tl"></span><span class="reticle-corner rc-tr"></span>
                                <span class="reticle-corner rc-bl"></span><span class="reticle-corner rc-br"></span>
                                {img_tag}
                              </div>
                              <div class="dossier-id">V#{vid:02d} · {_html.escape(vname)}</div>
                              <div class="dossier-sub">Videos: {item['videos_seen_count']} · Sightings: {len(item.get('sightings', []))}</div>
                              <div>{wl_b}{f_b}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"Manage V#{vid:02d}", expanded=False):
                            ed_name = st.text_input("Rename", value=item["display_name"] or "", key=f"v_ren_txt_{vid}")
                            ed_wl = st.checkbox("Watchlist", value=bool(item["is_watchlist"]), key=f"v_ren_wl_{vid}")
                            bc1, bc2 = st.columns(2)
                            if bc1.button("Save", key=f"v_ren_btn_{vid}", width="stretch"):
                                vault.rename_identity(vid, ed_name.strip() or None, is_watchlist=ed_wl)
                                st.rerun()
                            if bc2.button("Delete", key=f"v_del_btn_{vid}", width="stretch"):
                                vault.delete_identity(vid)
                                st.rerun()

            all_sightings = vault.get_sightings()
            if all_sightings:
                st.markdown("#### 🛰️ Cross-Video / Multi-Camera Sighting History")
                st.dataframe(pd.DataFrame(all_sightings), width="stretch", hide_index=True)

    # =======================================================================
    # TAB 6 — One-Click Forensic Export & Live CCTV (RTSP) / Webcam Monitor
    # =======================================================================
    with tab6:
        st.markdown("#### 📄 One-Click Forensic Report & Telemetry Export")
        if res is None:
            st.info("Process a video first to generate the multi-section Forensic PDF Report and Telemetry CSV.")
        else:
            zone_for_exp = st.session_state.get("last_zone_result_v6")
            hm_for_exp = st.session_state.get("last_heatmap_result_v6")
            if hm_for_exp is None and consolidated_timeline:
                fw = int(res.get("width", 1280))
                fh = int(res.get("height", 720))
                hm_for_exp = HeatmapGenerator(alpha=0.45).generate(
                    consolidated_timeline,
                    video_path=source_video_path if source_video_path.exists() else None,
                    frame_size=(fw, fh),
                )
                st.session_state["last_heatmap_result_v6"] = hm_for_exp

            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                pdf_bytes = generate_forensic_pdf(
                    job_result=res,
                    heatmap_rgb=hm_for_exp["rgb_image"] if hm_for_exp else None,
                    zone_result=zone_for_exp,
                    pose_result=pose_result,
                    vault_matches=vault_matches,
                )
                st.download_button(
                    label=f"📄 Download Forensic PDF Report ({len(pdf_bytes) / 1024:.0f} KB)",
                    data=pdf_bytes,
                    file_name=f"{source_video_path.stem or 'tracking'}_forensic_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )
            with exp_c2:
                csv_bytes = generate_forensic_csv(
                    job_result=res,
                    zone_result=zone_for_exp,
                    pose_result=pose_result,
                    vault_matches=vault_matches,
                )
                st.download_button(
                    label=f"📊 Download Telemetry CSV ({len(csv_bytes) / 1024:.1f} KB)",
                    data=csv_bytes,
                    file_name=f"{source_video_path.stem or 'tracking'}_telemetry.csv",
                    mime="text/csv",
                    width="stretch",
                )

        st.divider()
        st.markdown("#### 📡 Live CCTV (RTSP) / Webcam Real-Time Monitor")
        st.caption(
            "Connect a local webcam (`0`) or live IP camera (`rtsp://user:pass@ip:554/stream`) for fast Pass-1 "
            "tracking with Biometric Vault watchlist alerts and zero-cost Pose Fall / Hands-Raised / Running alerts."
        )
        lc1, lc2, lc3 = st.columns([1.2, 2.0, 1.0], vertical_alignment="bottom")
        with lc1:
            live_src_mode = st.radio(
                "Stream Source",
                options=["Webcam (Index 0)", "RTSP Stream URL"],
                index=0,
                horizontal=True,
                key="live_src_mode",
            )
        with lc2:
            if live_src_mode == "Webcam (Index 0)":
                live_source_val: Any = st.number_input("Webcam Device Index", min_value=0, max_value=8, value=0, step=1, key="live_cam_idx")
            else:
                live_source_val = st.text_input(
                    "RTSP URL",
                    placeholder="rtsp://username:password@192.168.1.50:554/stream1",
                    key="live_rtsp_url",
                )
        with lc3:
            live_max_frames = st.slider("Session Frame Cap", min_value=30, max_value=1500, value=240, step=30, key="live_max_frames")

        lb1, lb2 = st.columns(2)
        start_live = lb1.button("▶️ Start Live Monitor", type="primary", width="stretch", key="live_start_btn")
        stop_live = lb2.button("⏹️ Stop Live Monitor", width="stretch", key="live_stop_btn")

        if stop_live:
            st.session_state["live_monitor_running"] = False
        if start_live:
            st.session_state["live_monitor_running"] = True

        if st.session_state.get("live_monitor_running", False):
            if live_src_mode == "RTSP Stream URL" and not str(live_source_val).strip():
                st.warning("Enter a valid `rtsp://...` URL before starting Live Monitor.")
                st.session_state["live_monitor_running"] = False
            else:
                live_status = st.empty()
                live_alert_box = st.empty()
                live_frame_slot = st.empty()

                cap_live = cv2.VideoCapture(int(live_source_val) if live_src_mode == "Webcam (Index 0)" else str(live_source_val).strip())
                if not cap_live.isOpened():
                    st.error(f"Could not open live stream source `{live_source_val}`.")
                    st.session_state["live_monitor_running"] = False
                else:
                    try:
                        live_det = load_nextgen_detector(model_name=detector_weights, device=device_now)
                        live_det.classes = [0]
                        live_det.conf = conf_thresh
                        live_det.imgsz = min(int(detector_imgsz), 768)
                        live_face = load_face_analyzer(device=device_now) if match_vault else None
                        live_tracker = LongHorizonTracker(
                            config=TrackerConfig(
                                det_thresh=conf_thresh,
                                max_active_miss=25,
                                probation_frames=15,
                                stitch_min_sim=0.72,
                                margin_min_diff=0.08,
                            )
                        )
                        live_pose_analyzer = PoseBehaviorAnalyzer(smooth_window=5)
                        live_pose_history: List[Dict[str, Any]] = []
                        live_vault_names: Dict[int, str] = {}
                        live_fps_ema = 15.0

                        for f_i in range(1, int(live_max_frames) + 1):
                            if not st.session_state.get("live_monitor_running", False):
                                break
                            ok_l, fr_l = cap_live.read()
                            if not ok_l or fr_l is None:
                                live_status.warning("Live stream ended or frame read failed.")
                                break
                            t_l0 = time.perf_counter()
                            l_boxes, l_scores, l_cls, l_kpts = live_det.detect(fr_l)
                            n_ld = len(l_boxes)
                            l_dets: List[Detection] = []
                            for i in range(n_ld):
                                x1, y1, x2, y2 = [int(round(v)) for v in l_boxes[i]]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(fr_l.shape[1], max(x1 + 4, x2)), min(fr_l.shape[0], max(y1 + 4, y2))
                                emb_i = AppearanceEncoder._hsv_hog_embedding(fr_l[y1:y2, x1:x2])
                                d_y = float(y2) / float(max(1, fr_l.shape[0]))
                                l_dets.append(Detection(box=l_boxes[i], score=float(l_scores[i]), class_id=0, embedding=emb_i, depth=d_y))

                            l_res = live_tracker.update(fr_l, l_dets)
                            act_list_for_pose: List[Dict[str, Any]] = []
                            active_alerts_live: List[str] = []

                            for tr in l_res.active_tracks:
                                tb = tr.current_box()
                                mi = _match_box_to_detection(tb, l_boxes) if n_ld > 0 else None
                                tr_kp = l_kpts[mi] if mi is not None else None
                                act_list_for_pose.append({"track_id": tr.track_id, "box": tb, "keypoints": tr_kp})

                                # Watchlist ArcFace check every 8th frame for unnamed active tracks
                                if (
                                    match_vault
                                    and live_face is not None
                                    and live_face.available
                                    and tr.track_id not in live_vault_names
                                    and f_i % 8 == 1
                                ):
                                    hb = compute_head_box_above_shoulders(tb, tr_kp if tr_kp is not None else np.zeros((17, 3), dtype=np.float32), fr_l.shape[:2])
                                    f_obs = live_face.extract_from_head_box(fr_l, hb)
                                    if f_obs is not None:
                                        vm = vault.match_track_to_vault({"face_arcface": f_obs[0]}, face_thresh=0.42)
                                        if vm is not None:
                                            live_vault_names[tr.track_id] = vm[1] or f"Vault #{vm[0]}"
                                            active_alerts_live.append(f"WATCHLIST MATCH: {live_vault_names[tr.track_id]} (#{tr.track_id}, {vm[2]:.2f})")

                            live_pose_history.append({"frame_idx": f_i, "active": act_list_for_pose})
                            if len(live_pose_history) > 25:
                                live_pose_history = live_pose_history[-25:]
                            p_now = live_pose_analyzer.analyze_timeline(live_pose_history, src_fps=max(5.0, live_fps_ema))
                            f_lbls = p_now["frame_labels"].get(f_i, {})

                            vis_l = fr_l.copy()
                            for tr in l_res.active_tracks:
                                tb = tr.current_box()
                                x1, y1, x2, y2 = [int(round(v)) for v in tb]
                                col = track_color_bgr(tr.track_id)
                                cv2.rectangle(vis_l, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                                vtag = f" [{live_vault_names[tr.track_id]}]" if tr.track_id in live_vault_names else ""
                                blbl = f_lbls.get(tr.track_id, "")
                                btag = f" | {blbl}" if blbl in ("FALL_ALERT", "HANDS_RAISED", "RUNNING") else ""
                                if blbl in ("FALL_ALERT", "HANDS_RAISED", "RUNNING"):
                                    active_alerts_live.append(f"{blbl} on Person #{tr.track_id}{vtag}")
                                draw_label_pill(vis_l, f"LIVE #{tr.track_id}{vtag}{btag}", (x1, max(20, y1)), col, (15, 15, 15), font_scale=0.48)

                            dt_l = max(1e-4, time.perf_counter() - t_l0)
                            live_fps_ema = 0.85 * live_fps_ema + 0.15 * (1.0 / dt_l)
                            live_status.markdown(
                                f"**LIVE STREAM** · Frame `{f_i}/{live_max_frames}` · "
                                f"**FPS:** `{live_fps_ema:.1f}` · **Active Tracks:** `{len(l_res.active_tracks)}`"
                            )
                            if active_alerts_live:
                                live_alert_box.error(" | ".join(active_alerts_live))
                            live_frame_slot.image(cv2.cvtColor(vis_l, cv2.COLOR_BGR2RGB), width="stretch")
                    finally:
                        cap_live.release()
                        st.session_state["live_monitor_running"] = False


if __name__ == "__main__":
    main()
