"""Main Long-Horizon Tracker Pipeline + Explainable Diagnostics Overlay.

Runs object detection, appearance Re-ID embedding extraction, monocular depth
estimation, and the Long-Horizon Tracking Engine over a video, rendering an
explainable diagnostics video and writing a structured JSONL event audit log.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracker import (
    ActiveTrack,
    Detection,
    DiagnosticEvent,
    GhostConfig,
    GhostState,
    LongHorizonTracker,
    ProbationaryTrack,
    TrackerConfig,
    TrackingFrameResult,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def track_color_bgr(track_id: int) -> Tuple[int, int, int]:
    """Deterministic, high-contrast BGR color per track ID via golden-ratio hue spacing."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 180)
    hsv = np.uint8([[[hue, 225, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def extract_box_depth(depth_map: Optional[np.ndarray], box: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    """Robust central-core median depth in [0, 1] (1.0 = closest to camera)."""
    h, w = frame_shape[:2]
    if depth_map is None:
        # Ground-plane fallback prior: bottom of box closer to bottom of image => closer
        return float(np.clip(box[3] / max(1.0, float(h)), 0.0, 1.0))

    dm_h, dm_w = depth_map.shape[:2]
    sx = dm_w / max(1.0, float(w))
    sy = dm_h / max(1.0, float(h))

    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    x1 = int(np.clip((box[0] + 0.25 * bw) * sx, 0, dm_w - 1))
    x2 = int(np.clip((box[2] - 0.25 * bw) * sx, x1 + 1, dm_w))
    y1 = int(np.clip((box[1] + 0.20 * bh) * sy, 0, dm_h - 1))
    y2 = int(np.clip((box[3] - 0.25 * bh) * sy, y1 + 1, dm_h))

    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.5
    return float(np.clip(np.median(patch), 0.0, 1.0))


def draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    """Draw a crisp dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    step = dash_len + gap_len
    for (ax, ay), (bx, by) in edges:
        dist = math.hypot(bx - ax, by - ay)
        if dist < 1:
            continue
        dx = (bx - ax) / dist
        dy = (by - ay) / dist
        pos = 0.0
        while pos < dist:
            end = min(pos + dash_len, dist)
            p_start = (int(round(ax + dx * pos)), int(round(ay + dy * pos)))
            p_end = (int(round(ax + dx * end)), int(round(ay + dy * end)))
            cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)
            pos += step


def draw_label_pill(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.46,
    thickness: int = 1,
    alpha: float = 0.82,
) -> None:
    """Render a readable semi-transparent pill badge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 5, 3
    x, y = origin
    h, w = img.shape[:2]
    x1 = max(0, min(x, w - tw - 2 * pad_x - 1))
    y1 = max(0, min(y - th - 2 * pad_y, h - th - 2 * pad_y - 1))
    x2 = min(w - 1, x1 + tw + 2 * pad_x)
    y2 = min(h - 1, y1 + th + 2 * pad_y + baseline)

    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    rect = np.full_like(roi, bg_color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0.0, roi)
    cv2.putText(
        img,
        text,
        (x1 + pad_x, y2 - pad_y - baseline + 1),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Pluggable Front-End Backends (Detector, Re-ID Encoder, Depth Estimator)
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YOLODetector:
    """Ultralytics YOLO object detector wrapper."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        classes: Optional[Sequence[int]] = (0,),
        conf: float = 0.35,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.classes = list(classes) if classes is not None else None
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes_xyxy [N,4], scores [N], class_ids [N])."""
        results = self.model.predict(
            frame_bgr,
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
            )
        b = results[0].boxes
        boxes = b.xyxy.cpu().numpy().astype(np.float32)
        scores = b.conf.cpu().numpy().astype(np.float32)
        cls_ids = b.cls.cpu().numpy().astype(np.int32)
        return boxes, scores, cls_ids


def upload_frame_gpu(frame_bgr: np.ndarray, device: str) -> Any:
    """Upload BGR uint8 HxWx3 frame once to a (1, 3, H, W) RGB float32 [0, 1] tensor."""
    import torch

    t = torch.from_numpy(frame_bgr).to(device, non_blocking=(device == "cuda"))
    # BGR -> RGB, HWC -> NCHW, scale to [0, 1] on device
    return t.flip(-1).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)


def warp_depth_map(
    depth_map: np.ndarray,
    cmc_matrix: np.ndarray,
    full_frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp cached depth map forward using the frame-to-frame CMC affine matrix."""
    dm_h, dm_w = depth_map.shape[:2]
    full_h, full_w = full_frame_shape[:2]
    M_dm = cmc_matrix.astype(np.float32).copy()
    M_dm[0, 2] *= dm_w / max(1.0, float(full_w))
    M_dm[1, 2] *= dm_h / max(1.0, float(full_h))
    return cv2.warpAffine(
        depth_map,
        M_dm,
        (dm_w, dm_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def get_ffmpeg_binary() -> Optional[str]:
    """Locate standalone ffmpeg via imageio-ffmpeg first, falling back to system PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def transcode_to_h264(video_path: Path | str, crf: int = 23) -> bool:
    """Transcode an OpenCV mp4v file in-place to browser-compatible H.264 (yuv420p + faststart)."""
    src = Path(video_path)
    if not src.exists():
        return False
    ffmpeg_exe = get_ffmpeg_binary()
    if ffmpeg_exe is None:
        print(f"[H264] ffmpeg not found; keeping original codec for {src}")
        return False

    tmp = src.with_name(f"{src.stem}.h264_tmp.mp4")
    try:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-an",
                str(tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            os.replace(str(tmp), str(src))
            return True
    except Exception as exc:
        print(f"[H264] Transcode failed for {src} ({exc}); keeping original file")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return False


class AsyncVideoWriter:
    """Background producer-consumer video writer thread with automatic H.264 transcode on close."""

    def __init__(
        self,
        out_path: str,
        fourcc: int,
        fps: float,
        frame_size: Tuple[int, int],
        max_queue: int = 32,
        transcode_h264: bool = True,
    ) -> None:
        self._out_path = Path(out_path)
        self._transcode_h264 = transcode_h264
        self._writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue)
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._worker, name="AsyncVideoWriter", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                self._writer.write(item)
                self._queue.task_done()
        except BaseException as exc:  # pragma: no cover
            self._error = exc

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(frame_bgr)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._writer.release()
        if self._error is not None:
            raise self._error
        if self._transcode_h264:
            transcode_to_h264(self._out_path)


class AppearanceEncoder:
    """Batched GPU appearance Re-ID extractor using roi_align + FP16 autocast."""

    def __init__(self, backend: str = "mobilenet", device: str = "cpu") -> None:
        self.backend = backend
        self.device = device
        self._model = None
        self._torch = None
        self._roi_align = None
        self._mean = None
        self._std = None
        self._out_hw: Tuple[int, int] = (224, 224) if backend == "dinov2" else (224, 112)

        if backend != "hsv_hog":
            self._init_deep_backend(backend)

    def _init_deep_backend(self, backend: str) -> None:
        try:
            import torch
            import torchvision.models as models
            from torchvision.ops import roi_align

            self._torch = torch
            self._roi_align = roi_align
            if backend == "resnet50":
                weights = models.ResNet50_Weights.DEFAULT
                net = models.resnet50(weights=weights)
                net.fc = torch.nn.Identity()
                self._out_hw = (224, 112)
            elif backend == "dinov2":
                try:
                    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
                except Exception:
                    import timm

                    net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
                self._out_hw = (224, 224)
            else:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
                net = models.mobilenet_v3_large(weights=weights)
                net.classifier = torch.nn.Identity()
                self.backend = "mobilenet"
                self._out_hw = (224, 112)

            net.eval().to(self.device)
            self._model = net
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        except Exception as exc:
            print(f"[AppearanceEncoder] Falling back to hsv_hog ({exc})")
            self.backend = "hsv_hog"
            self._model = None

    @staticmethod
    def _hsv_hog_embedding(crop_bgr: np.ndarray) -> np.ndarray:
        """Fast spatial upper/lower HSV color + gradient orientation descriptor (256D)."""
        resized = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        parts = np.array_split(hsv, 4, axis=0)
        feats: List[np.ndarray] = []
        for part in parts:
            h_hist = cv2.calcHist([part], [0], None, [24], [0, 180]).flatten()
            s_hist = cv2.calcHist([part], [1], None, [16], [0, 256]).flatten()
            v_hist = cv2.calcHist([part], [2], None, [16], [0, 256]).flatten()
            strip = np.concatenate([h_hist, s_hist, v_hist])
            strip /= max(1e-6, float(np.linalg.norm(strip)))
            feats.append(strip)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ori_hist, _ = np.histogram(ang, bins=32, range=(0, 360), weights=mag)
        ori_hist = ori_hist.astype(np.float32)
        ori_hist /= max(1e-6, float(np.linalg.norm(ori_hist)))
        feats.append(ori_hist)

        vec = np.concatenate(feats).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-8)

    def extract(
        self,
        frame_bgr: np.ndarray,
        boxes: np.ndarray,
        frame_gpu: Optional[Any] = None,
    ) -> np.ndarray:
        """Return (N, D) L2-normalized float32 embeddings using batched GPU roi_align + FP16."""
        if len(boxes) == 0:
            return np.zeros((0, 256), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        if self._model is None:
            crops: List[np.ndarray] = []
            for box in boxes:
                x1 = max(0, int(np.floor(box[0])))
                y1 = max(0, int(np.floor(box[1])))
                x2 = min(w, int(np.ceil(box[2])))
                y2 = min(h, int(np.ceil(box[3])))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    crops.append(np.zeros((32, 16, 3), dtype=np.uint8))
                else:
                    crops.append(frame_bgr[y1:y2, x1:x2])
            return np.stack([self._hsv_hog_embedding(c) for c in crops], axis=0)

        torch = self._torch
        use_cuda = self.device == "cuda"

        # Build clamped (N, 5) ROIs [batch_idx=0, x1, y1, x2, y2] for single-kernel GPU cropping
        b_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b_arr[:, 0] = np.clip(b_arr[:, 0], 0.0, max(0.0, float(w) - 2.0))
        b_arr[:, 1] = np.clip(b_arr[:, 1], 0.0, max(0.0, float(h) - 2.0))
        b_arr[:, 2] = np.clip(b_arr[:, 2], b_arr[:, 0] + 1.0, float(w))
        b_arr[:, 3] = np.clip(b_arr[:, 3], b_arr[:, 1] + 1.0, float(h))

        rois_np = np.zeros((len(b_arr), 5), dtype=np.float32)
        rois_np[:, 1:] = b_arr

        with torch.inference_mode():
            img_t = frame_gpu if frame_gpu is not None else upload_frame_gpu(frame_bgr, self.device)
            rois_t = torch.from_numpy(rois_np).to(self.device, non_blocking=use_cuda)
            batch = self._roi_align(
                img_t,
                rois_t,
                output_size=self._out_hw,
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            batch.sub_(self._mean).div_(self._std)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)


class DepthEstimator:
    """Monocular depth estimator producing [0, 1] inverse depth (1.0 = closest) with CUDA FP16."""

    def __init__(self, backend: str = "midas_small", device: str = "cpu") -> None:
        self.backend = backend
        self.device = device
        self._model = None
        self._transform = None
        self._processor = None
        self._torch = None
        self._mean = None
        self._std = None

        if backend != "y_prior":
            self._init_model(backend)

    def _init_model(self, backend: str) -> None:
        try:
            import torch

            self._torch = torch
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
            if backend == "depth_anything":
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                model_id = "depth-anything/Depth-Anything-V2-Small-hf"
                self._processor = AutoImageProcessor.from_pretrained(model_id)
                self._model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
                return

            # Default: MiDaS_small via torch.hub (fast & reliable)
            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self._model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = midas_transforms.small_transform
            self.backend = "midas_small"
        except Exception as exc:
            print(f"[DepthEstimator] Falling back to y_prior ({exc})")
            self.backend = "y_prior"
            self._model = None

    @staticmethod
    def _normalize_depth_tensor(pred: Any) -> np.ndarray:
        """Robust GPU percentile normalization [2%, 98%] -> [0, 1] float32 numpy array."""
        import torch

        pred_f = pred.float()
        sample = pred_f[::2, ::2].reshape(-1)
        q = torch.quantile(sample, torch.tensor([0.02, 0.98], device=pred_f.device))
        lo, hi = q[0], q[1]
        span = hi - lo
        if float(span) < 1e-5:
            return np.full(tuple(pred_f.shape), 0.5, dtype=np.float32)
        norm = torch.clamp((pred_f - lo) / span, 0.0, 1.0)
        return norm.cpu().numpy().astype(np.float32)

    @staticmethod
    def _depth_anything_size(h: int, w: int, base: int = 518) -> Tuple[int, int]:
        """Match Depth-Anything-V2 aspect-preserving multiple-of-14 input size."""
        scale = base / float(min(h, w))
        nh = max(14, int(round((h * scale) / 14.0)) * 14)
        nw = max(14, int(round((w * scale) / 14.0)) * 14)
        return nh, nw

    def predict(
        self,
        frame_bgr: np.ndarray,
        frame_gpu: Optional[Any] = None,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._model is None:
            y_grad = np.linspace(0.05, 0.95, h, dtype=np.float32)[:, None]
            return np.broadcast_to(y_grad, (h, w)).copy()

        torch = self._torch
        use_cuda = self.device == "cuda"

        with torch.inference_mode():
            if self.backend == "depth_anything":
                img_t = frame_gpu if frame_gpu is not None else upload_frame_gpu(frame_bgr, self.device)
                th, tw = self._depth_anything_size(h, w, base=518)
                inp = torch.nn.functional.interpolate(
                    img_t, size=(th, tw), mode="bilinear", align_corners=False
                )
                inp = (inp - self._mean) / self._std
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    pred = self._model(pixel_values=inp).predicted_depth.squeeze(0)
                return self._normalize_depth_tensor(pred)

            # MiDaS_small
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self.device, non_blocking=use_cuda)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                pred = self._model(inp).squeeze(0)
            return self._normalize_depth_tensor(pred)


# ---------------------------------------------------------------------------
# Explainable Diagnostics Overlay Renderer
# ---------------------------------------------------------------------------


EVENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "STITCH_COMMIT": (255, 240, 60),     # Bright cyan-gold
    "MARGIN_VETO": (160, 40, 255),       # Vivid magenta-red
    "GHOST_SPAWN": (255, 130, 180),      # Soft violet-pink
    "DEPTH_OCCLUDED": (0, 140, 255),     # Bright orange
    "DEPTH_GATE_REJECT": (50, 220, 255), # Bright yellow
    "NEW_ID_AFTER_VETO": (80, 140, 255), # Orange
    "GHOST_EXPIRED": (160, 160, 160),    # Gray
    "NEW_TRACK": (120, 235, 120),        # Soft green
}


class DiagnosticsOverlay:
    """Renders ghosts, covariance ellipses, probation links, HUD telemetry, and event ticker."""

    def __init__(
        self,
        max_ticker_events: int = 6,
        show_depth_inset: bool = True,
        probation_frames: int = 15,
    ) -> None:
        self.ticker: Deque[DiagnosticEvent] = deque(maxlen=max_ticker_events)
        self.show_depth_inset = show_depth_inset
        self.probation_frames = max(15, int(probation_frames))

    def render(
        self,
        frame_bgr: np.ndarray,
        result: TrackingFrameResult,
        depth_map: Optional[np.ndarray],
        fps: float,
        backend_info: str = "",
        in_place: bool = False,
    ) -> np.ndarray:
        canvas = frame_bgr if in_place else frame_bgr.copy()
        h, w = canvas.shape[:2]

        # Push notable events into persistent ticker
        for ev in result.events:
            if ev.event_type in (
                "STITCH_COMMIT",
                "MARGIN_VETO",
                "GHOST_SPAWN",
                "DEPTH_OCCLUDED",
                "DEPTH_GATE_REJECT",
                "NEW_ID_AFTER_VETO",
                "GHOST_EXPIRED",
            ):
                self.ticker.append(ev)

        # 1. Render Ghost Particle Clouds & Covariance Ellipses (behind active boxes)
        self._draw_ghosts(canvas, result.ghosts)

        # 2. Render Active Tracks (trails, boxes, depth, vault meter, clean/occluded badge)
        self._draw_active_tracks(canvas, result.active_tracks)

        # 3. Render Probationary Tracks + Ghost Candidate Links + Margin-Ratio Readout
        ghost_map = {g.track_id: g for g in result.ghosts}
        self._draw_probationary_tracks(canvas, result.probationary_tracks, ghost_map)

        # 4. Top-Left Telemetry HUD
        self._draw_hud(canvas, result, fps, backend_info)

        # 5. Top-Right Depth Map Picture-in-Picture
        if self.show_depth_inset and depth_map is not None:
            self._draw_depth_inset(canvas, depth_map)

        # 6. Bottom-Left Explainable Event Ticker
        self._draw_event_ticker(canvas)

        return canvas

    def _draw_ghosts(self, canvas: np.ndarray, ghosts: List[GhostState]) -> None:
        """Render each ghost cloud with tight ROI-scoped alpha blending (no full-frame copy)."""
        if not ghosts:
            return
        h, w = canvas.shape[:2]

        for g in ghosts:
            color = track_color_bgr(g.track_id)
            p = g.particles
            weights = g.weights
            w_max = max(1e-6, float(np.max(weights)))

            cx = int(round(0.5 * (g.mean_box[0] + g.mean_box[2])))
            cy = int(round(0.5 * (g.mean_box[1] + g.mean_box[3])))
            eigvals, eigvecs = np.linalg.eigh(g.cov_xy)
            eigvals = np.maximum(eigvals, 16.0)
            # Cap drawn ellipse radii to <= 90px (internal particle corridor stays full-scale)
            max_ell = min(90, w // 2, h // 2)
            ax1 = int(np.clip(1.5 * math.sqrt(float(eigvals[1])), 8, max_ell))
            ax2 = int(np.clip(1.5 * math.sqrt(float(eigvals[0])), 8, max_ell))
            angle = float(math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1])))
            ell_r = max(ax1, ax2)

            x1, y1, x2, y2 = [int(round(v)) for v in g.mean_box]

            # Compact visual ROI around the capped ghost ellipse & mean box (<= 90px radius)
            vis_r = max(ell_r, 90)
            pad = 8
            rx1 = max(0, min(cx - vis_r, x1) - pad)
            ry1 = max(0, min(cy - vis_r, y1) - pad)
            rx2 = min(w, max(cx + vis_r, x2) + pad)
            ry2 = min(h, max(cy + vis_r, y2) + pad)

            if rx2 > rx1 and ry2 > ry1:
                roi = canvas[ry1:ry2, rx1:rx2]
                roi_overlay = roi.copy()
                rw = rx2 - rx1
                rh = ry2 - ry1

                # Draw particle cloud in ROI-local coordinates
                px_arr = np.rint(p[:, 0]).astype(np.int32) - rx1
                py_arr = np.rint(p[:, 1]).astype(np.int32) - ry1
                in_roi = (px_arr >= 0) & (px_arr < rw) & (py_arr >= 0) & (py_arr < rh)
                half_w_max = 0.5 * w_max
                for idx in np.flatnonzero(in_roi):
                    radius = 2 if weights[idx] < half_w_max else 3
                    cv2.circle(roi_overlay, (int(px_arr[idx]), int(py_arr[idx])), radius, color, -1, cv2.LINE_AA)

                # Draw 1.5-sigma covariance ellipse in ROI-local coordinates
                cv2.ellipse(
                    roi_overlay,
                    (cx - rx1, cy - ry1),
                    (ax1, ax2),
                    angle,
                    0,
                    360,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # Dashed mean box in ROI-local coordinates
                draw_dashed_rect(
                    roi_overlay,
                    (x1 - rx1, y1 - ry1),
                    (x2 - rx1, y2 - ry1),
                    color,
                    thickness=1,
                    dash_len=6,
                    gap_len=5,
                )

                cv2.addWeighted(roi_overlay, 0.65, roi, 0.35, 0.0, roi)

            occ_pct = int(round(100 * g.occluded_fraction))
            label = f"GHOST #{g.track_id} | {g.age}f | occ:{occ_pct}% | c:{g.confidence:.2f}"
            draw_label_pill(canvas, label, (x1, max(18, y1 - 2)), (45, 25, 65), color, font_scale=0.42)

    def _draw_active_tracks(self, canvas: np.ndarray, tracks: List[ActiveTrack]) -> None:
        for tr in tracks:
            color = track_color_bgr(tr.track_id)
            box = tr.current_box()
            x1, y1, x2, y2 = [int(round(v)) for v in box]

            # Draw trajectory trail
            if len(tr.history) >= 2:
                pts = list(tr.history)
                for i in range(1, len(pts)):
                    p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                    p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                    thick = 1 if i < len(pts) // 2 else 2
                    cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)

            # Bounding box (dashed if currently missing a frame, orange border if occluded)
            box_color = (0, 165, 255) if tr.is_occluded else color
            if tr.time_since_update > 0:
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), box_color, thickness=2)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            stitch_tag = f" [S{tr.stitched_count}]" if tr.stitched_count > 0 else ""
            header = f"#{tr.track_id}{stitch_tag} | d:{tr.depth_ema:.2f} | V:{len(tr.vault)}/{tr.vault.max_entries}"
            draw_label_pill(canvas, header, (x1, max(20, y1)), box_color, (15, 15, 15), font_scale=0.47, thickness=1)

            # Clean / Depth-Occluded vault state chip at bottom-left of box
            if tr.is_depth_occluded or tr.status == "DEPTH-OCCLUDED":
                badge = (
                    f"DEPTH-OCCLUDED (by #{tr.occluder_id}) | VAULT FROZEN"
                    if tr.occluder_id is not None
                    else "DEPTH-OCCLUDED | VAULT FROZEN"
                )
                draw_label_pill(canvas, badge, (x1, y2 + 16), (0, 75, 220), (255, 255, 255), font_scale=0.38)
            elif tr.is_occluded:
                draw_label_pill(canvas, "OVERLAP | VAULT LOCKED", (x1, y2 + 16), (0, 110, 200), (255, 255, 255), font_scale=0.38)
            elif tr.last_clean_reason in ("VAULT_STORE", "VAULT_STORE_EVICT"):
                draw_label_pill(canvas, "CLEAN -> VAULT+", (x1, y2 + 16), (30, 145, 50), (255, 255, 255), font_scale=0.38)

    def _draw_probationary_tracks(
        self,
        canvas: np.ndarray,
        prob_tracks: List[ProbationaryTrack],
        ghost_map: Dict[int, GhostState],
    ) -> None:
        amber = (40, 215, 255)
        veto_col = (180, 40, 255)

        for pr in prob_tracks:
            box = pr.current_box()
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            col = veto_col if pr.last_veto_reason else amber
            draw_dashed_rect(canvas, (x1, y1), (x2, y2), col, thickness=2, dash_len=7, gap_len=5)

            label = f"VERIFYING ({pr.hits}/{self.probation_frames})"
            draw_label_pill(canvas, label, (x1, max(20, y1)), (30, 30, 30), col, font_scale=0.42)

            if pr.last_veto_reason:
                draw_label_pill(canvas, pr.last_veto_reason, (x1, y2 + 16), veto_col, (255, 255, 255), font_scale=0.38)

            # Draw link line to candidate Ghost center
            if pr.best_candidate_ghost is not None and pr.best_candidate_ghost in ghost_map:
                g = ghost_map[pr.best_candidate_ghost]
                p_ctr = (int(round(0.5 * (box[0] + box[2]))), int(round(0.5 * (box[1] + box[3]))))
                g_ctr = (
                    int(round(0.5 * (g.mean_box[0] + g.mean_box[2]))),
                    int(round(0.5 * (g.mean_box[1] + g.mean_box[3]))),
                )
                cv2.line(canvas, p_ctr, g_ctr, col, 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        result: TrackingFrameResult,
        fps: float,
        backend_info: str,
    ) -> None:
        panel_w, panel_h = 425, 118
        x0, y0 = 12, 12
        h, w = canvas.shape[:2]
        x1, y1 = min(w - 1, x0 + panel_w), min(h - 1, y0 + panel_h)
        roi = canvas[y0:y1, x0:x1]
        bg = np.full_like(roi, (18, 20, 26), dtype=np.uint8)
        cv2.addWeighted(bg, 0.78, roi, 0.22, 0.0, roi)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (85, 100, 125), 1, cv2.LINE_AA)

        cmc = result.cmc
        st = result.stats
        cmc_txt = (
            "CMC: FALLBACK (identity)"
            if cmc.used_fallback
            else f"CMC: dx={cmc.dx:+.1f} dy={cmc.dy:+.1f} rot={cmc.dtheta_deg:+.2f}d inl={cmc.inlier_ratio*100:.0f}%"
        )
        lines = [
            (f"LONG-HORIZON TRACKER // {backend_info}", (255, 225, 110), 0.46),
            (f"Frame: {result.frame_idx:04d}  |  FPS: {fps:5.1f}  |  Dets: {st['detections_count']}", (235, 240, 245), 0.44),
            (cmc_txt, (120, 200, 255) if not cmc.used_fallback else (90, 160, 255), 0.42),
            (f"Active: {st['active_count']}  |  Prob: {st['probationary_count']}  |  Ghosts: {st['ghost_count']}", (185, 245, 185), 0.42),
            (
                f"Stitches: {st['stitches']}  |  Vetoes: {st['margin_vetoes']}  |  DepthGates: {st['depth_gate_rejects']}  |  Vault+: {st['vault_stores']}",
                (230, 210, 255),
                0.40,
            ),
        ]
        y_cursor = y0 + 21
        for text, col, scale in lines:
            cv2.putText(canvas, text, (x0 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
            y_cursor += 21

    def _draw_depth_inset(self, canvas: np.ndarray, depth_map: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        inset_w = min(220, max(120, w // 5))
        inset_h = int(round(inset_w * (h / max(1, w))))
        x1 = w - inset_w - 12
        y1 = 12
        if x1 < 445 or y1 + inset_h >= h:
            return

        d_u8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
        d_col = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
        d_small = cv2.resize(d_col, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        canvas[y1 : y1 + inset_h, x1 : x1 + inset_w] = d_small
        cv2.rectangle(canvas, (x1, y1), (x1 + inset_w, y1 + inset_h), (180, 180, 195), 1, cv2.LINE_AA)
        draw_label_pill(canvas, "DEPTH (BRIGHT=NEAR)", (x1 + 4, y1 + 18), (20, 20, 20), (240, 240, 240), font_scale=0.36)

    def _draw_event_ticker(self, canvas: np.ndarray) -> None:
        if not self.ticker:
            return
        h, w = canvas.shape[:2]
        line_h = 19
        pad = 8
        panel_h = pad * 2 + line_h * len(self.ticker) + 16
        panel_w = min(w - 24, 590)
        x0 = 12
        y0 = max(140, h - panel_h - 12)
        x1 = x0 + panel_w
        y1 = min(h - 1, y0 + panel_h)

        roi = canvas[y0:y1, x0:x1]
        bg = np.full_like(roi, (14, 16, 22), dtype=np.uint8)
        cv2.addWeighted(bg, 0.74, roi, 0.26, 0.0, roi)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (75, 85, 105), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "EXPLAINABLE DECISION LOG",
            (x0 + 8, y0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (180, 195, 215),
            1,
            cv2.LINE_AA,
        )
        y_cursor = y0 + 33
        for ev in self.ticker:
            col = EVENT_COLORS.get(ev.event_type, (220, 220, 220))
            txt = f"[F{ev.frame_idx:04d} {ev.event_type}] {ev.message}"
            if len(txt) > 84:
                txt = txt[:81] + "..."
            cv2.putText(canvas, txt, (x0 + 8, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1, cv2.LINE_AA)
            y_cursor += line_h


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-Horizon Occlusion-Resilient Tracker with Explainable Diagnostics"
    )
    parser.add_argument("--video", type=str, default="data/test_video.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="output/tracked_diagnostics.mp4", help="Output video path")
    parser.add_argument("--log", type=str, default="output/events.jsonl", help="Structured JSONL event log path")
    parser.add_argument("--detector", type=str, default="yolov8n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="COCO class IDs to track (0=person)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument(
        "--reid",
        type=str,
        default="mobilenet",
        choices=["mobilenet", "resnet50", "dinov2", "hsv_hog"],
        help="Appearance Re-ID backbone",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="midas_small",
        choices=["midas_small", "depth_anything", "y_prior"],
        help="Monocular depth estimation backend",
    )
    parser.add_argument("--depth-stride", type=int, default=2, help="Run depth model every K frames")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cuda, cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    parser.add_argument("--max-ghost-age", type=int, default=600, help="Maximum frames to keep a Ghost alive")
    parser.add_argument("--probation-frames", type=int, default=15, help="Minimum probation frames before stitching (>=15)")
    parser.add_argument("--no-depth-inset", action="store_true", help="Disable depth PiP overlay")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[Pipeline] Device={device} | Video={video_path}")

    detector = YOLODetector(
        model_name=args.detector,
        classes=args.classes if args.classes else None,
        conf=args.conf,
        device=device,
    )
    encoder = AppearanceEncoder(backend=args.reid, device=device)
    depth_est = DepthEstimator(backend=args.depth, device=device)

    tracker_cfg = TrackerConfig(
        det_thresh=args.conf,
        probation_frames=args.probation_frames,
        ghost_config=GhostConfig(max_ghost_age=args.max_ghost_age),
    )
    tracker = LongHorizonTracker(config=tracker_cfg)
    overlay = DiagnosticsOverlay(
        show_depth_inset=not args.no_depth_inset,
        probation_frames=tracker_cfg.probation_frames,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height} @ {src_fps:.2f} FPS ({total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(str(out_path), fourcc, src_fps, (width, height), max_queue=32)
    backend_info = f"{device.upper()} FP16 | {encoder.backend} | {depth_est.backend} (s={max(1, args.depth_stride)})"

    depth_stride = max(1, args.depth_stride)
    use_gpu_frame = device == "cuda" and (encoder._model is not None or depth_est._model is not None)
    last_depth_map: Optional[np.ndarray] = None
    frame_count = 0
    t_start = time.perf_counter()
    fps_ema = src_fps

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if args.max_frames > 0 and frame_count > args.max_frames:
                break

            t0 = time.perf_counter()

            # Single GPU upload shared by DepthEstimator and AppearanceEncoder
            frame_gpu = upload_frame_gpu(frame, device) if use_gpu_frame else None

            # 1. Object Detection (YOLO FP16 on CUDA)
            boxes, scores, cls_ids = detector.detect(frame)

            # 2. Monocular Depth: run model every `depth_stride` frames, else warp cached depth via CMC
            run_depth_now = last_depth_map is None or (frame_count - 1) % depth_stride == 0
            cmc_res = None
            if run_depth_now:
                last_depth_map = depth_est.predict(frame, frame_gpu=frame_gpu)
            else:
                cmc_res = tracker.cmc.update(frame, boxes)
                last_depth_map = warp_depth_map(last_depth_map, cmc_res.matrix, frame.shape[:2])

            # 3. Batched GPU Appearance Re-ID Embeddings (roi_align + FP16)
            embeddings = encoder.extract(frame, boxes, frame_gpu=frame_gpu)

            # 4. Assemble Detections with Central-Core Depth
            detections: List[Detection] = []
            for i in range(len(boxes)):
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

            # 5. Long-Horizon Tracker Step (reuses cmc_res if already computed for depth warping)
            result = tracker.update(frame, detections, depth_map=last_depth_map, cmc_result=cmc_res)

            dt = max(1e-4, time.perf_counter() - t0)
            fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt)

            # 6. Write JSONL diagnostic events
            for ev in result.events:
                rec = {
                    "frame": ev.frame_idx,
                    "event_type": ev.event_type,
                    "track_id": ev.track_id,
                    "message": ev.message,
                    "details": ev.details,
                }
                log_file.write(json.dumps(rec) + "\n")

            # 7. Render Explainable Diagnostics Overlay in-place & enqueue to background writer
            vis = overlay.render(
                frame,
                result,
                last_depth_map,
                fps_ema,
                backend_info=backend_info,
                in_place=True,
            )
            writer.write(vis)

            if args.show:
                cv2.imshow("Long-Horizon Tracker Diagnostics", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 50 == 0 or frame_count == 1:
                st = result.stats
                print(
                    f"[F{frame_count:04d}/{total_frames}] FPS={fps_ema:5.1f} | "
                    f"Act={st['active_count']} Prob={st['probationary_count']} Ghost={st['ghost_count']} | "
                    f"Stitches={st['stitches']} Vetoes={st['margin_vetoes']} DepthGates={st['depth_gate_rejects']}"
                )

    cap.release()
    writer.close()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    summary = {
        "video": str(video_path),
        "output": str(out_path),
        "frames_processed": frame_count,
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(frame_count / max(1e-3, elapsed), 2),
        "device": device,
        "reid_backend": encoder.backend,
        "depth_backend": depth_est.backend,
        "counters": tracker.counters,
    }
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Done] Processed {frame_count} frames in {elapsed:.1f}s ({summary['avg_fps']} FPS)")
    print(f"[Outputs] Video: {out_path} | Log: {log_path} | Summary: {summary_path}")


if __name__ == "__main__":
    main()
