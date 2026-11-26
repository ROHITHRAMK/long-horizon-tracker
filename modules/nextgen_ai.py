"""2026 Next-Gen AI Model Drop-Ins, Mask-Clean Cropping, Natural-Language CLIP Search & TensorRT Export.

Designed as clean, `@st.cache_resource`-friendly modular drop-ins with automatic graceful
fallbacks and zero modifications to `tracker/engine.py` or `tests/test_invariants.py`.

Components:
1. 2026 Model Catalog & Safe Loaders:
   - `NextGenPoseDetector`: YOLO26-Pose (`yolo26l/m/n-pose.pt`), YOLO11-Pose (`yolo11l/m/x/n-pose.pt`),
     and optional `RF-DETR` (`rfdetr` package), with automatic fallback to `yolo11l-pose.pt`.
   - `MaskCleaner`: optional `yolo26m-seg.pt` (fallback `yolo11m-seg.pt`) person segmentation
     neutralizing background pixels with ImageNet mean gray BGR `(104, 116, 124)`.
   - `NextGenReIDEncoder`: `dinov2_vitb14` (default, zero-auth) and `DINOv3`
     (`facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`),
     automatically falling back to `dinov2_vitb14` if HF auth/weights are unavailable.
   - `NextGenDepthEstimator`: `Depth Anything V2` (default), `YOLO26-Depth` (`yolo26n/m-depth.pt`),
     and `Depth Anything V3` (`depth-anything/DA3-SMALL`), falling back cleanly to Depth Anything V2.
2. `CLIPTargetSearcher`: FP16 `openai/clip-vit-base-patch32` natural-language search over top-3
   clearest full-body crops per confirmed `#ID`.
3. `export_to_tensorrt`: one-click NVIDIA TensorRT FP16 `.engine` exporter with clear diagnostics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from run_tracker import DepthEstimator, resolve_device, upload_frame_gpu

try:
    import streamlit as _st

    _cache_resource: Callable[..., Any] = _st.cache_resource
except Exception:  # pragma: no cover

    def _cache_resource(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# 2026 Model Catalog Constants
# ---------------------------------------------------------------------------

# ImageNet mean pixel in BGR order: RGB (0.485, 0.456, 0.406) * 255 = (124, 116, 104) -> BGR (104, 116, 124)
IMAGENET_MEAN_BGR: Tuple[int, int, int] = (104, 116, 124)

DETECTOR_CATALOG: List[Tuple[str, str]] = [
    ("yolo26l-pose.pt — YOLO26-Large Pose (2026)", "yolo26l-pose.pt"),
    ("yolo26m-pose.pt — YOLO26-Medium Pose (2026 Balanced)", "yolo26m-pose.pt"),
    ("yolo26n-pose.pt — YOLO26-Nano Pose (2026 Fast)", "yolo26n-pose.pt"),
    ("yolo11l-pose.pt — YOLO11-Large Pose (Proven)", "yolo11l-pose.pt"),
    ("yolo11m-pose.pt — YOLO11-Medium Pose (Balanced)", "yolo11m-pose.pt"),
    ("yolo11x-pose.pt — YOLO11-XLarge Pose (Max)", "yolo11x-pose.pt"),
    ("rfdetr — RF-DETR Transformer Detector (Optional)", "rfdetr"),
]

REID_CATALOG: List[Tuple[str, str]] = [
    ("dinov2_vitb14 — DINOv2 ViT-B/14 (Default, Zero-Auth)", "dinov2_vitb14"),
    ("dinov2_vits14 — DINOv2 ViT-S/14 (Fast, Zero-Auth)", "dinov2_vits14"),
    (
        "facebook/dinov3-vitb16-pretrain-lvd1689m — DINOv3 ViT-B/16 (2026, HF Token)",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ),
    (
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m — DINOv3 ConvNeXt-Tiny (2026, HF Token)",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ),
]

DEPTH_CATALOG: List[Tuple[str, str]] = [
    ("depth_anything — Depth Anything V2 Small (Default)", "depth_anything"),
    ("yolo26n-depth.pt — YOLO26-Nano Depth (2026 Fast)", "yolo26n-depth.pt"),
    ("yolo26m-depth.pt — YOLO26-Medium Depth (2026 Accurate)", "yolo26m-depth.pt"),
    ("depth-anything/DA3-SMALL — Depth Anything V3 Small (2026 Optional)", "depth-anything/DA3-SMALL"),
    ("midas_small — MiDaS Small", "midas_small"),
    ("y_prior — Fast Geometric Y-Prior (CPU)", "y_prior"),
]


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix [len(a), len(b)] between two xyxy float32 box arrays."""
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32).reshape(na, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(nb, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(1.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    area_b = np.maximum(1.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    union = np.maximum(1e-6, area_a[:, None] + area_b[None, :] - inter)
    return (inter / union).astype(np.float32)


def _laplacian_sharpness(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# 1A. 2026 Pose / Person Detector (YOLO26, YOLO11, Optional RF-DETR)
# ---------------------------------------------------------------------------


class NextGenPoseDetector:
    """Safe 2026 pose/object detector supporting YOLO26-Pose, YOLO11-Pose, TensorRT, & RF-DETR.

    Falls back automatically to `yolo11l-pose.pt` if a requested weight or optional `rfdetr`
    package is unavailable.
    """

    FALLBACK_WEIGHTS: str = "yolo11l-pose.pt"

    def __init__(
        self,
        model_name: str = "yolo11l-pose.pt",
        imgsz: int = 768,
        classes: Optional[Sequence[int]] = (0,),
        conf: float = 0.35,
        device: str = "auto",
    ) -> None:
        self.requested_model = str(model_name)
        self.model_name = str(model_name)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes is not None else None
        self.conf = float(conf)
        self.device = resolve_device(device)
        self._use_half = self.device == "cuda"
        self.used_fallback: bool = False
        self.status_note: str = ""
        self.backend_kind: str = "ultralytics"
        self.is_pose_model: bool = "pose" in self.model_name.lower()
        self.model: Any = None

        if self._use_half:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        self._load_with_fallback(self.requested_model)

    def _load_with_fallback(self, requested: str) -> None:
        req_low = requested.strip().lower()
        # Optional RF-DETR backend
        if req_low.startswith("rfdetr") or req_low == "rf-detr":
            try:
                import rfdetr  # type: ignore[import-not-found]

                det_cls = getattr(rfdetr, "RFDETRBase", None) or getattr(rfdetr, "RFDETRLarge", None)
                if det_cls is None:
                    raise ImportError("rfdetr installed but no RFDETRBase/RFDETRLarge class found.")
                self.model = det_cls()
                self.backend_kind = "rfdetr"
                self.model_name = "rfdetr"
                self.is_pose_model = False
                self.status_note = f"Loaded RF-DETR detector on {self.device.upper()}."
                return
            except Exception as exc:
                self.used_fallback = True
                self.status_note = (
                    f"RF-DETR unavailable ({exc}); automatically fell back to `{self.FALLBACK_WEIGHTS}`."
                )
                requested = self.FALLBACK_WEIGHTS

        # Ultralytics YOLO26 / YOLO11 / TensorRT .engine backend
        try:
            from ultralytics import YOLO

            self.model = YOLO(requested)
            self.backend_kind = "ultralytics"
            self.model_name = requested
            self.is_pose_model = "pose" in requested.lower()
            if not self.status_note:
                self.status_note = f"Loaded `{requested}` on {self.device.upper()}."
            return
        except Exception as exc:
            if requested != self.FALLBACK_WEIGHTS:
                try:
                    from ultralytics import YOLO

                    self.model = YOLO(self.FALLBACK_WEIGHTS)
                    self.backend_kind = "ultralytics"
                    self.model_name = self.FALLBACK_WEIGHTS
                    self.is_pose_model = True
                    self.used_fallback = True
                    self.status_note = (
                        f"`{requested}` unavailable ({exc}); automatically fell back to `{self.FALLBACK_WEIGHTS}`."
                    )
                    return
                except Exception as exc2:
                    self.used_fallback = True
                    self.model = None
                    self.status_note = f"Detector unavailable ({exc2})."
            else:
                self.used_fallback = True
                self.model = None
                self.status_note = f"Detector unavailable ({exc})."

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(boxes_xyxy [N,4], scores [N], class_ids [N], keypoints_xyc [N,17,3])`."""
        empty = (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 17, 3), dtype=np.float32),
        )
        if self.model is None or frame_bgr is None or frame_bgr.size == 0:
            return empty

        if self.backend_kind == "rfdetr":
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                preds = self.model.predict(rgb, threshold=self.conf)
                xyxy = np.asarray(getattr(preds, "xyxy", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
                scores = np.asarray(getattr(preds, "confidence", np.zeros((0,))), dtype=np.float32).reshape(-1)
                cls_ids = np.asarray(getattr(preds, "class_id", np.zeros((0,))), dtype=np.int32).reshape(-1)
                if len(xyxy) == 0:
                    return empty
                if self.classes is not None:
                    # RF-DETR uses COCO 1-indexed (person=1) or 0-indexed (person=0) depending on version
                    allowed = set(int(c) for c in self.classes)
                    if 0 in allowed:
                        allowed.add(1)
                    keep = np.array([int(c) in allowed for c in cls_ids], dtype=bool)
                    xyxy, scores, cls_ids = xyxy[keep], scores[keep], np.zeros(int(np.sum(keep)), dtype=np.int32)
                if len(xyxy) == 0:
                    return empty
                return xyxy, scores, cls_ids, np.zeros((len(xyxy), 17, 3), dtype=np.float32)
            except Exception:
                return empty

        # Ultralytics YOLO26 / YOLO11 / TensorRT path
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
            return empty

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


# ---------------------------------------------------------------------------
# 1B. Mask-Clean Segmentation Helper (`MaskCleaner`)
# ---------------------------------------------------------------------------


class MaskCleaner:
    """Instance-segmentation background neutralizer (`yolo26m-seg.pt`, fallback `yolo11m-seg.pt`).

    Replaces non-person background pixels inside matched person boxes with ImageNet mean gray
    `BGR (104, 116, 124)` so counters, shelves, and neighboring people never contaminate
    DINOv2/DINOv3 or skin tone embeddings.
    """

    PRIMARY_WEIGHTS: str = "yolo26m-seg.pt"
    FALLBACK_WEIGHTS: str = "yolo11m-seg.pt"

    def __init__(
        self,
        model_name: str = "yolo26m-seg.pt",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.30,
        load_model: bool = True,
    ) -> None:
        self.requested_model = str(model_name)
        self.model_name = str(model_name)
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self._use_half = self.device == "cuda"
        self.available: bool = False
        self.used_fallback: bool = False
        self.status_note: str = ""
        self._model: Any = None

        if load_model:
            self._init_model(self.requested_model)
        else:
            self.status_note = "MaskCleaner initialized without weights (manual mask mode)."

    def _init_model(self, requested: str) -> None:
        candidates = [requested]
        if self.FALLBACK_WEIGHTS not in candidates:
            candidates.append(self.FALLBACK_WEIGHTS)

        last_err: Optional[Exception] = None
        for idx, cand in enumerate(candidates):
            try:
                from ultralytics import YOLO

                self._model = YOLO(cand)
                self.model_name = cand
                self.available = True
                self.used_fallback = idx > 0
                if self.used_fallback:
                    self.status_note = (
                        f"`{requested}` unavailable ({last_err}); fell back to `{cand}` on {self.device.upper()}."
                    )
                else:
                    self.status_note = f"Loaded `{cand}` segmentation cleaner on {self.device.upper()}."
                return
            except Exception as exc:
                last_err = exc

        self._model = None
        self.available = False
        self.used_fallback = True
        self.status_note = f"Segmentation model unavailable ({last_err}); MaskCleaner operating as clean pass-through."

    def segment_persons(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run person instance segmentation and return `(seg_boxes [M,4], seg_masks_full_hw [M x (H,W) bool])`."""
        if not self.available or self._model is None or frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((0, 4), dtype=np.float32), []

        h, w = frame_bgr.shape[:2]
        try:
            results = self._model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=[0],
                device=self.device,
                half=self._use_half,
                retina_masks=True,
                verbose=False,
            )
        except Exception:
            if self._use_half:
                self._use_half = False
                results = self._model.predict(
                    frame_bgr,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    classes=[0],
                    device=self.device,
                    half=False,
                    retina_masks=True,
                    verbose=False,
                )
            else:
                return np.zeros((0, 4), dtype=np.float32), []

        if not results or results[0].boxes is None or results[0].masks is None or len(results[0].boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), []

        r0 = results[0]
        seg_boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
        raw_masks = r0.masks.data.cpu().numpy()  # [M, mh, mw]
        masks_out: List[np.ndarray] = []
        for m in raw_masks:
            if m.shape[0] != h or m.shape[1] != w:
                m_rs = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                masks_out.append(m_rs >= 0.5)
            else:
                masks_out.append(m >= 0.5)
        return seg_boxes, masks_out

    @staticmethod
    def apply_mask_to_crop(
        crop_bgr: np.ndarray,
        mask_crop: Optional[np.ndarray],
        fill_bgr: Tuple[int, int, int] = IMAGENET_MEAN_BGR,
    ) -> np.ndarray:
        """Neutralize non-person pixels in `crop_bgr` using ImageNet mean gray `(104, 116, 124)`."""
        if crop_bgr is None or crop_bgr.size == 0 or mask_crop is None:
            return crop_bgr
        ch, cw = crop_bgr.shape[:2]
        m = mask_crop
        if m.shape[0] != ch or m.shape[1] != cw:
            m = cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST) > 0
        else:
            m = m.astype(bool)
        if not np.any(m):
            return crop_bgr
        cleaned = np.empty_like(crop_bgr)
        cleaned[:, :] = fill_bgr
        cleaned[m] = crop_bgr[m]
        return cleaned

    def clean_person_crops(
        self,
        frame_bgr: np.ndarray,
        person_boxes: np.ndarray,
        seg_boxes: Optional[np.ndarray] = None,
        seg_masks: Optional[Sequence[np.ndarray]] = None,
        iou_thresh: float = 0.35,
    ) -> List[np.ndarray]:
        """Return mask-cleaned BGR crops aligned 1:1 with `person_boxes`."""
        n = len(person_boxes)
        if n == 0:
            return []
        h, w = frame_bgr.shape[:2]

        if seg_boxes is None or seg_masks is None:
            seg_boxes, seg_masks = self.segment_persons(frame_bgr)

        iou_mat = _box_iou_matrix(person_boxes, seg_boxes) if (seg_boxes is not None and len(seg_boxes) > 0) else None

        crops: List[np.ndarray] = []
        for i in range(n):
            x1, y1, x2, y2 = [int(round(float(v))) for v in person_boxes[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
            crop = frame_bgr[y1:y2, x1:x2]

            if iou_mat is not None and seg_masks is not None and len(seg_masks) > 0:
                best_j = int(np.argmax(iou_mat[i]))
                if float(iou_mat[i, best_j]) >= iou_thresh:
                    m_full = seg_masks[best_j]
                    m_crop = m_full[y1:y2, x1:x2]
                    crop = self.apply_mask_to_crop(crop, m_crop)
            crops.append(crop)
        return crops

    def build_mask_cleaned_frame(
        self,
        frame_bgr: np.ndarray,
        person_boxes: np.ndarray,
        seg_boxes: Optional[np.ndarray] = None,
        seg_masks: Optional[Sequence[np.ndarray]] = None,
        iou_thresh: float = 0.35,
        fill_bgr: Tuple[int, int, int] = IMAGENET_MEAN_BGR,
    ) -> Tuple[np.ndarray, int]:
        """Return a copy of `frame_bgr` where background inside matched person boxes is ImageNet-gray neutralized.

        Returns `(cleaned_frame_bgr, matched_mask_count)`.
        """
        if len(person_boxes) == 0:
            return frame_bgr, 0
        if seg_boxes is None or seg_masks is None:
            seg_boxes, seg_masks = self.segment_persons(frame_bgr)
        if seg_boxes is None or len(seg_boxes) == 0 or not seg_masks:
            return frame_bgr, 0

        h, w = frame_bgr.shape[:2]
        iou_mat = _box_iou_matrix(person_boxes, seg_boxes)
        union_fg = np.zeros((h, w), dtype=bool)
        matched_boxes: List[Tuple[int, int, int, int]] = []
        for i in range(len(person_boxes)):
            best_j = int(np.argmax(iou_mat[i]))
            if float(iou_mat[i, best_j]) >= iou_thresh:
                x1, y1, x2, y2 = [int(round(float(v))) for v in person_boxes[i]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
                matched_boxes.append((x1, y1, x2, y2))
                union_fg[y1:y2, x1:x2] |= seg_masks[best_j][y1:y2, x1:x2]

        if not matched_boxes:
            return frame_bgr, 0

        cleaned = frame_bgr.copy()
        for x1, y1, x2, y2 in matched_boxes:
            roi = cleaned[y1:y2, x1:x2]
            fg = union_fg[y1:y2, x1:x2]
            bg_mask = ~fg
            if np.any(bg_mask):
                roi[bg_mask] = fill_bgr
        return cleaned, len(matched_boxes)


# ---------------------------------------------------------------------------
# 1C. Re-ID Backbone Selector (DINOv2 Default + DINOv3 HF Drop-In)
# ---------------------------------------------------------------------------


class NextGenReIDEncoder:
    """Re-ID encoder supporting DINOv2 (`dinov2_vitb14`, default) and DINOv3 via HF `transformers`.

    Automatically falls back to `dinov2_vitb14` with a clear `status_note` if DINOv3 weights
    are gated / require a Hugging Face token or fail to load.
    """

    DEFAULT_FALLBACK: str = "dinov2_vitb14"

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        device: str = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        self.requested_variant = str(variant)
        self.backbone_name: str = str(variant)
        self.device = resolve_device(device)
        self.used_fallback: bool = False
        self.status_note: str = ""
        self._backend_kind: str = "dinov2"
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._roi_align: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._out_hw: Tuple[int, int] = (224, 224)
        self.embed_dim: int = 768

        self._init_backbone(self.requested_variant, hf_token=hf_token)

    def _init_dinov2(self, variant: str) -> None:
        import torch
        from torchvision.ops import roi_align

        self._torch = torch
        self._roi_align = roi_align
        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        model.eval().to(self.device)
        self._model = model
        self._backend_kind = "dinov2"
        self.backbone_name = variant
        self.embed_dim = int(getattr(model, "embed_dim", 768))
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

    def _init_dinov3(self, model_id: str, hf_token: Optional[str] = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        kwargs: Dict[str, Any] = {}
        if hf_token:
            kwargs["token"] = hf_token

        processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        model = AutoModel.from_pretrained(model_id, **kwargs)
        model.eval().to(self.device)
        self._processor = processor
        self._model = model
        self._backend_kind = "dinov3_hf"
        self.backbone_name = model_id
        self.embed_dim = int(getattr(model.config, "hidden_size", 768))

    def _init_backbone(self, variant: str, hf_token: Optional[str] = None) -> None:
        is_dinov3 = "dinov3" in variant.lower() or "/" in variant
        if is_dinov3:
            try:
                self._init_dinov3(variant, hf_token=hf_token)
                self.status_note = f"Loaded DINOv3 backbone `{variant}` on {self.device.upper()}."
                return
            except Exception as exc:
                err_txt = str(exc)
                token_hint = (
                    " (requires a Hugging Face access token / gated repo acceptance)"
                    if any(k in err_txt.lower() for k in ("401", "403", "gated", "token", "unauthorized"))
                    else f" ({err_txt[:140]})"
                )
                self.used_fallback = True
                self.status_note = (
                    f"DINOv3 `{variant}` unavailable{token_hint}; automatically fell back to `{self.DEFAULT_FALLBACK}`."
                )
                variant = self.DEFAULT_FALLBACK

        try:
            self._init_dinov2(variant)
            if not self.status_note:
                self.status_note = f"Loaded `{variant}` on {self.device.upper()}."
        except Exception as exc:
            self._model = None
            self._backend_kind = "cpu_fallback"
            self.backbone_name = "cpu_hog_fallback"
            self.embed_dim = 256
            self.used_fallback = True
            self.status_note = f"DINO backbone unavailable ({exc}); using CPU texture fallback."

    def _forward_hf_crops(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        torch = self._torch
        use_cuda = self.device == "cuda"
        inputs = self._processor(images=crops_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=use_cuda) for k, v in inputs.items()}
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                outputs = self._model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def extract_crops_batch(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a list of BGR crops into L2-normalized float32 embeddings `[N, D]`."""
        n = len(crops_bgr)
        if n == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        if self._model is None:
            from run_tracker import AppearanceEncoder

            vecs = [AppearanceEncoder._hsv_hog_embedding(c) for c in crops_bgr]
            return np.stack(vecs, axis=0).astype(np.float32)

        if self._backend_kind == "dinov3_hf":
            rgb_list = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
            return self._forward_hf_crops(rgb_list)

        # DINOv2 batched FP16 forward pass
        torch = self._torch
        use_cuda = self.device == "cuda"
        resized = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), self._out_hw, interpolation=cv2.INTER_LINEAR) for c in crops_bgr]
        arr = np.stack(resized, axis=0).astype(np.float32) / 255.0  # [N, H, W, 3]
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device, non_blocking=use_cuda)
        tensor = (tensor - self._mean) / self._std
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._model(tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def extract_biometric_dino(
        self,
        frame_bgr: np.ndarray,
        boxes: np.ndarray,
        head_boxes: np.ndarray,
        frame_gpu: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(head_hair_embs [N,D], grayscale_shape_embs [N,D])` compatible with `BiometricDINOv2Encoder`."""
        n = len(boxes)
        if n == 0:
            empty = np.zeros((0, self.embed_dim), dtype=np.float32)
            return empty, empty

        h, w = frame_bgr.shape[:2]

        # Fast GPU RoIAlign path for DINOv2
        if self._backend_kind == "dinov2" and self._model is not None and self._roi_align is not None:
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
                    img_t, rois_t, output_size=self._out_hw, spatial_scale=1.0, sampling_ratio=2, aligned=True
                )
                head_rgb = aligned[:n]
                body_rgb = aligned[n:]
                luma = (
                    0.299 * body_rgb[:, 0:1, :, :]
                    + 0.587 * body_rgb[:, 1:2, :, :]
                    + 0.114 * body_rgb[:, 2:3, :, :]
                )
                gray_body = luma.expand(-1, 3, -1, -1).contiguous()

                head_norm = (head_rgb - self._mean) / self._std
                head_flip = torch.flip(head_norm, dims=[3])
                gray_norm = (gray_body - self._mean) / self._std

                all_crops = torch.cat([head_norm, head_flip, gray_norm], dim=0)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    all_feats = self._model(all_crops)
                if isinstance(all_feats, (tuple, list)):
                    all_feats = all_feats[0]
                all_feats = torch.nn.functional.normalize(all_feats.float().reshape(all_feats.shape[0], -1), p=2, dim=1)

                head_avg = torch.nn.functional.normalize(0.5 * (all_feats[:n] + all_feats[n : 2 * n]), p=2, dim=1)
                gray_vec = torch.nn.functional.normalize(all_feats[2 * n :], p=2, dim=1)
            return head_avg.cpu().numpy().astype(np.float32), gray_vec.cpu().numpy().astype(np.float32)

        # Universal batched crop path for DINOv3 HF or CPU fallback
        head_crops: List[np.ndarray] = []
        head_flips: List[np.ndarray] = []
        gray_crops: List[np.ndarray] = []
        for i in range(n):
            hx1, hy1, hx2, hy2 = [int(round(float(v))) for v in head_boxes[i]]
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(w, max(hx1 + 4, hx2)), min(h, max(hy1 + 4, hy2))
            hc = frame_bgr[hy1:hy2, hx1:hx2]
            head_crops.append(hc)
            head_flips.append(cv2.flip(hc, 1))

            bx1, by1, bx2, by2 = [int(round(float(v))) for v in boxes[i]]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, max(bx1 + 4, bx2)), min(h, max(by1 + 4, by2))
            bc = frame_bgr[by1:by2, bx1:bx2]
            g = cv2.cvtColor(bc, cv2.COLOR_BGR2GRAY)
            gray_crops.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))

        all_vecs = self.extract_crops_batch(head_crops + head_flips + gray_crops)
        f_h = all_vecs[:n]
        f_hf = all_vecs[n : 2 * n]
        f_g = all_vecs[2 * n :]
        h_avg = 0.5 * (f_h + f_hf)
        h_avg = h_avg / np.maximum(np.linalg.norm(h_avg, axis=1, keepdims=True), 1e-8)
        g_vec = f_g / np.maximum(np.linalg.norm(f_g, axis=1, keepdims=True), 1e-8)
        return h_avg.astype(np.float32), g_vec.astype(np.float32)


# ---------------------------------------------------------------------------
# 1D. Depth Model Selector (Depth Anything V2, YOLO26-Depth, Depth Anything V3)
# ---------------------------------------------------------------------------


class NextGenDepthEstimator:
    """Drop-in depth estimator preserving AGENTS.md Invariant 2 (higher value = closer to camera).

    Supports:
    - `depth_anything` (Depth Anything V2, default)
    - `yolo26n-depth.pt` / `yolo26m-depth.pt` (YOLO26-Depth via Ultralytics)
    - `depth-anything/DA3-SMALL` (Depth Anything V3 via `depth_anything_3` if installed)
    Falls back cleanly to Depth Anything V2 on any missing package/weight.
    """

    def __init__(self, backend: str = "depth_anything", device: str = "auto") -> None:
        self.requested_backend = str(backend)
        self.backend: str = str(backend)
        self.device = resolve_device(device)
        self.used_fallback: bool = False
        self.status_note: str = ""
        self._kind: str = "v2"
        self._v2: Optional[DepthEstimator] = None
        self._yolo_depth: Any = None
        self._da3: Any = None
        self._model: Any = None

        self._init_depth(self.requested_backend)

    def _fallback_to_v2(self, reason: str) -> None:
        self._v2 = DepthEstimator(backend="depth_anything", device=self.device)
        self._kind = "v2"
        self.backend = self._v2.backend
        self._model = getattr(self._v2, "_model", None)
        self.used_fallback = True
        self.status_note = f"{reason}; automatically fell back to Depth Anything V2 ({self.backend})."

    def _init_depth(self, req: str) -> None:
        low = req.strip().lower()
        if low in ("depth_anything", "midas_small", "y_prior"):
            self._v2 = DepthEstimator(backend=low, device=self.device)
            self._kind = "v2"
            self.backend = self._v2.backend
            self._model = getattr(self._v2, "_model", None)
            self.status_note = f"Loaded `{self.backend}` on {self.device.upper()}."
            return

        if "yolo26" in low and "depth" in low:
            try:
                from ultralytics import YOLO

                self._yolo_depth = YOLO(req)
                self._kind = "yolo26_depth"
                self.backend = req
                self._model = self._yolo_depth
                self.status_note = f"Loaded YOLO26-Depth (`{req}`) on {self.device.upper()}."
                return
            except Exception as exc:
                self._fallback_to_v2(f"YOLO26-Depth `{req}` unavailable ({exc})")
                return

        if "da3" in low or "depth-anything-v3" in low or "depth_anything_3" in low:
            try:
                from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]

                model = DepthAnything3.from_pretrained(req)
                model = model.to(self.device)
                self._da3 = model
                self._kind = "da3"
                self.backend = req
                self._model = model
                self.status_note = f"Loaded Depth Anything V3 (`{req}`) on {self.device.upper()}."
                return
            except Exception as exc:
                self._fallback_to_v2(f"Depth Anything V3 `{req}` unavailable ({exc})")
                return

        self._fallback_to_v2(f"Unknown depth backend `{req}`")

    @staticmethod
    def _normalize_disparity(raw_map: np.ndarray, invert_distance: bool = False) -> np.ndarray:
        arr = np.asarray(raw_map, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if invert_distance:
            # Convert metric depth/distance (smaller = closer) to relative disparity (larger = closer)
            arr = 1.0 / np.maximum(arr, 1e-3)
        lo = float(np.percentile(arr, 2.0))
        hi = float(np.percentile(arr, 98.0))
        if hi - lo < 1e-6:
            return np.full(arr.shape[:2], 0.5, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def predict(self, frame_bgr: np.ndarray, frame_gpu: Optional[Any] = None) -> np.ndarray:
        """Predict relative disparity map `[H', W']` float32 in `[0, 1]` (higher = closer to camera)."""
        if self._kind == "v2" and self._v2 is not None:
            return self._v2.predict(frame_bgr, frame_gpu=frame_gpu)

        if self._kind == "yolo26_depth" and self._yolo_depth is not None:
            try:
                res = self._yolo_depth.predict(
                    frame_bgr,
                    device=self.device,
                    half=(self.device == "cuda"),
                    verbose=False,
                )
                if res:
                    r0 = res[0]
                    depth_attr = getattr(r0, "depth", None)
                    if depth_attr is None and getattr(r0, "probs", None) is None and hasattr(r0, "orig_img"):
                        depth_attr = getattr(r0, "masks", None)
                    if depth_attr is not None:
                        raw = depth_attr.cpu().numpy() if hasattr(depth_attr, "cpu") else np.asarray(depth_attr)
                        return self._normalize_disparity(raw, invert_distance=False)
            except Exception:
                pass

        if self._kind == "da3" and self._da3 is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pred = self._da3.inference([rgb])
                depth_arr = np.asarray(pred.depth[0], dtype=np.float32)
                # DA3 outputs metric/relative depth (smaller = closer), so invert to disparity (higher = closer)
                return self._normalize_disparity(depth_arr, invert_distance=True)
            except Exception:
                pass

        # Safe fallback to Depth Anything V2 if runtime inference fails
        if self._v2 is None:
            self._v2 = DepthEstimator(backend="depth_anything", device=self.device)
        return self._v2.predict(frame_bgr, frame_gpu=frame_gpu)


# ---------------------------------------------------------------------------
# 2. Natural-Language Target Search (`CLIPTargetSearcher`)
# ---------------------------------------------------------------------------


class CLIPTargetSearcher:
    """Zero-shot natural-language person search via FP16 `openai/clip-vit-base-patch32`."""

    MODEL_ID: str = "openai/clip-vit-base-patch32"

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto") -> None:
        self.model_id = str(model_id)
        self.device = resolve_device(device)
        self.available: bool = False
        self.status_note: str = ""
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

        # Indexed per-track cache: track_id -> {"embs": [K, D], "crops_rgb": [K x HxWx3]}
        self._index: Dict[int, Dict[str, Any]] = {}
        self._init_clip()

    def _init_clip(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            processor = CLIPProcessor.from_pretrained(self.model_id)
            try:
                model = CLIPModel.from_pretrained(self.model_id, dtype=dtype)
            except TypeError:
                model = CLIPModel.from_pretrained(self.model_id, torch_dtype=dtype)
            model.eval().to(self.device)
            self._processor = processor
            self._model = model
            self.available = True
            prec = "FP16 CUDA" if self.device == "cuda" else "FP32 CPU"
            self.status_note = f"Loaded `{self.model_id}` ({prec})."
        except Exception as exc:
            self.available = False
            self._model = None
            self._processor = None
            self.status_note = f"CLIP model `{self.model_id}` unavailable ({exc})."

    @staticmethod
    def _extract_feature_tensor(out: Any) -> Any:
        """Handle both raw tensors and `BaseModelOutputWithPooling` across transformers versions."""
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        return out

    def _encode_images_rgb(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not self.available or self._model is None or not crops_rgb:
            return np.zeros((0, 512), dtype=np.float32)
        torch = self._torch
        use_cuda = self.device == "cuda"
        inputs = self._processor(images=crops_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            self.device,
            dtype=torch.float16 if use_cuda else torch.float32,
            non_blocking=use_cuda,
        )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._extract_feature_tensor(self._model.get_image_features(pixel_values=pixel_values))
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def _encode_text_query(self, query: str) -> Optional[np.ndarray]:
        if not self.available or self._model is None:
            return None
        q = (query or "").strip()
        if not q:
            return None
        torch = self._torch
        use_cuda = self.device == "cuda"
        prompts = [q, f"a surveillance photo of {q}", f"a person {q}"]
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device, non_blocking=use_cuda) for k, v in inputs.items()}
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                feats = self._extract_feature_tensor(self._model.get_text_features(**inputs))
            feats = feats.float().reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            # Weighted average favoring the user's literal phrasing
            weights = torch.tensor([[0.55], [0.25], [0.20]], device=feats.device, dtype=feats.dtype)
            combined = torch.sum(feats * weights, dim=0, keepdim=True)
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        return combined[0].cpu().numpy().astype(np.float32)

    @staticmethod
    def extract_top3_crops_from_timeline(
        source_video_path: Union[str, Path],
        consolidated_timeline: Sequence[Any],
        top_k: int = 3,
        max_side: int = 288,
    ) -> Dict[int, List[np.ndarray]]:
        """Collect top-3 clearest full-body BGR crops per confirmed `#ID` from the Pass-2 timeline."""
        # First pass over timeline metadata to select top-K candidate frames per track_id (fast O(frames) scan)
        candidates: Dict[int, List[Tuple[float, int, np.ndarray]]] = {}
        for snap in consolidated_timeline:
            fidx = int(getattr(snap, "frame_idx", 0))
            for tr in getattr(snap, "active", []):
                tid = int(tr.track_id)
                box = np.asarray(tr.box, dtype=np.float32).reshape(4)
                bw = max(1.0, float(box[2] - box[0]))
                bh = max(1.0, float(box[3] - box[1]))
                obs_ok = getattr(tr, "time_since_update", 0) == 0
                not_occ = not bool(getattr(tr, "is_depth_occluded", False))
                bonus = (1.0 if obs_ok else 0.2) * (1.0 if not_occ else 0.4)
                score = bonus * max(0.1, float(getattr(tr, "score", 0.5))) * math.sqrt(bw * bh)
                candidates.setdefault(tid, []).append((score, fidx, box.copy()))

        # Pick up to `top_k * 3` temporally spaced top frames per track so we can rank by real Laplacian sharpness
        wanted_frames: Dict[int, List[Tuple[int, float, np.ndarray]]] = {}
        for tid, items in candidates.items():
            items.sort(key=lambda x: -x[0])
            picked: List[Tuple[float, int, np.ndarray]] = []
            for sc, fidx, box in items:
                if all(abs(fidx - pf) >= 8 for _, pf, _ in picked):
                    picked.append((sc, fidx, box))
                if len(picked) >= max(top_k * 3, 6):
                    break
            if not picked and items:
                picked = items[:top_k]
            for sc, fidx, box in picked:
                wanted_frames.setdefault(fidx, []).append((tid, sc, box))

        if not wanted_frames:
            return {}

        cap = cv2.VideoCapture(str(source_video_path))
        if not cap.isOpened():
            return {}

        scored_crops: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        try:
            for fidx in sorted(wanted_frames.keys()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fidx - 1))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                for tid, base_sc, box in wanted_frames[fidx]:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    bw, bh = max(4.0, x2 - x1), max(4.0, y2 - y1)
                    px, py = 0.08 * bw, 0.06 * bh
                    cx1 = max(0, int(round(x1 - px)))
                    cy1 = max(0, int(round(y1 - py)))
                    cx2 = min(w, int(round(x2 + px)))
                    cy2 = min(h, int(round(y2 + py)))
                    if cx2 - cx1 < 12 or cy2 - cy1 < 12:
                        continue
                    crop = frame[cy1:cy2, cx1:cx2]
                    sharp = _laplacian_sharpness(crop)
                    rank = base_sc * (1.0 + 0.25 * math.log1p(max(0.0, sharp)))
                    ch, cw = crop.shape[:2]
                    if max(ch, cw) > max_side:
                        scale = float(max_side) / float(max(ch, cw))
                        crop = cv2.resize(
                            crop,
                            (max(16, int(round(cw * scale))), max(16, int(round(ch * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    scored_crops.setdefault(tid, []).append((rank, crop.copy()))
        finally:
            cap.release()

        out: Dict[int, List[np.ndarray]] = {}
        for tid, clist in scored_crops.items():
            clist.sort(key=lambda x: -x[0])
            out[tid] = [c for _, c in clist[:top_k]]
        return out

    def index_track_crops(
        self,
        track_crops: Dict[int, Sequence[np.ndarray]],
        crops_are_bgr: bool = True,
    ) -> int:
        """Index up to top-3 crops per confirmed `#ID` in a single batched CLIP forward pass."""
        self._index.clear()
        if not self.available or not track_crops:
            return 0

        flat_rgb: List[np.ndarray] = []
        slices: List[Tuple[int, int, int]] = []
        for tid, crops in sorted(track_crops.items()):
            valid_rgb: List[np.ndarray] = []
            for c in list(crops)[:3]:
                if c is None or not isinstance(c, np.ndarray) or c.size == 0:
                    continue
                valid_rgb.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if crops_are_bgr else c.copy())
            if not valid_rgb:
                continue
            start = len(flat_rgb)
            flat_rgb.extend(valid_rgb)
            slices.append((int(tid), start, len(flat_rgb)))

        if not flat_rgb:
            return 0

        embs = self._encode_images_rgb(flat_rgb)
        for tid, s, e in slices:
            self._index[tid] = {
                "embs": embs[s:e],
                "crops_rgb": flat_rgb[s:e],
            }
        return len(self._index)

    def index_from_timeline(
        self,
        source_video_path: Union[str, Path],
        consolidated_timeline: Sequence[Any],
        top_k: int = 3,
    ) -> int:
        """Convenience helper: extract top-3 crops per `#ID` from video + timeline and index them."""
        crops_map = self.extract_top3_crops_from_timeline(source_video_path, consolidated_timeline, top_k=top_k)
        return self.index_track_crops(crops_map, crops_are_bgr=True)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """Rank indexed tracks by natural-language query -> `[(track_id, match_score_pct, best_crop_rgb), ...]`."""
        if not self.available or not self._index:
            return []
        text_vec = self._encode_text_query(query)
        if text_vec is None:
            return []

        raw_matches: List[Tuple[int, float, np.ndarray]] = []
        for tid, entry in self._index.items():
            embs = entry["embs"]
            crops_rgb = entry["crops_rgb"]
            sims = (embs @ text_vec).astype(np.float32)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            raw_matches.append((tid, best_sim, crops_rgb[best_idx]))

        if not raw_matches:
            return []

        raw_matches.sort(key=lambda item: -item[1])

        # Calibrate CLIP cosine similarity (~0.18-0.36 range for ViT-B/32) into intuitive 0-100% score
        # while strictly preserving exact cosine ranking order.
        results: List[Tuple[int, float, np.ndarray]] = []
        for tid, sim, crop_rgb in raw_matches[: max(1, int(top_k))]:
            # Map cosine 0.16..0.34 -> ~5%..98% via smooth logistic curve centered at 0.25
            pct = 100.0 / (1.0 + math.exp(-26.0 * (sim - 0.25)))
            results.append((tid, round(float(np.clip(pct, 0.1, 99.9)), 1), crop_rgb))
        return results


# ---------------------------------------------------------------------------
# 3. One-Click NVIDIA TensorRT Exporter (`export_to_tensorrt`)
# ---------------------------------------------------------------------------


def export_to_tensorrt(
    weights_path: Union[str, Path],
    imgsz: int = 768,
    half: bool = True,
    device: int = 0,
) -> Dict[str, Any]:
    """Export an Ultralytics YOLO `.pt` model to an FP16 NVIDIA TensorRT `.engine` file.

    Returns a structured dict `{"ok": bool, "engine_path": Optional[str], "message": str}`
    with clear diagnostics if CUDA or `tensorrt` is not installed.
    """
    try:
        import torch
    except Exception as exc:
        return {
            "ok": False,
            "engine_path": None,
            "message": f"PyTorch is not available ({exc}). TensorRT export requires PyTorch + CUDA.",
        }

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "engine_path": None,
            "message": "No CUDA-capable NVIDIA GPU detected (`torch.cuda.is_available() == False`). TensorRT export requires an NVIDIA GPU.",
        }

    try:
        import tensorrt as _trt  # type: ignore[import-not-found]
        _ = _trt.__version__
    except Exception:
        return {
            "ok": False,
            "engine_path": None,
            "message": (
                "NVIDIA `tensorrt` Python package is not installed in this environment. "
                "Install it with `pip install tensorrt` (matching your CUDA 12.x toolkit) to enable one-click `.engine` export."
            ),
        }

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        exported = model.export(
            format="engine",
            imgsz=int(imgsz),
            half=bool(half),
            device=device,
        )
        engine_path = str(exported) if exported else str(Path(weights_path).with_suffix(".engine"))
        return {
            "ok": True,
            "engine_path": engine_path,
            "message": f"Successfully exported FP16 TensorRT engine: `{engine_path}` (imgsz={int(imgsz)}).",
        }
    except Exception as exc:
        return {
            "ok": False,
            "engine_path": None,
            "message": f"TensorRT export failed for `{weights_path}`: {exc}",
        }


# ---------------------------------------------------------------------------
# `@st.cache_resource`-Friendly Cached Loader Functions
# ---------------------------------------------------------------------------


@_cache_resource(show_spinner=False)
def load_nextgen_detector(model_name: str = "yolo11l-pose.pt", device: str = "auto") -> NextGenPoseDetector:
    return NextGenPoseDetector(model_name=model_name, device=device)


@_cache_resource(show_spinner=False)
def load_mask_cleaner(model_name: str = "yolo26m-seg.pt", device: str = "auto") -> MaskCleaner:
    return MaskCleaner(model_name=model_name, device=device)


@_cache_resource(show_spinner=False)
def load_nextgen_reid_encoder(variant: str = "dinov2_vitb14", device: str = "auto") -> NextGenReIDEncoder:
    return NextGenReIDEncoder(variant=variant, device=device)


@_cache_resource(show_spinner=False)
def load_nextgen_depth_estimator(backend: str = "depth_anything", device: str = "auto") -> NextGenDepthEstimator:
    return NextGenDepthEstimator(backend=backend, device=device)


@_cache_resource(show_spinner=False)
def load_clip_searcher(model_id: str = CLIPTargetSearcher.MODEL_ID, device: str = "auto") -> CLIPTargetSearcher:
    return CLIPTargetSearcher(model_id=model_id, device=device)
