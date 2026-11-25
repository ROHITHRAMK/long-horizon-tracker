"""Core Long-Horizon Tracking Engine.

Implements:
- CMC-warped 8D Kalman box tracking
- Depth-gated short-term Hungarian association (prevents crossing-path ID swaps)
- Clean-view gated Episodic Vault updates & Online Hard-Negative Metric learning
- Occlusion-aware Ghost Particle Filter spawning
- Probationary Stitching with Lowe-style Margin-Ratio Veto for long-horizon Re-ID
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cmc import CMCResult, CameraMotionCompensator, warp_kalman_state
from .ghosts import GhostConfig, GhostState, SceneConstrainedGhostFilter
from .vault import (
    CleanViewGate,
    EpisodicVault,
    OnlineHardNegativeMetric,
    compute_iou_1_to_n,
    l2_normalize,
)


LARGE_COST = 1e5


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between (M, 4) and (N, 4) xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(1e-6, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    area_b = np.maximum(1e-6, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    union = np.maximum(1e-6, area_a[:, None] + area_b[None, :] - inter)
    return (inter / union).astype(np.float32)


def box_to_xywh(box: np.ndarray) -> np.ndarray:
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    w = max(4.0, float(box[2] - box[0]))
    h = max(4.0, float(box[3] - box[1]))
    return np.array([cx, cy, w, h], dtype=np.float64)


def xywh_to_box(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = float(xywh[0]), float(xywh[1]), max(4.0, float(xywh[2])), max(4.0, float(xywh[3]))
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)


class KalmanBoxTracker:
    """8-state constant-velocity Kalman Filter: [cx, cy, w, h, vx, vy, vw, vh]."""

    def __init__(self, initial_box: np.ndarray) -> None:
        xywh = box_to_xywh(initial_box)
        self.mean = np.zeros((8,), dtype=np.float64)
        self.mean[:4] = xywh

        std_pos = np.array([
            0.05 * xywh[2],
            0.05 * xywh[3],
            0.05 * xywh[2],
            0.05 * xywh[3],
            0.12 * xywh[2],
            0.12 * xywh[3],
            0.05 * xywh[2],
            0.05 * xywh[3],
        ], dtype=np.float64)
        self.cov = np.diag(np.maximum(std_pos, 1.0) ** 2)

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

    def warp(self, cmc_matrix: np.ndarray) -> None:
        """Apply Camera Motion Compensation to state mean and covariance."""
        self.mean, self.cov = warp_kalman_state(self.mean, self.cov, cmc_matrix)
        self.mean[2] = max(6.0, self.mean[2])
        self.mean[3] = max(6.0, self.mean[3])

    def predict(self) -> np.ndarray:
        """Advance state one step with adaptive size-proportional process noise."""
        w = max(12.0, float(self.mean[2]))
        h = max(12.0, float(self.mean[3]))
        q_std = np.array([
            0.04 * w,
            0.04 * h,
            0.025 * w,
            0.025 * h,
            0.06 * w,
            0.06 * h,
            0.015 * w,
            0.015 * h,
        ], dtype=np.float64)
        Q = np.diag(np.maximum(q_std, 0.5) ** 2)

        # Mild velocity damping on box size derivative
        self.mean[6] *= 0.85
        self.mean[7] *= 0.85

        self.mean = self.F @ self.mean
        self.mean[2] = max(6.0, self.mean[2])
        self.mean[3] = max(6.0, self.mean[3])
        self.cov = self.F @ self.cov @ self.F.T + Q
        return self.get_box()

    def update(self, box: np.ndarray) -> np.ndarray:
        """Kalman measurement update from an observed xyxy bounding box."""
        z = box_to_xywh(box)
        w = max(12.0, float(self.mean[2]))
        h = max(12.0, float(self.mean[3]))
        r_std = np.array([0.045 * w, 0.045 * h, 0.06 * w, 0.06 * h], dtype=np.float64)
        R = np.diag(np.maximum(r_std, 1.0) ** 2)

        y = z - (self.H @ self.mean)
        S = self.H @ self.cov @ self.H.T + R
        K = self.cov @ self.H.T @ np.linalg.inv(S)

        self.mean = self.mean + K @ y
        self.mean[2] = max(6.0, self.mean[2])
        self.mean[3] = max(6.0, self.mean[3])
        I = np.eye(8, dtype=np.float64)
        self.cov = (I - K @ self.H) @ self.cov
        return self.get_box()

    def get_box(self) -> np.ndarray:
        return xywh_to_box(self.mean[:4])

    def get_velocity(self) -> np.ndarray:
        return self.mean[4:6].astype(np.float32)


@dataclass
class Detection:
    """Single object detection input to the tracking engine."""

    box: np.ndarray  # (4,) float32 xyxy
    score: float
    class_id: int
    embedding: np.ndarray  # (D,) float32
    depth: float = 0.5  # [0, 1] where 1.0 = closest


@dataclass
class DiagnosticEvent:
    """Structured explainable tracking decision event."""

    frame_idx: int
    event_type: str
    track_id: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerConfig:
    """Hyperparameters governing depth gating, occlusion locking, probation & margin abstention."""

    # Short-term association & depth gating
    det_thresh: float = 0.35
    iou_weight: float = 0.52
    app_weight: float = 0.48
    active_cost_thresh: float = 0.68
    min_active_iou_without_app: float = 0.15
    depth_gate_thresh: float = 0.22
    depth_ema_alpha: float = 0.25
    depth_occluder_margin: float = 0.06
    # AGENTS.md Invariant 2: pairwise active track IoU >= 0.25 locks farther track as DEPTH-OCCLUDED
    depth_occlusion_iou_thresh: float = 0.25
    # Stage 1 kinematic/IoU fallback for isolated flicker / in-place clothing/posture changes
    kinematic_fallback_iou: float = 0.25
    kinematic_fallback_norm_dist: float = 0.45

    # Active -> LOST (Ghost) transition
    max_active_miss: int = 25
    occluded_ghost_delay: int = 3
    min_hits_for_ghost: int = 4

    # AGENTS.md Invariant 3 & 4: Probationary Stitching (>= 15f) & Margin-Ratio Abstention
    probation_frames: int = 15
    max_probation_frames: int = 40
    probation_max_miss: int = 2
    prob_match_cost_thresh: float = 0.65
    ghost_depth_gate: float = 0.35
    stitch_min_sim: float = 0.72
    margin_min_diff: float = 0.08

    # Sub-module configs
    vault_max_entries: int = 16
    vault_min_frame_gap: int = 4
    ghost_config: GhostConfig = field(default_factory=GhostConfig)
    trail_length: int = 32

    def __post_init__(self) -> None:
        # Enforce AGENTS.md Invariant 3 minimum 15-frame probation
        self.probation_frames = max(15, int(self.probation_frames))
        self.max_probation_frames = max(self.probation_frames, int(self.max_probation_frames))


@dataclass
class ActiveTrack:
    """A confirmed, actively tracked object identity."""

    track_id: int
    class_id: int
    kf: KalmanBoxTracker
    vault: EpisodicVault
    depth_ema: float
    score: float
    created_frame: int
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    status: str = "ACTIVE"  # "ACTIVE" or "DEPTH-OCCLUDED"
    is_depth_occluded: bool = False
    is_occluded: bool = False
    occluder_id: Optional[int] = None
    last_clean_reason: str = "INIT"
    last_box: np.ndarray = None
    stitched_count: int = 0
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=32))

    def current_box(self) -> np.ndarray:
        return self.last_box if self.last_box is not None else self.kf.get_box()


@dataclass
class ProbationaryTrack:
    """Tentative track accumulating multi-frame gallery before set-to-set LOST matching."""

    prob_id: int
    class_id: int
    kf: KalmanBoxTracker
    depth_ema: float
    score: float
    created_frame: int
    state: str = "PROBATIONARY"
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_box: np.ndarray = None
    # All frames stored for continuity: (embedding, det_conf, is_clean)
    embeddings: List[Tuple[np.ndarray, float, bool]] = field(default_factory=list)
    # Clean-View gated subset (IoU <= 0.05, conf >= 0.50, sharpness >= 18.0): (embedding, quality)
    clean_embeddings: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    best_candidate_ghost: Optional[int] = None
    best_score: float = 0.0
    second_best_score: float = 0.0
    margin_ratio: float = 1.0
    last_veto_reason: Optional[str] = None

    def current_box(self) -> np.ndarray:
        return self.last_box if self.last_box is not None else self.kf.get_box()

    def add_observation(
        self,
        embedding: np.ndarray,
        det_conf: float,
        quality: float,
        is_clean: bool,
    ) -> None:
        """Record frame embedding for continuity and separately tag Clean-View frames."""
        emb = l2_normalize(embedding)
        self.embeddings.append((emb, float(det_conf), bool(is_clean)))
        if is_clean:
            self.clean_embeddings.append((emb, float(quality)))

    def get_gallery_embeddings(self, min_clean: int = 3, fallback_top_k: int = 5) -> np.ndarray:
        """Prioritize clean probationary embeddings so pillar-edge/half-occluded crops don't dilute Re-ID.

        Falls back to top-confidence embeddings ONLY when fewer than `min_clean` clean views exist.
        """
        if len(self.clean_embeddings) >= min_clean:
            return np.stack([emb for emb, _ in self.clean_embeddings], axis=0).astype(np.float32)

        if not self.embeddings:
            return np.zeros((0, 0), dtype=np.float32)

        # Rank by (is_clean first, then detection confidence descending) and keep top-k
        ranked = sorted(self.embeddings, key=lambda item: (item[2], item[1]), reverse=True)
        chosen = [emb for emb, _, _ in ranked[: max(min_clean, min(fallback_top_k, len(ranked)))]]
        return np.stack(chosen, axis=0).astype(np.float32)


@dataclass
class TrackingFrameResult:
    """Full structured output for a single processed frame."""

    frame_idx: int
    cmc: CMCResult
    active_tracks: List[ActiveTrack]
    probationary_tracks: List[ProbationaryTrack]
    ghosts: List[GhostState]
    events: List[DiagnosticEvent]
    stats: Dict[str, Any]


class LongHorizonTracker:
    """Complete Long-Horizon Occlusion-Resilient Multi-Object Tracker."""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        cmc: Optional[CameraMotionCompensator] = None,
        clean_gate: Optional[CleanViewGate] = None,
        metric: Optional[OnlineHardNegativeMetric] = None,
        ghost_filter: Optional[SceneConstrainedGhostFilter] = None,
    ) -> None:
        self.cfg = config or TrackerConfig()
        self.cmc = cmc or CameraMotionCompensator()
        self.clean_gate = clean_gate or CleanViewGate(
            depth_occluder_margin=self.cfg.depth_occluder_margin
        )
        self.metric = metric or OnlineHardNegativeMetric()
        self.ghost_filter = ghost_filter or SceneConstrainedGhostFilter(self.cfg.ghost_config)

        self.frame_idx: int = 0
        self._next_track_id: int = 1
        self._next_prob_id: int = 1

        self.active_tracks: Dict[int, ActiveTrack] = {}
        self.probationary_tracks: Dict[int, ProbationaryTrack] = {}
        self.ghost_vaults: Dict[int, EpisodicVault] = {}
        self.ghost_meta: Dict[int, Dict[str, Any]] = {}

        # Cumulative audit counters
        self.counters: Dict[str, int] = {
            "stitches": 0,
            "margin_vetoes": 0,
            "depth_gate_rejects": 0,
            "depth_occlusions": 0,
            "ghosts_spawned": 0,
            "ghosts_expired": 0,
            "vault_stores": 0,
            "vault_rejects": 0,
            "new_ids": 0,
        }

    def _emit(
        self,
        events: List[DiagnosticEvent],
        event_type: str,
        track_id: int,
        message: str,
        **details: Any,
    ) -> None:
        events.append(
            DiagnosticEvent(
                frame_idx=self.frame_idx,
                event_type=event_type,
                track_id=track_id,
                message=message,
                details=details,
            )
        )

    def _spawn_active_track(
        self,
        det: Detection,
        frame_bgr: Optional[np.ndarray],
        all_boxes: np.ndarray,
        all_depths: np.ndarray,
        det_idx: int,
        frame_shape: Tuple[int, int],
        track_id: Optional[int] = None,
        vault: Optional[EpisodicVault] = None,
        stitched_count: int = 0,
    ) -> ActiveTrack:
        tid = track_id if track_id is not None else self._next_track_id
        if track_id is None:
            self._next_track_id += 1
            self.counters["new_ids"] += 1

        other_mask = np.arange(len(all_boxes)) != det_idx
        other_boxes = all_boxes[other_mask] if np.any(other_mask) else None
        other_depths = all_depths[other_mask] if np.any(other_mask) else None

        assessment = self.clean_gate.assess(
            frame_bgr,
            det.box,
            det.score,
            det.depth,
            other_boxes,
            other_depths,
            frame_shape,
        )
        has_overlap = assessment.max_iou > self.clean_gate.max_iou_overlap

        if vault is None:
            vault = EpisodicVault(
                max_entries=self.cfg.vault_max_entries,
                min_frame_gap=self.cfg.vault_min_frame_gap,
            )
            # Never seed an unverified embedding during neighbor overlap
            vault.seed_unverified(
                det.embedding,
                self.frame_idx,
                det.depth,
                det.box,
                has_overlap=has_overlap,
            )

        kf = KalmanBoxTracker(det.box)
        tr = ActiveTrack(
            track_id=tid,
            class_id=det.class_id,
            kf=kf,
            vault=vault,
            depth_ema=float(det.depth),
            score=float(det.score),
            created_frame=self.frame_idx,
            last_box=det.box.copy(),
            stitched_count=stitched_count,
            history=deque(maxlen=self.cfg.trail_length),
        )
        cx = 0.5 * (float(det.box[0]) + float(det.box[2]))
        cy = 0.5 * (float(det.box[1]) + float(det.box[3]))
        tr.history.append((cx, cy))

        stored, reason = tr.vault.try_add(det.embedding, self.frame_idx, assessment, det.depth, det.box)
        tr.last_clean_reason = reason
        if stored:
            self.counters["vault_stores"] += 1
        elif not assessment.is_clean:
            self.counters["vault_rejects"] += 1
        return tr

    def _apply_depth_occlusion_locking(self, events: List[DiagnosticEvent]) -> None:
        """Enforce AGENTS.md Invariant 2: Depth-Triggered Occlusion Locking.

        When two active tracks overlap (IoU >= 0.25), the lower-disparity (farther)
        track MUST be flagged `DEPTH-OCCLUDED` and have its EpisodicVault frozen.
        """
        for tr in self.active_tracks.values():
            tr.status = "ACTIVE"
            tr.is_depth_occluded = False
            tr.is_occluded = False
            tr.occluder_id = None
            tr.vault.unfreeze()

        tids = list(self.active_tracks.keys())
        if len(tids) < 2:
            return

        boxes = np.stack([self.active_tracks[tid].current_box() for tid in tids], axis=0)
        pair_iou = iou_matrix(boxes, boxes)
        n = len(tids)

        for i in range(n):
            for j in range(i + 1, n):
                ov = float(pair_iou[i, j])
                if ov < self.cfg.depth_occlusion_iou_thresh:
                    continue
                tr_i = self.active_tracks[tids[i]]
                tr_j = self.active_tracks[tids[j]]
                # Higher disparity = closer to camera; lower disparity = farther (occluded)
                if tr_i.depth_ema < tr_j.depth_ema - 1e-6:
                    farther, closer = tr_i, tr_j
                elif tr_j.depth_ema < tr_i.depth_ema - 1e-6:
                    farther, closer = tr_j, tr_i
                else:
                    continue

                if not farther.is_depth_occluded:
                    self.counters["depth_occlusions"] += 1
                    self._emit(
                        events,
                        "DEPTH_OCCLUDED",
                        farther.track_id,
                        f"#{farther.track_id} DEPTH-OCCLUDED by #{closer.track_id} (IoU={ov:.2f}, d={farther.depth_ema:.2f}<{closer.depth_ema:.2f})",
                        occluder_id=closer.track_id,
                        iou=ov,
                        farther_depth=farther.depth_ema,
                        closer_depth=closer.depth_ema,
                    )
                farther.status = "DEPTH-OCCLUDED"
                farther.is_depth_occluded = True
                farther.is_occluded = True
                farther.occluder_id = closer.track_id
                farther.vault.freeze()

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        depth_map: Optional[np.ndarray] = None,
        cmc_result: Optional[CMCResult] = None,
    ) -> TrackingFrameResult:
        """Process one video frame and return full explainable tracking state."""
        self.frame_idx += 1
        events: List[DiagnosticEvent] = []
        frame_shape = frame_bgr.shape[:2]

        # Filter detections by confidence threshold
        dets = [d for d in detections if d.score >= self.cfg.det_thresh]
        det_boxes = (
            np.stack([d.box for d in dets], axis=0).astype(np.float32)
            if dets
            else np.zeros((0, 4), dtype=np.float32)
        )
        det_depths = (
            np.array([d.depth for d in dets], dtype=np.float32)
            if dets
            else np.zeros((0,), dtype=np.float32)
        )

        # ------------------------------------------------------------------
        # 1. Camera Motion Compensation (foreground-masked optical flow)
        # ------------------------------------------------------------------
        cmc_res = cmc_result if cmc_result is not None else self.cmc.update(frame_bgr, det_boxes)
        M = cmc_res.matrix

        for tr in self.active_tracks.values():
            tr.kf.warp(M)
            if tr.time_since_update >= 2:
                # Damp coasting velocity during multi-frame YOLO flickers so stationary targets stay anchored
                tr.kf.mean[4:6] *= 0.85
            tr.kf.predict()
            tr.age += 1
            tr.time_since_update += 1

        for pr in self.probationary_tracks.values():
            pr.kf.warp(M)
            pr.kf.predict()
            pr.age += 1
            pr.time_since_update += 1

        # Advance scene-constrained Ghost Particle Filters with CMC & depth
        pruned_ghosts = self.ghost_filter.step_all(
            cmc_matrix=M,
            depth_map=depth_map,
            active_boxes=det_boxes,
            active_depths=det_depths,
            frame_shape=frame_shape,
            frame_idx=self.frame_idx,
        )
        for gid in pruned_ghosts:
            self.ghost_vaults.pop(gid, None)
            self.ghost_meta.pop(gid, None)
            self.counters["ghosts_expired"] += 1
            self._emit(events, "GHOST_EXPIRED", gid, f"Ghost #{gid} expired (age/visibility)")

        # ------------------------------------------------------------------
        # 2. Stage 1: Depth-Gated Active Track Association
        # ------------------------------------------------------------------
        active_ids = list(self.active_tracks.keys())
        matched_active: Dict[int, int] = {}
        unmatched_det_indices = set(range(len(dets)))

        if active_ids and dets:
            pred_boxes = np.stack([self.active_tracks[tid].kf.get_box() for tid in active_ids], axis=0)
            ious = iou_matrix(pred_boxes, det_boxes)
            # Pairwise normalized center distance (relative to predicted box diagonal) for isolated kinematic fallback
            pred_cx = 0.5 * (pred_boxes[:, 0] + pred_boxes[:, 2])
            pred_cy = 0.5 * (pred_boxes[:, 1] + pred_boxes[:, 3])
            pred_diag = np.maximum(
                1.0,
                np.hypot(pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]),
            )
            det_cx = 0.5 * (det_boxes[:, 0] + det_boxes[:, 2])
            det_cy = 0.5 * (det_boxes[:, 1] + det_boxes[:, 3])
            norm_dist = (
                np.hypot(pred_cx[:, None] - det_cx[None, :], pred_cy[:, None] - det_cy[None, :])
                / pred_diag[:, None]
            )
            cost = np.full((len(active_ids), len(dets)), LARGE_COST, dtype=np.float32)

            for r, tid in enumerate(active_ids):
                tr = self.active_tracks[tid]
                for c, det in enumerate(dets):
                    if det.class_id != tr.class_id:
                        continue
                    iou_val = float(ious[r, c])
                    ndist = float(norm_dist[r, c])
                    depth_diff = abs(float(det.depth) - tr.depth_ema)

                    # Depth Gate: reject crossing-path mismatches at incompatible depths
                    if depth_diff > self.cfg.depth_gate_thresh:
                        if iou_val >= 0.25:
                            self.counters["depth_gate_rejects"] += 1
                            self._emit(
                                events,
                                "DEPTH_GATE_REJECT",
                                tid,
                                f"#{tid} depth-gated (|dd|={depth_diff:.2f}, IoU={iou_val:.2f})",
                                depth_diff=depth_diff,
                                iou=iou_val,
                            )
                        continue

                    app_dist = self.metric.distance(det.embedding, tr.vault, track_id=tid)
                    combined = self.cfg.iou_weight * (1.0 - iou_val) + self.cfg.app_weight * app_dist
                    if (
                        (iou_val >= self.cfg.min_active_iou_without_app or app_dist <= 0.28)
                        and combined <= self.cfg.active_cost_thresh
                    ):
                        cost[r, c] = combined
                    elif (
                        iou_val >= self.cfg.kinematic_fallback_iou
                        or ndist <= self.cfg.kinematic_fallback_norm_dist
                    ):
                        # Isolated kinematic/IoU fallback (handles in-place clothing/posture changes)
                        other_track_competes = any(
                            r2 != r and (ious[r2, c] >= 0.15 or norm_dist[r2, c] <= 0.50)
                            for r2 in range(len(active_ids))
                        )
                        other_det_competes = any(
                            c2 != c and (ious[r, c2] >= 0.15 or norm_dist[r, c2] <= 0.50)
                            for c2 in range(len(dets))
                        )
                        if not other_track_competes and not other_det_competes:
                            kin_cost = 0.50 * (1.0 - iou_val) + 0.50 * min(1.0, ndist)
                            cost[r, c] = min(self.cfg.active_cost_thresh - 0.01, float(kin_cost))

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.cfg.active_cost_thresh:
                    tid = active_ids[r]
                    matched_active[tid] = c
                    unmatched_det_indices.discard(c)

        # ------------------------------------------------------------------
        # 3. Update Matched Active Tracks -> Depth-Triggered Occlusion Lock -> Clean-View Vault
        # ------------------------------------------------------------------
        # 3a. Update kinematics & disparity for all matched tracks first
        for tid, det_idx in matched_active.items():
            tr = self.active_tracks[tid]
            det = dets[det_idx]

            tr.kf.update(det.box)
            tr.last_box = det.box.copy()
            tr.score = float(det.score)
            tr.hits += 1
            tr.time_since_update = 0
            tr.depth_ema = float(
                (1.0 - self.cfg.depth_ema_alpha) * tr.depth_ema
                + self.cfg.depth_ema_alpha * det.depth
            )
            cx = 0.5 * (float(det.box[0]) + float(det.box[2]))
            cy = 0.5 * (float(det.box[1]) + float(det.box[3]))
            tr.history.append((cx, cy))

        # 3b. Depth-Triggered Occlusion Locking BEFORE any vault insertion
        # Unmatched active tracks use their CMC-warped predicted box for overlap checks
        for tid, tr in self.active_tracks.items():
            if tid not in matched_active:
                tr.last_box = tr.kf.get_box()
        self._apply_depth_occlusion_locking(events)

        # 3c. Clean-View Gate & EpisodicVault updates (blocked if vault.frozen or overlap > 0.05)
        for tid, det_idx in matched_active.items():
            tr = self.active_tracks[tid]
            det = dets[det_idx]

            other_mask = np.arange(len(dets)) != det_idx
            other_boxes = det_boxes[other_mask] if np.any(other_mask) else None
            other_depths = det_depths[other_mask] if np.any(other_mask) else None

            assessment = self.clean_gate.assess(
                frame_bgr,
                det.box,
                det.score,
                det.depth,
                other_boxes,
                other_depths,
                frame_shape,
            )
            if not tr.is_depth_occluded and assessment.reason in ("REJECT_OCCLUDED_DEPTH", "REJECT_OVERLAP_IOU"):
                tr.is_occluded = True

            stored, reason = tr.vault.try_add(
                det.embedding, self.frame_idx, assessment, det.depth, det.box
            )
            tr.last_clean_reason = reason
            if stored:
                self.counters["vault_stores"] += 1
            elif tr.vault.frozen or not assessment.is_clean:
                self.counters["vault_rejects"] += 1

        # Update Online Hard-Negative Metric with co-occurring clean, unfrozen active tracks
        cooccurring_vaults = {
            tid: self.active_tracks[tid].vault
            for tid in matched_active
            if not self.active_tracks[tid].is_occluded and not self.active_tracks[tid].vault.frozen
        }
        self.metric.observe_cooccurring(cooccurring_vaults, self.frame_idx)

        # ------------------------------------------------------------------
        # 4. Handle Unmatched Active Tracks -> Occlusion Check & Ghost Spawn
        # ------------------------------------------------------------------
        for tid in list(self.active_tracks.keys()):
            if tid in matched_active:
                continue
            tr = self.active_tracks[tid]
            pred_box = tr.kf.get_box()
            tr.last_box = pred_box

            # Check if predicted location is overlapped by a closer active object or closer static depth
            occ_by_obj = False
            occluder_id = tr.occluder_id
            for other_tid, other_det_idx in matched_active.items():
                other_det = dets[other_det_idx]
                if other_det.depth > tr.depth_ema + self.cfg.depth_occluder_margin:
                    _, ios_val = compute_iou_1_to_n(pred_box, other_det.box.reshape(1, 4))
                    if len(ios_val) > 0 and ios_val[0] > 0.15:
                        occ_by_obj = True
                        occluder_id = other_tid
                        break

            occ_by_scene = False
            if depth_map is not None:
                dm_h, dm_w = depth_map.shape[:2]
                cx = int(np.clip(0.5 * (pred_box[0] + pred_box[2]) * dm_w / max(1, frame_shape[1]), 0, dm_w - 1))
                cy = int(np.clip(0.5 * (pred_box[1] + pred_box[3]) * dm_h / max(1, frame_shape[0]), 0, dm_h - 1))
                if float(depth_map[cy, cx]) > tr.depth_ema + self.cfg.depth_occluder_margin:
                    occ_by_scene = True

            is_occluded_loss = tr.is_occluded or occ_by_obj or occ_by_scene
            tr.is_occluded = is_occluded_loss
            tr.occluder_id = occluder_id

            miss_limit = self.cfg.occluded_ghost_delay if is_occluded_loss else self.cfg.max_active_miss
            if tr.time_since_update >= miss_limit:
                del self.active_tracks[tid]
                if tr.hits >= self.cfg.min_hits_for_ghost and tr.vault.get_embeddings().size > 0:
                    self.ghost_filter.spawn_ghost(
                        track_id=tid,
                        class_id=tr.class_id,
                        box=pred_box,
                        velocity=tr.kf.get_velocity(),
                        depth=tr.depth_ema,
                        frame_idx=self.frame_idx,
                        occluded_spawn=is_occluded_loss,
                        occluder_id=occluder_id,
                    )
                    self.ghost_vaults[tid] = tr.vault
                    self.ghost_meta[tid] = {
                        "stitched_count": tr.stitched_count,
                        "history": tr.history,
                    }
                    self.counters["ghosts_spawned"] += 1
                    cause = (
                        f"occluded by #{occluder_id}"
                        if occluder_id is not None
                        else ("scene occlusion" if is_occluded_loss else "lost detection")
                    )
                    used_fallback = len(tr.vault) == 0
                    v_tag = "V=0+seed" if used_fallback else f"V={len(tr.vault)}"
                    self._emit(
                        events,
                        "GHOST_SPAWN",
                        tid,
                        f"Ghost #{tid} spawned ({cause}, {v_tag})",
                        occluded=is_occluded_loss,
                        occluder_id=occluder_id,
                        clean_entries=len(tr.vault),
                        used_fallback=used_fallback,
                    )
                elif tr.hits >= self.cfg.min_hits_for_ghost:
                    self._emit(
                        events,
                        "TRACK_DROPPED_NO_VAULT",
                        tid,
                        f"Dropped #{tid} without ghost (hits={tr.hits}, V=0, last={tr.last_clean_reason})",
                        hits=tr.hits,
                        last_clean_reason=tr.last_clean_reason,
                        vault_rejected=tr.vault.total_rejected,
                    )

        # ------------------------------------------------------------------
        # 5. Stage 2: Probationary Association & Multi-Frame Stitching
        # ------------------------------------------------------------------
        prob_ids = list(self.probationary_tracks.keys())
        rem_det_list = sorted(unmatched_det_indices)
        matched_prob: Dict[int, int] = {}

        if prob_ids and rem_det_list:
            p_boxes = np.stack([self.probationary_tracks[pid].kf.get_box() for pid in prob_ids], axis=0)
            r_boxes = det_boxes[rem_det_list]
            ious_p = iou_matrix(p_boxes, r_boxes)
            cost_p = np.full((len(prob_ids), len(rem_det_list)), LARGE_COST, dtype=np.float32)

            for r, pid in enumerate(prob_ids):
                pr = self.probationary_tracks[pid]
                for c, det_idx in enumerate(rem_det_list):
                    det = dets[det_idx]
                    if det.class_id != pr.class_id:
                        continue
                    if abs(det.depth - pr.depth_ema) > self.cfg.depth_gate_thresh:
                        continue
                    iou_val = float(ious_p[r, c])
                    if pr.embeddings:
                        last_emb = pr.embeddings[-1][0]
                        app_d = float(1.0 - np.clip(float(np.dot(last_emb, l2_normalize(det.embedding))), 0.0, 1.0))
                    else:
                        app_d = 0.5
                    comb = 0.55 * (1.0 - iou_val) + 0.45 * app_d
                    if comb <= self.cfg.prob_match_cost_thresh and iou_val >= 0.08:
                        cost_p[r, c] = comb

            row_ind, col_ind = linear_sum_assignment(cost_p)
            for r, c in zip(row_ind, col_ind):
                if cost_p[r, c] <= self.cfg.prob_match_cost_thresh:
                    pid = prob_ids[r]
                    det_idx = rem_det_list[c]
                    matched_prob[pid] = det_idx
                    unmatched_det_indices.discard(det_idx)

        # Update matched probationary tracks (accumulate embeddings; NO gallery match until >= probation_frames)
        for pid, det_idx in matched_prob.items():
            pr = self.probationary_tracks[pid]
            det = dets[det_idx]
            pr.kf.update(det.box)
            pr.last_box = det.box.copy()
            pr.score = float(det.score)
            pr.hits += 1
            pr.time_since_update = 0
            pr.depth_ema = float(
                (1.0 - self.cfg.depth_ema_alpha) * pr.depth_ema
                + self.cfg.depth_ema_alpha * det.depth
            )
            other_mask = np.arange(len(dets)) != det_idx
            assess = self.clean_gate.assess(
                frame_bgr,
                det.box,
                det.score,
                det.depth,
                det_boxes[other_mask] if np.any(other_mask) else None,
                det_depths[other_mask] if np.any(other_mask) else None,
                frame_shape,
            )
            pr.add_observation(
                det.embedding,
                det_conf=det.score,
                quality=assess.quality if assess.is_clean else 0.0,
                is_clean=assess.is_clean,
            )

        # Spawn new PROBATIONARY tracks for unmatched detections (frame 1 cold-start initializes scene directly)
        for det_idx in sorted(unmatched_det_indices):
            det = dets[det_idx]
            if self.frame_idx == 1 and not self.ghost_filter.ghosts:
                tr = self._spawn_active_track(
                    det, frame_bgr, det_boxes, det_depths, det_idx, frame_shape
                )
                self.active_tracks[tr.track_id] = tr
                continue

            other_mask = np.arange(len(dets)) != det_idx
            assess = self.clean_gate.assess(
                frame_bgr,
                det.box,
                det.score,
                det.depth,
                det_boxes[other_mask] if np.any(other_mask) else None,
                det_depths[other_mask] if np.any(other_mask) else None,
                frame_shape,
            )
            pid = self._next_prob_id
            self._next_prob_id += 1
            pr = ProbationaryTrack(
                prob_id=pid,
                class_id=det.class_id,
                kf=KalmanBoxTracker(det.box),
                depth_ema=float(det.depth),
                score=float(det.score),
                created_frame=self.frame_idx,
                last_box=det.box.copy(),
            )
            pr.add_observation(
                det.embedding,
                det_conf=det.score,
                quality=assess.quality if assess.is_clean else 0.0,
                is_clean=assess.is_clean,
            )
            self.probationary_tracks[pid] = pr
            matched_prob[pid] = det_idx

        # Evaluate Probationary Tracks once hits >= probation_frames (15) via set-to-set gallery matching
        self._resolve_probationary_tracks(
            frame_bgr, dets, det_boxes, det_depths, matched_prob, frame_shape, events
        )

        # Refresh depth-triggered occlusion lock if any cold-start/graduated active tracks overlap
        if self.frame_idx == 1 and len(self.active_tracks) >= 2:
            self._apply_depth_occlusion_locking(events)

        stats = {
            "active_count": len(self.active_tracks),
            "probationary_count": len(self.probationary_tracks),
            "ghost_count": len(self.ghost_filter.ghosts),
            "detections_count": len(dets),
            **self.counters,
        }

        return TrackingFrameResult(
            frame_idx=self.frame_idx,
            cmc=cmc_res,
            active_tracks=list(self.active_tracks.values()),
            probationary_tracks=list(self.probationary_tracks.values()),
            ghosts=list(self.ghost_filter.ghosts.values()),
            events=events,
            stats=stats,
        )

    def _evaluate_stitch_candidates(
        self,
        pr: ProbationaryTrack,
        cur_box: np.ndarray,
        cur_depth: float,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[int], float, float, float]:
        """Set-to-set gallery matching against LOST tracks (only called once hits >= probation_frames).

        Applies depth gating AND time-expanded spatial corridor gating before gallery matching.
        Returns (best_gid, best_sim, second_best_sim, margin_ratio).
        """
        query_gallery = pr.get_gallery_embeddings()
        if query_gallery.size == 0 or not self.ghost_filter.ghosts:
            return None, 0.0, 0.0, 1.0

        ghost_scores: List[Tuple[int, float]] = []
        for gid, ghost in self.ghost_filter.ghosts.items():
            if ghost.class_id != pr.class_id:
                continue
            vault = self.ghost_vaults.get(gid)
            if vault is None:
                continue

            _, _, depth_diff = self.ghost_filter.score_detection(gid, cur_box, cur_depth)
            if depth_diff > self.cfg.ghost_depth_gate:
                continue

            in_corridor, _, _ = self.ghost_filter.is_in_corridor(gid, cur_box, frame_shape)
            if not in_corridor:
                continue

            set_sim = self.metric.set_to_set_similarity(query_gallery, vault, track_id=gid)
            ghost_scores.append((gid, set_sim))

        if not ghost_scores:
            return None, 0.0, 0.0, 1.0

        ghost_scores.sort(key=lambda kv: -kv[1])
        best_gid, best_sim = ghost_scores[0]
        second_best_sim = ghost_scores[1][1] if len(ghost_scores) > 1 else 0.0

        # Also check competing ACTIVE tracks as look-alike distractors for second_best_sim
        for tid, active_tr in self.active_tracks.items():
            if active_tr.class_id != pr.class_id:
                continue
            active_sim = self.metric.set_to_set_similarity(query_gallery, active_tr.vault, track_id=tid)
            if active_sim > second_best_sim:
                second_best_sim = active_sim

        d1 = max(1e-6, 1.0 - best_sim)
        d2 = max(1e-6, 1.0 - second_best_sim)
        margin_ratio = float(d1 / d2)
        return best_gid, best_sim, second_best_sim, margin_ratio

    def _resolve_probationary_tracks(
        self,
        frame_bgr: np.ndarray,
        dets: List[Detection],
        det_boxes: np.ndarray,
        det_depths: np.ndarray,
        matched_prob: Dict[int, int],
        frame_shape: Tuple[int, int],
        events: List[DiagnosticEvent],
    ) -> None:
        """Enforce AGENTS.md Invariants 3 & 4: >=15f probation + Margin-Ratio Abstention."""
        claimed_ghosts: set[int] = set()
        ready_candidates: List[Tuple[float, int, Optional[int], float, float, float]] = []

        for pid, pr in list(self.probationary_tracks.items()):
            if pr.time_since_update > self.cfg.probation_max_miss:
                del self.probationary_tracks[pid]
                continue

            # Invariant 3: NEVER perform set-to-set LOST gallery matching before probation_frames (15)
            if pr.hits < self.cfg.probation_frames:
                pr.best_candidate_ghost = None
                pr.best_score = 0.0
                pr.second_best_score = 0.0
                pr.margin_ratio = 1.0
                continue

            det_idx = matched_prob.get(pid)
            if det_idx is None:
                continue
            det = dets[det_idx]

            best_gid, best_sim, second_best_sim, ratio = self._evaluate_stitch_candidates(
                pr, det.box, det.depth, frame_shape
            )
            pr.best_candidate_ghost = best_gid
            pr.best_score = best_sim
            pr.second_best_score = second_best_sim
            pr.margin_ratio = ratio
            ready_candidates.append((best_sim, pid, best_gid, best_sim, second_best_sim, ratio))

        # Resolve highest-similarity set-to-set candidates first
        for _, pid, _, _, _, _ in sorted(ready_candidates, key=lambda x: -x[0]):
            pr = self.probationary_tracks.get(pid)
            if pr is None:
                continue

            det_idx = matched_prob.get(pid)
            if det_idx is None:
                continue
            det = dets[det_idx]

            # Re-evaluate against remaining unclaimed LOST ghosts
            best_gid, best_sim, second_best_sim, ratio = self._evaluate_stitch_candidates(
                pr, det.box, det.depth, frame_shape
            )
            margin = best_sim - second_best_sim
            pr.best_candidate_ghost = best_gid
            pr.best_score = best_sim
            pr.second_best_score = second_best_sim
            pr.margin_ratio = ratio

            # Case A: Candidate LOST track satisfies best_sim >= 0.72
            if best_gid is not None and best_gid not in claimed_ghosts and best_sim >= self.cfg.stitch_min_sim:
                # Invariant 4: Margin-Ratio Abstention -> require (best_sim - second_best_sim) >= 0.08
                if margin >= self.cfg.margin_min_diff:
                    claimed_ghosts.add(best_gid)
                    ghost = self.ghost_filter.remove_ghost(best_gid)
                    restored_vault = self.ghost_vaults.pop(best_gid, None)
                    meta = self.ghost_meta.pop(best_gid, {})
                    ghost_age = ghost.age if ghost is not None else 0

                    tr = self._spawn_active_track(
                        det=det,
                        frame_bgr=frame_bgr,
                        all_boxes=det_boxes,
                        all_depths=det_depths,
                        det_idx=det_idx,
                        frame_shape=frame_shape,
                        track_id=best_gid,
                        vault=restored_vault,
                        stitched_count=meta.get("stitched_count", 0) + 1,
                    )
                    if "history" in meta:
                        for pt in meta["history"]:
                            tr.history.append(pt)
                        cx = 0.5 * (float(det.box[0]) + float(det.box[2]))
                        cy = 0.5 * (float(det.box[1]) + float(det.box[3]))
                        tr.history.append((cx, cy))

                    self.active_tracks[best_gid] = tr
                    del self.probationary_tracks[pid]
                    self.counters["stitches"] += 1
                    self._emit(
                        events,
                        "STITCH_COMMIT",
                        best_gid,
                        f"Stitched P{pid} -> #{best_gid} after {ghost_age}f (sim={best_sim:.2f}, m={margin:.2f}, r={ratio:.2f})",
                        prob_id=pid,
                        ghost_age=ghost_age,
                        best_sim=best_sim,
                        second_best_sim=second_best_sim,
                        margin=margin,
                        ratio=ratio,
                    )
                    continue

                # MARGIN_VETO: ambiguous look-alike -> abstain and preserve precision!
                pr.last_veto_reason = f"MARGIN_VETO(m={margin:.2f}<{self.cfg.margin_min_diff:.2f})"
                self.counters["margin_vetoes"] += 1
                self._emit(
                    events,
                    "MARGIN_VETO",
                    best_gid,
                    f"VETO P{pid} -> #{best_gid}: ambiguous (s1={best_sim:.2f}, s2={second_best_sim:.2f}, m={margin:.2f})",
                    prob_id=pid,
                    best_sim=best_sim,
                    second_best_sim=second_best_sim,
                    margin=margin,
                    ratio=ratio,
                )
                if pr.hits < self.cfg.max_probation_frames:
                    continue

                # Max probation reached while still ambiguous -> safely assign new ID (0 false merges)
                tr = self._spawn_active_track(
                    det, frame_bgr, det_boxes, det_depths, det_idx, frame_shape
                )
                self.active_tracks[tr.track_id] = tr
                del self.probationary_tracks[pid]
                self._emit(
                    events,
                    "NEW_ID_AFTER_VETO",
                    tr.track_id,
                    f"P{pid} -> NEW #{tr.track_id} after persistent margin veto",
                    prob_id=pid,
                    vetoed_ghost=best_gid,
                )
                continue

            # Case B: Candidate is near threshold (0.58 <= best_sim < 0.72) and still gathering clean views
            clean_view_count = len(pr.clean_embeddings)
            if (
                best_gid is not None
                and best_gid not in claimed_ghosts
                and best_sim >= 0.58
                and clean_view_count < 5
                and pr.hits < self.cfg.max_probation_frames
            ):
                continue

            # Case C: No lost track reached best_sim >= 0.72 -> graduate as a brand-new active track
            tr = self._spawn_active_track(
                det, frame_bgr, det_boxes, det_depths, det_idx, frame_shape
            )
            self.active_tracks[tr.track_id] = tr
            del self.probationary_tracks[pid]
            if best_gid is not None:
                msg = (
                    f"Graduated P{pid} -> NEW #{tr.track_id} "
                    f"(best=#{best_gid}, sim={best_sim:.2f}, s2={second_best_sim:.2f}, m={margin:.2f})"
                )
            else:
                msg = f"Graduated P{pid} -> NEW #{tr.track_id} (no ghost candidate)"
            self._emit(
                events,
                "NEW_TRACK",
                tr.track_id,
                msg,
                prob_id=pid,
                best_candidate_ghost=best_gid,
                best_sim=best_sim,
                second_best_sim=second_best_sim,
                margin=margin,
                ratio=ratio,
                clean_views=clean_view_count,
            )
