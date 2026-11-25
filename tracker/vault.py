"""Clean-View Episodic Vault and Online Hard-Negative Metric.

Guarantees appearance gallery purity by gating out occluded, overlapping, blurry,
or border-clipped crops, stores diverse viewpoint prototypes per identity, and
learns an online discriminative metric from co-occurring distinct identities
(hard negatives) using fast dual/Woodbury within-class scatter whitening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_iou_1_to_n(box: np.ndarray, others: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute IoU and Intersection-over-Self (IoS) between one box and N boxes."""
    if others is None or len(others) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty

    others = np.asarray(others, dtype=np.float32).reshape(-1, 4)
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_self = max(1e-6, float((box[2] - box[0]) * (box[3] - box[1])))
    area_other = np.maximum(1e-6, (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))
    union = np.maximum(1e-6, area_self + area_other - inter)

    iou = inter / union
    ios = inter / area_self
    return iou.astype(np.float32), ios.astype(np.float32)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 vector."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return v
    return v / norm


@dataclass
class VaultEntry:
    """A single verified clean-view appearance snapshot."""

    embedding: np.ndarray  # (D,) L2-normalized
    frame_idx: int
    quality: float
    sharpness: float
    depth: float
    aspect_ratio: float
    area: float
    is_anchor: bool = False


@dataclass
class CleanViewAssessment:
    """Explainable cleanliness verdict for a single detection crop."""

    is_clean: bool
    quality: float
    reason: str
    sharpness: float = 0.0
    max_iou: float = 0.0
    max_closer_ios: float = 0.0


class CleanViewGate:
    """Strict gate preventing occluded or degraded crops from entering an identity vault.

    Enforces AGENTS.md Invariant 1:
    - Zero neighbor overlap (neighbor IoU <= 0.05)
    - High detection confidence (conf >= 0.50)
    - Laplacian sharpness gating (sharpness >= 18.0)
    """

    def __init__(
        self,
        min_conf: float = 0.50,
        min_area: float = 500.0,
        max_iou_overlap: float = 0.05,
        max_closer_ios: float = 0.05,
        depth_occluder_margin: float = 0.06,
        min_sharpness: float = 18.0,
        border_margin_px: int = 0,
    ) -> None:
        self.min_conf = min_conf
        self.min_area = min_area
        self.max_iou_overlap = max_iou_overlap
        self.max_closer_ios = max_closer_ios
        self.depth_occluder_margin = depth_occluder_margin
        self.min_sharpness = min_sharpness
        self.border_margin_px = border_margin_px

    @staticmethod
    def _crop_sharpness(frame_bgr: np.ndarray, box: np.ndarray) -> float:
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(np.floor(box[0])))
        y1 = max(0, int(np.floor(box[1])))
        x2 = min(w, int(np.ceil(box[2])))
        y2 = min(h, int(np.ceil(box[3])))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0
        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # Normalize crop scale slightly so sharpness is comparable across box sizes
        if max(gray.shape[:2]) > 128:
            scale = 128.0 / max(gray.shape[:2])
            gray = cv2.resize(
                gray,
                (max(8, int(gray.shape[1] * scale)), max(8, int(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def assess(
        self,
        frame_bgr: Optional[np.ndarray],
        box: np.ndarray,
        conf: float,
        det_depth: float,
        other_boxes: Optional[np.ndarray],
        other_depths: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> CleanViewAssessment:
        """Evaluate whether a detection crop qualifies as a Clean View."""
        h, w = frame_shape[:2]
        bw = max(0.0, float(box[2] - box[0]))
        bh = max(0.0, float(box[3] - box[1]))
        area = bw * bh

        if conf < self.min_conf:
            return CleanViewAssessment(False, 0.0, "REJECT_LOW_CONF")

        if area < self.min_area or bw < 14.0 or bh < 18.0:
            return CleanViewAssessment(False, 0.0, "REJECT_SMALL_BOX")

        # Optional border clipping gate (disabled when border_margin_px <= 0 so edge targets store clean views)
        m = self.border_margin_px
        touches = 0
        if m > 0:
            touches = int(box[0] <= m) + int(box[1] <= m) + int(box[2] >= w - m) + int(box[3] >= h - m)
            if touches >= 2:
                return CleanViewAssessment(False, 0.0, "REJECT_BORDER_CLIPPED")

        max_iou = 0.0
        max_closer_ios = 0.0
        if other_boxes is not None and len(other_boxes) > 0:
            ious, ioss = compute_iou_1_to_n(box, other_boxes)
            max_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

            if other_depths is not None and len(other_depths) == len(other_boxes):
                # Depth convention: higher value = closer to camera
                closer_mask = np.asarray(other_depths, dtype=np.float32) > (
                    det_depth + self.depth_occluder_margin
                )
                if np.any(closer_mask):
                    max_closer_ios = float(np.max(ioss[closer_mask]))
                    if max_closer_ios > self.max_closer_ios:
                        return CleanViewAssessment(
                            False,
                            0.0,
                            "REJECT_OCCLUDED_DEPTH",
                            max_iou=max_iou,
                            max_closer_ios=max_closer_ios,
                        )

            if max_iou > self.max_iou_overlap:
                return CleanViewAssessment(
                    False,
                    0.0,
                    "REJECT_OVERLAP_IOU",
                    max_iou=max_iou,
                    max_closer_ios=max_closer_ios,
                )

        sharpness = self._crop_sharpness(frame_bgr, box) if frame_bgr is not None else 100.0
        if sharpness < self.min_sharpness:
            return CleanViewAssessment(
                False,
                0.0,
                "REJECT_BLURRY",
                sharpness=sharpness,
                max_iou=max_iou,
                max_closer_ios=max_closer_ios,
            )

        # Composite cleanliness quality in [0, 1]
        conf_score = np.clip((conf - self.min_conf) / max(1e-3, 1.0 - self.min_conf), 0.0, 1.0)
        sharp_score = np.clip(sharpness / (self.min_sharpness * 5.0), 0.0, 1.0)
        isol_score = np.clip(1.0 - max_iou / max(1e-3, self.max_iou_overlap), 0.0, 1.0)
        border_penalty = 0.85 if touches == 1 else 1.0
        quality = float(border_penalty * (0.40 * conf_score + 0.30 * sharp_score + 0.30 * isol_score))
        quality = float(np.clip(quality, 0.05, 1.0))

        return CleanViewAssessment(
            is_clean=True,
            quality=quality,
            reason="CLEAN",
            sharpness=sharpness,
            max_iou=max_iou,
            max_closer_ios=max_closer_ios,
        )


class EpisodicVault:
    """Diversity-preserving Clean-View episodic appearance bank for a single identity.

    Enforces AGENTS.md Invariant 1 (No Naive EMA Poisoning) & Invariant 2 (Vault Freezing):
    - Never applies a rolling EMA update during overlap or outside verified Clean-View inserts.
    - When `frozen` is True (e.g. DEPTH-OCCLUDED), all insertions are blocked.
    - Prototype (`ema_embedding`) is recomputed strictly as the quality-weighted centroid of
      verified `entries` that passed CleanViewGate, never from unverified or overlapping frames.
    """

    _uid_counter: int = 0

    def __init__(
        self,
        max_entries: int = 16,
        min_frame_gap: int = 4,
        redundancy_cos_thresh: float = 0.965,
    ) -> None:
        EpisodicVault._uid_counter += 1
        self._uid: int = EpisodicVault._uid_counter
        self.max_entries = max(4, max_entries)
        self.min_frame_gap = max(1, min_frame_gap)
        self.redundancy_cos_thresh = redundancy_cos_thresh

        self.entries: List[VaultEntry] = []
        self.ema_embedding: Optional[np.ndarray] = None  # Clean-entry centroid (ONLY set via CleanViewGate)
        self._fallback_embedding: Optional[np.ndarray] = None  # Temporary matcher seed before first clean entry
        self.frozen: bool = False
        self.last_store_frame: int = -99999
        self.total_rejected: int = 0
        self.total_stored: int = 0
        self._version: int = 0
        self._cached_mat: Optional[np.ndarray] = None
        self._cached_mat_version: int = -1

    def __len__(self) -> int:
        return len(self.entries)

    def freeze(self) -> None:
        """Lock the vault against any appearance updates (e.g. DEPTH-OCCLUDED)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unlock the vault once occlusion clears."""
        self.frozen = False

    def _recompute_clean_centroid(self) -> None:
        """Recompute L2-normalized quality-weighted centroid exclusively from clean entries."""
        self._version += 1
        if not self.entries:
            self.ema_embedding = None
            return
        weights = np.array([max(0.05, e.quality) for e in self.entries], dtype=np.float32)
        embs = np.stack([e.embedding for e in self.entries], axis=0)
        centroid = np.sum(embs * weights[:, None], axis=0)
        self.ema_embedding = l2_normalize(centroid)
        self._fallback_embedding = None

    def seed_unverified(
        self,
        embedding: np.ndarray,
        frame_idx: int,
        depth: float,
        box: np.ndarray,
        has_overlap: bool = False,
    ) -> None:
        """Store an initial temporary matcher embedding ONLY if vault is empty and unoverlapped.

        Does NOT modify `ema_embedding`, which is updated exclusively after passing CleanViewGate.
        """
        if self.frozen or has_overlap or self.entries or self._fallback_embedding is not None:
            return
        emb = l2_normalize(embedding)
        self._fallback_embedding = emb.copy()
        self._version += 1

    def try_add(
        self,
        embedding: np.ndarray,
        frame_idx: int,
        assessment: CleanViewAssessment,
        depth: float,
        box: np.ndarray,
    ) -> Tuple[bool, str]:
        """Attempt to store a candidate clean view with diversity-preserving eviction."""
        if self.frozen:
            self.total_rejected += 1
            return False, "VAULT_FROZEN"

        emb = l2_normalize(embedding)
        if not assessment.is_clean:
            self.total_rejected += 1
            return False, assessment.reason

        if len(self.entries) >= 3 and (frame_idx - self.last_store_frame) < self.min_frame_gap:
            return False, "SKIP_TEMPORAL_GAP"

        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
        candidate = VaultEntry(
            embedding=emb,
            frame_idx=frame_idx,
            quality=assessment.quality,
            sharpness=assessment.sharpness,
            depth=depth,
            aspect_ratio=bw / bh,
            area=bw * bh,
            is_anchor=(len(self.entries) == 0),
        )

        if len(self.entries) < self.max_entries:
            # Skip near-duplicate frames once we have a few seed views
            if len(self.entries) >= 3:
                mat = self.get_embeddings()
                max_sim = float(np.max(mat @ emb))
                if max_sim > self.redundancy_cos_thresh:
                    return False, "SKIP_REDUNDANT_VIEW"
            self.entries.append(candidate)
            self.last_store_frame = frame_idx
            self.total_stored += 1
            self._recompute_clean_centroid()
            return True, "VAULT_STORE"

        # Vault is full: perform diversity-preserving eviction (protect anchor index 0)
        non_anchor_indices = [i for i, e in enumerate(self.entries) if not e.is_anchor]
        if not non_anchor_indices:
            return False, "SKIP_ANCHOR_LOCKED"

        mat = self.get_embeddings()  # (K, D)
        sims_to_cand = mat @ emb
        max_sim_cand = float(np.max(sims_to_cand))

        # Find most redundant pair among existing non-anchor entries
        sub = mat[non_anchor_indices]
        pair_sim = sub @ sub.T
        np.fill_diagonal(pair_sim, -2.0)
        flat_idx = int(np.argmax(pair_sim))
        ri, rj = divmod(flat_idx, len(non_anchor_indices))
        best_pair_sim = float(pair_sim[ri, rj])

        idx_a = non_anchor_indices[ri]
        idx_b = non_anchor_indices[rj]
        evict_idx = idx_a if self.entries[idx_a].quality <= self.entries[idx_b].quality else idx_b

        # Accept if candidate is more diverse than the most redundant pair OR higher quality
        if max_sim_cand < best_pair_sim or candidate.quality > self.entries[evict_idx].quality + 0.08:
            self.entries[evict_idx] = candidate
            self.last_store_frame = frame_idx
            self.total_stored += 1
            self._recompute_clean_centroid()
            return True, "VAULT_STORE_EVICT"

        return False, "SKIP_LOW_NOVELTY"

    def get_embeddings(self) -> np.ndarray:
        """Return (K, D) float32 array of stored clean embeddings (or temporary fallback), cached by version."""
        if self._cached_mat_version == self._version and self._cached_mat is not None:
            return self._cached_mat
        if self.entries:
            mat = np.stack([e.embedding for e in self.entries], axis=0)
        elif self._fallback_embedding is not None:
            mat = self._fallback_embedding.reshape(1, -1)
        else:
            mat = np.zeros((0, 0), dtype=np.float32)
        self._cached_mat = mat
        self._cached_mat_version = self._version
        return mat

    def get_qualities(self) -> np.ndarray:
        if self.entries:
            return np.array([e.quality for e in self.entries], dtype=np.float32)
        return np.array([0.5], dtype=np.float32) if self._fallback_embedding is not None else np.zeros((0,), dtype=np.float32)

    def raw_cosine_similarity(self, query_emb: np.ndarray, top_k: int = 3) -> float:
        """Quality-aware top-k + centroid cosine similarity clamped to [0, 1]."""
        q = l2_normalize(query_emb)
        mat = self.get_embeddings()
        if mat.size == 0:
            return 0.0
        sims = mat @ q
        k = min(top_k, len(sims))
        top_sims = np.sort(sims)[-k:]
        top_mean = float(np.mean(top_sims))
        if self.ema_embedding is not None:
            ema_sim = float(np.dot(self.ema_embedding, q))
            score = 0.65 * top_mean + 0.35 * ema_sim
        else:
            score = top_mean
        return float(np.clip(score, 0.0, 1.0))


class OnlineHardNegativeMetric:
    """Online discriminative metric learned from co-occurring distinct tracks.

    Combines:
    1. Online scene-mean centering (removes shared illumination/camera color cast).
    2. Dual Woodbury within-class scatter whitening over intra-vault viewpoint pairs
       so nuisance pose/lighting directions are downweighted in O(M^2 D) time.
    3. Explicit co-occurring hard-negative ledger: records look-alike pairs seen
       simultaneously in the scene and penalizes queries that are equally close to
       a known hard-negative competitor.
    """

    def __init__(
        self,
        max_within_pairs: int = 64,
        whitening_strength: float = 0.45,
        hard_neg_sim_thresh: float = 0.72,
        hard_neg_penalty_weight: float = 0.22,
        scene_mean_momentum: float = 0.05,
        recompute_interval: int = 8,
    ) -> None:
        self.max_within_pairs = max_within_pairs
        self.whitening_strength = float(np.clip(whitening_strength, 0.0, 0.9))
        self.hard_neg_sim_thresh = hard_neg_sim_thresh
        self.hard_neg_penalty_weight = hard_neg_penalty_weight
        self.scene_mean_momentum = scene_mean_momentum
        self.recompute_interval = max(1, recompute_interval)

        self._scene_mean: Optional[np.ndarray] = None
        self._within_diffs: List[np.ndarray] = []
        # Exact dual factors for S^{-1/2} = lam^{-1/2} I + Q^T diag(delta) Q
        self._Q: Optional[np.ndarray] = None  # (R, D) orthonormal right singular vectors
        self._delta: Optional[np.ndarray] = None  # (R,) spectral shrink factors
        self._inv_sqrt_lam: float = 1.0
        self._last_recompute_frame: int = -99999

        # Co-occurring hard negatives: track_id -> {other_track_id: (prototype_emb, max_sim)}
        self._hard_negatives: Dict[int, Dict[int, Tuple[np.ndarray, float]]] = {}
        self._cooccur_counts: Dict[Tuple[int, int], int] = {}
        # Cache transformed vault galleries keyed by (id(vault), metric_version, vault_version)
        self._metric_version: int = 0
        self._vault_t_cache: Dict[int, Tuple[int, int, np.ndarray, Optional[np.ndarray]]] = {}

    def observe_cooccurring(
        self,
        active_vaults: Dict[int, EpisodicVault],
        frame_idx: int,
    ) -> None:
        """Update scene statistics, within-class scatter, and co-occurring hard negatives."""
        valid_ids = [tid for tid, v in active_vaults.items() if v.ema_embedding is not None]
        if not valid_ids:
            return

        # 1. Update running scene mean embedding
        protos = np.stack([active_vaults[tid].ema_embedding for tid in valid_ids], axis=0)
        batch_mean = np.mean(protos, axis=0)
        if self._scene_mean is None or self._scene_mean.shape != batch_mean.shape:
            self._scene_mean = batch_mean.astype(np.float32)
        else:
            m = self.scene_mean_momentum
            self._scene_mean = ((1.0 - m) * self._scene_mean + m * batch_mean).astype(np.float32)
        self._metric_version += 1

        # 2. Sample within-class viewpoint differences from vaults with >= 2 clean views
        for tid in valid_ids:
            vault = active_vaults[tid]
            if len(vault.entries) >= 2:
                e_newest = vault.entries[-1].embedding
                e_anchor = vault.entries[0].embedding
                diff = (e_newest - e_anchor).astype(np.float32)
                norm = float(np.linalg.norm(diff))
                if norm > 1e-4:
                    self._within_diffs.append(diff)
                    if len(self._within_diffs) > self.max_within_pairs:
                        self._within_diffs.pop(0)

        # 3. Record co-occurring hard negatives (simultaneously visible => guaranteed distinct IDs)
        if len(valid_ids) >= 2:
            sim_mat = protos @ protos.T
            n = len(valid_ids)
            for i in range(n):
                tid_i = valid_ids[i]
                for j in range(i + 1, n):
                    tid_j = valid_ids[j]
                    pair_key = (min(tid_i, tid_j), max(tid_i, tid_j))
                    self._cooccur_counts[pair_key] = self._cooccur_counts.get(pair_key, 0) + 1
                    sim_ij = float(np.clip(sim_mat[i, j], 0.0, 1.0))
                    if sim_ij >= self.hard_neg_sim_thresh:
                        self._register_hard_negative(tid_i, tid_j, protos[j], sim_ij)
                        self._register_hard_negative(tid_j, tid_i, protos[i], sim_ij)

        # 4. Periodically refresh dual Woodbury inverse for within-class whitening
        if (
            len(self._within_diffs) >= 4
            and (frame_idx - self._last_recompute_frame) >= self.recompute_interval
        ):
            self._recompute_woodbury()
            self._last_recompute_frame = frame_idx

    def _register_hard_negative(
        self,
        tid: int,
        neg_id: int,
        neg_proto: np.ndarray,
        sim_val: float,
    ) -> None:
        bucket = self._hard_negatives.setdefault(tid, {})
        prev = bucket.get(neg_id)
        if prev is None or sim_val >= prev[1]:
            bucket[neg_id] = (neg_proto.copy(), sim_val)
        else:
            # Keep prototype fresh while retaining peak similarity memory
            updated = l2_normalize(0.8 * prev[0] + 0.2 * neg_proto)
            bucket[neg_id] = (updated, prev[1])
        # Keep top-6 hardest negatives per track
        if len(bucket) > 6:
            weakest = min(bucket.items(), key=lambda kv: kv[1][1])[0]
            del bucket[weakest]

    def _recompute_woodbury(self) -> None:
        """Compute exact S^{-1/2} for S = lam*I + alpha*(1/M)*U^T U in O(M^2 D) via dual eigh."""
        U = np.stack(self._within_diffs, axis=0).astype(np.float64)  # (M, D)
        M = U.shape[0]
        alpha = self.whitening_strength
        lam = max(1e-3, 1.0 - alpha)
        U_scaled = U * np.sqrt(alpha / max(1, M))

        # Dual Gram matrix G = U_scaled U_scaled^T (M x M, M <= 64)
        gram = U_scaled @ U_scaled.T
        try:
            eigvals, eigvecs = np.linalg.eigh(gram)
            keep = eigvals > 1e-6
            if not np.any(keep):
                if self._Q is not None or self._delta is not None:
                    self._Q = None
                    self._delta = None
                    self._metric_version += 1
                return
            s2 = eigvals[keep]
            P = eigvecs[:, keep]  # (M, R)
            # Orthonormal right singular vectors Q = diag(1/sqrt(s2)) P^T U_scaled -> (R, D)
            Q = (P.T @ U_scaled) / np.sqrt(s2)[:, None]
            inv_sqrt_lam = 1.0 / np.sqrt(lam)
            delta = (1.0 / np.sqrt(lam + s2)) - inv_sqrt_lam

            self._Q = Q.astype(np.float32)
            self._delta = delta.astype(np.float32)
            self._inv_sqrt_lam = float(inv_sqrt_lam)
            self._metric_version += 1
        except np.linalg.LinAlgError:
            self._Q = None
            self._delta = None
            self._metric_version += 1

    def _apply_metric_transform(self, vecs: np.ndarray) -> np.ndarray:
        """Apply light scene centering and exact S^{-1/2} within-class whitening."""
        x = np.asarray(vecs, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Gentle centering preserves absolute angular structure while suppressing scene cast
        if self._scene_mean is not None and self._scene_mean.shape[0] == x.shape[1]:
            x = x - 0.15 * self._scene_mean[None, :]

        if self._Q is not None and self._delta is not None and self._Q.shape[1] == x.shape[1]:
            # S^{-1/2} x = lam^{-1/2} x + ((x Q^T) * delta) Q
            proj = x @ self._Q.T  # (N, R)
            x = self._inv_sqrt_lam * x + (proj * self._delta[None, :]) @ self._Q

        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-8)
        return x[0] if single else x

    def _get_transformed_vault(
        self, vault: EpisodicVault
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return cached (mat_t, ema_t) keyed by (vault._uid, _metric_version, vault._version)."""
        vid = vault._uid
        cached = self._vault_t_cache.get(vid)
        if cached is not None and cached[0] == self._metric_version and cached[1] == vault._version:
            return cached[2], cached[3]

        mat = vault.get_embeddings()
        if mat.size == 0:
            return mat, None
        mat_t = self._apply_metric_transform(mat)
        ema_t = (
            self._apply_metric_transform(vault.ema_embedding)
            if vault.ema_embedding is not None
            else None
        )
        if len(self._vault_t_cache) > 512:
            self._vault_t_cache.clear()
        self._vault_t_cache[vid] = (self._metric_version, vault._version, mat_t, ema_t)
        return mat_t, ema_t

    def similarity(
        self,
        query_emb: np.ndarray,
        vault: EpisodicVault,
        track_id: Optional[int] = None,
        top_k: int = 3,
        apply_hn_penalty: bool = True,
    ) -> float:
        """Compute whitened cosine similarity in [0.0, 1.0] (optionally hard-negative calibrated)."""
        mat_t, ema_t = self._get_transformed_vault(vault)
        if mat_t.size == 0:
            return 0.0

        q_t = self._apply_metric_transform(l2_normalize(query_emb))

        sims = mat_t @ q_t
        k = min(top_k, len(sims))
        top_mean = float(np.mean(np.sort(sims)[-k:]))

        if ema_t is not None:
            ema_sim = float(np.dot(ema_t, q_t))
            base_sim = 0.65 * top_mean + 0.35 * ema_sim
        else:
            base_sim = top_mean

        sim_01 = float(np.clip(base_sim, 0.0, 1.0))

        # Hard-negative calibration for short-term association
        if apply_hn_penalty and track_id is not None and track_id in self._hard_negatives:
            neg_bucket = self._hard_negatives[track_id]
            if neg_bucket:
                neg_protos = np.stack([p for p, _ in neg_bucket.values()], axis=0)
                neg_t = self._apply_metric_transform(neg_protos)
                max_neg_01 = float(np.clip(np.max(neg_t @ q_t), 0.0, 1.0))
                if max_neg_01 > sim_01 - 0.08:
                    overlap_pressure = np.clip((max_neg_01 - (sim_01 - 0.08)) / 0.20, 0.0, 1.0)
                    sim_01 = float(np.clip(sim_01 - self.hard_neg_penalty_weight * overlap_pressure, 0.0, 1.0))

        return sim_01

    def set_to_set_similarity(
        self,
        query_embs: np.ndarray,
        vault: EpisodicVault,
        track_id: Optional[int] = None,
        top_k: int = 3,
    ) -> float:
        """Set-to-set whitened cosine gallery similarity in [0.0, 1.0] for probationary stitching.

        Computes pairwise similarity between all probationary tracklet embeddings Q (M, D)
        and all clean vault entries G (K, D), aggregating the top-k matching pairs per query
        so pose changes are tolerated without collapsing ambiguity for Margin-Ratio Abstention.
        """
        if query_embs is None or len(query_embs) == 0:
            return 0.0
        g_t, _ = self._get_transformed_vault(vault)
        if g_t.size == 0:
            return 0.0

        q_arr = np.asarray(query_embs, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
        q_arr = q_arr / np.maximum(q_norms, 1e-8)

        q_t = self._apply_metric_transform(q_arr)  # (M, D)

        pair_sim = q_t @ g_t.T  # (M, K)
        # For each query frame, take top-k gallery matches; then average the top half of query frames
        k_g = min(top_k, pair_sim.shape[1])
        per_query = np.mean(np.sort(pair_sim, axis=1)[:, -k_g:], axis=1)  # (M,)
        k_q = max(1, (len(per_query) + 1) // 2)
        set_sim = float(np.mean(np.sort(per_query)[-k_q:]))
        return float(np.clip(set_sim, 0.0, 1.0))

    def distance(
        self,
        query_emb: np.ndarray,
        vault: EpisodicVault,
        track_id: Optional[int] = None,
        top_k: int = 3,
    ) -> float:
        """Compute calibrated appearance distance in [0.0, 1.0] (lower = closer)."""
        return float(1.0 - self.similarity(query_emb, vault, track_id=track_id, top_k=top_k))

    def get_hard_negatives(self, track_id: int) -> List[Tuple[int, float]]:
        """Return known look-alike competitor IDs and their peak similarity."""
        bucket = self._hard_negatives.get(track_id, {})
        return sorted(((nid, sim) for nid, (_, sim) in bucket.items()), key=lambda x: -x[1])
