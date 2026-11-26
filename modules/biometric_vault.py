"""Persistent SQLite Biometric Vault for Cross-Video / Multi-Camera Identity Memory & Watchlist Enrollment.

Stores clothing-invariant biometric profiles across videos and cameras without modifying
any core `tracker/engine.py` invariants:
- `identities`: persistent `vault_id`, optional `display_name`, `is_watchlist`, timestamps,
  `videos_seen_count`, and portrait `thumbnail_jpeg` (BLOB).
- `embeddings`: per-modality float32 vectors (`face_arcface`, `head_front`, `head_side`,
  `head_back`, `body_shape`, `skin_chroma`) with quality scores and top-K pruning.
- `sightings`: cross-video / multi-camera appearance log (`video_name`, `camera_label`,
  `first_sec`, `last_sec`, `timestamp`).
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


VALID_MODALITIES: Tuple[str, ...] = (
    "face_arcface",
    "head_front",
    "head_side",
    "head_back",
    "body_shape",
    "skin_chroma",
)

HEAD_MODALITIES: Tuple[str, str, str] = ("head_front", "head_side", "head_back")

MAX_EMBEDDINGS_PER_MODALITY: Dict[str, int] = {
    "face_arcface": 12,
    "head_front": 10,
    "head_side": 10,
    "head_back": 10,
    "body_shape": 6,
    "skin_chroma": 10,
}

# Matching body ratio scales used in upload_tracker.py for 8D skeletal proportions
_BODY_RATIO_SCALES = np.array([0.16, 0.16, 0.14, 0.18, 0.18, 0.18, 0.12, 0.18], dtype=np.float32)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _l2_normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _top3_set_to_set_similarity(gal_a: np.ndarray, gal_b: np.ndarray) -> float:
    """Symmetric set-to-set top-3 mean cosine similarity in [0, 1]."""
    if gal_a.size == 0 or gal_b.size == 0:
        return 0.0
    a = np.asarray(gal_a, dtype=np.float32)
    b = np.asarray(gal_b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.shape[1] != b.shape[1]:
        return 0.0
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)

    sim_mat = np.clip(a @ b.T, 0.0, 1.0)
    row_best = np.max(sim_mat, axis=1)
    col_best = np.max(sim_mat, axis=0)
    ka = min(3, len(row_best))
    kb = min(3, len(col_best))
    return float(
        np.clip(
            0.5 * (float(np.mean(np.sort(row_best)[-ka:])) + float(np.mean(np.sort(col_best)[-kb:]))),
            0.0,
            1.0,
        )
    )


def _body_shape_similarity(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
    """Compute similarity in [0, 1] between two body-shape representations.

    Supports both:
    - 8D raw scale-invariant skeletal body ratios (running median + exponential ratio error kernel)
    - General L2-normalized body-shape vectors (cosine similarity)
    """
    if vecs_a.size == 0 or vecs_b.size == 0:
        return 0.0
    a = np.asarray(vecs_a, dtype=np.float32)
    b = np.asarray(vecs_b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.shape[1] != b.shape[1]:
        return 0.0

    if a.shape[1] == 8:
        med_a = np.median(a, axis=0)
        med_b = np.median(b, axis=0)
        # Treat positive entries as valid ratios (entries <= 0 indicate missing keypoints)
        shared = (med_a > 1e-3) & (med_b > 1e-3)
        if not np.any(shared):
            return 0.5
        rel_diff = np.abs(med_a[shared] - med_b[shared]) / _BODY_RATIO_SCALES[shared]
        return float(np.clip(math.exp(-0.65 * float(np.mean(rel_diff))), 0.0, 1.0))

    return _top3_set_to_set_similarity(a, b)


def _extract_skin_chroma_from_region(
    image_bgr: np.ndarray,
    face_box: Optional[np.ndarray] = None,
    face_kps: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract lighting-normalized 34-D CIE Lab (a*, b*) skin chroma vector (L* dropped) from a portrait."""
    h, w = image_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    patches: List[np.ndarray] = []
    if face_kps is not None and np.asarray(face_kps).shape[0] >= 3:
        kps = np.asarray(face_kps, dtype=np.float32)
        # InsightFace 5-point kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
        nose = kps[2]
        eye_dist = float(np.linalg.norm(kps[0] - kps[1]))
        patch_r = max(4, min(18, int(round(0.22 * max(12.0, eye_dist)))))
        centers = [
            (float(nose[0]), float(nose[1])),
            (0.55 * float(nose[0]) + 0.45 * float(kps[0, 0]), float(nose[1]) + 0.15 * eye_dist),
            (0.55 * float(nose[0]) + 0.45 * float(kps[1, 0]), float(nose[1]) + 0.15 * eye_dist),
        ]
        for cx, cy in centers:
            x1 = max(0, int(round(cx)) - patch_r)
            y1 = max(0, int(round(cy)) - patch_r)
            x2 = min(w, int(round(cx)) + patch_r + 1)
            y2 = min(h, int(round(cy)) + patch_r + 1)
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                patches.append(image_bgr[y1:y2, x1:x2])

    if not patches:
        if face_box is not None:
            fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
        else:
            fx1, fy1, fx2, fy2 = 0.25 * w, 0.20 * h, 0.75 * w, 0.75 * h
        fw = max(4.0, fx2 - fx1)
        fh = max(4.0, fy2 - fy1)
        cx1 = max(0, int(round(fx1 + 0.25 * fw)))
        cy1 = max(0, int(round(fy1 + 0.30 * fh)))
        cx2 = min(w, int(round(fx2 - 0.25 * fw)))
        cy2 = min(h, int(round(fy2 - 0.22 * fh)))
        if cx2 - cx1 >= 4 and cy2 - cy1 >= 4:
            patches.append(image_bgr[cy1:cy2, cx1:cx2])

    if not patches:
        return None

    ab_list: List[np.ndarray] = []
    for patch in patches:
        pf = patch.astype(np.float32)
        mean_luma = float(np.mean(pf))
        if mean_luma > 8.0:
            pf = np.clip(pf * (145.0 / mean_luma), 0.0, 255.0)
        lab = cv2.cvtColor(pf.astype(np.uint8), cv2.COLOR_BGR2LAB)
        ab_list.append(lab[:, :, 1:3].reshape(-1, 2))

    ab_all = np.concatenate(ab_list, axis=0).astype(np.float32)
    if ab_all.shape[0] == 0:
        return None

    med_ab = np.median(ab_all, axis=0).astype(np.float32)
    ha, _ = np.histogram(ab_all[:, 0], bins=16, range=(110.0, 174.0))
    hb, _ = np.histogram(ab_all[:, 1], bins=16, range=(110.0, 174.0))
    kernel = np.array([0.08, 0.22, 0.40, 0.22, 0.08], dtype=np.float32)
    ha_f = np.convolve(ha.astype(np.float32), kernel, mode="same")
    hb_f = np.convolve(hb.astype(np.float32), kernel, mode="same")
    ha_f /= max(1e-6, float(np.sum(ha_f)))
    hb_f /= max(1e-6, float(np.sum(hb_f)))
    med_scaled = (med_ab - 128.0) / 30.0
    skin_vec = np.concatenate([ha_f, hb_f, med_scaled], axis=0).astype(np.float32)
    return _l2_normalize(skin_vec)


def _encode_portrait_jpeg(
    image_bgr: np.ndarray,
    crop_box: Optional[np.ndarray] = None,
    max_side: int = 224,
) -> Optional[bytes]:
    """Encode a clean JPEG portrait thumbnail for SQLite BLOB storage."""
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if crop_box is not None:
        x1, y1, x2, y2 = [float(v) for v in crop_box]
        bw = max(4.0, x2 - x1)
        bh = max(4.0, y2 - y1)
        px = 0.28 * bw
        py = 0.32 * bh
        cx1 = max(0, int(round(x1 - px)))
        cy1 = max(0, int(round(y1 - py)))
        cx2 = min(w, int(round(x2 + px)))
        cy2 = min(h, int(round(y2 + py)))
        crop = image_bgr[cy1:cy2, cx1:cx2] if (cx2 > cx1 + 4 and cy2 > cy1 + 4) else image_bgr
    else:
        crop = image_bgr

    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_side:
        scale = float(max_side) / float(max(ch, cw))
        crop = cv2.resize(
            crop,
            (max(16, int(round(cw * scale))), max(16, int(round(ch * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes() if ok else None


def _normalize_track_biometrics(track_biometrics: Any) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """Extract normalized `{modality: [(vec, quality), ...]}` from a TrackBiometricProfile or dict."""
    out: Dict[str, List[Tuple[np.ndarray, float]]] = {m: [] for m in VALID_MODALITIES}
    if track_biometrics is None:
        return out

    # Case 1: Duck-typed `TrackBiometricProfile` from upload_tracker.py
    if hasattr(track_biometrics, "face_gallery") and hasattr(track_biometrics, "head_gallery"):
        fg = getattr(track_biometrics, "face_gallery", None)
        if fg is not None and hasattr(fg, "_items"):
            for rank, _fidx, emb in fg._items:
                nvec = _l2_normalize(emb)
                if nvec is not None:
                    out["face_arcface"].append((nvec, float(rank)))

        hg = getattr(track_biometrics, "head_gallery", {})
        bucket_map = {
            "FRONT": "head_front",
            "SIDE": "head_side",
            "BACK": "head_back",
        }
        if isinstance(hg, dict):
            for vp, mod in bucket_map.items():
                gal = hg.get(vp)
                if gal is not None and hasattr(gal, "_items"):
                    for rank, _fidx, emb in gal._items:
                        nvec = _l2_normalize(emb)
                        if nvec is not None:
                            out[mod].append((nvec, float(rank)))

        if hasattr(track_biometrics, "body_ratio_median"):
            try:
                med, valid = track_biometrics.body_ratio_median()
                med_arr = np.asarray(med, dtype=np.float32).reshape(-1)
                val_arr = np.asarray(valid, dtype=bool).reshape(-1)
                if med_arr.size == 8 and np.any(val_arr):
                    stored = np.where(val_arr, med_arr, 0.0).astype(np.float32)
                    ratio_q = float(np.sum(val_arr)) / 8.0
                    out["body_shape"].append((stored, max(0.25, ratio_q)))
            except Exception:
                pass

        sg = getattr(track_biometrics, "skin_chroma_gallery", None)
        if sg is not None and hasattr(sg, "_items"):
            for rank, _fidx, emb in sg._items:
                nvec = _l2_normalize(emb)
                if nvec is not None:
                    out["skin_chroma"].append((nvec, float(rank)))
        return out

    # Case 2: Dictionary input `{modality: vec | matrix | [(vec, quality), ...]}`
    if isinstance(track_biometrics, dict):
        alias_map = {
            "face": "face_arcface",
            "face_arcface": "face_arcface",
            "arcface": "face_arcface",
            "head_front": "head_front",
            "head_side": "head_side",
            "head_back": "head_back",
            "body_shape": "body_shape",
            "body_ratios": "body_shape",
            "skin_chroma": "skin_chroma",
            "skin": "skin_chroma",
        }
        for key, raw in track_biometrics.items():
            if raw is None:
                continue
            mod = alias_map.get(str(key).lower())
            if mod is None:
                continue
            is_raw_ratios = mod == "body_shape"
            if isinstance(raw, np.ndarray):
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim == 1:
                    if arr.size == 8 and is_raw_ratios:
                        out[mod].append((arr.copy(), 1.0))
                    else:
                        nv = _l2_normalize(arr)
                        if nv is not None:
                            out[mod].append((nv, 1.0))
                elif arr.ndim == 2:
                    for row in arr:
                        if row.size == 8 and is_raw_ratios:
                            out[mod].append((row.copy(), 1.0))
                        else:
                            nv = _l2_normalize(row)
                            if nv is not None:
                                out[mod].append((nv, 1.0))
            elif isinstance(raw, (list, tuple)):
                if len(raw) == 2 and isinstance(raw[0], (np.ndarray, list)) and isinstance(raw[1], (int, float)):
                    # Single (vec, quality) tuple
                    vec_arr = np.asarray(raw[0], dtype=np.float32).reshape(-1)
                    q = float(raw[1])
                    if vec_arr.size == 8 and is_raw_ratios:
                        out[mod].append((vec_arr, q))
                    else:
                        nv = _l2_normalize(vec_arr)
                        if nv is not None:
                            out[mod].append((nv, q))
                else:
                    for item in raw:
                        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
                            vec_arr = np.asarray(item[0], dtype=np.float32).reshape(-1)
                            q = float(item[1])
                        else:
                            vec_arr = np.asarray(item, dtype=np.float32).reshape(-1)
                            q = 1.0
                        if vec_arr.size == 8 and is_raw_ratios:
                            out[mod].append((vec_arr, q))
                        else:
                            nv = _l2_normalize(vec_arr)
                            if nv is not None:
                                out[mod].append((nv, q))
    return out


class BiometricVault:
    """SQLite-backed cross-video / multi-camera biometric identity vault & watchlist."""

    def __init__(self, db_path: Union[str, Path] = "data/biometric_vault.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    vault_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name      TEXT,
                    is_watchlist      INTEGER NOT NULL DEFAULT 0,
                    first_seen_ts     TEXT NOT NULL,
                    last_seen_ts      TEXT NOT NULL,
                    videos_seen_count INTEGER NOT NULL DEFAULT 0,
                    thumbnail_jpeg    BLOB
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    vault_id      INTEGER NOT NULL,
                    modality      TEXT NOT NULL CHECK (
                        modality IN (
                            'face_arcface',
                            'head_front',
                            'head_side',
                            'head_back',
                            'body_shape',
                            'skin_chroma'
                        )
                    ),
                    vector_blob   BLOB NOT NULL,
                    quality_score REAL NOT NULL DEFAULT 1.0,
                    FOREIGN KEY (vault_id) REFERENCES identities(vault_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_vault_mod
                    ON embeddings (vault_id, modality, quality_score DESC);

                CREATE TABLE IF NOT EXISTS sightings (
                    sighting_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    vault_id     INTEGER NOT NULL,
                    video_name   TEXT NOT NULL,
                    camera_label TEXT NOT NULL DEFAULT 'Camera 1',
                    first_sec    REAL NOT NULL DEFAULT 0.0,
                    last_sec     REAL NOT NULL DEFAULT 0.0,
                    timestamp    TEXT NOT NULL,
                    FOREIGN KEY (vault_id) REFERENCES identities(vault_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sightings_vault_ts
                    ON sightings (vault_id, timestamp DESC);
                """
            )
            conn.commit()

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _insert_embeddings_and_prune(
        self,
        conn: sqlite3.Connection,
        vault_id: int,
        modality_vectors: Dict[str, List[Tuple[np.ndarray, float]]],
    ) -> List[str]:
        stored_modalities: List[str] = []
        for modality, pairs in modality_vectors.items():
            if modality not in VALID_MODALITIES or not pairs:
                continue
            rows_to_insert: List[Tuple[int, str, bytes, float]] = []
            for vec, q in pairs:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    continue
                if modality != "body_shape" or arr.size != 8:
                    normed = _l2_normalize(arr)
                    if normed is None:
                        continue
                    arr = normed
                rows_to_insert.append((vault_id, modality, _vec_to_blob(arr), float(q)))

            if not rows_to_insert:
                continue

            conn.executemany(
                "INSERT INTO embeddings (vault_id, modality, vector_blob, quality_score) VALUES (?, ?, ?, ?);",
                rows_to_insert,
            )
            stored_modalities.append(modality)

            # Prune lowest-quality overflow embeddings for this (vault_id, modality)
            max_keep = MAX_EMBEDDINGS_PER_MODALITY.get(modality, 10)
            conn.execute(
                """
                DELETE FROM embeddings
                WHERE vault_id = ? AND modality = ?
                  AND embedding_id NOT IN (
                      SELECT embedding_id FROM embeddings
                      WHERE vault_id = ? AND modality = ?
                      ORDER BY quality_score DESC, embedding_id DESC
                      LIMIT ?
                  );
                """,
                (vault_id, modality, vault_id, modality, max_keep),
            )
        return stored_modalities

    def _load_all_profiles(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        ident_rows = conn.execute(
            "SELECT vault_id, display_name, is_watchlist FROM identities ORDER BY vault_id ASC;"
        ).fetchall()
        profiles: Dict[int, Dict[str, Any]] = {}
        for row in ident_rows:
            vid = int(row["vault_id"])
            profiles[vid] = {
                "vault_id": vid,
                "display_name": row["display_name"],
                "is_watchlist": bool(row["is_watchlist"]),
                "modalities": {m: [] for m in VALID_MODALITIES},
            }

        if not profiles:
            return []

        emb_rows = conn.execute(
            "SELECT vault_id, modality, vector_blob, quality_score FROM embeddings ORDER BY quality_score DESC;"
        ).fetchall()
        for er in emb_rows:
            vid = int(er["vault_id"])
            prof = profiles.get(vid)
            if prof is None:
                continue
            mod = str(er["modality"])
            if mod in prof["modalities"]:
                prof["modalities"][mod].append(_blob_to_vec(er["vector_blob"]))

        return list(profiles.values())

    @staticmethod
    def _extract_face_from_photo(
        image_bgr: np.ndarray,
        face_app: Any,
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run ArcFace on a reference photo using either ArcFaceBiometricAnalyzer or FaceAnalysis.

        Returns `(embedding_512, det_score, face_bbox_xyxy, face_kps_5x2)`.
        """
        if face_app is None or image_bgr is None or image_bgr.size == 0:
            return None, 0.0, None, None

        # Resolve underlying InsightFace FaceAnalysis app if wrapped by ArcFaceBiometricAnalyzer
        raw_app = None
        if hasattr(face_app, "_app") and getattr(face_app, "available", False):
            raw_app = face_app._app
        elif hasattr(face_app, "get") and callable(face_app.get):
            raw_app = face_app

        h, w = image_bgr.shape[:2]
        if raw_app is not None:
            # Pad small or tightly-cropped portraits so SCRFD detects reliably
            pad_x = max(16, int(round(0.20 * w)))
            pad_y = max(16, int(round(0.20 * h)))
            padded = cv2.copyMakeBorder(
                image_bgr, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
            )
            ph, pw = padded.shape[:2]
            scale = 1.0
            if min(ph, pw) < 256:
                scale = 256.0 / float(max(1, min(ph, pw)))
                padded = cv2.resize(
                    padded,
                    (int(round(pw * scale)), int(round(ph * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            try:
                faces = raw_app.get(padded)
            except Exception:
                faces = []

            best_face = None
            best_rank = -1.0
            for face in faces or []:
                score = float(getattr(face, "det_score", 0.0))
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(face, "embedding", None)
                if emb is None:
                    continue
                bbox = getattr(face, "bbox", None)
                area = 1.0
                if bbox is not None and len(bbox) >= 4:
                    area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
                rank = score * math.sqrt(area)
                if rank > best_rank:
                    best_rank = rank
                    best_face = face

            if best_face is not None:
                emb = getattr(best_face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(best_face, "embedding", None)
                nvec = _l2_normalize(np.asarray(emb, dtype=np.float32))
                score = float(getattr(best_face, "det_score", 0.90))
                fbox_orig = None
                if getattr(best_face, "bbox", None) is not None:
                    fb = np.asarray(best_face.bbox, dtype=np.float32).reshape(-1)[:4]
                    fbox_orig = np.array(
                        [
                            np.clip(fb[0] / scale - pad_x, 0.0, float(w)),
                            np.clip(fb[1] / scale - pad_y, 0.0, float(h)),
                            np.clip(fb[2] / scale - pad_x, 0.0, float(w)),
                            np.clip(fb[3] / scale - pad_y, 0.0, float(h)),
                        ],
                        dtype=np.float32,
                    )
                fkps_orig = None
                if getattr(best_face, "kps", None) is not None:
                    fk = np.asarray(best_face.kps, dtype=np.float32).copy()
                    if fk.ndim == 2 and fk.shape[1] >= 2:
                        fk[:, 0] = np.clip(fk[:, 0] / scale - pad_x, 0.0, float(w))
                        fk[:, 1] = np.clip(fk[:, 1] / scale - pad_y, 0.0, float(h))
                        fkps_orig = fk
                if nvec is not None:
                    return nvec, score, fbox_orig, fkps_orig

        # Fallback: wrapper exposing extract_from_head_box(frame_bgr, head_box)
        if hasattr(face_app, "extract_from_head_box") and callable(face_app.extract_from_head_box):
            full_box = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
            res = face_app.extract_from_head_box(image_bgr, full_box, min_det_score=0.25)
            if res is not None:
                emb, score = res
                nvec = _l2_normalize(emb)
                if nvec is not None:
                    return nvec, float(score), full_box, None

        return None, 0.0, None, None

    @staticmethod
    def _extract_head_from_photo(
        image_bgr: np.ndarray,
        head_box: np.ndarray,
        dinov2_encoder: Any,
    ) -> Optional[np.ndarray]:
        """Extract L2-normalized head+hair+ears DINOv2 embedding from a reference photo."""
        if dinov2_encoder is None or image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        hb = np.asarray(head_box, dtype=np.float32).reshape(1, 4)
        fb = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)

        if hasattr(dinov2_encoder, "extract_biometric_dino") and callable(dinov2_encoder.extract_biometric_dino):
            try:
                head_embs, _gray_embs = dinov2_encoder.extract_biometric_dino(image_bgr, fb, hb)
                if head_embs is not None and len(head_embs) > 0:
                    return _l2_normalize(head_embs[0])
            except Exception:
                return None

        if callable(dinov2_encoder):
            x1, y1, x2, y2 = [int(round(v)) for v in hb[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, max(x1 + 4, x2)), min(h, max(y1 + 4, y2))
            crop = image_bgr[y1:y2, x1:x2]
            try:
                vec = dinov2_encoder(crop)
                if vec is not None:
                    return _l2_normalize(np.asarray(vec, dtype=np.float32))
            except Exception:
                return None

        return None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def enroll_reference_photo(
        self,
        image_bgr: np.ndarray,
        display_name: str,
        face_app: Any = None,
        dinov2_encoder: Any = None,
        is_watchlist: bool = True,
    ) -> Dict[str, Any]:
        """Extract ArcFace + head/hair + skin embeddings from a reference photo and insert a named profile."""
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("image_bgr must be a non-empty BGR numpy image.")

        clean_name = (display_name or "").strip() or None
        h, w = image_bgr.shape[:2]

        face_emb, face_score, face_box, face_kps = self._extract_face_from_photo(image_bgr, face_app)

        # Build a clean head+hair+ears crop box from detected face (or upper/full portrait fallback)
        if face_box is not None:
            fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
            fw = max(8.0, fx2 - fx1)
            fh = max(8.0, fy2 - fy1)
            head_box = np.array(
                [
                    max(0.0, fx1 - 0.30 * fw),
                    max(0.0, fy1 - 0.45 * fh),
                    min(float(w), fx2 + 0.30 * fw),
                    min(float(h), fy2 + 0.15 * fh),
                ],
                dtype=np.float32,
            )
        else:
            head_box = np.array(
                [0.10 * w, 0.04 * h, 0.90 * w, min(float(h), 0.78 * h)],
                dtype=np.float32,
            )

        head_emb = self._extract_head_from_photo(image_bgr, head_box, dinov2_encoder)
        skin_vec = _extract_skin_chroma_from_region(image_bgr, face_box=face_box, face_kps=face_kps)
        thumb_jpeg = _encode_portrait_jpeg(image_bgr, crop_box=head_box if face_box is not None else None)

        modality_vectors: Dict[str, List[Tuple[np.ndarray, float]]] = {m: [] for m in VALID_MODALITIES}
        if face_emb is not None:
            modality_vectors["face_arcface"].append((face_emb, max(0.50, float(face_score))))
        if head_emb is not None:
            modality_vectors["head_front"].append((head_emb, 1.0))
        if skin_vec is not None:
            modality_vectors["skin_chroma"].append((skin_vec, 0.90))

        now_ts = _utc_now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO identities (
                    display_name, is_watchlist, first_seen_ts, last_seen_ts, videos_seen_count, thumbnail_jpeg
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (clean_name, 1 if is_watchlist else 0, now_ts, now_ts, 0, thumb_jpeg),
            )
            vault_id = int(cur.lastrowid)
            stored_mods = self._insert_embeddings_and_prune(conn, vault_id, modality_vectors)
            conn.commit()

        return {
            "vault_id": vault_id,
            "display_name": clean_name,
            "is_watchlist": bool(is_watchlist),
            "face_extracted": face_emb is not None,
            "face_score": round(float(face_score), 4),
            "modalities_stored": stored_mods,
        }

    def match_track_to_vault(
        self,
        track_biometrics: Any,
        face_thresh: float = 0.42,
        bio_thresh: float = 0.70,
        face_margin: float = 0.05,
        bio_margin: float = 0.04,
    ) -> Optional[Tuple[int, Optional[str], float, str]]:
        """Match a confirmed video track against all stored vault profiles.

        Tier 1: Direct ArcFace face lock (`face_arcface`, cosine >= `face_thresh`).
        Tier 2: Angle-aware head (`head_front`/`head_side`/`head_back`) + `body_shape` + `skin_chroma`
                weighted fusion (>= `bio_thresh`).

        Returns `(vault_id, display_name, confidence, cue)` or `None`.
        """
        query = _normalize_track_biometrics(track_biometrics)
        with self._lock, self._connect() as conn:
            profiles = self._load_all_profiles(conn)

        if not profiles:
            return None

        # --- TIER 1: Direct ArcFace Face Lock ---
        q_faces = np.stack([v for v, _ in query["face_arcface"]], axis=0) if query["face_arcface"] else np.zeros((0, 0), dtype=np.float32)
        if q_faces.size > 0:
            face_scores: List[Tuple[float, Dict[str, Any]]] = []
            for prof in profiles:
                p_list = prof["modalities"]["face_arcface"]
                if not p_list:
                    continue
                p_mat = np.stack(p_list, axis=0)
                sim = _top3_set_to_set_similarity(q_faces, p_mat)
                face_scores.append((sim, prof))

            if face_scores:
                face_scores.sort(key=lambda item: -item[0])
                best_sim, best_prof = face_scores[0]
                second_sim = face_scores[1][0] if len(face_scores) >= 2 else 0.0
                margin_ok = (len(face_scores) == 1) or ((best_sim - second_sim) >= face_margin)
                if best_sim >= face_thresh and margin_ok:
                    return (
                        int(best_prof["vault_id"]),
                        best_prof["display_name"],
                        round(float(best_sim), 4),
                        "FACE_ARCFACE",
                    )

        # --- TIER 2: Angle-Aware Head + Body Shape + Skin Chroma Fusion ---
        q_heads: Dict[str, np.ndarray] = {
            m: (np.stack([v for v, _ in query[m]], axis=0) if query[m] else np.zeros((0, 0), dtype=np.float32))
            for m in HEAD_MODALITIES
        }
        q_all_heads_list = [mat for mat in q_heads.values() if mat.size > 0]
        q_all_heads = np.concatenate(q_all_heads_list, axis=0) if q_all_heads_list else np.zeros((0, 0), dtype=np.float32)
        q_body = np.stack([v for v, _ in query["body_shape"]], axis=0) if query["body_shape"] else np.zeros((0, 0), dtype=np.float32)
        q_skin = np.stack([v for v, _ in query["skin_chroma"]], axis=0) if query["skin_chroma"] else np.zeros((0, 0), dtype=np.float32)

        if q_all_heads.size == 0:
            return None

        bio_scores: List[Tuple[float, Dict[str, Any], str]] = []
        for prof in profiles:
            p_mods = prof["modalities"]
            # Angle-bucketed head similarity (same angle preferred; damped cross-angle fallback)
            same_angle_sims: List[Tuple[float, str]] = []
            for m in HEAD_MODALITIES:
                if q_heads[m].size > 0 and p_mods[m]:
                    p_hmat = np.stack(p_mods[m], axis=0)
                    s = _top3_set_to_set_similarity(q_heads[m], p_hmat)
                    same_angle_sims.append((s, m.upper()))

            head_sim = 0.0
            head_cue = "HEAD"
            if same_angle_sims:
                same_angle_sims.sort(key=lambda x: -x[0])
                head_sim, head_cue = same_angle_sims[0]
            else:
                p_all_list = [np.stack(p_mods[m], axis=0) for m in HEAD_MODALITIES if p_mods[m]]
                if p_all_list:
                    p_all_heads = np.concatenate(p_all_list, axis=0)
                    head_sim = 0.94 * _top3_set_to_set_similarity(q_all_heads, p_all_heads)
                    head_cue = "HEAD_XANGLE"
                else:
                    continue

            parts: List[Tuple[float, float]] = [(0.60, head_sim)]
            cue_tokens: List[str] = [head_cue]

            if q_body.size > 0 and p_mods["body_shape"]:
                p_bmat = np.stack(p_mods["body_shape"], axis=0)
                b_sim = _body_shape_similarity(q_body, p_bmat)
                parts.append((0.25, b_sim))
                cue_tokens.append("BODY")

            if q_skin.size > 0 and p_mods["skin_chroma"]:
                p_smat = np.stack(p_mods["skin_chroma"], axis=0)
                s_sim = _top3_set_to_set_similarity(q_skin, p_smat)
                parts.append((0.15, s_sim))
                cue_tokens.append("SKIN")

            total_w = sum(w for w, _ in parts)
            fused = sum(w * val for w, val in parts) / max(1e-6, total_w)
            bio_scores.append((float(fused), prof, "+".join(cue_tokens)))

        if not bio_scores:
            return None

        bio_scores.sort(key=lambda item: -item[0])
        best_score, best_prof, best_cue = bio_scores[0]
        second_score = bio_scores[1][0] if len(bio_scores) >= 2 else 0.0
        margin_ok = (len(bio_scores) == 1) or ((best_score - second_score) >= bio_margin)
        if best_score >= bio_thresh and margin_ok:
            return (
                int(best_prof["vault_id"]),
                best_prof["display_name"],
                round(float(best_score), 4),
                best_cue,
            )

        return None

    def upsert_track_identity(
        self,
        track_biometrics: Any,
        vault_id: Optional[int] = None,
        display_name: Optional[str] = None,
        is_watchlist: Optional[bool] = None,
        video_name: Optional[str] = None,
        camera_label: str = "Camera 1",
        first_sec: float = 0.0,
        last_sec: float = 0.0,
        thumbnail_jpeg: Optional[bytes] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert a new tracked identity or update an existing vault profile + append sighting."""
        modality_vectors = _normalize_track_biometrics(track_biometrics)
        ts = timestamp or _utc_now_iso()
        clean_name = display_name.strip() if isinstance(display_name, str) and display_name.strip() else None

        with self._lock, self._connect() as conn:
            existing_row = None
            if vault_id is not None:
                existing_row = conn.execute(
                    "SELECT vault_id, display_name, is_watchlist, videos_seen_count, thumbnail_jpeg "
                    "FROM identities WHERE vault_id = ?;",
                    (int(vault_id),),
                ).fetchone()

            if existing_row is None:
                cur = conn.execute(
                    """
                    INSERT INTO identities (
                        display_name, is_watchlist, first_seen_ts, last_seen_ts, videos_seen_count, thumbnail_jpeg
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        clean_name,
                        1 if bool(is_watchlist) else 0,
                        ts,
                        ts,
                        1 if video_name else 0,
                        thumbnail_jpeg,
                    ),
                )
                target_id = int(cur.lastrowid)
            else:
                target_id = int(existing_row["vault_id"])
                new_name = clean_name if clean_name is not None else existing_row["display_name"]
                new_watch = (
                    (1 if is_watchlist else 0)
                    if is_watchlist is not None
                    else int(existing_row["is_watchlist"])
                )
                new_thumb = (
                    thumbnail_jpeg
                    if (thumbnail_jpeg is not None and existing_row["thumbnail_jpeg"] is None)
                    else existing_row["thumbnail_jpeg"]
                )

                new_vid_count = int(existing_row["videos_seen_count"])
                if video_name:
                    already_seen_vid = conn.execute(
                        "SELECT 1 FROM sightings WHERE vault_id = ? AND video_name = ? LIMIT 1;",
                        (target_id, str(video_name)),
                    ).fetchone()
                    if already_seen_vid is None:
                        new_vid_count += 1

                conn.execute(
                    """
                    UPDATE identities
                    SET display_name = ?,
                        is_watchlist = ?,
                        last_seen_ts = ?,
                        videos_seen_count = ?,
                        thumbnail_jpeg = ?
                    WHERE vault_id = ?;
                    """,
                    (new_name, new_watch, ts, new_vid_count, new_thumb, target_id),
                )

            self._insert_embeddings_and_prune(conn, target_id, modality_vectors)

            if video_name:
                conn.execute(
                    """
                    INSERT INTO sightings (
                        vault_id, video_name, camera_label, first_sec, last_sec, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        target_id,
                        str(video_name),
                        str(camera_label or "Camera 1"),
                        float(first_sec),
                        float(last_sec),
                        ts,
                    ),
                )

            conn.commit()
            return target_id

    def rename_identity(
        self,
        vault_id: int,
        new_name: Optional[str],
        is_watchlist: bool = False,
    ) -> bool:
        """Update display name and watchlist flag for an existing vault identity."""
        clean_name = new_name.strip() if isinstance(new_name, str) and new_name.strip() else None
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE identities SET display_name = ?, is_watchlist = ? WHERE vault_id = ?;",
                (clean_name, 1 if is_watchlist else 0, int(vault_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_identities(
        self,
        watchlist_only: bool = False,
        include_sightings: bool = True,
    ) -> List[Dict[str, Any]]:
        """List stored vault identities with modality counts and optional sighting history."""
        with self._lock, self._connect() as conn:
            if watchlist_only:
                rows = conn.execute(
                    """
                    SELECT vault_id, display_name, is_watchlist, first_seen_ts, last_seen_ts,
                           videos_seen_count, thumbnail_jpeg
                    FROM identities
                    WHERE is_watchlist = 1
                    ORDER BY last_seen_ts DESC, vault_id ASC;
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT vault_id, display_name, is_watchlist, first_seen_ts, last_seen_ts,
                           videos_seen_count, thumbnail_jpeg
                    FROM identities
                    ORDER BY is_watchlist DESC, last_seen_ts DESC, vault_id ASC;
                    """
                ).fetchall()

            emb_counts: Dict[int, Dict[str, int]] = {}
            for er in conn.execute(
                "SELECT vault_id, modality, COUNT(*) AS cnt FROM embeddings GROUP BY vault_id, modality;"
            ).fetchall():
                emb_counts.setdefault(int(er["vault_id"]), {})[str(er["modality"])] = int(er["cnt"])

            sightings_map: Dict[int, List[Dict[str, Any]]] = {}
            if include_sightings:
                for sr in conn.execute(
                    """
                    SELECT sighting_id, vault_id, video_name, camera_label, first_sec, last_sec, timestamp
                    FROM sightings
                    ORDER BY timestamp DESC, sighting_id DESC;
                    """
                ).fetchall():
                    sightings_map.setdefault(int(sr["vault_id"]), []).append(
                        {
                            "sighting_id": int(sr["sighting_id"]),
                            "vault_id": int(sr["vault_id"]),
                            "video_name": str(sr["video_name"]),
                            "camera_label": str(sr["camera_label"]),
                            "first_sec": float(sr["first_sec"]),
                            "last_sec": float(sr["last_sec"]),
                            "timestamp": str(sr["timestamp"]),
                        }
                    )

        results: List[Dict[str, Any]] = []
        for row in rows:
            vid = int(row["vault_id"])
            counts = emb_counts.get(vid, {})
            entry: Dict[str, Any] = {
                "vault_id": vid,
                "display_name": row["display_name"],
                "is_watchlist": bool(row["is_watchlist"]),
                "first_seen_ts": str(row["first_seen_ts"]),
                "last_seen_ts": str(row["last_seen_ts"]),
                "videos_seen_count": int(row["videos_seen_count"]),
                "thumbnail_jpeg": bytes(row["thumbnail_jpeg"]) if row["thumbnail_jpeg"] is not None else None,
                "embedding_counts": counts,
                "has_face": counts.get("face_arcface", 0) > 0,
            }
            if include_sightings:
                entry["sightings"] = sightings_map.get(vid, [])
            results.append(entry)
        return results

    def get_identity(self, vault_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single identity record with its modality counts and sightings."""
        for item in self.list_identities(watchlist_only=False, include_sightings=True):
            if item["vault_id"] == int(vault_id):
                return item
        return None

    def get_sightings(self, vault_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch sighting logs for a specific vault_id or across the entire vault."""
        with self._lock, self._connect() as conn:
            if vault_id is not None:
                rows = conn.execute(
                    """
                    SELECT s.sighting_id, s.vault_id, i.display_name, i.is_watchlist,
                           s.video_name, s.camera_label, s.first_sec, s.last_sec, s.timestamp
                    FROM sightings s
                    JOIN identities i ON i.vault_id = s.vault_id
                    WHERE s.vault_id = ?
                    ORDER BY s.timestamp DESC, s.sighting_id DESC;
                    """,
                    (int(vault_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT s.sighting_id, s.vault_id, i.display_name, i.is_watchlist,
                           s.video_name, s.camera_label, s.first_sec, s.last_sec, s.timestamp
                    FROM sightings s
                    JOIN identities i ON i.vault_id = s.vault_id
                    ORDER BY s.timestamp DESC, s.sighting_id DESC;
                    """
                ).fetchall()
        return [
            {
                "sighting_id": int(r["sighting_id"]),
                "vault_id": int(r["vault_id"]),
                "display_name": r["display_name"],
                "is_watchlist": bool(r["is_watchlist"]),
                "video_name": str(r["video_name"]),
                "camera_label": str(r["camera_label"]),
                "first_sec": float(r["first_sec"]),
                "last_sec": float(r["last_sec"]),
                "timestamp": str(r["timestamp"]),
            }
            for r in rows
        ]

    def delete_identity(self, vault_id: int) -> bool:
        """Delete an identity and cascade-delete all of its embeddings and sightings."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM identities WHERE vault_id = ?;", (int(vault_id),))
            conn.commit()
            return cur.rowcount > 0

    def clear_vault(self) -> int:
        """Wipe all stored identities, embeddings, and sightings; return count of removed identities."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM identities;").fetchone()
            removed = int(row["cnt"]) if row is not None else 0
            conn.execute("DELETE FROM sightings;")
            conn.execute("DELETE FROM embeddings;")
            conn.execute("DELETE FROM identities;")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('identities', 'embeddings', 'sightings');")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            return removed
