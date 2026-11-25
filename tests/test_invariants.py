"""Deterministic synthetic verification suite for AGENTS.md invariants.

Verifies without heavy model downloads:
1. test_cmc_pan_recovery: foreground-masked affine CMC recovers (+14, -9)px pan within 1.5px.
2. test_depth_occlusion_vault_freeze: IoU >= 0.35 crossing locks farther track as
   DEPTH-OCCLUDED and freezes EpisodicVault with zero EMA poisoning.
3. test_60_frame_blackout_recovery: 60-frame blackout under camera panning keeps ghost alive,
   enforces >= 15f PROBATIONARY state, and recovers exact original ID (STITCH_COMMIT).
4. test_lookalike_margin_veto: two near-identical lost tracks (sim gap < 0.04) trigger
   MARGIN_VETO with 0 false identity merges.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from tracker import (
    CameraMotionCompensator,
    CleanViewAssessment,
    CleanViewGate,
    Detection,
    EpisodicVault,
    GhostConfig,
    LongHorizonTracker,
    OnlineHardNegativeMetric,
    TrackerConfig,
)
from tracker.engine import iou_matrix
from tracker.vault import l2_normalize


# ---------------------------------------------------------------------------
# Synthetic Frame & Embedding Helpers
# ---------------------------------------------------------------------------


def make_textured_canvas(height: int = 900, width: int = 1400, seed: int = 7) -> np.ndarray:
    """Create a rich, high-contrast textured synthetic background for sharp crops & LK flow."""
    rng = np.random.default_rng(seed)
    canvas = rng.integers(35, 220, size=(height, width, 3), dtype=np.uint8)
    # Add geometric corners/edges for reliable Shi-Tomasi keypoints and high Laplacian sharpness
    for _ in range(420):
        x1 = int(rng.integers(0, width - 40))
        y1 = int(rng.integers(0, height - 40))
        x2 = x1 + int(rng.integers(12, 42))
        y2 = y1 + int(rng.integers(12, 42))
        col = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, thickness=2)
    for _ in range(300):
        cx = int(rng.integers(15, width - 15))
        cy = int(rng.integers(15, height - 15))
        r = int(rng.integers(4, 18))
        col = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(canvas, (cx, cy), r, col, thickness=-1)
    return canvas


def crop_panned_frame(
    canvas: np.ndarray,
    cam_origin_x: float,
    cam_origin_y: float,
    out_h: int = 480,
    out_w: int = 640,
) -> np.ndarray:
    """Crop a viewport from the textured canvas using sub-pixel warpAffine for smooth panning."""
    M = np.array(
        [[1.0, 0.0, -float(cam_origin_x)], [0.0, 1.0, -float(cam_origin_y)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(canvas, M, (out_w, out_h), flags=cv2.INTER_LINEAR)


def make_unit_embedding(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(dim).astype(np.float32))


def perturb_embedding(base: np.ndarray, target_cos: float, seed: int) -> np.ndarray:
    """Return a unit vector with exact cosine similarity `target_cos` to `base`."""
    rng = np.random.default_rng(seed)
    base_u = l2_normalize(base)
    ortho = rng.standard_normal(base_u.shape).astype(np.float32)
    ortho = ortho - float(np.dot(ortho, base_u)) * base_u
    ortho = l2_normalize(ortho)
    sin_part = float(np.sqrt(max(0.0, 1.0 - target_cos * target_cos)))
    return l2_normalize(target_cos * base_u + sin_part * ortho)


# ---------------------------------------------------------------------------
# 1. CMC Pan Recovery
# ---------------------------------------------------------------------------


def test_cmc_pan_recovery() -> None:
    """Shift a textured synthetic frame by (+14px, -9px) with a moving foreground box
    and verify CMC recovers the shift within 1.5px.
    """
    canvas = make_textured_canvas(height=720, width=960, seed=101)
    h, w = 480, 640

    # To shift background image pixels by (+14, -9), shift viewport origin by (-14, +9)
    ox0, oy0 = 120.0, 120.0
    expected_dx, expected_dy = 14.0, -9.0

    frame0 = crop_panned_frame(canvas, ox0, oy0, out_h=h, out_w=w)
    frame1 = crop_panned_frame(canvas, ox0 - expected_dx, oy0 - expected_dy, out_h=h, out_w=w)

    # Inject a strongly textured foreground object moving in the OPPOSITE direction (-38, +26)
    fg_box0 = np.array([220.0, 160.0, 360.0, 340.0], dtype=np.float32)
    fg_box1 = np.array([182.0, 186.0, 322.0, 366.0], dtype=np.float32)
    fg_rng = np.random.default_rng(999)
    fg_patch = fg_rng.integers(0, 255, size=(180, 140, 3), dtype=np.uint8)
    frame0[160:340, 220:360] = fg_patch
    frame1[186:366, 182:322] = fg_patch

    cmc = CameraMotionCompensator(downscale=0.5)
    res0 = cmc.update(frame0, foreground_boxes=fg_box0.reshape(1, 4))
    assert res0.used_fallback is True  # First frame has no predecessor

    res1 = cmc.update(frame1, foreground_boxes=fg_box1.reshape(1, 4))
    assert res1.used_fallback is False, f"CMC unexpectedly fell back to identity: {res1}"
    assert abs(res1.dx - expected_dx) <= 1.5, f"Expected dx={expected_dx}, got {res1.dx:.2f}"
    assert abs(res1.dy - expected_dy) <= 1.5, f"Expected dy={expected_dy}, got {res1.dy:.2f}"
    assert res1.inlier_ratio >= 0.70


# ---------------------------------------------------------------------------
# 2. Depth-Triggered Occlusion Locking & EpisodicVault Freeze
# ---------------------------------------------------------------------------


def test_depth_occlusion_vault_freeze() -> None:
    """Pass a closer track (higher disparity) across a farther track at IoU >= 0.35;
    assert the farther track is flagged DEPTH-OCCLUDED and its EpisodicVault episode
    count and centroid do not change during overlap.
    """
    canvas = make_textured_canvas(height=600, width=800, seed=202)
    frame = crop_panned_frame(canvas, 50, 50, out_h=480, out_w=640)
    depth_map = np.full((480, 640), 0.20, dtype=np.float32)

    tracker = LongHorizonTracker(TrackerConfig(probation_frames=15))

    emb_far = make_unit_embedding(128, seed=11)
    emb_near = make_unit_embedding(128, seed=22)
    emb_poison = make_unit_embedding(128, seed=33)  # Orthogonal distractor fed during overlap

    box_far = np.array([280.0, 160.0, 360.0, 320.0], dtype=np.float32)  # 80x160
    depth_far = 0.35
    depth_near = 0.85

    # Phase 1: 12 clean, non-overlapping frames so both tracks populate clean vaults
    for f in range(12):
        bx_near = np.array([40.0 + f * 12.0, 160.0, 120.0 + f * 12.0, 320.0], dtype=np.float32)
        dets = [
            Detection(box=box_far.copy(), score=0.92, class_id=0, embedding=perturb_embedding(emb_far, 0.96, 100 + f), depth=depth_far),
            Detection(box=bx_near, score=0.92, class_id=0, embedding=perturb_embedding(emb_near, 0.96, 200 + f), depth=depth_near),
        ]
        res = tracker.update(frame, dets, depth_map=depth_map)

    assert len(tracker.active_tracks) == 2
    far_tid = min(tracker.active_tracks.values(), key=lambda t: t.depth_ema).track_id
    near_tid = max(tracker.active_tracks.values(), key=lambda t: t.depth_ema).track_id
    far_track = tracker.active_tracks[far_tid]

    pre_overlap_len = len(far_track.vault)
    pre_overlap_stored = far_track.vault.total_stored
    pre_overlap_centroid = far_track.vault.ema_embedding.copy()
    assert pre_overlap_len >= 2, "Far track should have clean vault entries before overlap"

    # Phase 2: Closer track sweeps right across farther track with IoU >= 0.35
    # Move smoothly so Kalman tracks stay locked while crossing at high IoU
    near_x_positions = [195.0, 225.0, 250.0, 265.0, 278.0, 290.0, 305.0]
    overlap_frames_checked = 0
    saw_depth_occluded_event = False

    for step, nx in enumerate(near_x_positions):
        bx_near = np.array([nx, 160.0, nx + 80.0, 320.0], dtype=np.float32)
        cur_iou = float(iou_matrix(box_far.reshape(1, 4), bx_near.reshape(1, 4))[0, 0])

        # Deliberately feed poisoned embedding to the farther track during overlap
        far_emb = emb_poison if cur_iou >= 0.25 else perturb_embedding(emb_far, 0.95, 300 + step)
        dets = [
            Detection(box=box_far.copy(), score=0.90, class_id=0, embedding=far_emb, depth=depth_far),
            Detection(box=bx_near, score=0.92, class_id=0, embedding=perturb_embedding(emb_near, 0.96, 400 + step), depth=depth_near),
        ]
        res = tracker.update(frame, dets, depth_map=depth_map)

        if any(ev.event_type == "DEPTH_OCCLUDED" and ev.track_id == far_tid for ev in res.events):
            saw_depth_occluded_event = True

        if cur_iou >= 0.35:
            overlap_frames_checked += 1
            tr_far = tracker.active_tracks[far_tid]
            assert tr_far.status == "DEPTH-OCCLUDED", f"Expected DEPTH-OCCLUDED at IoU={cur_iou:.2f}"
            assert tr_far.is_depth_occluded is True
            assert tr_far.occluder_id == near_tid
            assert tr_far.vault.frozen is True
            assert len(tr_far.vault) == pre_overlap_len, "Vault episode count changed during overlap!"
            assert tr_far.vault.total_stored == pre_overlap_stored
            assert np.allclose(tr_far.vault.ema_embedding, pre_overlap_centroid), "EMA poisoned during overlap!"

    assert overlap_frames_checked >= 4, "Test must exercise multiple frames at IoU >= 0.35"
    assert saw_depth_occluded_event is True

    # Phase 3: Separate tracks again -> farther track unfreezes cleanly
    bx_clear = np.array([460.0, 160.0, 540.0, 320.0], dtype=np.float32)
    dets = [
        Detection(box=box_far.copy(), score=0.92, class_id=0, embedding=perturb_embedding(emb_far, 0.95, 500), depth=depth_far),
        Detection(box=bx_clear, score=0.92, class_id=0, embedding=perturb_embedding(emb_near, 0.96, 501), depth=depth_near),
    ]
    tracker.update(frame, dets, depth_map=depth_map)
    assert tracker.active_tracks[far_tid].status == "ACTIVE"
    assert tracker.active_tracks[far_tid].vault.frozen is False


# ---------------------------------------------------------------------------
# 3. 60-Frame Blackout Recovery under Camera Panning
# ---------------------------------------------------------------------------


def test_60_frame_blackout_recovery() -> None:
    """Track Target A for 30 frames under camera panning, drop all detections for 60
    consecutive frames, then reintroduce Target A inside its reachable corridor with
    slight appearance drift; assert >= 15f PROBATIONARY and exact original ID recovery.
    """
    canvas = make_textured_canvas(height=900, width=1600, seed=303)
    h, w = 480, 640
    depth_map = np.full((h, w), 0.25, dtype=np.float32)

    cfg = TrackerConfig(
        probation_frames=15,
        stitch_min_sim=0.72,
        margin_min_diff=0.08,
        ghost_config=GhostConfig(max_ghost_age=180),
    )
    tracker = LongHorizonTracker(cfg)

    emb_a = make_unit_embedding(128, seed=77)
    # Slow steady camera pan across the textured canvas (world-to-camera shift = (-pan_x, -pan_y))
    cam_x, cam_y = 450.0, 200.0
    pan_vx, pan_vy = -2.0, 0.4  # camera viewport moves left -> image pixels shift (+2.0, -0.4) per frame

    # Target A world position (moves slowly right in world coordinates)
    world_cx, world_cy = 720.0, 420.0
    world_vx, world_vy = 0.6, 0.0
    bw, bh = 84.0, 168.0
    depth_a = 0.55

    def target_image_box(wx: float, wy: float, cx_cam: float, cy_cam: float) -> np.ndarray:
        ix = wx - cx_cam
        iy = wy - cy_cam
        return np.array([ix - bw / 2, iy - bh / 2, ix + bw / 2, iy + bh / 2], dtype=np.float32)

    # Phase 1: Track Target A for 30 frames under camera panning
    orig_id = None
    for f in range(30):
        frame = crop_panned_frame(canvas, cam_x, cam_y, out_h=h, out_w=w)
        box = target_image_box(world_cx, world_cy, cam_x, cam_y)
        det = Detection(
            box=box,
            score=0.92,
            class_id=0,
            embedding=perturb_embedding(emb_a, 0.95, 1000 + f),
            depth=depth_a,
        )
        tracker.update(frame, [det], depth_map=depth_map)
        if orig_id is None:
            assert len(tracker.active_tracks) == 1
            orig_id = next(iter(tracker.active_tracks.keys()))
        cam_x += pan_vx
        cam_y += pan_vy
        world_cx += world_vx
        world_cy += world_vy

    assert orig_id in tracker.active_tracks
    assert len(tracker.active_tracks[orig_id].vault) >= 4

    # Phase 2: 60-frame complete blackout while camera continues panning
    ghost_spawned = False
    for f in range(60):
        frame = crop_panned_frame(canvas, cam_x, cam_y, out_h=h, out_w=w)
        res = tracker.update(frame, [], depth_map=depth_map)
        if any(ev.event_type == "GHOST_SPAWN" and ev.track_id == orig_id for ev in res.events):
            ghost_spawned = True
        cam_x += pan_vx
        cam_y += pan_vy
        world_cx += world_vx
        world_cy += world_vy

    assert ghost_spawned is True, "Target A must transition to a LOST Ghost during blackout"
    assert orig_id in tracker.ghost_filter.ghosts, "Ghost must survive full 60-frame blackout"
    assert orig_id not in tracker.active_tracks

    # Phase 3: Reintroduce Target A with slight appearance drift (cos ~ 0.87 >= 0.72)
    stitch_frame = None
    stitch_events = []
    for f in range(1, 21):
        frame = crop_panned_frame(canvas, cam_x, cam_y, out_h=h, out_w=w)
        box = target_image_box(world_cx, world_cy, cam_x, cam_y)
        # Keep box comfortably inside viewport
        box[[0, 2]] = np.clip(box[[0, 2]], 20.0, w - 20.0)
        box[[1, 3]] = np.clip(box[[1, 3]], 20.0, h - 20.0)

        det = Detection(
            box=box,
            score=0.90,
            class_id=0,
            embedding=perturb_embedding(emb_a, 0.87, 2000 + f),
            depth=depth_a,
        )
        res = tracker.update(frame, [det], depth_map=depth_map)

        for ev in res.events:
            if ev.event_type == "STITCH_COMMIT":
                stitch_events.append(ev)
                if stitch_frame is None:
                    stitch_frame = f

        if f < 15:
            # Invariant 3: MUST stay PROBATIONARY for at least 15 frames (no instant merge)
            assert orig_id not in tracker.active_tracks, f"Premature merge at probation frame {f}!"
            assert len(tracker.probationary_tracks) == 1
            assert len(stitch_events) == 0

        cam_x += pan_vx
        cam_y += pan_vy
        world_cx += world_vx
        world_cy += world_vy

    assert stitch_frame == 15, f"Expected STITCH_COMMIT at probation frame 15, got {stitch_frame}"
    assert len(stitch_events) == 1
    assert stitch_events[0].track_id == orig_id
    assert orig_id in tracker.active_tracks
    assert orig_id not in tracker.ghost_filter.ghosts
    assert tracker.counters["stitches"] == 1


# ---------------------------------------------------------------------------
# 4. Look-Alike Margin-Ratio Abstention (MARGIN_VETO, 0 False Merges)
# ---------------------------------------------------------------------------


def test_lookalike_margin_veto() -> None:
    """Create two lost tracks with near-identical embeddings (similarity gap < 0.04);
    reintroduce one candidate and assert MARGIN_VETO triggers with 0 false identity merges.
    """
    canvas = make_textured_canvas(height=600, width=800, seed=404)
    frame = crop_panned_frame(canvas, 50, 50, out_h=480, out_w=640)
    depth_map = np.full((480, 640), 0.25, dtype=np.float32)

    cfg = TrackerConfig(
        probation_frames=15,
        max_probation_frames=22,
        stitch_min_sim=0.72,
        margin_min_diff=0.08,
    )
    tracker = LongHorizonTracker(cfg)

    # Two look-alike identities with cosine similarity 0.96 -> gap to midpoint query < 0.04
    emb_1 = make_unit_embedding(128, seed=501)
    emb_2 = perturb_embedding(emb_1, target_cos=0.96, seed=502)
    # Candidate sits between emb_1 and emb_2 (slightly closer to emb_1 by < 0.02)
    emb_query = l2_normalize(0.56 * emb_1 + 0.44 * emb_2)
    sim_gap = abs(float(np.dot(emb_query, emb_1)) - float(np.dot(emb_query, emb_2)))
    assert sim_gap < 0.04, f"Expected similarity gap < 0.04, got {sim_gap:.4f}"

    box_1 = np.array([100.0, 150.0, 180.0, 310.0], dtype=np.float32)
    box_2 = np.array([420.0, 150.0, 500.0, 310.0], dtype=np.float32)

    # Phase 1: Establish both clean tracks for 18 frames
    for f in range(18):
        dets = [
            Detection(box=box_1.copy(), score=0.92, class_id=0, embedding=perturb_embedding(emb_1, 0.995, 600 + f), depth=0.55),
            Detection(box=box_2.copy(), score=0.92, class_id=0, embedding=perturb_embedding(emb_2, 0.995, 700 + f), depth=0.55),
        ]
        tracker.update(frame, dets, depth_map=depth_map)

    assert len(tracker.active_tracks) == 2
    lost_ids = set(tracker.active_tracks.keys())

    # Phase 2: Drop both tracks for max_active_miss (25) frames so both become LOST ghosts
    for _ in range(cfg.max_active_miss):
        tracker.update(frame, [], depth_map=depth_map)

    assert lost_ids.issubset(set(tracker.ghost_filter.ghosts.keys()))
    assert len(tracker.active_tracks) == 0

    # Phase 3: Reintroduce one ambiguous look-alike candidate for 28 frames
    box_cand = np.array([260.0, 150.0, 340.0, 310.0], dtype=np.float32)
    veto_events = []
    stitch_events = []
    graduated_after_veto = False

    for f in range(28):
        det = Detection(
            box=box_cand.copy(),
            score=0.90,
            class_id=0,
            embedding=perturb_embedding(emb_query, 0.995, 800 + f),
            depth=0.55,
        )
        res = tracker.update(frame, [det], depth_map=depth_map)
        for ev in res.events:
            if ev.event_type == "MARGIN_VETO":
                veto_events.append(ev)
            elif ev.event_type == "STITCH_COMMIT":
                stitch_events.append(ev)
            elif ev.event_type == "NEW_ID_AFTER_VETO":
                graduated_after_veto = True

    assert len(veto_events) >= 1, "Expected MARGIN_VETO on ambiguous look-alike candidates"
    assert tracker.counters["margin_vetoes"] >= 1
    assert len(stitch_events) == 0, "Expected 0 false identity merges!"
    assert tracker.counters["stitches"] == 0
    assert graduated_after_veto is True
    # Neither lost look-alike ID was falsely hijacked
    assert lost_ids.isdisjoint(set(tracker.active_tracks.keys()))
    assert len(tracker.active_tracks) == 1


# ---------------------------------------------------------------------------
# 5. Vault Cache Equivalence & Clean-View Centroid Purity
# ---------------------------------------------------------------------------


def test_metric_vault_cache_equivalence() -> None:
    """Verify _get_transformed_vault cache matches uncached computation to 1e-6 after
    both a vault eviction (constant len(entries)) and an online metric whitening update,
    empty query_embs short-circuits before cache lookup, and vault.ema_embedding only
    updates after passing CleanViewGate.
    """
    box = np.array([100.0, 100.0, 180.0, 260.0], dtype=np.float32)
    vault = EpisodicVault(max_entries=4, min_frame_gap=1, redundancy_cos_thresh=0.999)
    peer_vault = EpisodicVault(max_entries=4, min_frame_gap=1, redundancy_cos_thresh=0.999)
    metric = OnlineHardNegativeMetric(recompute_interval=1)

    # 1. vault.ema_embedding must NOT be populated by seed_unverified or rejected/frozen views
    e_seed = make_unit_embedding(64, seed=900)
    vault.seed_unverified(e_seed, frame_idx=0, depth=0.5, box=box, has_overlap=False)
    assert vault.ema_embedding is None, "seed_unverified must not set ema_embedding"

    rej_overlap = CleanViewAssessment(False, 0.0, "REJECT_OVERLAP_IOU", max_iou=0.20)
    ok, reason = vault.try_add(make_unit_embedding(64, seed=901), 1, rej_overlap, 0.5, box)
    assert not ok and reason == "REJECT_OVERLAP_IOU"
    assert vault.ema_embedding is None

    vault.freeze()
    clean_ok = CleanViewAssessment(True, 0.85, "CLEAN", sharpness=120.0, max_iou=0.0)
    ok, reason = vault.try_add(make_unit_embedding(64, seed=902), 2, clean_ok, 0.5, box)
    assert not ok and reason == "VAULT_FROZEN"
    assert vault.ema_embedding is None
    vault.unfreeze()

    # 2. Short-circuit empty query_embs BEFORE touching _get_transformed_vault cache
    assert metric.set_to_set_similarity(None, vault) == 0.0
    assert metric.set_to_set_similarity(np.zeros((0, 64), dtype=np.float32), vault) == 0.0
    assert len(metric._vault_t_cache) == 0, "Empty query_embs must short-circuit before cache lookup"

    # 3. Populate vault to capacity (4 entries) with clean views -> ema_embedding updates cleanly
    base = make_unit_embedding(64, seed=910)
    peer_base = make_unit_embedding(64, seed=920)
    init_embs = [
        base,
        perturb_embedding(base, 0.98, seed=911),
        perturb_embedding(base, 0.98, seed=912),
        perturb_embedding(base, 0.97, seed=913),
    ]
    for i, e in enumerate(init_embs):
        added, _ = vault.try_add(e, frame_idx=10 + i, assessment=clean_ok, depth=0.5, box=box)
        assert added is True
    assert len(vault) == 4
    assert vault.ema_embedding is not None

    # Populate peer vault for online co-occurring scatter updates
    for i in range(4):
        pe = perturb_embedding(peer_base, 0.80, seed=930 + i)
        peer_vault.try_add(pe, frame_idx=10 + i, assessment=clean_ok, depth=0.6, box=box)

    q_single = perturb_embedding(base, 0.82, seed=940)
    q_set = np.stack([q_single, perturb_embedding(base, 0.78, seed=941), perturb_embedding(base, 0.85, seed=942)], axis=0)

    # Helper computing uncached reference directly from raw embeddings & metric transform
    def uncached_single(q: np.ndarray, v: EpisodicVault) -> float:
        mat_t = metric._apply_metric_transform(v.get_embeddings())
        q_t = metric._apply_metric_transform(l2_normalize(q))
        sims = mat_t @ q_t
        k = min(3, len(sims))
        top_mean = float(np.mean(np.sort(sims)[-k:]))
        ema_sim = float(np.dot(metric._apply_metric_transform(v.ema_embedding), q_t))
        return float(np.clip(0.65 * top_mean + 0.35 * ema_sim, 0.0, 1.0))

    def uncached_set(qs: np.ndarray, v: EpisodicVault) -> float:
        q_t = metric._apply_metric_transform(qs)
        g_t = metric._apply_metric_transform(v.get_embeddings())
        pair = q_t @ g_t.T
        k_g = min(3, pair.shape[1])
        per_q = np.mean(np.sort(pair, axis=1)[:, -k_g:], axis=1)
        k_q = max(1, (len(per_q) + 1) // 2)
        return float(np.clip(np.mean(np.sort(per_q)[-k_q:]), 0.0, 1.0))

    # Warm cache before eviction
    _ = metric.similarity(q_single, vault, apply_hn_penalty=False)
    assert vault._uid in metric._vault_t_cache

    # 4. Trigger vault EVICTION (len(vault) stays 4, but vault._version MUST increment!)
    v_before = vault._version
    novel_emb = perturb_embedding(base, 0.45, seed=950)
    evicted, ev_reason = vault.try_add(
        novel_emb,
        frame_idx=20,
        assessment=CleanViewAssessment(True, 0.99, "CLEAN", sharpness=160.0, max_iou=0.0),
        depth=0.5,
        box=box,
    )
    assert evicted is True and ev_reason == "VAULT_STORE_EVICT"
    assert len(vault) == 4, "Length stays constant on eviction"
    assert vault._version > v_before, "Mutation version must increment on eviction"

    cached_s1 = metric.similarity(novel_emb, vault, apply_hn_penalty=False)
    ref_s1 = uncached_single(novel_emb, vault)
    assert abs(cached_s1 - ref_s1) <= 1e-6, f"Post-eviction single mismatch: {cached_s1} vs {ref_s1}"

    cached_set1 = metric.set_to_set_similarity(q_set, vault)
    ref_set1 = uncached_set(q_set, vault)
    assert abs(cached_set1 - ref_set1) <= 1e-6, f"Post-eviction set mismatch: {cached_set1} vs {ref_set1}"

    # 5. Trigger online metric whitening/scatter update (metric._metric_version increments)
    m_before = metric._metric_version
    for f in range(30, 36):
        metric.observe_cooccurring({1: vault, 2: peer_vault}, frame_idx=f)
    assert metric._metric_version > m_before
    assert metric._Q is not None, "Dual whitening factors should be active after scatter updates"

    cached_s2 = metric.similarity(q_single, vault, apply_hn_penalty=False)
    ref_s2 = uncached_single(q_single, vault)
    assert abs(cached_s2 - ref_s2) <= 1e-6, f"Post-metric-update single mismatch: {cached_s2} vs {ref_s2}"

    cached_set2 = metric.set_to_set_similarity(q_set, vault)
    ref_set2 = uncached_set(q_set, vault)
    assert abs(cached_set2 - ref_set2) <= 1e-6, f"Post-metric-update set mismatch: {cached_set2} vs {ref_set2}"

    # Verify repeated hit from cache remains exact to 1e-6
    assert abs(metric.similarity(q_single, vault, apply_hn_penalty=False) - ref_s2) <= 1e-6
    assert abs(metric.set_to_set_similarity(q_set, vault) - ref_set2) <= 1e-6


# ---------------------------------------------------------------------------
# 6. 3-Person 5-Second (150-Frame) Static Pillar Occlusion Recovery
# ---------------------------------------------------------------------------


def test_three_person_5sec_pillar_recovery() -> None:
    """Simulate 3 distinct people (IDs 1, 2, 3): Person A (ID 1) walks behind a closer
    static depth pillar (disparity 0.90 vs person 0.45) and disappears completely for
    150 frames (5 seconds @ 30 FPS) while Person B (ID 2) and Person C (ID 3) remain
    visible and move around; Person A steps out from the opposite side of the pillar
    with noisy half-occluded edge frames followed by clean slightly drifted views.

    Assert:
    - Ghost #1 survives the full 150-frame pillar occlusion with high confidence.
    - Half-occluded pillar-edge frames do NOT dilute clean probationary query gallery.
    - After >= 15 probationary frames, Person A recovers exact ID 1 (STITCH_COMMIT).
    - 0 new IDs are minted (only IDs 1, 2, 3 ever exist), and Person A is never merged
      into Person B (ID 2) or Person C (ID 3).
    """
    canvas = make_textured_canvas(height=600, width=800, seed=606)
    frame = crop_panned_frame(canvas, 50, 50, out_h=480, out_w=640)

    # Static depth map: background 0.25, wide closer pillar at x=[250..410] with disparity 0.90
    depth_map = np.full((480, 640), 0.25, dtype=np.float32)
    depth_map[:, 250:410] = 0.90

    cfg = TrackerConfig(
        probation_frames=15,
        max_probation_frames=40,
        stitch_min_sim=0.72,
        margin_min_diff=0.08,
        ghost_config=GhostConfig(max_ghost_age=600),
    )
    tracker = LongHorizonTracker(cfg)

    # 3 distinct person identities
    emb_a = make_unit_embedding(128, seed=611)
    emb_b = make_unit_embedding(128, seed=612)
    emb_c = make_unit_embedding(128, seed=613)
    emb_pillar_noise = make_unit_embedding(128, seed=699)  # Half-occluded pillar-edge noise

    def box_xy(cx: float, cy: float, bw: float = 76.0, bh: float = 152.0) -> np.ndarray:
        return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)

    def moving_b_box(f: int) -> np.ndarray:
        # Person B walks smoothly on left side (never overlapping A or C)
        cx = 110.0 + 35.0 * np.sin(f * 0.05)
        return box_xy(cx, 370.0)

    def moving_c_box(f: int) -> np.ndarray:
        # Person C walks smoothly on right-bottom side (never overlapping A's exit)
        cx = 525.0 + 30.0 * np.cos(f * 0.04)
        return box_xy(cx, 380.0)

    # Phase 1 (Frames 1..25): All 3 people visible; Person A walks toward left edge of pillar
    for f in range(1, 26):
        ax = 110.0 + 5.6 * (f - 1)  # 110 -> 244 (steps into pillar edge at x=250)
        dets = [
            Detection(box=box_xy(ax, 180.0), score=0.92, class_id=0, embedding=perturb_embedding(emb_a, 0.96, 6100 + f), depth=0.45),
            Detection(box=moving_b_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_b, 0.96, 6200 + f), depth=0.55),
            Detection(box=moving_c_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_c, 0.96, 6300 + f), depth=0.62),
        ]
        tracker.update(frame, dets, depth_map=depth_map)

    assert set(tracker.active_tracks.keys()) == {1, 2, 3}
    assert len(tracker.active_tracks[1].vault) >= 4

    # Phase 2 (Frames 26..175): 150-frame (5-second) complete disappearance of Person A behind pillar
    ghost_a_spawned = False
    for f in range(26, 176):
        dets = [
            Detection(box=moving_b_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_b, 0.96, 6200 + f), depth=0.55),
            Detection(box=moving_c_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_c, 0.96, 6300 + f), depth=0.62),
        ]
        res = tracker.update(frame, dets, depth_map=depth_map)
        if any(ev.event_type == "GHOST_SPAWN" and ev.track_id == 1 for ev in res.events):
            ghost_a_spawned = True
        assert {2, 3}.issubset(set(tracker.active_tracks.keys())), f"B/C lost at frame {f}"

    assert ghost_a_spawned is True, "Person A must spawn a LOST ghost behind the pillar"
    assert 1 in tracker.ghost_filter.ghosts, "Ghost #1 must survive full 150-frame (5s) pillar occlusion"
    assert tracker.ghost_filter.ghosts[1].confidence >= 0.40, (
        f"Pillar-occluded ghost decayed too fast: conf={tracker.ghost_filter.ghosts[1].confidence:.3f}"
    )

    # Phase 3 (Frames 176..200): Person A emerges from opposite (right) side of pillar (x=425 -> 545)
    # First 5 frames are half-occluded/low-conf pillar-edge crops (conf=0.42, heavily noisy embedding);
    # remaining frames are clean views with slight appearance drift (cos=0.84 to emb_a).
    stitch_events = []
    stitch_prob_hit = None
    for idx, f in enumerate(range(176, 201), start=1):
        ax = min(560.0, 425.0 + 4.5 * (idx - 1))
        if idx <= 5:
            # Half-occluded pillar-edge frame: fails CleanViewGate (conf < 0.50), noisy embedding
            det_a = Detection(
                box=box_xy(ax, 180.0),
                score=0.42,
                class_id=0,
                embedding=l2_normalize(0.45 * emb_a + 0.55 * emb_pillar_noise),
                depth=0.45,
            )
        else:
            # Clean full-body view after clearing pillar edge, with slight appearance drift
            det_a = Detection(
                box=box_xy(ax, 180.0),
                score=0.90,
                class_id=0,
                embedding=perturb_embedding(emb_a, 0.84, 6400 + f),
                depth=0.45,
            )

        dets = [
            det_a,
            Detection(box=moving_b_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_b, 0.96, 6200 + f), depth=0.55),
            Detection(box=moving_c_box(f), score=0.92, class_id=0, embedding=perturb_embedding(emb_c, 0.96, 6300 + f), depth=0.62),
        ]
        res = tracker.update(frame, dets, depth_map=depth_map)

        for ev in res.events:
            if ev.event_type == "STITCH_COMMIT":
                stitch_events.append(ev)
                if stitch_prob_hit is None:
                    stitch_prob_hit = idx

        if idx < 15:
            # Invariant 3: MUST remain PROBATIONARY for at least 15 frames
            assert 1 not in tracker.active_tracks, f"Premature merge at probation frame {idx}"
            assert len(stitch_events) == 0

    assert stitch_prob_hit == 15, f"Expected STITCH_COMMIT at probation frame 15, got {stitch_prob_hit}"
    assert len(stitch_events) == 1
    assert stitch_events[0].track_id == 1, f"Expected recovery to ID 1, got #{stitch_events[0].track_id}"
    assert set(tracker.active_tracks.keys()) == {1, 2, 3}
    assert tracker.counters["new_ids"] == 3, f"Expected 0 new IDs minted (only 1, 2, 3), got {tracker.counters['new_ids']}"
    assert tracker.counters["stitches"] == 1
    assert 1 not in tracker.ghost_filter.ghosts
