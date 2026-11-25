"""Side-by-side comparison video renderer: NaiveEMATracker vs. LongHorizonTracker.

Runs both trackers on the exact same YOLOv8n detections, DINOv2 embeddings, and
Depth-Anything-V2 depth maps, rendering a synchronized side-by-side comparison to
`output/comparison_side_by_side.mp4`.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from eval_metrics import NaiveEMATracker
from run_tracker import (
    AppearanceEncoder,
    AsyncVideoWriter,
    DepthEstimator,
    YOLODetector,
    draw_dashed_rect,
    draw_label_pill,
    extract_box_depth,
    resolve_device,
    track_color_bgr,
    upload_frame_gpu,
    warp_depth_map,
)
from tracker import (
    ActiveTrack,
    Detection,
    GhostConfig,
    GhostState,
    LongHorizonTracker,
    ProbationaryTrack,
    TrackerConfig,
    TrackingFrameResult,
)


def _scale_box(box: np.ndarray, sx: float, sy: float) -> Tuple[int, int, int, int]:
    return (
        int(round(float(box[0]) * sx)),
        int(round(float(box[1]) * sy)),
        int(round(float(box[2]) * sx)),
        int(round(float(box[3]) * sy)),
    )


def _draw_header_bar(
    panel: np.ndarray,
    title: str,
    subtitle: str,
    accent_bgr: Tuple[int, int, int],
) -> None:
    h, w = panel.shape[:2]
    bar_h = 54
    roi = panel[0:bar_h, 0:w]
    bg = np.full_like(roi, (18, 20, 26), dtype=np.uint8)
    cv2.addWeighted(bg, 0.84, roi, 0.16, 0.0, roi)
    cv2.line(panel, (0, bar_h - 1), (w, bar_h - 1), accent_bgr, 2, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (6, bar_h - 1), accent_bgr, -1)

    cv2.putText(
        panel,
        title,
        (16, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (16, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        accent_bgr,
        1,
        cv2.LINE_AA,
    )


def render_baseline_panel(
    panel: np.ndarray,
    preds: List[Tuple[int, np.ndarray]],
    trails: Dict[int, Deque[Tuple[float, float]]],
    unique_ids_count: int,
    instant_merges: int,
    sx: float,
    sy: float,
) -> None:
    """Render left panel: Baseline (2D IoU + EMA)."""
    for tid, box in preds:
        color = track_color_bgr(tid)
        x1, y1, x2, y2 = _scale_box(box, sx, sy)

        # Update & draw trail
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        tr = trails.setdefault(tid, deque(maxlen=24))
        tr.append((cx, cy))
        if len(tr) >= 2:
            pts = list(tr)
            for i in range(1, len(pts)):
                cv2.line(
                    panel,
                    (int(round(pts[i - 1][0])), int(round(pts[i - 1][1]))),
                    (int(round(pts[i][0])), int(round(pts[i][1]))),
                    color,
                    1 if i < len(pts) // 2 else 2,
                    cv2.LINE_AA,
                )

        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_label_pill(
            panel,
            f"ID #{tid}",
            (x1, max(72, y1)),
            color,
            (15, 15, 15),
            font_scale=0.45,
            thickness=1,
        )

    _draw_header_bar(
        panel,
        "Baseline (2D IoU + EMA)",
        f"Active: {len(preds)}  |  Total Unique IDs Created: {unique_ids_count}  |  Instant Merges: {instant_merges}",
        accent_bgr=(90, 110, 255),
    )


def _draw_ghosts_scaled(
    panel: np.ndarray,
    ghosts: List[GhostState],
    sx: float,
    sy: float,
) -> None:
    if not ghosts:
        return
    h, w = panel.shape[:2]

    for g in ghosts:
        color = track_color_bgr(g.track_id)
        p = g.particles
        weights = g.weights
        w_max = max(1e-6, float(np.max(weights)))

        px_s = p[:, 0] * sx
        py_s = p[:, 1] * sy

        x1, y1, x2, y2 = _scale_box(g.mean_box, sx, sy)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cov_s = np.array(
            [
                [g.cov_xy[0, 0] * sx * sx, g.cov_xy[0, 1] * sx * sy],
                [g.cov_xy[1, 0] * sx * sy, g.cov_xy[1, 1] * sy * sy],
            ],
            dtype=np.float32,
        )
        eigvals, eigvecs = np.linalg.eigh(cov_s)
        eigvals = np.maximum(eigvals, 9.0)
        ax1 = int(np.clip(1.5 * math.sqrt(float(eigvals[1])), 6, w // 2))
        ax2 = int(np.clip(1.5 * math.sqrt(float(eigvals[0])), 6, h // 2))
        angle = float(math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1])))
        ell_r = max(ax1, ax2)

        pad = 6
        rx1 = max(0, int(min(float(np.min(px_s)), cx - ell_r, x1)) - pad)
        ry1 = max(54, int(min(float(np.min(py_s)), cy - ell_r, y1)) - pad)
        rx2 = min(w, int(max(float(np.max(px_s)), cx + ell_r, x2)) + pad)
        ry2 = min(h, int(max(float(np.max(py_s)), cy + ell_r, y2)) + pad)

        if rx2 > rx1 and ry2 > ry1:
            roi = panel[ry1:ry2, rx1:rx2]
            roi_ov = roi.copy()
            rw, rh = rx2 - rx1, ry2 - ry1
            px_i = np.rint(px_s).astype(np.int32) - rx1
            py_i = np.rint(py_s).astype(np.int32) - ry1
            in_roi = (px_i >= 0) & (px_i < rw) & (py_i >= 0) & (py_i < rh)
            half_w = 0.5 * w_max
            for idx in np.flatnonzero(in_roi):
                r = 1 if weights[idx] < half_w else 2
                cv2.circle(roi_ov, (int(px_i[idx]), int(py_i[idx])), r, color, -1, cv2.LINE_AA)

            cv2.ellipse(roi_ov, (cx - rx1, cy - ry1), (ax1, ax2), angle, 0, 360, color, 2, cv2.LINE_AA)
            draw_dashed_rect(roi_ov, (x1 - rx1, y1 - ry1), (x2 - rx1, y2 - ry1), color, thickness=1, dash_len=5, gap_len=4)
            cv2.addWeighted(roi_ov, 0.65, roi, 0.35, 0.0, roi)

        draw_label_pill(
            panel,
            f"GHOST #{g.track_id} ({g.age}f)",
            (x1, max(72, y1 - 2)),
            (45, 25, 65),
            color,
            font_scale=0.38,
        )


def render_ours_panel(
    panel: np.ndarray,
    result: TrackingFrameResult,
    unique_ids_count: int,
    probation_frames: int,
    sx: float,
    sy: float,
) -> None:
    """Render right panel: LongHorizonTracker (Ours) with state badges & ghost clouds."""
    # 1. Ghost particle clouds
    _draw_ghosts_scaled(panel, result.ghosts, sx, sy)

    # 2. Active tracks with explicit ACTIVE / DEPTH-OCCLUDED / RECOVERED badges
    for tr in result.active_tracks:
        color = track_color_bgr(tr.track_id)
        x1, y1, x2, y2 = _scale_box(tr.current_box(), sx, sy)

        if len(tr.history) >= 2:
            pts = list(tr.history)
            for i in range(1, len(pts)):
                p0 = (int(round(pts[i - 1][0] * sx)), int(round(pts[i - 1][1] * sy)))
                p1 = (int(round(pts[i][0] * sx)), int(round(pts[i][1] * sy)))
                cv2.line(panel, p0, p1, color, 1 if i < len(pts) // 2 else 2, cv2.LINE_AA)

        is_depth_occ = tr.is_depth_occluded or tr.status == "DEPTH-OCCLUDED"
        is_recovered = tr.stitched_count > 0

        if is_depth_occ:
            box_col = (0, 125, 255)
            badge_txt = "DEPTH-OCCLUDED"
            badge_bg = (0, 75, 220)
            badge_fg = (255, 255, 255)
        elif is_recovered:
            box_col = (255, 235, 50)
            badge_txt = f"RECOVERED (x{tr.stitched_count})"
            badge_bg = (180, 145, 0)
            badge_fg = (255, 255, 255)
        else:
            box_col = color
            badge_txt = "ACTIVE"
            badge_bg = (30, 135, 55)
            badge_fg = (255, 255, 255)

        if tr.time_since_update > 0:
            draw_dashed_rect(panel, (x1, y1), (x2, y2), box_col, thickness=2)
        else:
            cv2.rectangle(panel, (x1, y1), (x2, y2), box_col, 2, cv2.LINE_AA)

        header = f"#{tr.track_id} | d:{tr.depth_ema:.2f} | V:{len(tr.vault)}"
        draw_label_pill(panel, header, (x1, max(72, y1)), box_col, (15, 15, 15), font_scale=0.43, thickness=1)
        draw_label_pill(panel, badge_txt, (x1, y2 + 15), badge_bg, badge_fg, font_scale=0.37, thickness=1)

    # 3. Probationary tracks with explicit PROBATIONARY badge & candidate link
    ghost_map = {g.track_id: g for g in result.ghosts}
    amber = (40, 215, 255)
    veto_col = (180, 40, 255)
    for pr in result.probationary_tracks:
        x1, y1, x2, y2 = _scale_box(pr.current_box(), sx, sy)
        col = veto_col if pr.last_veto_reason else amber
        draw_dashed_rect(panel, (x1, y1), (x2, y2), col, thickness=2, dash_len=6, gap_len=4)
        p_label = f"PROBATIONARY P{pr.prob_id} ({pr.hits}/{probation_frames})"
        draw_label_pill(panel, p_label, (x1, max(72, y1)), (30, 30, 30), col, font_scale=0.39)
        if pr.last_veto_reason:
            draw_label_pill(panel, "ABSTAINED (VETO)", (x1, y2 + 15), veto_col, (255, 255, 255), font_scale=0.36)

        if pr.best_candidate_ghost is not None and pr.best_candidate_ghost in ghost_map:
            g = ghost_map[pr.best_candidate_ghost]
            gx1, gy1, gx2, gy2 = _scale_box(g.mean_box, sx, sy)
            cv2.line(panel, ((x1 + x2) // 2, (y1 + y2) // 2), ((gx1 + gx2) // 2, (gy1 + gy2) // 2), col, 1, cv2.LINE_AA)

    st = result.stats
    _draw_header_bar(
        panel,
        "LongHorizonTracker (Ours)",
        (
            f"Unique IDs: {unique_ids_count}  |  Recovered: {st['stitches']}  |  "
            f"Abstained: {st['margin_vetoes']}  |  Ghosts: {st['ghost_count']}  |  "
            f"DepthLock: {st['depth_occlusions']}"
        ),
        accent_bgr=(255, 225, 80),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render side-by-side Baseline vs LongHorizonTracker video")
    parser.add_argument("--video", type=str, default="data/test_video.mp4", help="Input video path")
    parser.add_argument(
        "--output",
        type=str,
        default="output/comparison_side_by_side.mp4",
        help="Output side-by-side comparison video path",
    )
    parser.add_argument("--detector", type=str, default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="COCO class IDs (0=person)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--depth-stride", type=int, default=2, help="Run Depth-Anything-V2 every K frames")
    parser.add_argument("--panel-width", type=int, default=960, help="Width of each half panel (total = 2x + 4)")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cuda, cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap (0 = full video)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[Comparison] Device={device} | Video={video_path} | Output={out_path}")

    # Shared front-end models (exact same inputs fed to both trackers!)
    detector = YOLODetector(
        model_name=args.detector,
        classes=args.classes if args.classes else None,
        conf=args.conf,
        device=device,
    )
    encoder = AppearanceEncoder(backend="dinov2", device=device)
    depth_est = DepthEstimator(backend="depth_anything", device=device)

    # Both trackers
    baseline = NaiveEMATracker()
    lh_cfg = TrackerConfig(
        det_thresh=args.conf,
        probation_frames=15,
        ghost_config=GhostConfig(max_ghost_age=180),
    )
    ours = LongHorizonTracker(config=lh_cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_w = args.panel_width
    panel_h = int(round(panel_w * (src_h / max(1, src_w))))
    sx = panel_w / max(1.0, float(src_w))
    sy = panel_h / max(1.0, float(src_h))
    divider_w = 4
    out_w = panel_w * 2 + divider_w
    out_h = panel_h

    print(f"[Video] Source={src_w}x{src_h} @ {src_fps:.2f} FPS ({total_frames} frames) -> Side-by-side={out_w}x{out_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h), max_queue=32)

    depth_stride = max(1, args.depth_stride)
    use_gpu_frame = device == "cuda" and (encoder._model is not None or depth_est._model is not None)
    last_depth_map: Optional[np.ndarray] = None

    baseline_unique_ids: Set[int] = set()
    ours_unique_ids: Set[int] = set()
    baseline_trails: Dict[int, Deque[Tuple[float, float]]] = {}

    frame_count = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        # 1. Shared GPU upload + YOLOv8n + Depth-Anything-V2 (CMC-warped on stride) + DINOv2
        frame_gpu = upload_frame_gpu(frame, device) if use_gpu_frame else None
        boxes, scores, cls_ids = detector.detect(frame)

        run_depth_now = last_depth_map is None or (frame_count - 1) % depth_stride == 0
        cmc_res = None
        if run_depth_now:
            last_depth_map = depth_est.predict(frame, frame_gpu=frame_gpu)
        else:
            cmc_res = ours.cmc.update(frame, boxes)
            last_depth_map = warp_depth_map(last_depth_map, cmc_res.matrix, frame.shape[:2])

        embeddings = encoder.extract(frame, boxes, frame_gpu=frame_gpu)

        detections: List[Detection] = []
        for i in range(len(boxes)):
            if float(scores[i]) < args.conf:
                continue
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

        # 2. Step BOTH trackers on the exact same detections & depth map
        base_preds = baseline.update(frame, detections, depth_map=last_depth_map)
        for tid, _ in base_preds:
            baseline_unique_ids.add(tid)

        ours_res = ours.update(frame, detections, depth_map=last_depth_map, cmc_result=cmc_res)
        for tr in ours_res.active_tracks:
            ours_unique_ids.add(tr.track_id)

        # 3. Render synchronized side-by-side panels
        base_small = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        left_panel = base_small.copy()
        right_panel = base_small

        render_baseline_panel(
            left_panel,
            base_preds,
            baseline_trails,
            unique_ids_count=len(baseline_unique_ids),
            instant_merges=len(baseline.recovery_events),
            sx=sx,
            sy=sy,
        )
        render_ours_panel(
            right_panel,
            ours_res,
            unique_ids_count=len(ours_unique_ids),
            probation_frames=lh_cfg.probation_frames,
            sx=sx,
            sy=sy,
        )

        combined = np.empty((out_h, out_w, 3), dtype=np.uint8)
        combined[:, :panel_w] = left_panel
        combined[:, panel_w : panel_w + divider_w] = (230, 235, 245)
        combined[:, panel_w + divider_w :] = right_panel

        # Shared centered frame badge at bottom
        draw_label_pill(
            combined,
            f"Frame {frame_count:04d} / {total_frames}",
            (panel_w - 64, out_h - 10),
            (18, 20, 26),
            (235, 240, 248),
            font_scale=0.45,
        )

        writer.write(combined)

        if frame_count % 100 == 0 or frame_count == 1:
            elapsed_now = max(1e-3, time.perf_counter() - t_start)
            print(
                f"[F{frame_count:04d}/{total_frames}] {frame_count / elapsed_now:5.1f} FPS | "
                f"Baseline IDs={len(baseline_unique_ids)} (merges={len(baseline.recovery_events)}) | "
                f"Ours IDs={len(ours_unique_ids)} (recovered={ours.counters['stitches']}, abstained={ours.counters['margin_vetoes']})"
            )

    cap.release()
    writer.close()

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_count / max(1e-3, elapsed)
    summary = {
        "video": str(video_path),
        "output": str(out_path),
        "frames_processed": frame_count,
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(avg_fps, 2),
        "baseline": {
            "total_unique_ids": len(baseline_unique_ids),
            "next_id_minus_1": baseline._next_id - 1,
            "instant_reid_merges": len(baseline.recovery_events),
        },
        "long_horizon_tracker": {
            "total_unique_ids": len(ours_unique_ids),
            "new_ids_counter": ours.counters["new_ids"],
            "recovered_stitches": ours.counters["stitches"],
            "abstained_margin_vetoes": ours.counters["margin_vetoes"],
            "depth_occlusions": ours.counters["depth_occlusions"],
            "ghosts_spawned": ours.counters["ghosts_spawned"],
            "ghosts_expired": ours.counters["ghosts_expired"],
        },
    }
    summary_path = out_path.parent / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[Done] Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.2f} FPS)")
    print(f"[Baseline] Total Unique IDs Created: {len(baseline_unique_ids)} (instant merges: {len(baseline.recovery_events)})")
    print(
        f"[Ours]     Total Unique IDs Created: {len(ours_unique_ids)} "
        f"(recovered: {ours.counters['stitches']}, abstained: {ours.counters['margin_vetoes']}, "
        f"depth-occluded: {ours.counters['depth_occlusions']})"
    )
    print(f"[Outputs]  Video: {out_path} | Summary: {summary_path}")


if __name__ == "__main__":
    main()
