"""Spatial Analytics, Traffic Heatmaps, and Zero-Cost Pose Behavior/Anomaly Detection.

All modules operate purely on Pass-2 consolidated tracking outputs (boxes + 17 COCO keypoints)
with zero extra neural-network forward passes and zero changes to `tracker/engine.py`.

Components:
1. `ZoneAnalyzer` + `draw_zones_overlay`: normalized polygon zones, preset templates,
   foot-anchor `cv2.pointPolygonTest`, dwell/entries/exits/line-crossings, `LOITERING_ALERT`.
2. `HeatmapGenerator`: Gaussian foot-point density accumulator -> `cv2.COLORMAP_TURBO`
   blended (`alpha=0.45`) over a clean median/first background frame -> RGB preview + PNG bytes.
3. `PoseBehaviorAnalyzer`: 17-keypoint + velocity classifier with 5-frame temporal smoothing
   (`FALL_ALERT`, `RUNNING`, `HANDS_RAISED`, `SITTING`, `STANDING`) + structured anomaly log.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Custom Polygon Zones & Dwell / Loitering Engine
# ---------------------------------------------------------------------------

ZONE_PRESETS: Dict[str, List[Tuple[float, float]]] = {
    "Full Frame": [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)],
    "Left Half / Counter": [(0.02, 0.05), (0.48, 0.05), (0.48, 0.95), (0.02, 0.95)],
    "Right Half / Entrance": [(0.52, 0.05), (0.98, 0.05), (0.98, 0.95), (0.52, 0.95)],
    "Center Box": [(0.22, 0.18), (0.78, 0.18), (0.78, 0.88), (0.22, 0.88)],
}

ZONE_PALETTE_BGR: Tuple[Tuple[int, int, int], ...] = (
    (212, 182, 6),    # Cyan (#06B6D4)
    (11, 158, 245),   # Amber (#F59E0B)
    (129, 185, 16),   # Emerald (#10B981)
    (246, 92, 139),   # Violet (#8B5CF6)
    (113, 113, 248),  # Rose (#F87171)
)


@dataclass
class PolygonZone:
    """A named normalized polygon zone with coordinates in [0.0, 1.0]."""

    name: str
    points: List[Tuple[float, float]]
    color_bgr: Tuple[int, int, int] = (212, 182, 6)
    loiter_threshold_sec: Optional[float] = None

    def to_pixel_contour(self, width: int, height: int) -> np.ndarray:
        pts = [
            (
                int(round(float(np.clip(x, 0.0, 1.0)) * float(max(1, width - 1)))),
                int(round(float(np.clip(y, 0.0, 1.0)) * float(max(1, height - 1)))),
            )
            for x, y in self.points
        ]
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def parse_polygon_coords(text: str) -> List[Tuple[float, float]]:
    """Parse custom normalized polygon coordinates from JSON or `'x1,y1; x2,y2; ...'` text."""
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        pts = [(float(np.clip(p[0], 0.0, 1.0)), float(np.clip(p[1], 0.0, 1.0))) for p in data if len(p) >= 2]
        return pts if len(pts) >= 3 else []

    pts_out: List[Tuple[float, float]] = []
    for chunk in raw.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) >= 2:
            x = float(np.clip(float(parts[0]), 0.0, 1.0))
            y = float(np.clip(float(parts[1]), 0.0, 1.0))
            pts_out.append((x, y))
    return pts_out if len(pts_out) >= 3 else []


def format_polygon_coords(points: Sequence[Tuple[float, float]]) -> str:
    """Format normalized polygon coordinates into an editable `'x1,y1; x2,y2; ...'` string."""
    return "; ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in points)


def _coerce_zones(zones: Optional[Sequence[Union[PolygonZone, Dict[str, Any]]]]) -> List[PolygonZone]:
    if not zones:
        return [
            PolygonZone(
                name="Left Half / Counter",
                points=list(ZONE_PRESETS["Left Half / Counter"]),
                color_bgr=ZONE_PALETTE_BGR[0],
            )
        ]
    coerced: List[PolygonZone] = []
    for idx, z in enumerate(zones):
        if isinstance(z, PolygonZone):
            if len(z.points) >= 3:
                coerced.append(z)
        elif isinstance(z, dict):
            pts = [(float(p[0]), float(p[1])) for p in z.get("points", []) if len(p) >= 2]
            if len(pts) >= 3:
                col = z.get("color_bgr", ZONE_PALETTE_BGR[idx % len(ZONE_PALETTE_BGR)])
                coerced.append(
                    PolygonZone(
                        name=str(z.get("name", f"Zone {idx + 1}")),
                        points=pts,
                        color_bgr=(int(col[0]), int(col[1]), int(col[2])),
                        loiter_threshold_sec=(
                            float(z["loiter_threshold_sec"])
                            if z.get("loiter_threshold_sec") is not None
                            else None
                        ),
                    )
                )
    return coerced


def _iter_timeline_tracks(snap: Any) -> Tuple[int, List[Tuple[int, np.ndarray, Optional[np.ndarray]]]]:
    """Extract `(frame_idx, [(track_id, box_xyxy, keypoints_or_none), ...])` from FrameSnapshot or dict."""
    if isinstance(snap, dict):
        fidx = int(snap.get("frame_idx", 0))
        raw_tracks = snap.get("active", snap.get("tracks", []))
        items: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = []
        for tr in raw_tracks:
            if isinstance(tr, dict):
                tid = int(tr["track_id"])
                box = np.asarray(tr["box"], dtype=np.float32).reshape(4)
                kpts = tr.get("keypoints")
                k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None
                items.append((tid, box, k_arr))
            else:
                tid = int(tr.track_id)
                box = np.asarray(tr.box, dtype=np.float32).reshape(4)
                kpts = getattr(tr, "keypoints", None)
                k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None
                items.append((tid, box, k_arr))
        return fidx, items

    fidx = int(getattr(snap, "frame_idx", 0))
    items = []
    for tr in getattr(snap, "active", []):
        tid = int(tr.track_id)
        box = np.asarray(tr.box, dtype=np.float32).reshape(4)
        kpts = getattr(tr, "keypoints", None)
        k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None
        items.append((tid, box, k_arr))
    return fidx, items


def _foot_anchor(box: np.ndarray) -> Tuple[float, float]:
    """Bottom-center foot-anchor point `((x1 + x2) / 2, y2)`."""
    return (0.5 * (float(box[0]) + float(box[2])), float(box[3]))


def draw_zones_overlay(
    frame: np.ndarray,
    zones: Sequence[Union[PolygonZone, Dict[str, Any]]],
    active_counts: Optional[Dict[str, int]] = None,
    alert_zones: Optional[Set[str]] = None,
    alpha: float = 0.22,
) -> np.ndarray:
    """Render translucent shaded polygon zones, glowing borders, and live occupancy counts on `frame`."""
    coerced = _coerce_zones(zones)
    if not coerced or frame is None or frame.size == 0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    counts = active_counts or {}
    alerts = alert_zones or set()

    for zone in coerced:
        contour = zone.to_pixel_contour(w, h)
        is_alert = zone.name in alerts
        fill_col = (48, 52, 235) if is_alert else zone.color_bgr
        cv2.fillPoly(overlay, [contour], fill_col)

    cv2.addWeighted(overlay, float(np.clip(alpha, 0.05, 0.65)), out, 1.0 - float(np.clip(alpha, 0.05, 0.65)), 0.0, out)

    for zone in coerced:
        contour = zone.to_pixel_contour(w, h)
        is_alert = zone.name in alerts
        border_col = (68, 68, 255) if is_alert else zone.color_bgr
        cv2.polylines(out, [contour], isClosed=True, color=border_col, thickness=2, lineType=cv2.LINE_AA)
        for pt in contour.reshape(-1, 2):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 3, border_col, -1, cv2.LINE_AA)

        # Label pill near top-left vertex of polygon
        pts2 = contour.reshape(-1, 2)
        min_x = int(np.min(pts2[:, 0]))
        min_y = int(np.min(pts2[:, 1]))
        occ = int(counts.get(zone.name, 0))
        tag = f"{zone.name} | Live: {occ}" + (" | LOITER ALERT" if is_alert else "")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thick = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, scale, thick)
        lx = max(6, min(w - tw - 16, min_x + 6))
        ly = max(th + 10, min(h - 10, min_y + th + 10))
        bg_col = (30, 34, 190) if is_alert else (18, 26, 40)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 8, ly + baseline + 4), bg_col, -1)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 8, ly + baseline + 4), border_col, 1, cv2.LINE_AA)
        cv2.putText(out, tag, (lx + 1, ly), font, scale, (242, 248, 255), thick, cv2.LINE_AA)

    return out


class ZoneAnalyzer:
    """Custom polygon zone dwell, entry/exit, line-crossing, and loitering alert engine."""

    PRESETS: Dict[str, List[Tuple[float, float]]] = ZONE_PRESETS

    def __init__(
        self,
        zones: Optional[Sequence[Union[PolygonZone, Dict[str, Any]]]] = None,
        loiter_threshold_sec: float = 10.0,
        src_fps: float = 30.0,
    ) -> None:
        self.zones: List[PolygonZone] = _coerce_zones(zones)
        self.loiter_threshold_sec = max(0.5, float(loiter_threshold_sec))
        self.src_fps = max(1.0, float(src_fps))

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        loiter_threshold_sec: float = 10.0,
        src_fps: float = 30.0,
    ) -> "ZoneAnalyzer":
        pts = ZONE_PRESETS.get(preset_name, ZONE_PRESETS["Full Frame"])
        zone = PolygonZone(name=preset_name, points=list(pts), color_bgr=ZONE_PALETTE_BGR[0])
        return cls(zones=[zone], loiter_threshold_sec=loiter_threshold_sec, src_fps=src_fps)

    @staticmethod
    def draw_zones_overlay(
        frame: np.ndarray,
        zones: Sequence[Union[PolygonZone, Dict[str, Any]]],
        active_counts: Optional[Dict[str, int]] = None,
        alert_zones: Optional[Set[str]] = None,
        alpha: float = 0.22,
    ) -> np.ndarray:
        return draw_zones_overlay(
            frame=frame,
            zones=zones,
            active_counts=active_counts,
            alert_zones=alert_zones,
            alpha=alpha,
        )

    def analyze_timeline(
        self,
        timeline: Sequence[Any],
        frame_size: Tuple[int, int],
        src_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run full zone occupancy, dwell, crossing, and loitering analysis across a timeline."""
        width, height = int(frame_size[0]), int(frame_size[1])
        fps = max(1.0, float(src_fps if src_fps is not None else self.src_fps))

        contours: Dict[str, np.ndarray] = {
            z.name: z.to_pixel_contour(width, height) for z in self.zones
        }
        thresholds_frames: Dict[str, int] = {
            z.name: max(
                1,
                int(round((z.loiter_threshold_sec if z.loiter_threshold_sec is not None else self.loiter_threshold_sec) * fps)),
            )
            for z in self.zones
        }

        # Per-(zone_name, track_id) running state
        state: Dict[Tuple[str, int], Dict[str, Any]] = {}
        peak_occupancy: Dict[str, int] = {z.name: 0 for z in self.zones}
        unique_visitors: Dict[str, Set[int]] = {z.name: set() for z in self.zones}

        per_frame_occupancy: Dict[int, Dict[str, int]] = {}
        per_frame_alert_zones: Dict[int, Set[str]] = {}
        loitering_events: List[Dict[str, Any]] = []
        transition_events: List[Dict[str, Any]] = []

        # Track open loitering event dicts so duration_sec stays exact until exit
        open_loiter_event: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for seq_idx, snap in enumerate(timeline, start=1):
            fidx, tracks = _iter_timeline_tracks(snap)
            if fidx <= 0:
                fidx = seq_idx
            t_sec = (fidx - 1) / fps

            present_ids: Set[int] = set()
            frame_occ: Dict[str, int] = {z.name: 0 for z in self.zones}
            frame_alerts: Set[str] = set()

            for tid, box, _kpts in tracks:
                present_ids.add(tid)
                fx, fy = _foot_anchor(box)
                for z in self.zones:
                    key = (z.name, tid)
                    st = state.setdefault(
                        key,
                        {
                            "inside": False,
                            "entries": 0,
                            "exits": 0,
                            "total_frames": 0,
                            "cur_streak": 0,
                            "max_streak": 0,
                            "entry_frame": None,
                            "loiter_count": 0,
                        },
                    )
                    inside_now = cv2.pointPolygonTest(contours[z.name], (fx, fy), False) >= 0.0

                    if inside_now:
                        frame_occ[z.name] += 1
                        unique_visitors[z.name].add(tid)
                        if not st["inside"]:
                            st["inside"] = True
                            st["entries"] += 1
                            st["cur_streak"] = 1
                            st["entry_frame"] = fidx
                            transition_events.append(
                                {
                                    "timestamp_sec": round(t_sec, 2),
                                    "frame_idx": fidx,
                                    "track_id": tid,
                                    "zone_name": z.name,
                                    "event_type": "ZONE_ENTRY",
                                }
                            )
                        else:
                            st["cur_streak"] += 1

                        st["total_frames"] += 1
                        if st["cur_streak"] > st["max_streak"]:
                            st["max_streak"] = st["cur_streak"]

                        if st["cur_streak"] >= thresholds_frames[z.name]:
                            frame_alerts.add(z.name)
                            dwell_sec = st["cur_streak"] / fps
                            if key not in open_loiter_event:
                                st["loiter_count"] += 1
                                ev = {
                                    "timestamp_sec": round(t_sec, 2),
                                    "entry_sec": round(((st["entry_frame"] or fidx) - 1) / fps, 2),
                                    "frame_idx": fidx,
                                    "track_id": tid,
                                    "zone_name": z.name,
                                    "event_type": "LOITERING_ALERT",
                                    "duration_sec": round(dwell_sec, 2),
                                    "severity": "HIGH",
                                }
                                loitering_events.append(ev)
                                open_loiter_event[key] = ev
                            else:
                                open_loiter_event[key]["duration_sec"] = round(dwell_sec, 2)
                    else:
                        if st["inside"]:
                            st["inside"] = False
                            st["exits"] += 1
                            st["cur_streak"] = 0
                            st["entry_frame"] = None
                            open_loiter_event.pop(key, None)
                            transition_events.append(
                                {
                                    "timestamp_sec": round(t_sec, 2),
                                    "frame_idx": fidx,
                                    "track_id": tid,
                                    "zone_name": z.name,
                                    "event_type": "ZONE_EXIT",
                                }
                            )

            # Close active inside streaks for tracks that disappeared on this frame
            for key, st in state.items():
                zname, tid = key
                if st["inside"] and tid not in present_ids:
                    st["inside"] = False
                    st["exits"] += 1
                    st["cur_streak"] = 0
                    st["entry_frame"] = None
                    open_loiter_event.pop(key, None)
                    transition_events.append(
                        {
                            "timestamp_sec": round(t_sec, 2),
                            "frame_idx": fidx,
                            "track_id": tid,
                            "zone_name": zname,
                            "event_type": "ZONE_EXIT",
                        }
                    )

            for z in self.zones:
                if frame_occ[z.name] > peak_occupancy[z.name]:
                    peak_occupancy[z.name] = frame_occ[z.name]

            per_frame_occupancy[fidx] = frame_occ
            per_frame_alert_zones[fidx] = frame_alerts

        # Build clean zone summaries and per-track stats
        track_zone_stats: List[Dict[str, Any]] = []
        zone_summaries: List[Dict[str, Any]] = []

        for z in self.zones:
            z_entries = 0
            z_exits = 0
            z_dwells: List[float] = []
            z_alerts = 0
            for (zname, tid), st in sorted(state.items(), key=lambda x: (x[0][0], x[0][1])):
                if zname != z.name or st["total_frames"] == 0:
                    continue
                total_sec = st["total_frames"] / fps
                max_cont_sec = st["max_streak"] / fps
                z_entries += int(st["entries"])
                z_exits += int(st["exits"])
                z_dwells.append(total_sec)
                z_alerts += int(st["loiter_count"])
                track_zone_stats.append(
                    {
                        "zone_name": z.name,
                        "track_id": tid,
                        "entries": int(st["entries"]),
                        "exits": int(st["exits"]),
                        "line_crossings": int(st["entries"] + st["exits"]),
                        "total_dwell_sec": round(total_sec, 2),
                        "max_continuous_dwell_sec": round(max_cont_sec, 2),
                        "loitering_alerts": int(st["loiter_count"]),
                    }
                )

            zone_summaries.append(
                {
                    "zone_name": z.name,
                    "unique_visitors": len(unique_visitors[z.name]),
                    "entries": z_entries,
                    "exits": z_exits,
                    "line_crossings": z_entries + z_exits,
                    "peak_occupancy": peak_occupancy[z.name],
                    "mean_dwell_sec": round(float(np.mean(z_dwells)), 2) if z_dwells else 0.0,
                    "max_dwell_sec": round(float(np.max(z_dwells)), 2) if z_dwells else 0.0,
                    "loitering_alerts": z_alerts,
                }
            )

        return {
            "zone_summaries": zone_summaries,
            "track_zone_stats": track_zone_stats,
            "events": loitering_events,
            "transition_events": transition_events,
            "per_frame_occupancy": per_frame_occupancy,
            "per_frame_alert_zones": per_frame_alert_zones,
        }


# ---------------------------------------------------------------------------
# 2. Traffic Movement Heatmap (`HeatmapGenerator`)
# ---------------------------------------------------------------------------


class HeatmapGenerator:
    """Gaussian foot-point traffic density heatmap generator (`COLORMAP_TURBO`, alpha=0.45)."""

    def __init__(self, alpha: float = 0.45, sigma_frac: float = 0.032) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 0.90))
        self.sigma_frac = float(np.clip(sigma_frac, 0.01, 0.10))

    @staticmethod
    def sample_clean_background(
        video_path: Union[str, Path],
        max_samples: int = 15,
    ) -> Optional[np.ndarray]:
        """Compute a clean per-pixel median background frame (or first frame fallback) from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[np.ndarray] = []
        try:
            if total > 1:
                n_samples = max(1, min(int(max_samples), total))
                indices = np.linspace(0, total - 1, num=n_samples, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        frames.append(fr)
            if not frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
        finally:
            cap.release()

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def generate(
        self,
        timeline: Sequence[Any],
        background_bgr: Optional[np.ndarray] = None,
        video_path: Optional[Union[str, Path]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
        filter_track_ids: Optional[Set[int]] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Accumulate Gaussian foot-point density across `timeline` and blend over a clean background.

        Returns dict with `rgb_image`, `bgr_image`, `png_bytes`, `density_map`, `total_points`.
        """
        bg = background_bgr
        if bg is None and video_path is not None:
            bg = self.sample_clean_background(video_path)
        if bg is None:
            if frame_size is None:
                raise ValueError("Provide background_bgr, video_path, or frame_size.")
            bg = np.full((int(frame_size[1]), int(frame_size[0]), 3), (15, 20, 32), dtype=np.uint8)

        h, w = bg.shape[:2]
        density = np.zeros((h, w), dtype=np.float32)

        sigma = max(10.0, min(44.0, self.sigma_frac * float(min(h, w))))
        radius = int(math.ceil(2.5 * sigma))
        grid = np.arange(-radius, radius + 1, dtype=np.float32)
        gx, gy = np.meshgrid(grid, grid)
        kernel = np.exp(-0.5 * (gx * gx + gy * gy) / (sigma * sigma)).astype(np.float32)

        total_points = 0
        for snap in timeline:
            _fidx, tracks = _iter_timeline_tracks(snap)
            for tid, box, _kpts in tracks:
                if filter_track_ids is not None and tid not in filter_track_ids:
                    continue
                fx, fy = _foot_anchor(box)
                cx = int(round(fx))
                cy = int(round(fy))
                if cx < -radius or cx >= w + radius or cy < -radius or cy >= h + radius:
                    continue

                x1 = max(0, cx - radius)
                y1 = max(0, cy - radius)
                x2 = min(w, cx + radius + 1)
                y2 = min(h, cy + radius + 1)
                kx1 = x1 - (cx - radius)
                ky1 = y1 - (cy - radius)
                kx2 = kx1 + (x2 - x1)
                ky2 = ky1 + (y2 - y1)

                if x2 > x1 and y2 > y1:
                    density[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
                    total_points += 1

        blend_alpha = self.alpha if alpha is None else float(np.clip(alpha, 0.05, 0.90))
        blended_bgr = bg.copy()
        norm_map = np.zeros((h, w), dtype=np.float32)

        if total_points > 0:
            pos = density[density > 1e-5]
            if pos.size > 0:
                p_hi = float(np.percentile(pos, 99.2))
                denom = max(1e-6, p_hi)
                norm_map = np.clip(density / denom, 0.0, 1.0)
                # Gentle gamma curve so secondary corridors stay readable alongside peak dwell spots
                vis_map = np.power(norm_map, 0.72)
                heat_u8 = np.clip(vis_map * 255.0, 0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

                # Per-pixel soft mask keeps zero-traffic areas crystal clear instead of blue-tinted
                mask = np.clip((vis_map - 0.04) / 0.18, 0.0, 1.0)[:, :, None] * blend_alpha
                blended_bgr = np.clip(
                    bg.astype(np.float32) * (1.0 - mask) + colored.astype(np.float32) * mask,
                    0.0,
                    255.0,
                ).astype(np.uint8)

        rgb_image = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        ok, png_buf = cv2.imencode(".png", blended_bgr)
        png_bytes = png_buf.tobytes() if ok else b""

        return {
            "rgb_image": rgb_image,
            "bgr_image": blended_bgr,
            "png_bytes": png_bytes,
            "density_map": norm_map,
            "total_points": total_points,
        }


# ---------------------------------------------------------------------------
# 3. Zero-Cost Pose Action & Anomaly Classifier (`PoseBehaviorAnalyzer`)
# ---------------------------------------------------------------------------

ANOMALY_SEVERITY: Dict[str, str] = {
    "FALL_ALERT": "CRITICAL",
    "HANDS_RAISED": "HIGH",
    "RUNNING": "MEDIUM",
}

_LABEL_PRIORITY: Dict[str, int] = {
    "FALL_ALERT": 5,
    "HANDS_RAISED": 4,
    "RUNNING": 3,
    "SITTING": 2,
    "STANDING": 1,
}


class PoseBehaviorAnalyzer:
    """Zero-cost 17-COCO-keypoint + velocity behavior & anomaly classifier with 5-frame smoothing."""

    def __init__(
        self,
        kpt_conf_thresh: float = 0.35,
        smooth_window: int = 5,
        running_speed_ratio: float = 1.35,
        fall_angle_deg: float = 35.0,
        fall_aspect_ratio: float = 1.15,
        fall_downward_speed_ratio: float = 0.55,
    ) -> None:
        self.kpt_conf_thresh = float(kpt_conf_thresh)
        self.smooth_window = max(1, int(smooth_window))
        self.running_speed_ratio = float(running_speed_ratio)
        self.fall_angle_deg = float(fall_angle_deg)
        self.fall_aspect_ratio = float(fall_aspect_ratio)
        self.fall_downward_speed_ratio = float(fall_downward_speed_ratio)

    def _kpt_ok(self, kpts: Optional[np.ndarray], idx: int) -> bool:
        return (
            kpts is not None
            and kpts.ndim == 2
            and kpts.shape[0] >= 17
            and kpts.shape[1] >= 3
            and float(kpts[idx, 2]) >= self.kpt_conf_thresh
        )

    def _mean_kpt(self, kpts: Optional[np.ndarray], indices: Sequence[int]) -> Optional[np.ndarray]:
        pts = [kpts[i, :2].astype(np.float32) for i in indices if self._kpt_ok(kpts, i)]
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)

    def _classify_raw(
        self,
        box: np.ndarray,
        kpts: Optional[np.ndarray],
        ref_height: float,
        speed_px_per_sec: float,
        recent_peak_down_speed: float,
        prev_was_fall: bool,
    ) -> str:
        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
        h_ref = max(24.0, float(ref_height))

        # --- 1. FALL_ALERT: rapid downward head/hip velocity spike + horizontal torso (<=35 deg) & wide bbox ---
        sh_mid = self._mean_kpt(kpts, (5, 6))
        hip_mid = self._mean_kpt(kpts, (11, 12))
        torso_horizontal = False
        if sh_mid is not None and hip_mid is not None:
            dx = abs(float(sh_mid[0] - hip_mid[0]))
            dy = abs(float(sh_mid[1] - hip_mid[1]))
            if math.hypot(dx, dy) >= 6.0:
                angle_from_horiz = math.degrees(math.atan2(dy, max(1e-6, dx)))
                torso_horizontal = angle_from_horiz <= self.fall_angle_deg

        wide_bbox = bw > (bh * self.fall_aspect_ratio)
        has_downward_spike = recent_peak_down_speed >= (self.fall_downward_speed_ratio * h_ref)

        if wide_bbox and (torso_horizontal or sh_mid is None or hip_mid is None):
            if has_downward_spike or prev_was_fall:
                return "FALL_ALERT"

        # --- 2. HANDS_RAISED: both wrists clearly above shoulder/nose keypoints ---
        if self._kpt_ok(kpts, 9) and self._kpt_ok(kpts, 10):
            lw_y = float(kpts[9, 1])
            rw_y = float(kpts[10, 1])
            l_ref_y: Optional[float] = None
            r_ref_y: Optional[float] = None

            if self._kpt_ok(kpts, 0):
                nose_y = float(kpts[0, 1])
                sh_y = float(sh_mid[1]) if sh_mid is not None else nose_y + 0.12 * h_ref
                # Require both wrists clearly above shoulders and at/above nose-ear band
                thresh_y = min(sh_y - 0.04 * h_ref, nose_y + 0.04 * h_ref)
                if lw_y < thresh_y and rw_y < thresh_y:
                    return "HANDS_RAISED"
            elif sh_mid is not None:
                l_ref_y = float(kpts[5, 1]) if self._kpt_ok(kpts, 5) else float(sh_mid[1])
                r_ref_y = float(kpts[6, 1]) if self._kpt_ok(kpts, 6) else float(sh_mid[1])
                if lw_y < (l_ref_y - 0.06 * h_ref) and rw_y < (r_ref_y - 0.06 * h_ref):
                    return "HANDS_RAISED"

        # --- 3. RUNNING: normalized center displacement per second > 1.35 * body_height ---
        if speed_px_per_sec > (self.running_speed_ratio * h_ref):
            return "RUNNING"

        # --- 4. SITTING vs STANDING: knee-hip-ankle vertical compression ratio ---
        knee_mid = self._mean_kpt(kpts, (13, 14))
        ankle_mid = self._mean_kpt(kpts, (15, 16))
        if hip_mid is not None and knee_mid is not None and ankle_mid is not None:
            hip_knee_dy = max(0.0, float(knee_mid[1] - hip_mid[1]))
            knee_ankle_dy = max(1.0, float(ankle_mid[1] - knee_mid[1]))
            thigh_compress_ratio = hip_knee_dy / knee_ankle_dy

            hip_ankle_dy = max(0.0, float(ankle_mid[1] - hip_mid[1]))
            torso_dy = max(1.0, float(hip_mid[1] - sh_mid[1])) if sh_mid is not None else 0.0

            if thigh_compress_ratio < 0.58:
                if torso_dy <= 1.0 or (hip_ankle_dy / torso_dy) < 1.85:
                    return "SITTING"
        elif hip_mid is not None and knee_mid is not None and sh_mid is not None:
            torso_dy = max(1.0, float(hip_mid[1] - sh_mid[1]))
            hip_knee_dy = max(0.0, float(knee_mid[1] - hip_mid[1]))
            if (hip_knee_dy / torso_dy) < 0.38:
                return "SITTING"

        return "STANDING"

    @staticmethod
    def _smooth_label(window: Deque[str]) -> str:
        """5-frame temporal majority-vote smoothing with deterministic severity tie-breaking."""
        counts = Counter(window)
        best_label = max(
            counts.items(),
            key=lambda kv: (kv[1], _LABEL_PRIORITY.get(kv[0], 0)),
        )[0]
        return best_label

    def analyze_timeline(
        self,
        timeline: Sequence[Any],
        src_fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Classify per-frame pose states and extract structured anomaly events across `timeline`."""
        fps = max(1.0, float(src_fps))
        spike_lookback = max(6, int(round(0.65 * fps)))

        # Per-track kinematic & smoothing buffers
        track_state: Dict[int, Dict[str, Any]] = {}
        frame_labels: Dict[int, Dict[int, str]] = {}
        label_frame_counts: Dict[int, Counter] = {}
        peak_norm_speed: Dict[int, float] = {}

        # Active contiguous anomaly episodes per track_id
        open_episode: Dict[int, Dict[str, Any]] = {}
        anomaly_events: List[Dict[str, Any]] = []

        last_frame_idx = 0

        for seq_idx, snap in enumerate(timeline, start=1):
            fidx, tracks = _iter_timeline_tracks(snap)
            if fidx <= 0:
                fidx = seq_idx
            last_frame_idx = fidx
            t_sec = (fidx - 1) / fps

            present_ids: Set[int] = set()
            cur_frame_labels: Dict[int, str] = {}

            for tid, box, kpts in tracks:
                present_ids.add(tid)
                bw = max(1.0, float(box[2] - box[0]))
                bh = max(1.0, float(box[3] - box[1]))
                cx = 0.5 * (float(box[0]) + float(box[2]))
                cy = 0.5 * (float(box[1]) + float(box[3]))

                head_pt = self._mean_kpt(kpts, (0, 1, 2, 3, 4))
                hip_pt = self._mean_kpt(kpts, (11, 12))
                vert_anchor_y = (
                    0.5 * (float(head_pt[1]) + float(hip_pt[1]))
                    if (head_pt is not None and hip_pt is not None)
                    else (
                        float(head_pt[1])
                        if head_pt is not None
                        else (float(hip_pt[1]) if hip_pt is not None else cy)
                    )
                )

                st = track_state.get(tid)
                if st is None:
                    st = {
                        "ref_height": max(bh, 0.85 * bw),
                        "prev_fidx": fidx,
                        "prev_center": (cx, cy),
                        "prev_anchor_y": vert_anchor_y,
                        "speed_hist": deque(maxlen=3),
                        "down_hist": deque(maxlen=spike_lookback),
                        "label_win": deque(maxlen=self.smooth_window),
                        "last_smoothed": "STANDING",
                    }
                    track_state[tid] = st
                    speed_px_sec = 0.0
                    down_px_sec = 0.0
                else:
                    df = max(1, fidx - int(st["prev_fidx"]))
                    dt = df / fps
                    if df <= max(10, int(round(0.5 * fps))):
                        disp = math.hypot(cx - st["prev_center"][0], cy - st["prev_center"][1])
                        speed_px_sec = disp / dt
                        down_px_sec = max(0.0, (vert_anchor_y - float(st["prev_anchor_y"])) / dt)
                    else:
                        speed_px_sec = 0.0
                        down_px_sec = 0.0

                    # Update reference standing height only while upright (bbox height >= bbox width)
                    if bh >= bw:
                        st["ref_height"] = 0.90 * float(st["ref_height"]) + 0.10 * bh

                    st["prev_fidx"] = fidx
                    st["prev_center"] = (cx, cy)
                    st["prev_anchor_y"] = vert_anchor_y

                st["speed_hist"].append(speed_px_sec)
                st["down_hist"].append(down_px_sec)
                avg_speed = float(np.mean(st["speed_hist"]))
                peak_down = float(np.max(st["down_hist"])) if st["down_hist"] else 0.0

                norm_spd = avg_speed / max(24.0, float(st["ref_height"]))
                if norm_spd > peak_norm_speed.get(tid, 0.0):
                    peak_norm_speed[tid] = norm_spd

                raw_label = self._classify_raw(
                    box=box,
                    kpts=kpts,
                    ref_height=float(st["ref_height"]),
                    speed_px_per_sec=avg_speed,
                    recent_peak_down_speed=peak_down,
                    prev_was_fall=(st["last_smoothed"] == "FALL_ALERT"),
                )
                st["label_win"].append(raw_label)
                smoothed = self._smooth_label(st["label_win"])
                st["last_smoothed"] = smoothed

                cur_frame_labels[tid] = smoothed
                label_frame_counts.setdefault(tid, Counter())[smoothed] += 1

                # Maintain contiguous anomaly event episodes
                if smoothed in ANOMALY_SEVERITY:
                    ep = open_episode.get(tid)
                    if ep is not None and ep["event_type"] == smoothed:
                        ep["end_frame"] = fidx
                        ep["duration_sec"] = round((ep["end_frame"] - ep["frame_idx"] + 1) / fps, 2)
                    else:
                        if ep is not None:
                            anomaly_events.append(self._finalize_episode(ep))
                        open_episode[tid] = {
                            "timestamp_sec": round(t_sec, 2),
                            "frame_idx": fidx,
                            "end_frame": fidx,
                            "track_id": tid,
                            "event_type": smoothed,
                            "duration_sec": round(1.0 / fps, 2),
                            "severity": ANOMALY_SEVERITY[smoothed],
                        }
                else:
                    if tid in open_episode:
                        anomaly_events.append(self._finalize_episode(open_episode.pop(tid)))

            # Close episodes for tracks no longer present
            for tid in list(open_episode.keys()):
                if tid not in present_ids:
                    anomaly_events.append(self._finalize_episode(open_episode.pop(tid)))

            frame_labels[fidx] = cur_frame_labels

        # Finalize any still-open episodes at end of video
        for tid in list(open_episode.keys()):
            anomaly_events.append(self._finalize_episode(open_episode.pop(tid)))

        anomaly_events.sort(key=lambda e: (e["frame_idx"], e["track_id"]))

        # Build concise per-track summaries
        track_summaries: Dict[int, Dict[str, Any]] = {}
        for tid, cnt in sorted(label_frame_counts.items()):
            dominant = cnt.most_common(1)[0][0] if cnt else "STANDING"
            dist_sec = {lbl: round(c / fps, 2) for lbl, c in cnt.items()}
            tid_anoms = [e for e in anomaly_events if e["track_id"] == tid]
            track_summaries[tid] = {
                "track_id": tid,
                "dominant_state": dominant,
                "state_seconds": dist_sec,
                "peak_speed_bh_per_sec": round(peak_norm_speed.get(tid, 0.0), 2),
                "anomaly_count": len(tid_anoms),
                "anomalies": sorted({e["event_type"] for e in tid_anoms}),
            }

        return {
            "frame_labels": frame_labels,
            "events": anomaly_events,
            "track_summaries": track_summaries,
            "total_frames": last_frame_idx,
        }

    @staticmethod
    def _finalize_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp_sec": float(ep["timestamp_sec"]),
            "frame_idx": int(ep["frame_idx"]),
            "track_id": int(ep["track_id"]),
            "event_type": str(ep["event_type"]),
            "duration_sec": float(ep["duration_sec"]),
            "severity": str(ep["severity"]),
        }
