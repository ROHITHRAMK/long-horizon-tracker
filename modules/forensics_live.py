"""Privacy Anonymization & One-Click Forensic PDF / CSV Report Export.

Components:
1. `apply_privacy_blur`: Selective head/face anonymization (`Gaussian Blur` or `Pixelate`)
   with modes `Off`, `Blur Everyone Except Selected Target` (`Blur Non-Targets in Spotlight`),
   and `Anonymize All Faces`.
2. `generate_forensic_pdf`: Multi-section enterprise PDF report via `fpdf2` containing
   Executive Summary, embedded Movement Heatmap, Zone Dwell & Behavior Anomaly tables,
   Per-Person Biometric Dossier with embedded portrait thumbnails, and full Recovery Log.
3. `generate_forensic_csv`: Structured telemetry CSV bytes for spreadsheet/SIEM ingestion.
"""

from __future__ import annotations

import csv
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Privacy & Selective Anonymization (`apply_privacy_blur`)
# ---------------------------------------------------------------------------


def _compute_privacy_head_rect(
    box: np.ndarray,
    kpts: Optional[np.ndarray],
    frame_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Compute tight head/face rectangle `(x1, y1, x2, y2)` from 17 COCO keypoints or upper box."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(4.0, x2 - x1)
    bh = max(4.0, y2 - y1)

    head_pts: List[Tuple[float, float]] = []
    if kpts is not None and kpts.ndim == 2 and kpts.shape[0] >= 7 and kpts.shape[1] >= 3:
        for idx in (0, 1, 2, 3, 4):
            if float(kpts[idx, 2]) >= 0.30:
                head_pts.append((float(kpts[idx, 0]), float(kpts[idx, 1])))
        sh_ys = [float(kpts[idx, 1]) for idx in (5, 6) if float(kpts[idx, 2]) >= 0.30]
        shoulder_y = float(np.min(sh_ys)) if sh_ys else (y1 + 0.26 * bh)
    else:
        shoulder_y = y1 + 0.26 * bh

    if head_pts:
        xs = [p[0] for p in head_pts]
        ys = [p[1] for p in head_pts]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        span_x = max(float(max(xs) - min(xs)), 0.22 * bw)
        span_y = max(float(max(ys) - min(ys)), 0.14 * bh)
        rad_x = max(10.0, 0.85 * span_x)
        rad_y = max(12.0, 0.95 * span_y)
        hx1 = cx - rad_x
        hx2 = cx + rad_x
        hy1 = min(cy - rad_y, y1)
        hy2 = min(max(cy + rad_y, shoulder_y), y1 + 0.34 * bh)
    else:
        hx1 = x1 + 0.14 * bw
        hx2 = x2 - 0.14 * bw
        hy1 = y1
        hy2 = min(shoulder_y, y1 + 0.28 * bh)

    ix1 = max(0, int(round(hx1)))
    iy1 = max(0, int(round(hy1)))
    ix2 = min(w, int(round(hx2)))
    iy2 = min(h, int(round(hy2)))
    if ix2 - ix1 < 6 or iy2 - iy1 < 6:
        return None
    return ix1, iy1, ix2, iy2


def apply_privacy_blur(
    frame_bgr: np.ndarray,
    tracks: Sequence[Any],
    mode: str = "Off",
    style: str = "Gaussian Blur",
    target_id: Optional[int] = None,
) -> np.ndarray:
    """Apply selective head/face anonymization (`Gaussian Blur` or `Pixelate`) on `frame_bgr`.

    Supported `mode` values:
    - `"Off"` (no-op)
    - `"Blur Everyone Except Selected Target"` / `"Blur Non-Targets in Spotlight"`
    - `"Anonymize All Faces"`
    """
    if frame_bgr is None or frame_bgr.size == 0 or not tracks:
        return frame_bgr

    mode_low = (mode or "Off").strip().lower()
    if mode_low in ("", "off", "none", "disabled"):
        return frame_bgr

    blur_all = "all" in mode_low
    blur_non_target = ("except" in mode_low) or ("non-target" in mode_low) or ("spotlight" in mode_low)
    if not blur_all and not blur_non_target:
        return frame_bgr

    use_pixelate = "pixel" in (style or "Gaussian Blur").strip().lower()
    out = frame_bgr.copy()
    fh, fw = out.shape[:2]

    for tr in tracks:
        if isinstance(tr, dict):
            tid = int(tr.get("track_id", -1))
            box = np.asarray(tr["box"], dtype=np.float32).reshape(4)
            kpts = tr.get("keypoints")
            k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None
        else:
            tid = int(getattr(tr, "track_id", -1))
            box = np.asarray(tr.box, dtype=np.float32).reshape(4)
            kpts = getattr(tr, "keypoints", None)
            k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None

        if blur_non_target and target_id is not None and tid == int(target_id):
            continue

        rect = _compute_privacy_head_rect(box, k_arr, (fh, fw))
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        roi = out[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 4 or rw < 4:
            continue

        if use_pixelate:
            blocks_w = max(4, min(10, rw // 7))
            blocks_h = max(4, min(10, rh // 7))
            small = cv2.resize(roi, (blocks_w, blocks_h), interpolation=cv2.INTER_LINEAR)
            anon = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            k = max(21, int(round(max(rw, rh) * 0.55)))
            if k % 2 == 0:
                k += 1
            k = min(k, 99)
            anon = cv2.GaussianBlur(roi, (k, k), 0)

        # Clean elliptical mask so anonymization hugs the head cleanly
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(
            mask,
            center=(rw // 2, rh // 2),
            axes=(max(2, rw // 2), max(2, rh // 2)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )
        m_bool = mask > 0
        roi[m_bool] = anon[m_bool]

    return out


# ---------------------------------------------------------------------------
# 2. Forensic CSV & Multi-Section PDF Exporters (`fpdf2`)
# ---------------------------------------------------------------------------


def _pdf_safe(text: Any, max_len: int = 120) -> str:
    """Convert any text safely into Latin-1 printable characters for standard Helvetica PDF fonts."""
    s = str(text if text is not None else "")
    replacements = {
        "—": "-",
        "–": "-",
        "→": "->",
        "←": "<-",
        "·": "|",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        "°": " deg",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = s.encode("latin-1", errors="ignore").decode("latin-1").strip()
    if len(s) > max_len:
        s = s[: max(1, max_len - 3)] + "..."
    return s


def generate_forensic_csv(
    job_result: Dict[str, Any],
    zone_result: Optional[Dict[str, Any]] = None,
    pose_result: Optional[Dict[str, Any]] = None,
    vault_matches: Optional[Dict[int, Dict[str, Any]]] = None,
) -> bytes:
    """Export per-person telemetry, zone dwell summaries, behavior anomalies, and recoveries as CSV bytes."""
    person_stats: Dict[int, Dict[str, Any]] = job_result.get("person_stats", {})
    recoveries: List[Dict[str, Any]] = job_result.get("recoveries", [])
    src_fps = max(1.0, float(job_result.get("src_fps", 30.0)))
    vmap = vault_matches or {}
    pose_summaries = (pose_result or {}).get("track_summaries", {})

    zone_by_tid: Dict[int, List[str]] = {}
    for row in (zone_result or {}).get("track_zone_stats", []):
        tid = int(row["track_id"])
        zone_by_tid.setdefault(tid, []).append(
            f"{row['zone_name']}:{row['total_dwell_sec']:.1f}s({row['entries']}in)"
        )

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["SECTION", "PERSON_BIOMETRIC_DOSSIER"])
    writer.writerow(
        [
            "Person ID",
            "Vault ID",
            "Display Name",
            "Watchlist",
            "First Seen (s)",
            "Last Seen (s)",
            "Frames Visible",
            "Duration (s)",
            "Face Locked",
            "Angles",
            "Recoveries",
            "Baseline IDs Used",
            "Dominant Behavior",
            "Behavior Alerts",
            "Zone Dwell Summary",
            "Appearance Intervals",
        ]
    )
    for pid in sorted(person_stats.keys()):
        ps = person_stats[pid]
        vm = vmap.get(pid, {})
        bsum = pose_summaries.get(pid, {})
        first_s = round(max(0, int(ps.get("first_frame", 1)) - 1) / src_fps, 2)
        last_s = round(max(0, int(ps.get("last_frame", 1)) - 1) / src_fps, 2)
        writer.writerow(
            [
                f"#{pid:02d}",
                vm.get("vault_id", ""),
                vm.get("display_name", "") or "",
                "YES" if vm.get("is_watchlist") else "NO",
                first_s,
                last_s,
                ps.get("frames_visible", 0),
                ps.get("duration_sec", 0.0),
                "YES" if ps.get("has_face_lock") else "NO",
                ps.get("angle_code", ""),
                ps.get("recovery_count", 0),
                len(ps.get("baseline_fragment_ids", [])),
                bsum.get("dominant_state", "N/A"),
                "; ".join(bsum.get("anomalies", [])),
                "; ".join(zone_by_tid.get(pid, [])),
                ps.get("intervals_text", ""),
            ]
        )

    # Anomaly events (Behavior + Loitering)
    all_anomalies: List[Dict[str, Any]] = []
    if zone_result:
        all_anomalies.extend(zone_result.get("events", []))
    if pose_result:
        all_anomalies.extend(pose_result.get("events", []))
    all_anomalies.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))

    writer.writerow([])
    writer.writerow(["SECTION", "BEHAVIOR_AND_ZONE_ALERTS"])
    writer.writerow(["Timestamp (s)", "Frame", "Person ID", "Event Type", "Zone", "Duration (s)", "Severity"])
    for ev in all_anomalies:
        writer.writerow(
            [
                ev.get("timestamp_sec", 0.0),
                ev.get("frame_idx", ""),
                f"#{int(ev.get('track_id', 0)):02d}",
                ev.get("event_type", ""),
                ev.get("zone_name", ""),
                ev.get("duration_sec", 0.0),
                ev.get("severity", ""),
            ]
        )

    writer.writerow([])
    writer.writerow(["SECTION", "IDENTITY_RECOVERY_LOG"])
    writer.writerow(
        ["Frame", "Recovery Type", "Recovered ID", "Biometric Cue", "Occlusion Frames", "Occlusion Sec", "Best Sim", "Margin"]
    )
    for rec in recoveries:
        writer.writerow(
            [
                rec.get("Frame", ""),
                rec.get("Recovery Type", ""),
                rec.get("Recovered ID", ""),
                rec.get("Biometric Cue", ""),
                rec.get("Occlusion Frames", 0),
                rec.get("Occlusion Sec", 0.0),
                rec.get("Best Similarity", 0.0),
                rec.get("Margin", 0.0),
            ]
        )

    return buf.getvalue().encode("utf-8")


def generate_forensic_pdf(
    job_result: Dict[str, Any],
    heatmap_rgb: Optional[np.ndarray] = None,
    zone_result: Optional[Dict[str, Any]] = None,
    pose_result: Optional[Dict[str, Any]] = None,
    vault_matches: Optional[Dict[int, Dict[str, Any]]] = None,
) -> bytes:
    """Generate a polished multi-section Forensic Analysis PDF report via `fpdf2` and return raw bytes."""
    from fpdf import FPDF

    class _ForensicPDF(FPDF):
        def header(self) -> None:
            self.set_fill_color(13, 19, 35)
            self.rect(0, 0, 210, 18, "F")
            self.set_fill_color(6, 182, 212)
            self.rect(0, 18, 210, 1.0, "F")
            self.set_xy(10, 4.5)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(240, 248, 255)
            self.cell(130, 8, "LONG-HORIZON BIOMETRIC MOT // FORENSIC REPORT", border=0, align="L")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(148, 163, 184)
            self.cell(60, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), border=0, align="R")
            self.ln(16)

        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 116, 139)
            self.cell(
                0,
                8,
                f"Long-Horizon Biometric Multi-Object Tracker  |  Page {self.page_no()}/{{nb}}",
                align="C",
            )

    pdf = _ForensicPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    person_stats: Dict[int, Dict[str, Any]] = job_result.get("person_stats", {})
    person_thumbs: Dict[int, bytes] = job_result.get("person_thumbnails", {})
    recoveries: List[Dict[str, Any]] = job_result.get("recoveries", [])
    vmap = vault_matches or {}
    pose_summaries = (pose_result or {}).get("track_summaries", {})

    src_name = Path(str(job_result.get("source_video_path", job_result.get("input_path", "uploaded_video.mp4")))).name
    src_fps = max(1.0, float(job_result.get("src_fps", 30.0)))
    confirmed_ids = int(job_result.get("confirmed_unique_ids", len(person_stats)))
    baseline_ids = int(job_result.get("baseline_unique_ids", confirmed_ids))
    frag_red = (
        100.0 * (baseline_ids - confirmed_ids) / float(max(1, baseline_ids))
        if baseline_ids > 0
        else 0.0
    )
    total_rec = int(job_result.get("total_recoveries", len(recoveries)))
    max_occ_f = int(job_result.get("max_occ_frames", 0))
    max_occ_s = float(job_result.get("max_occ_sec", 0.0))

    def _section_title(title: str) -> None:
        pdf.ln(2)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(6, 182, 212)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7.5, f"  {_pdf_safe(title, 90)}", border="L", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # --- SECTION 1: EXECUTIVE SUMMARY ---
    _section_title("1. EXECUTIVE SUMMARY")

    # 4 KPI Cards across 190mm printable width
    kpi_y = pdf.get_y()
    kpi_w = 45.5
    kpi_gap = 2.6
    kpis = [
        ("CONFIRMED IDs", f"{confirmed_ids}", f"vs {baseline_ids} Baseline"),
        ("FRAG REDUCTION", f"{frag_red:.1f}%", f"{max(0, baseline_ids - confirmed_ids)} splits avoided"),
        ("RECOVERIES", f"{total_rec}", f"Max gap {max_occ_f}f ({max_occ_s:.1f}s)"),
        ("THROUGHPUT", f"{float(job_result.get('pass1_fps', 0.0)):.1f} FPS", f"End-to-end {float(job_result.get('avg_fps', 0.0)):.1f} FPS"),
    ]
    for idx, (lbl, val, sub) in enumerate(kpis):
        x = 10.0 + idx * (kpi_w + kpi_gap)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(x, kpi_y, kpi_w, 20, "DF")
        pdf.set_fill_color(6, 182, 212)
        pdf.rect(x, kpi_y, kpi_w, 1.4, "F")
        pdf.set_xy(x + 2, kpi_y + 2.5)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(kpi_w - 4, 4, lbl, align="L")
        pdf.set_xy(x + 2, kpi_y + 7.2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(kpi_w - 4, 6, val, align="L")
        pdf.set_xy(x + 2, kpi_y + 14.0)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(kpi_w - 4, 4, _pdf_safe(sub, 30), align="L")
    pdf.set_y(kpi_y + 23)

    meta_rows = [
        ("Source Video", src_name, "Resolution / FPS", f"{job_result.get('resolution', 'N/A')} @ {src_fps:.2f} FPS ({job_result.get('frames_processed', 0)} frames)"),
        ("Detector Stack", f"{job_result.get('detector_weights', 'N/A')} (imgsz={job_result.get('imgsz', 'N/A')})", "Re-ID Backbone", str(job_result.get("reid_backend", "N/A"))),
        ("Facial Biometrics", str(job_result.get("face_backend", "N/A")), "Depth Guard", str(job_result.get("depth_backend", "N/A"))),
        ("Compute Device", str(job_result.get("device", "cpu")).upper(), "Completed At", str(job_result.get("completed_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))),
    ]
    pdf.set_font("Helvetica", "", 8.2)
    for k1, v1, k2, v2 in meta_rows:
        pdf.set_text_color(71, 85, 105)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(28, 5.5, f"{k1}:", border=0)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(67, 5.5, _pdf_safe(v1, 42), border=0)
        pdf.set_text_color(71, 85, 105)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(28, 5.5, f"{k2}:", border=0)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(67, 5.5, _pdf_safe(v2, 42), border=0, new_x="LMARGIN", new_y="NEXT")

    # --- SECTION 2: MOVEMENT HEATMAP & SPATIAL / BEHAVIORAL ALERTS ---
    _section_title("2. MOVEMENT HEATMAP, ZONE DWELL & BEHAVIOR ANOMALIES")
    if heatmap_rgb is not None and isinstance(heatmap_rgb, np.ndarray) and heatmap_rgb.size > 0:
        bgr_hm = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        ok_png, png_buf = cv2.imencode(".png", bgr_hm)
        if ok_png:
            hm_stream = io.BytesIO(png_buf.tobytes())
            hh, hw = heatmap_rgb.shape[:2]
            img_w = 150.0
            img_h = min(82.0, img_w * (float(hh) / float(max(1, hw))))
            if pdf.get_y() + img_h + 10 > 275:
                pdf.add_page()
            x_center = 10.0 + (190.0 - img_w) / 2.0
            pdf.image(hm_stream, x=x_center, y=pdf.get_y(), w=img_w, h=img_h)
            pdf.set_y(pdf.get_y() + img_h + 3)

    # Zone Dwell Table
    zone_summaries = (zone_result or {}).get("zone_summaries", [])
    if zone_summaries:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5.5, "Zone Occupancy & Dwell Summary", new_x="LMARGIN", new_y="NEXT")
        headers = [("Zone Name", 48), ("Visitors", 20), ("Entries", 20), ("Crossings", 22), ("Peak Occ", 22), ("Mean Dwell", 28), ("Loiter Alerts", 30)]
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 7.8)
        for htxt, hw_mm in headers:
            pdf.cell(hw_mm, 6, htxt, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 7.8)
        pdf.set_text_color(15, 23, 42)
        for zs in zone_summaries:
            pdf.cell(48, 5.5, _pdf_safe(zs["zone_name"], 26), border=1)
            pdf.cell(20, 5.5, str(zs["unique_visitors"]), border=1, align="C")
            pdf.cell(20, 5.5, str(zs["entries"]), border=1, align="C")
            pdf.cell(22, 5.5, str(zs["line_crossings"]), border=1, align="C")
            pdf.cell(22, 5.5, str(zs["peak_occupancy"]), border=1, align="C")
            pdf.cell(28, 5.5, f"{zs['mean_dwell_sec']:.1f}s", border=1, align="C")
            pdf.cell(30, 5.5, str(zs["loitering_alerts"]), border=1, align="C")
            pdf.ln()
        pdf.ln(2)

    # Combined Anomaly Alerts Table (Loitering + Pose Anomalies)
    all_anomalies: List[Dict[str, Any]] = []
    if zone_result:
        all_anomalies.extend(zone_result.get("events", []))
    if pose_result:
        all_anomalies.extend(pose_result.get("events", []))
    all_anomalies.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 5.5, f"Behavior & Spatial Alert Feed ({len(all_anomalies)} event(s))", new_x="LMARGIN", new_y="NEXT")
    if all_anomalies:
        a_headers = [("Time (s)", 22), ("Person", 22), ("Alert Type", 44), ("Zone / Context", 48), ("Duration", 26), ("Severity", 28)]
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 7.8)
        for htxt, hw_mm in a_headers:
            pdf.cell(hw_mm, 6, htxt, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 7.8)
        pdf.set_text_color(15, 23, 42)
        for ev in all_anomalies[:40]:
            pdf.cell(22, 5.5, f"{float(ev.get('timestamp_sec', 0.0)):.2f}s", border=1, align="C")
            pdf.cell(22, 5.5, f"#{int(ev.get('track_id', 0)):02d}", border=1, align="C")
            pdf.cell(44, 5.5, _pdf_safe(ev.get("event_type", ""), 24), border=1, align="C")
            pdf.cell(48, 5.5, _pdf_safe(ev.get("zone_name", "Full Scene"), 26), border=1, align="C")
            pdf.cell(26, 5.5, f"{float(ev.get('duration_sec', 0.0)):.2f}s", border=1, align="C")
            pdf.cell(28, 5.5, _pdf_safe(ev.get("severity", "INFO"), 14), border=1, align="C")
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5.5, "No loitering, fall, running, or hands-raised anomalies detected.", new_x="LMARGIN", new_y="NEXT")

    # --- SECTION 3: PER-PERSON BIOMETRIC DOSSIER ---
    if pdf.get_y() > 235:
        pdf.add_page()
    _section_title(f"3. PER-PERSON BIOMETRIC DOSSIER ({len(person_stats)} CONFIRMED IDENTITIES)")

    if person_stats:
        for pid in sorted(person_stats.keys()):
            ps = person_stats[pid]
            vm = vmap.get(pid, {})
            bsum = pose_summaries.get(pid, {})
            if pdf.get_y() + 26 > 278:
                pdf.add_page()

            card_y = pdf.get_y()
            pdf.set_fill_color(248, 250, 252)
            pdf.set_draw_color(203, 213, 225)
            pdf.rect(10.0, card_y, 190.0, 24.0, "DF")

            # Portrait Thumbnail
            thumb = person_thumbs.get(pid)
            if thumb:
                try:
                    pdf.image(io.BytesIO(thumb), x=12.0, y=card_y + 2.0, w=16.0, h=20.0)
                except Exception:
                    pass
            else:
                pdf.set_xy(12.0, card_y + 8.0)
                pdf.set_font("Helvetica", "", 6.5)
                pdf.set_text_color(148, 163, 184)
                pdf.cell(16.0, 6.0, "NO CROP", align="C")

            v_name = vm.get("display_name")
            wl_tag = " [WATCHLIST]" if vm.get("is_watchlist") else ""
            title_txt = f"Person #{pid:02d}" + (f" - {v_name}{wl_tag}" if v_name else "")
            first_s = round(max(0, int(ps.get("first_frame", 1)) - 1) / src_fps, 2)
            last_s = round(max(0, int(ps.get("last_frame", 1)) - 1) / src_fps, 2)
            face_txt = "ArcFace LOCKED" if ps.get("has_face_lock") else "Head/Body/Skin Only"
            anoms = ", ".join(bsum.get("anomalies", [])) if bsum.get("anomalies") else "None"
            dom_beh = bsum.get("dominant_state", "STANDING")

            pdf.set_xy(31.0, card_y + 2.2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(165.0, 5.0, _pdf_safe(title_txt, 85))

            pdf.set_xy(31.0, card_y + 7.6)
            pdf.set_font("Helvetica", "", 7.8)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(
                165.0,
                4.8,
                _pdf_safe(
                    f"First Seen: {first_s:.1f}s (F{ps.get('first_frame', 1)}) | Last Seen: {last_s:.1f}s (F{ps.get('last_frame', 1)}) | "
                    f"Duration: {ps.get('duration_sec', 0.0):.1f}s ({ps.get('frames_visible', 0)}f)",
                    105,
                ),
            )

            pdf.set_xy(31.0, card_y + 12.6)
            pdf.cell(
                165.0,
                4.8,
                _pdf_safe(
                    f"Biometrics: {face_txt} | Angles: {ps.get('angle_code', '-')} | "
                    f"Recoveries: {ps.get('recovery_count', 0)} | Baseline IDs Replaced: {len(ps.get('baseline_fragment_ids', []))}",
                    105,
                ),
            )

            pdf.set_xy(31.0, card_y + 17.6)
            pdf.cell(
                165.0,
                4.8,
                _pdf_safe(f"Dominant State: {dom_beh} | Behavior Flags: {anoms}", 105),
            )
            pdf.set_y(card_y + 26.0)
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No confirmed identities recorded.", new_x="LMARGIN", new_y="NEXT")

    # --- SECTION 4: COMPLETE IDENTITY RECOVERY EVENT LOG ---
    if pdf.get_y() > 240:
        pdf.add_page()
    _section_title(f"4. IDENTITY RECOVERY EVENT LOG ({len(recoveries)} EVENT(S))")

    if recoveries:
        r_headers = [("Frame", 16), ("Recovery Type", 44), ("ID", 14), ("Biometric Cue", 56), ("Gap (s)", 18), ("Best Sim", 21), ("Margin", 21)]
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 7.6)
        for htxt, hw_mm in r_headers:
            pdf.cell(hw_mm, 6, htxt, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 7.4)
        pdf.set_text_color(15, 23, 42)
        for rec in recoveries:
            if pdf.get_y() > 274:
                pdf.add_page()
            pdf.cell(16, 5.2, str(rec.get("Frame", "")), border=1, align="C")
            pdf.cell(44, 5.2, _pdf_safe(rec.get("Recovery Type", ""), 26), border=1)
            pdf.cell(14, 5.2, _pdf_safe(rec.get("Recovered ID", ""), 8), border=1, align="C")
            pdf.cell(56, 5.2, _pdf_safe(rec.get("Biometric Cue", ""), 34), border=1)
            pdf.cell(18, 5.2, f"{float(rec.get('Occlusion Sec', 0.0)):.2f}s", border=1, align="C")
            pdf.cell(21, 5.2, f"{float(rec.get('Best Similarity', 0.0)):.3f}", border=1, align="C")
            pdf.cell(21, 5.2, f"{float(rec.get('Margin', 0.0)):.3f}", border=1, align="C")
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No identity recoveries were needed in this clip.", new_x="LMARGIN", new_y="NEXT")

    raw_out = pdf.output()
    return bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else str(raw_out).encode("latin-1")
