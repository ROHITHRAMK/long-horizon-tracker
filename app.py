"""Local Explainable Diagnostics & Benchmark Dashboard for Long-Horizon Tracker.

Launch with:
    .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
COMPARISON_VIDEO = OUTPUT_DIR / "comparison_side_by_side.mp4"
DIAGNOSTICS_VIDEO = OUTPUT_DIR / "tracked_diagnostics.mp4"
EVENTS_LOG = OUTPUT_DIR / "events.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
COMPARISON_SUMMARY_JSON = OUTPUT_DIR / "comparison_summary.json"
BENCHMARK_MD = ROOT / "benchmark_report.md"


st.set_page_config(
    page_title="Long-Horizon Tracker // Explainable Diagnostics",
    page_icon="🎯",
    layout="wide",
)


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_json(path: Path, mtime: float = 0.0) -> Dict[str, Any]:
    _ = mtime  # Included in cache key so regenerating outputs invalidates cache automatically
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_events_df(path: Path, mtime: float = 0.0) -> pd.DataFrame:
    _ = mtime  # Included in cache key so regenerating outputs invalidates cache automatically
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        details = rec.get("details") or {}
        rows.append(
            {
                "frame": int(rec.get("frame", 0)),
                "event_type": rec.get("event_type", ""),
                "track_id": int(rec.get("track_id", -1)),
                "prob_id": details.get("prob_id"),
                "best_candidate_ghost": details.get("best_candidate_ghost"),
                "best_sim": details.get("best_sim"),
                "second_best_sim": details.get("second_best_sim"),
                "margin": details.get("margin"),
                "ratio": details.get("ratio"),
                "ghost_age": details.get("ghost_age"),
                "clean_entries": details.get("clean_entries"),
                "occluder_id": details.get("occluder_id"),
                "iou": details.get("iou"),
                "message": rec.get("message", ""),
            }
        )
    return pd.DataFrame(rows)


def _get_ffmpeg_exe() -> Optional[str]:
    """Locate standalone ffmpeg via imageio-ffmpeg first, falling back to system PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _is_h264_mp4(video_path: Path) -> bool:
    """Fast header check for H.264 (avc1) atom inside an MP4 container."""
    try:
        header = video_path.read_bytes()[:65536]
        return b"avc1" in header or b"avcC" in header
    except Exception:
        return False


def ensure_web_playable_video(video_path: Path) -> Optional[Path]:
    """Ensure the video is H.264 (yuv420p + faststart) for HTML5 browser playback."""
    if not video_path.exists():
        return None
    if _is_h264_mp4(video_path):
        return video_path

    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        return video_path

    web_path = video_path.with_name(f"{video_path.stem}_h264.mp4")
    if web_path.exists() and web_path.stat().st_mtime >= video_path.stat().st_mtime:
        return web_path

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",
                str(web_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return web_path
    except Exception:
        return video_path


def render_video_card(title: str, caption: str, video_path: Path) -> None:
    head_l, head_r = st.columns([3, 2])
    head_l.subheader(title)
    if not video_path.exists():
        st.caption(caption)
        st.warning(f"Video not found at `{video_path.relative_to(ROOT)}`. Run the pipeline first.")
        return

    playable = ensure_web_playable_video(video_path)
    stat = playable.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    h264_badge = "H.264 (`avc1`)" if _is_h264_mp4(playable) else "non-H.264"

    head_r.markdown(
        f"<div style='text-align:right; padding-top:0.6rem; font-size:0.9rem;'>"
        f"<b>Updated:</b> <code>{mtime_str}</code> &nbsp;|&nbsp; "
        f"<b>Size:</b> <code>{size_mb:.2f} MB</code> &nbsp;|&nbsp; {h264_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(caption)

    # Pass raw bytes (not a static file path string) so browser/Streamlit never plays a stale cached video
    video_bytes = playable.read_bytes()
    st.video(video_bytes, format="video/mp4")
    st.caption(f"File: `{playable.relative_to(ROOT)}` | Modified: `{mtime_str}` | Size: `{size_mb:.2f} MB`")


def main() -> None:
    top_l, top_r = st.columns([5, 1])
    top_l.title("Long-Horizon Multi-Object Tracker — Diagnostics & Benchmark Dashboard")
    if top_r.button("Reload Latest Outputs", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "Occlusion-resilient MOT with **CMC Affine Warping**, **Depth-Triggered Occlusion Locking (`IoU >= 0.25`)**, "
        "**Clean-View Episodic Vault (`neighbor IoU <= 0.05`)**, **Online Hard-Negative Metric**, "
        "**Scene-Constrained Ghost Particle Filter**, and **≥15f Probationary Stitching + Margin-Ratio Abstention (`best_sim >= 0.72`, `margin >= 0.08`)**."
    )

    summary = load_json(SUMMARY_JSON, _file_mtime(SUMMARY_JSON))
    comp = load_json(COMPARISON_SUMMARY_JSON, _file_mtime(COMPARISON_SUMMARY_JSON))
    counters = summary.get("counters", {})
    base_stats = comp.get("baseline", {})
    ours_stats = comp.get("long_horizon_tracker", {})

    # -----------------------------------------------------------------------
    # Top KPI Row
    # -----------------------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ours_ids = ours_stats.get("total_unique_ids", counters.get("new_ids", "—"))
    base_ids = base_stats.get("total_unique_ids")
    id_delta = f"-{base_ids - ours_ids} vs Baseline ({base_ids})" if isinstance(base_ids, int) and isinstance(ours_ids, int) else None

    c1.metric("Unique IDs (Ours)", ours_ids, delta=id_delta, delta_color="inverse")
    c2.metric("STITCH_COMMIT", counters.get("stitches", "—"))
    c3.metric("MARGIN_VETO", counters.get("margin_vetoes", "—"))
    c4.metric("DEPTH_OCCLUDED", counters.get("depth_occlusions", "—"))
    c5.metric("Clean Vault Stores", counters.get("vault_stores", "—"), delta=f"{counters.get('vault_rejects', 0)} rejected", delta_color="off")
    c6.metric("End-to-End FPS", f"{summary.get('avg_fps', 0.0):.2f}", delta=f"{summary.get('frames_processed', 0)} frames", delta_color="off")

    tab_videos, tab_events, tab_bench = st.tabs(
        ["Videos (Side-by-Side & Diagnostics)", "Explainable Event Log & Similarity/Margin", "Stress Benchmark Report"]
    )

    # -----------------------------------------------------------------------
    # Tab 1: Videos
    # -----------------------------------------------------------------------
    with tab_videos:
        if _get_ffmpeg_exe() is None:
            st.info(
                "Install `imageio-ffmpeg` into `.venv` for automatic browser-compatible H.264 transcoding."
            )
        render_video_card(
            "Side-by-Side Comparison: Baseline (2D IoU + EMA) vs. LongHorizonTracker",
            "Identical YOLOv8n detections, DINOv2 embeddings, and Depth-Anything-V2 depth maps fed to both trackers.",
            COMPARISON_VIDEO,
        )
        st.divider()
        render_video_card(
            "Full Explainable Diagnostics Overlay",
            "Ghost particle clouds, 1.5σ covariance ellipses, depth PiP, vault occupancy, and live decision ticker.",
            DIAGNOSTICS_VIDEO,
        )

    # -----------------------------------------------------------------------
    # Tab 2: Filterable Events & Similarity/Margin Charts
    # -----------------------------------------------------------------------
    with tab_events:
        df = load_events_df(EVENTS_LOG, _file_mtime(EVENTS_LOG))
        if df.empty:
            st.warning(f"No events found in `{EVENTS_LOG.relative_to(ROOT)}`.")
        else:
            st.subheader("Filterable Explainable Decision Log")
            f1, f2, f3 = st.columns([2, 2, 2])
            all_types = sorted(df["event_type"].unique().tolist())
            sel_types = f1.multiselect("Event Types", options=all_types, default=all_types)

            all_tids = sorted(df["track_id"].unique().tolist())
            sel_tids = f2.multiselect("Track IDs (empty = all)", options=all_tids, default=[])

            min_f, max_f = int(df["frame"].min()), int(df["frame"].max())
            frame_rng = f3.slider("Frame Range", min_value=min_f, max_value=max(min_f + 1, max_f), value=(min_f, max_f))

            filt = df[
                df["event_type"].isin(sel_types)
                & (df["frame"] >= frame_rng[0])
                & (df["frame"] <= frame_rng[1])
            ]
            if sel_tids:
                filt = filt[filt["track_id"].isin(sel_tids)]

            st.caption(f"Showing **{len(filt)}** of **{len(df)}** logged events")
            display_cols = [
                "frame",
                "event_type",
                "track_id",
                "prob_id",
                "best_sim",
                "second_best_sim",
                "margin",
                "ratio",
                "ghost_age",
                "message",
            ]
            st.dataframe(filt[display_cols], use_container_width=True, hide_index=True)

            # Similarity & Margin-Ratio Abstention Charts
            st.subheader("Gallery Matching: Similarity & Margin-Ratio Abstention Boundaries")
            sim_df = filt.dropna(subset=["best_sim", "margin"]).copy()
            if sim_df.empty:
                st.info("Select `STITCH_COMMIT`, `MARGIN_VETO`, or `NEW_TRACK` events to view similarity & margin plots.")
            else:
                ch1, ch2 = st.columns(2)

                with ch1:
                    pts = (
                        alt.Chart(sim_df)
                        .mark_circle(size=90)
                        .encode(
                            x=alt.X("frame:Q", title="Frame"),
                            y=alt.Y("best_sim:Q", title="Best Set-to-Set Cosine Similarity", scale=alt.Scale(domain=[0.0, 1.0])),
                            color=alt.Color("event_type:N", title="Decision"),
                            tooltip=["frame", "event_type", "track_id", "prob_id", "best_sim", "second_best_sim", "margin", "message"],
                        )
                    )
                    thresh_sim = (
                        alt.Chart(pd.DataFrame({"y": [0.72]}))
                        .mark_rule(color="#ef4444", strokeDash=[6, 4], size=2)
                        .encode(y="y:Q")
                    )
                    st.altair_chart((pts + thresh_sim).properties(height=340, title="best_sim vs. Frame (threshold = 0.72)"), use_container_width=True)

                with ch2:
                    pts_m = (
                        alt.Chart(sim_df)
                        .mark_circle(size=90)
                        .encode(
                            x=alt.X("best_sim:Q", title="Best Similarity (best_sim)", scale=alt.Scale(domain=[0.0, 1.0])),
                            y=alt.Y("margin:Q", title="Margin (best_sim - second_best_sim)", scale=alt.Scale(domain=[0.0, 0.5])),
                            color=alt.Color("event_type:N", title="Decision"),
                            tooltip=["frame", "event_type", "track_id", "prob_id", "best_sim", "second_best_sim", "margin", "ratio", "message"],
                        )
                    )
                    rule_x = (
                        alt.Chart(pd.DataFrame({"x": [0.72]}))
                        .mark_rule(color="#ef4444", strokeDash=[6, 4], size=2)
                        .encode(x="x:Q")
                    )
                    rule_y = (
                        alt.Chart(pd.DataFrame({"y": [0.08]}))
                        .mark_rule(color="#f59e0b", strokeDash=[6, 4], size=2)
                        .encode(y="y:Q")
                    )
                    st.altair_chart(
                        (pts_m + rule_x + rule_y).properties(
                            height=340,
                            title="Decision Boundary: best_sim >= 0.72 AND margin >= 0.08",
                        ),
                        use_container_width=True,
                    )

            # Event type breakdown bar chart
            st.subheader("Event Distribution")
            counts_df = df["event_type"].value_counts().reset_index()
            counts_df.columns = ["event_type", "count"]
            bar = (
                alt.Chart(counts_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("event_type:N", sort="-y", title="Event Type"),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("event_type:N", legend=None),
                    tooltip=["event_type", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(bar, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 3: Benchmark Report
    # -----------------------------------------------------------------------
    with tab_bench:
        if BENCHMARK_MD.exists():
            st.markdown(BENCHMARK_MD.read_text(encoding="utf-8"))
        else:
            st.warning("`benchmark_report.md` not found. Run `eval_metrics.py` first.")


if __name__ == "__main__":
    main()
