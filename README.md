# Long-Horizon Multi-Object Tracker

An occlusion-resilient, explainable Multi-Object Tracking (MOT) engine designed to preserve object identities across **handheld/panning cameras**, **prolonged 60+ frame occlusions**, **severe appearance/illumination drift**, and **look-alike crowding**—with **0 false identity merges**.

---

## Problem Statement

Conventional 2D MOT pipelines (e.g., 2D IoU + rolling-EMA Re-ID such as DeepSORT/ByteTrack-style baselines) fail systematically under real-world long-horizon tracking stress:

1. **Camera Ego-Motion Breaks Spatial Matching**: Uncompensated camera pans shift predicted 2D boxes off targets, triggering unnecessary ID switches.
2. **Naive Rolling-EMA Poisoning**: Updating a track's appearance embedding with a rolling average during partial overlap or motion blur contaminates its identity representation with the foreground occluder.
3. **Crossing-Path ID Swaps**: Pure 2D overlap cannot distinguish which target is in front vs. behind when visually similar objects cross.
4. **Premature Look-Alike Merges**: Single-frame instant Re-ID matching against lost tracks causes catastrophic identity conflation whenever two visually similar targets appear in the same scene.

---

## Core Innovations

1. **Camera Motion Compensation (`tracker/cmc.py`)**
   - Estimates frame-to-frame partial-affine ego-motion on **foreground-masked background pixels** using Shi-Tomasi corners, forward-backward Lucas-Kanade optical flow consistency filtering, and RANSAC `estimateAffinePartial2D` with sanity gating and safe identity fallback.
   - Warps all active/probationary 8D Kalman states (`warp_kalman_state`) and ghost particle clouds (`warp_particles`) **before** every association step.

2. **Depth-Triggered Occlusion Locking (`tracker/engine.py`)**
   - Consumes monocular relative disparity maps (`Depth-Anything-V2-Small`, where higher disparity = closer to camera).
   - Whenever two active tracks overlap with `IoU >= 0.25`, the lower-disparity (farther) track is immediately flagged **`DEPTH-OCCLUDED`** and its `EpisodicVault` is **frozen** (`vault.freeze()`), preventing occluder feature bleed.

3. **Clean-View Episodic Vault (`tracker/vault.py`)**
   - Eliminates naive rolling-EMA updates. Candidate crops enter a track's fixed-capacity `EpisodicVault` **only** after passing `CleanViewGate`:
     - Zero neighbor overlap (`neighbor IoU <= 0.05`, closer-object `IoS <= 0.05`)
     - High detection confidence (`conf >= 0.60`)
     - High Laplacian sharpness (`var >= 50.0`)
   - Protects the initial clean anchor view and uses diversity-preserving eviction when at capacity; `vault.ema_embedding` is recomputed strictly as the quality-weighted centroid of verified clean entries.

4. **Online Hard-Negative Metric (`tracker/vault.py`)**
   - Treats simultaneously visible distinct active tracks as guaranteed **hard negatives**.
   - Performs online scene-mean centering and exact dual $S^{-1/2}$ within-class scatter whitening (in $O(M^2 D)$ time) with version-keyed vault transform caching (`(vault._uid, _metric_version, vault._version)`), plus a co-occurring hard-negative ledger.

5. **Scene-Constrained Ghost Particle Filter (`tracker/ghosts.py`)**
   - Represents temporarily lost/occluded tracks as weighted 7D particle clouds `[cx, cy, vx, vy, w, h, depth]` rather than blind linear extrapolation.
   - Incorporates **observation-of-absence** reweighting: particles hidden behind closer active objects or closer static scene geometry (pillars, walls, vehicles in the depth map) retain high survival weight, while particles in unoccluded open space receive visibility downweighting, with ESS-triggered systematic resampling.

6. **Probationary Stitching with Margin-Ratio Abstention (`tracker/engine.py`)**
   - Newly appearing unmatched detections enter a **`PROBATIONARY`** state for **at least 15 consecutive frames** before any set-to-set gallery matching against `LOST` tracks—eliminating instant single-frame false merges.
   - Commits a `STITCH_COMMIT` recovery **only** when both absolute similarity and competitor margin hold:
     - `best_sim >= 0.72` **AND** `(best_sim - second_best_sim) >= 0.08` (checked across both `LOST` ghosts and active look-alike distractors).
   - On ambiguity, logs **`MARGIN_VETO`** and abstains (`NEW_ID_AFTER_VETO`) to preserve identity precision.

---

## Project Structure

```text
long-horizon-tracker/
├── AGENTS.md                 # Persistent non-negotiable architecture invariants
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── tracker/
│   ├── __init__.py           # Public API exports
│   ├── cmc.py                # Camera Motion Compensation (foreground-masked affine warp)
│   ├── vault.py              # Clean-View Episodic Vault + Online Hard-Negative Metric
│   ├── ghosts.py             # Scene-Constrained Ghost Particle Filter
│   └── engine.py             # Depth-gated tracking, Probationary stitching & Margin-Ratio veto
├── tests/
│   └── test_invariants.py    # Deterministic synthetic unit & invariant verification suite
├── run_tracker.py            # RTX-optimized main pipeline + Explainable Diagnostics Overlay
├── eval_metrics.py           # Controlled 3-sequence stress benchmark vs. NaiveEMATracker
├── render_comparison.py      # Synchronized side-by-side Baseline vs. Ours video renderer
└── benchmark_report.md       # Generated benchmark comparison report
```

---

## Quickstart

### 1. Environment & Dependencies (Windows PowerShell, CUDA RTX GPU)

```powershell
# Create and activate virtual environment (Python 3.12 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install CUDA-enabled PyTorch (cu124)
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies + pytest
pip install -r requirements.txt pytest
```

### 2. Run Invariant Verification Suite (`pytest`)

```powershell
.\.venv\Scripts\pytest.exe -v
```

### 3. Run the Main Tracker + Explainable Diagnostics Overlay

Uses `YOLOv8n`, `DINOv2 (dinov2_vits14)` with batched GPU `roi_align` + FP16 autocast, `Depth-Anything-V2-Small` with CMC-warped `--depth-stride 2`, tight ROI ghost blending, and an asynchronous `VideoWriter` thread:

```powershell
.\.venv\Scripts\python.exe run_tracker.py `
  --video data/test_video.mp4 `
  --reid dinov2 `
  --depth depth_anything `
  --depth-stride 2 `
  --output output/tracked_diagnostics.mp4 `
  --log output/events.jsonl
```

### 4. Run Synthetic Stress Benchmark (`eval_metrics.py`)

```powershell
.\.venv\Scripts\python.exe eval_metrics.py
```

### 5. Render Side-by-Side Comparison Video (`Baseline` vs. `LongHorizonTracker`)

```powershell
.\.venv\Scripts\python.exe render_comparison.py `
  --video data/test_video.mp4 `
  --output output/comparison_side_by_side.mp4
```

---

## Verified Benchmark & Test Results

### Controlled Synthetic Stress Benchmark ([benchmark_report.md](benchmark_report.md))

| Stress Sequence | Tracker | IDF1 ↑ | IDSW ↓ | False Merges ↓ | Recovery Precision ↑ | Abstention Rate ↑ | Margin Vetoes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. Camera Pan + 60-Frame Blackout** | NaiveEMATracker | 79.7% | 2 | 0 | N/A (0/0) | N/A | 0 |
| **1. Camera Pan + 60-Frame Blackout** | **LongHorizonTracker** | **95.8%** | **0** | **0** | **100.0% (2/2)** | **N/A** | **0** |
| **2. Lookalike Crossing & Swaps** | NaiveEMATracker | 75.5% | 0 | 1 | 0.0% (0/1) | 0.0% | 0 |
| **2. Lookalike Crossing & Swaps** | **LongHorizonTracker** | **84.9%** | **1** | **0** | **N/A (0/0)** | **100.0%** | **16** |
| **3. Severe Appearance/Illumination Drift** | NaiveEMATracker | 83.6% | 1 | 0 | N/A (0/0) | N/A | 0 |
| **3. Severe Appearance/Illumination Drift** | **LongHorizonTracker** | **95.7%** | **0** | **0** | **100.0% (1/1)** | **N/A** | **0** |
| **Overall Aggregate** | NaiveEMATracker | 79.6% | 3 | 1 | 0.0% (0/1) | 0.0% | 0 |
| **Overall Aggregate** | **LongHorizonTracker** | **92.1%** | **1** | **0** | **100.0% (3/3)** | **100.0%** | **16** |

### Real Video Performance (`data/test_video.mp4` — 1920×1080, 556 frames on RTX 4070 Laptop GPU)

- **Main Diagnostics Pipeline (`run_tracker.py`)**: **`15.67 FPS`** end-to-end (`35.49s`, `2.24×` speedup over initial `6.98 FPS` FP32/PIL baseline; core loop `20.0–25.7 FPS`), achieving `3` verified `STITCH_COMMIT` recoveries (up to `50f` occlusion), `11` `DEPTH_OCCLUDED` vault freezes, and `11` `MARGIN_VETO` abstentions.
- **Side-by-Side Comparison (`render_comparison.py`)**: **`16.32 FPS`** (`34.1s`) on identical detections, embeddings, and depth maps:
  - **Baseline (`2D IoU + EMA`)**: **`42`** total unique IDs created (`41` unverified instant merges).
  - **`LongHorizonTracker` (Ours)**: **`32`** total unique IDs created (**23.8% fewer fragmented IDs**, `3` verified multi-frame recoveries, `11` margin-ratio abstentions, `11` depth-occlusion locks).

### Invariant Unit Test Suite (`tests/test_invariants.py`)

```text
tests/test_invariants.py::test_cmc_pan_recovery PASSED                   [ 20%]
tests/test_invariants.py::test_depth_occlusion_vault_freeze PASSED       [ 40%]
tests/test_invariants.py::test_60_frame_blackout_recovery PASSED         [ 60%]
tests/test_invariants.py::test_lookalike_margin_veto PASSED              [ 80%]
tests/test_invariants.py::test_metric_vault_cache_equivalence PASSED     [100%]

5 passed in 1.61s
```
