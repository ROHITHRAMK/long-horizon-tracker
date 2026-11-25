# Long-Horizon Multi-Object Tracker Rules

## Non-Negotiable Architecture Invariants
1. **No Naive EMA Poisoning:** Never update a track's identity embedding with a rolling average during overlap. Only insert embeddings into `EpisodicVault` when a detection passes the Clean-View Gate (zero neighbor overlap, high detection confidence, high Laplacian sharpness).
2. **Depth-Triggered Occlusion Locking:** Depth Anything V2 outputs relative disparity (higher value means closer to camera). When two tracks overlap (IoU >= 0.25), the lower-disparity (farther) track MUST be flagged `DEPTH-OCCLUDED` and have its appearance vault frozen.
3. **No Instant Re-ID Merges:** Newly appearing unmatched detections MUST enter a `PROBATIONARY` state for at least 15 frames before set-to-set gallery matching against `LOST` tracks.
4. **Margin-Ratio Abstention:** Never stitch a probationary tracklet to a lost track unless `best_sim >= 0.72` AND `(best_sim - second_best_sim) >= 0.08`. When ambiguous, abstain and preserve precision.
5. **Camera Motion Compensation:** Always warp predicted track boxes and ghost particle clouds using frame-to-frame affine homography before association.

## Verification Standard
- Every change to `tracker/` must pass a synthetic 60-frame blackout and lookalike crossing test with 0 false identity merges.
