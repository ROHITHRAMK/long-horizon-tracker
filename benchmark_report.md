# Long-Horizon Tracker Benchmark Report

Controlled synthetic stress benchmark comparing **`LongHorizonTracker`** (CMC affine warping, Clean-View `EpisodicVault`, Depth-Triggered Occlusion Locking, Scene-Constrained Ghost Particle Filter, ≥15f Probationary Stitching, Margin-Ratio Abstention) against **`NaiveEMATracker`** (2D IoU + rolling-EMA Re-ID with instant lost-track merge and no safeguards).

## Results Comparison Table

| Stress Sequence | Tracker | IDF1 | IDSW ↓ | False Merges ↓ | Recovery Precision ↑ | Abstention Rate ↑ | Margin Vetoes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Camera Pan + 60-Frame Blackout | NaiveEMATracker | 79.7% | 2 | 0 | N/A (0/0) | N/A | 0 |
| 1. Camera Pan + 60-Frame Blackout | **LongHorizonTracker** | **95.8%** | **0** | **0** | **100.0% (2/2)** | **N/A** | **0** |
| 2. Lookalike Crossing & Swaps | NaiveEMATracker | 75.5% | 0 | 1 | 0.0% (0/1) | 0.0% | 0 |
| 2. Lookalike Crossing & Swaps | **LongHorizonTracker** | **84.9%** | **1** | **0** | **N/A (0/0)** | **100.0%** | **16** |
| 3. Severe Appearance/Illumination Drift | NaiveEMATracker | 83.6% | 1 | 0 | N/A (0/0) | N/A | 0 |
| 3. Severe Appearance/Illumination Drift | **LongHorizonTracker** | **95.7%** | **0** | **0** | **100.0% (1/1)** | **N/A** | **0** |
| **Overall Aggregate** | NaiveEMATracker | 79.6% | 3 | 1 | 0.0% (0/1) | 0.0% | 0 |
| **Overall Aggregate** | **LongHorizonTracker** | **92.1%** | **1** | **0** | **100.0% (3/3)** | **100.0%** | **16** |

## Metric Definitions
- **IDF1**: Global optimal GT-to-Pred bipartite assignment `2*IDTP / (2*IDTP + IDFP + IDFN)` at `IoU >= 0.50`.
- **IDSW**: CLEAR MOT identity switches whenever a GT target's matched predicted ID changes.
- **False Merges**: Distinct `(pred_id, wrong_gt_id)` events where a predicted track hijacks a different GT identity than its original owner.
- **Recovery Precision**: Fraction of lost-track ID recoveries that stitched back to the exact original GT identity.
- **Abstention Rate**: Fraction of ambiguous look-alike newcomers safely vetoed (`MARGIN_VETO`) rather than falsely merged into a lost track.

## Key Takeaways
1. **Zero False Identity Merges (0 vs 1)**: Across all three stress scenarios—including look-alike crossings, overlap feature bleed, and ambiguous newcomers—`LongHorizonTracker` eliminated false merges completely, whereas `NaiveEMATracker` suffered 1 false merge(s) and 3 ID switch(es).
2. **Camera Pan + 60-Frame Blackout Continuity**: Foreground-masked CMC affine warping prevented track breakage during sudden camera pan jolts, and the Scene-Constrained Ghost Particle Filter preserved lost identities across the full 60-frame blackout for clean post-probation recovery (`100.0%` precision).
3. **Depth-Triggered Vault Freeze Prevents EMA Poisoning**: In both Sequence 2 (look-alike crossing) and Sequence 3 (partial-occlusion corridor + static pillar + severe illumination drift), freezing the farther track's `EpisodicVault` at `IoU >= 0.25` and rejecting overlapping/blurry crops preserved clean identity prototypes—enabling accurate recovery even after severe illumination drift, where the rolling-EMA baseline was poisoned.
4. **Margin-Ratio Abstention Protects Precision (100.0% vs 0.0%)**: When an ambiguous look-alike entered while two near-identical ghosts were lost, `LongHorizonTracker` fired `16` `MARGIN_VETO` abstentions (`margin < 0.08`) and safely minted a new ID instead of guessing; the naive baseline immediately merged the wrong identity.
5. **Intentional Probation Recall Tradeoff**: Enforcing the non-negotiable 15-frame `PROBATIONARY` buffer slightly reduces raw frame-level recall during the first 14 frames of reappearance, trading a few unconfirmed frames for **100% Recovery Precision and 0 False Identity Merges**.
