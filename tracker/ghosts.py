"""Scene-Constrained Ghost Particle Filter for Long-Horizon Occlusion Survival.

Represents each temporarily lost/occluded track as a weighted particle cloud in
state space [cx, cy, vx, vy, w, h, depth]. Every frame:
1. Particles are warped by Camera Motion Compensation (CMC).
2. Dynamics propagate with damped velocity and bounded diffusion.
3. Observation-of-Absence reweighting boosts particles hiding behind closer active
   foreground objects or closer static scene geometry (via depth map) and penalizes
   particles sitting in unoccluded open space where the detector should have fired.
4. Systematic resampling preserves multi-modal hypotheses without particle collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cmc import warp_points, warp_vectors


@dataclass
class GhostConfig:
    """Configuration for the Scene-Constrained Ghost Particle Filter."""

    num_particles: int = 128
    max_ghost_age: int = 600
    pos_noise_px: float = 3.0
    vel_noise_px: float = 1.4
    vel_damping: float = 0.96
    max_speed_px: float = 35.0
    size_noise_frac: float = 0.012
    depth_noise: float = 0.012
    depth_occluder_margin: float = 0.06
    static_occluder_margin: float = 0.05
    open_space_penalty: float = 0.86
    occluded_support_boost: float = 1.12
    border_support_factor: float = 0.97
    border_margin_px: float = 28.0
    resample_ess_ratio: float = 0.5
    min_confidence: float = 0.01
    open_space_conf_decay: float = 0.995
    occluded_conf_decay: float = 0.9995
    corridor_base_px: float = 140.0
    corridor_expand_px_per_frame: float = 4.5


@dataclass
class GhostState:
    """Live state of a single occluded/lost identity ghost hypothesis."""

    track_id: int
    class_id: int
    particles: np.ndarray  # (N, 7) float32: [cx, cy, vx, vy, w, h, depth]
    weights: np.ndarray  # (N,) float32 summing to 1
    spawn_frame: int
    last_update_frame: int
    age: int = 0
    occluded_spawn: bool = True
    occluder_id: Optional[int] = None
    mean_box: np.ndarray = None  # (4,) xyxy
    cov_xy: np.ndarray = None  # (2, 2)
    mean_depth: float = 0.5
    occluded_fraction: float = 1.0
    confidence: float = 1.0
    ess: float = 128.0

    def recompute_moments(self) -> None:
        """Update weighted mean bounding box, 2D covariance, depth, and ESS."""
        w = self.weights
        p = self.particles
        mean_cx = float(np.sum(w * p[:, 0]))
        mean_cy = float(np.sum(w * p[:, 1]))
        mean_w = max(8.0, float(np.sum(w * p[:, 4])))
        mean_h = max(8.0, float(np.sum(w * p[:, 5])))
        self.mean_depth = float(np.clip(np.sum(w * p[:, 6]), 0.0, 1.0))

        self.mean_box = np.array(
            [
                mean_cx - 0.5 * mean_w,
                mean_cy - 0.5 * mean_h,
                mean_cx + 0.5 * mean_w,
                mean_cy + 0.5 * mean_h,
            ],
            dtype=np.float32,
        )

        dx = p[:, 0] - mean_cx
        dy = p[:, 1] - mean_cy
        c_xx = float(np.sum(w * dx * dx)) + 4.0
        c_xy = float(np.sum(w * dx * dy))
        c_yy = float(np.sum(w * dy * dy)) + 4.0
        self.cov_xy = np.array([[c_xx, c_xy], [c_xy, c_yy]], dtype=np.float32)
        self.ess = float(1.0 / max(1e-9, float(np.sum(w * w))))

    def warp_particles(self, cmc_matrix: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        """Apply frame-to-frame CMC affine homography to particle positions, velocities, and scales."""
        h, w = frame_shape[:2]
        scale = float(np.sqrt(max(1e-6, abs(np.linalg.det(cmc_matrix[:, :2])))))
        p = self.particles
        p[:, 0:2] = warp_points(p[:, 0:2], cmc_matrix)
        p[:, 2:4] = warp_vectors(p[:, 2:4], cmc_matrix)
        p[:, 4] = np.clip(p[:, 4] * scale, 10.0, float(w) * 1.5)
        p[:, 5] = np.clip(p[:, 5] * scale, 10.0, float(h) * 1.5)
        self.recompute_moments()


# Alias matching architecture terminology
GhostParticleCloud = GhostState


class SceneConstrainedGhostFilter:
    """Manages all active Ghost Particle Filters in the scene."""

    def __init__(self, config: Optional[GhostConfig] = None, seed: int = 42) -> None:
        self.cfg = config or GhostConfig()
        self.rng = np.random.default_rng(seed)
        self.ghosts: Dict[int, GhostState] = {}

    def spawn_ghost(
        self,
        track_id: int,
        class_id: int,
        box: np.ndarray,
        velocity: np.ndarray,
        depth: float,
        frame_idx: int,
        occluded_spawn: bool = True,
        occluder_id: Optional[int] = None,
    ) -> GhostState:
        """Initialize a new Ghost Particle Cloud from a track's last known state."""
        N = self.cfg.num_particles
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        bw = max(12.0, float(box[2] - box[0]))
        bh = max(12.0, float(box[3] - box[1]))
        vx = float(np.clip(velocity[0], -self.cfg.max_speed_px, self.cfg.max_speed_px))
        vy = float(np.clip(velocity[1], -self.cfg.max_speed_px, self.cfg.max_speed_px))
        d = float(np.clip(depth, 0.0, 1.0))

        particles = np.zeros((N, 7), dtype=np.float32)
        particles[:, 0] = cx + self.rng.normal(0.0, max(2.0, 0.08 * bw), size=N)
        particles[:, 1] = cy + self.rng.normal(0.0, max(2.0, 0.06 * bh), size=N)
        # Multi-modal velocity hypothesis: 75% preserve momentum, 25% near-stopped behind occluder
        n_stop = N // 4
        particles[: n_stop, 2] = self.rng.normal(0.0, max(1.0, 0.25 * abs(vx) + 1.5), size=n_stop)
        particles[: n_stop, 3] = self.rng.normal(0.0, max(1.0, 0.25 * abs(vy) + 1.0), size=n_stop)
        particles[n_stop:, 2] = vx + self.rng.normal(0.0, max(1.5, 0.25 * abs(vx) + 1.2), size=N - n_stop)
        particles[n_stop:, 3] = vy + self.rng.normal(0.0, max(1.5, 0.25 * abs(vy) + 1.2), size=N - n_stop)
        particles[:, 4] = np.clip(bw * (1.0 + self.rng.normal(0.0, 0.04, size=N)), 10.0, 2000.0)
        particles[:, 5] = np.clip(bh * (1.0 + self.rng.normal(0.0, 0.04, size=N)), 10.0, 2000.0)
        particles[:, 6] = np.clip(d + self.rng.normal(0.0, 0.025, size=N), 0.0, 1.0)

        weights = np.full((N,), 1.0 / N, dtype=np.float32)
        ghost = GhostState(
            track_id=track_id,
            class_id=class_id,
            particles=particles,
            weights=weights,
            spawn_frame=frame_idx,
            last_update_frame=frame_idx,
            age=0,
            occluded_spawn=occluded_spawn,
            occluder_id=occluder_id,
            confidence=1.0 if occluded_spawn else 0.78,
        )
        ghost.recompute_moments()
        self.ghosts[track_id] = ghost
        return ghost

    def remove_ghost(self, track_id: int) -> Optional[GhostState]:
        return self.ghosts.pop(track_id, None)

    def _systematic_resample(self, ghost: GhostState) -> None:
        N = len(ghost.weights)
        positions = (self.rng.random() + np.arange(N, dtype=np.float64)) / N
        cumsum = np.cumsum(ghost.weights.astype(np.float64))
        cumsum[-1] = 1.0
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        ghost.particles = ghost.particles[indices].copy()
        ghost.weights.fill(1.0 / N)

        # Mild roughening jitter prevents particle impoverishment
        ghost.particles[:, 0] += self.rng.normal(0.0, 1.2, size=N).astype(np.float32)
        ghost.particles[:, 1] += self.rng.normal(0.0, 1.2, size=N).astype(np.float32)
        ghost.particles[:, 2] += self.rng.normal(0.0, 0.45, size=N).astype(np.float32)
        ghost.particles[:, 3] += self.rng.normal(0.0, 0.45, size=N).astype(np.float32)

    def step_all(
        self,
        cmc_matrix: np.ndarray,
        depth_map: Optional[np.ndarray],
        active_boxes: np.ndarray,
        active_depths: np.ndarray,
        frame_shape: Tuple[int, int],
        frame_idx: int,
    ) -> List[int]:
        """CMC-warp, propagate, and scene-constrain all active ghosts. Returns pruned IDs."""
        h, w = frame_shape[:2]
        pruned: List[int] = []

        has_active = active_boxes is not None and len(active_boxes) > 0
        act_boxes = (
            np.asarray(active_boxes, dtype=np.float32).reshape(-1, 4)
            if has_active
            else np.zeros((0, 4), dtype=np.float32)
        )
        act_depths = (
            np.asarray(active_depths, dtype=np.float32).reshape(-1)
            if has_active and active_depths is not None
            else np.zeros((0,), dtype=np.float32)
        )

        for tid, ghost in list(self.ghosts.items()):
            ghost.age += 1
            ghost.last_update_frame = frame_idx
            p = ghost.particles
            N = len(p)

            # 1. Camera Motion Compensation warp (always applied before propagation/association)
            ghost.warp_particles(cmc_matrix, frame_shape)

            # 2. Constant-velocity propagation with damped diffusion
            age_factor = min(2.0, 1.0 + 0.01 * ghost.age)
            p[:, 0] += p[:, 2] + self.rng.normal(0.0, self.cfg.pos_noise_px * age_factor, size=N)
            p[:, 1] += p[:, 3] + self.rng.normal(0.0, self.cfg.pos_noise_px * age_factor, size=N)
            p[:, 2] = np.clip(
                p[:, 2] * self.cfg.vel_damping + self.rng.normal(0.0, self.cfg.vel_noise_px, size=N),
                -self.cfg.max_speed_px,
                self.cfg.max_speed_px,
            )
            p[:, 3] = np.clip(
                p[:, 3] * self.cfg.vel_damping + self.rng.normal(0.0, self.cfg.vel_noise_px, size=N),
                -self.cfg.max_speed_px,
                self.cfg.max_speed_px,
            )
            p[:, 4] = np.clip(
                p[:, 4] * (1.0 + self.rng.normal(0.0, self.cfg.size_noise_frac, size=N)),
                10.0,
                float(w) * 1.5,
            )
            p[:, 5] = np.clip(
                p[:, 5] * (1.0 + self.rng.normal(0.0, self.cfg.size_noise_frac, size=N)),
                10.0,
                float(h) * 1.5,
            )
            p[:, 6] = np.clip(
                p[:, 6] + self.rng.normal(0.0, self.cfg.depth_noise, size=N),
                0.0,
                1.0,
            )

            # Keep particles within generous scene boundary margin
            margin_out = 0.25 * max(w, h)
            p[:, 0] = np.clip(p[:, 0], -margin_out, w + margin_out)
            p[:, 1] = np.clip(p[:, 1], -margin_out, h + margin_out)

            # 3. Scene-Constrained Observation-of-Absence Reweighting
            px = p[:, 0]
            py = p[:, 1]
            pd = p[:, 6]
            occluded_mask = np.zeros((N,), dtype=bool)

            # 3a. Occlusion by closer active foreground objects
            if len(act_boxes) > 0:
                in_x = (px[:, None] >= act_boxes[None, :, 0]) & (px[:, None] <= act_boxes[None, :, 2])
                in_y = (py[:, None] >= act_boxes[None, :, 1]) & (py[:, None] <= act_boxes[None, :, 3])
                if len(act_depths) == len(act_boxes):
                    closer = act_depths[None, :] > (pd[:, None] + self.cfg.depth_occluder_margin)
                else:
                    closer = np.ones((N, len(act_boxes)), dtype=bool)
                occluded_mask |= np.any(in_x & in_y & closer, axis=1)

            # 3b. Occlusion by closer static scene geometry (pillars/walls: scene_d > ghost_d + 0.05)
            in_frame = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if depth_map is not None and np.any(in_frame):
                dm_h, dm_w = depth_map.shape[:2]
                ix = np.clip((px[in_frame] * (dm_w / max(1.0, float(w)))).astype(np.int32), 0, dm_w - 1)
                iy = np.clip((py[in_frame] * (dm_h / max(1.0, float(h)))).astype(np.int32), 0, dm_h - 1)
                scene_d = depth_map[iy, ix]
                static_occ = scene_d > (pd[in_frame] + self.cfg.static_occluder_margin)
                occluded_mask[in_frame] |= static_occ

            # 3c. Border exit / near-edge zone
            bm = self.cfg.border_margin_px
            near_border = (px < bm) | (px > w - bm) | (py < bm) | (py > h - bm)

            # Compute per-particle likelihood multipliers
            mult = np.full((N,), self.cfg.open_space_penalty, dtype=np.float32)
            mult[near_border] = self.cfg.border_support_factor
            mult[occluded_mask] = self.cfg.occluded_support_boost

            ghost.weights *= mult
            w_sum = float(np.sum(ghost.weights))
            if w_sum <= 1e-12 or not np.isfinite(w_sum):
                ghost.weights.fill(1.0 / N)
            else:
                ghost.weights /= w_sum

            # Track weighted occluded fraction & update survival confidence
            occ_frac = float(np.sum(ghost.weights[occluded_mask | near_border]))
            ghost.occluded_fraction = occ_frac
            decay = (
                occ_frac * self.cfg.occluded_conf_decay
                + (1.0 - occ_frac) * self.cfg.open_space_conf_decay
            )
            ghost.confidence = float(np.clip(ghost.confidence * decay, 0.0, 1.0))

            # 4. Systematic resampling when ESS drops
            ghost.recompute_moments()
            if ghost.ess < self.cfg.resample_ess_ratio * N:
                self._systematic_resample(ghost)
                ghost.recompute_moments()

            # 5. Prune expired or disproven ghosts
            if ghost.age > self.cfg.max_ghost_age or ghost.confidence < self.cfg.min_confidence:
                del self.ghosts[tid]
                pruned.append(tid)

        return pruned

    def score_detection(
        self,
        ghost_id: int,
        det_box: np.ndarray,
        det_depth: float,
    ) -> Tuple[float, float, float]:
        """Score how well a re-emerging detection is supported by a ghost's particle cloud.

        Returns:
            (combined_support [0,1], mahalanobis_dist, depth_diff)
        """
        ghost = self.ghosts.get(ghost_id)
        if ghost is None:
            return 0.0, 999.0, 1.0

        det_cx = 0.5 * (float(det_box[0]) + float(det_box[2]))
        det_cy = 0.5 * (float(det_box[1]) + float(det_box[3]))
        det_w = max(8.0, float(det_box[2] - det_box[0]))
        det_h = max(8.0, float(det_box[3] - det_box[1]))

        p = ghost.particles
        w = ghost.weights

        # Scale-normalized Gaussian mixture kernel over particle positions
        sigma_x = np.maximum(24.0, 0.75 * p[:, 4])
        sigma_y = np.maximum(24.0, 0.75 * p[:, 5])
        norm_sq = ((p[:, 0] - det_cx) / sigma_x) ** 2 + ((p[:, 1] - det_cy) / sigma_y) ** 2
        spatial_kernel = np.exp(-0.5 * np.minimum(norm_sq, 50.0))
        spatial_support = float(np.sum(w * spatial_kernel))

        # Also reward high local density (peak particle cluster support for multi-modal clouds)
        top_idx = np.argsort(spatial_kernel)[-max(1, len(p) // 8) :]
        peak_support = float(np.mean(spatial_kernel[top_idx]))
        spatial_score = float(np.clip(0.55 * spatial_support + 0.45 * peak_support, 0.0, 1.0))

        # Scale consistency
        size_ratio_w = np.minimum(det_w, p[:, 4]) / np.maximum(det_w, p[:, 4])
        size_ratio_h = np.minimum(det_h, p[:, 5]) / np.maximum(det_h, p[:, 5])
        scale_score = float(np.sum(w * (0.5 * (size_ratio_w + size_ratio_h))))

        # Depth consistency
        depth_diff = abs(float(det_depth) - ghost.mean_depth)
        depth_score = float(np.exp(-0.5 * (depth_diff / 0.14) ** 2))

        # Mahalanobis distance to ghost mean for diagnostics
        mean_cx = 0.5 * (float(ghost.mean_box[0]) + float(ghost.mean_box[2]))
        mean_cy = 0.5 * (float(ghost.mean_box[1]) + float(ghost.mean_box[3]))
        diff = np.array([det_cx - mean_cx, det_cy - mean_cy], dtype=np.float64)
        cov = ghost.cov_xy.astype(np.float64) + np.diag([max(25.0, 0.25 * det_w) ** 2, max(25.0, 0.25 * det_h) ** 2])
        try:
            mahal = float(np.sqrt(max(0.0, diff @ np.linalg.inv(cov) @ diff)))
        except np.linalg.LinAlgError:
            mahal = 999.0

        combined = float(
            np.clip(
                (0.55 * spatial_score + 0.20 * scale_score + 0.25 * depth_score)
                * (0.4 + 0.6 * ghost.confidence),
                0.0,
                1.0,
            )
        )
        return combined, mahal, depth_diff

    def is_in_corridor(
        self,
        ghost_id: int,
        det_box: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[bool, float, float]:
        """Time-expanded spatial gating corridor for long-horizon pillar/occlusion exits.

        Expands reachable radius proportionally to `ghost.age` so a target walking behind
        a wide pillar and exiting on either side reaches set-to-set gallery matching while
        still gating out physically impossible cross-frame jumps on short occlusions.
        """
        ghost = self.ghosts.get(ghost_id)
        if ghost is None:
            return False, 1e9, 0.0

        h, w = frame_shape[:2]
        frame_diag = float(np.hypot(w, h))
        det_cx = 0.5 * (float(det_box[0]) + float(det_box[2]))
        det_cy = 0.5 * (float(det_box[1]) + float(det_box[3]))
        det_diag = float(np.hypot(max(8.0, float(det_box[2] - det_box[0])), max(8.0, float(det_box[3] - det_box[1]))))
        ghost_diag = float(
            np.hypot(
                max(8.0, float(ghost.mean_box[2] - ghost.mean_box[0])),
                max(8.0, float(ghost.mean_box[3] - ghost.mean_box[1])),
            )
        )

        # Base gate covers local box scale + time-expanded corridor proportional to occlusion age
        base_radius = max(self.cfg.corridor_base_px, 1.5 * max(det_diag, ghost_diag))
        corridor_radius = min(
            frame_diag,
            base_radius + float(ghost.age) * self.cfg.corridor_expand_px_per_frame,
        )

        # Measure shortest distance from detection to ghost mean OR any live particle hypothesis
        mean_cx = 0.5 * (float(ghost.mean_box[0]) + float(ghost.mean_box[2]))
        mean_cy = 0.5 * (float(ghost.mean_box[1]) + float(ghost.mean_box[3]))
        dist_mean = float(np.hypot(det_cx - mean_cx, det_cy - mean_cy))
        p_dists = np.hypot(ghost.particles[:, 0] - det_cx, ghost.particles[:, 1] - det_cy)
        min_dist = min(dist_mean, float(np.min(p_dists))) if len(p_dists) > 0 else dist_mean
        return min_dist <= corridor_radius, min_dist, corridor_radius
