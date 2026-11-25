"""Long-Horizon Occlusion-Resilient Multi-Object Tracker."""

from .cmc import (
    CMCResult,
    CameraMotionCompensator,
    warp_box,
    warp_boxes,
    warp_kalman_state,
    warp_points,
    warp_vectors,
)
from .vault import (
    CleanViewAssessment,
    CleanViewGate,
    EpisodicVault,
    OnlineHardNegativeMetric,
    VaultEntry,
)
from .ghosts import (
    GhostConfig,
    GhostParticleCloud,
    GhostState,
    SceneConstrainedGhostFilter,
)
from .engine import (
    ActiveTrack,
    Detection,
    DiagnosticEvent,
    KalmanBoxTracker,
    LongHorizonTracker,
    ProbationaryTrack,
    TrackerConfig,
    TrackingFrameResult,
)

__all__ = [
    "CMCResult",
    "CameraMotionCompensator",
    "warp_box",
    "warp_boxes",
    "warp_kalman_state",
    "warp_points",
    "warp_vectors",
    "CleanViewAssessment",
    "CleanViewGate",
    "EpisodicVault",
    "OnlineHardNegativeMetric",
    "VaultEntry",
    "GhostConfig",
    "GhostParticleCloud",
    "GhostState",
    "SceneConstrainedGhostFilter",
    "ActiveTrack",
    "Detection",
    "DiagnosticEvent",
    "KalmanBoxTracker",
    "LongHorizonTracker",
    "ProbationaryTrack",
    "TrackerConfig",
    "TrackingFrameResult",
]
