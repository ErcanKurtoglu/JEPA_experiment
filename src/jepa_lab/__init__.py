"""Small, inspectable JEPA learning components.

The package intentionally keeps Meta's pinned upstream repositories unmodified.
Its tiny models are mechanism checks, not paper-scale reproductions.
"""

from .action_world_model import ActionWorldModel, BlockCausalPredictor
from .cem import CEMPlanner
from .image_jepa import ImageJEPA
from .simulator import PlanarReachEnv
from .types import PlanResult, RawVideo, VideoBatch, VisualEncoder
from .video_jepa import TinyVideoJEPA

__all__ = [
    "ActionWorldModel",
    "BlockCausalPredictor",
    "CEMPlanner",
    "ImageJEPA",
    "PlanResult",
    "PlanarReachEnv",
    "RawVideo",
    "TinyVideoJEPA",
    "VideoBatch",
    "VisualEncoder",
]

__version__ = "0.1.0"
