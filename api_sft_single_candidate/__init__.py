"""Single-candidate multimodal SFT trajectory generation pipeline."""

from .trajectories import (
    GENERATION_AUDIT_FORMAT_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    generate_trajectories,
)

__all__ = [
    "GENERATION_AUDIT_FORMAT_VERSION",
    "TRAJECTORY_FORMAT_VERSION",
    "generate_trajectories",
]

__version__ = "1.0.0"
