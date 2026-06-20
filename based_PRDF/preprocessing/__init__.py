"""Preprocessing components for gaze-target HDF5 generation."""

from .bbox import BBoxFilterConfig, MaskCandidate
from .image_source import ImageRecord, discover_images
from .pipeline import GazePreprocessingPipeline

__all__ = [
    "BBoxFilterConfig",
    "GazePreprocessingPipeline",
    "ImageRecord",
    "MaskCandidate",
    "discover_images",
]
