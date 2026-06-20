"""Compatibility exports for head detection."""

from preprocessing.head_detector import (
    GroundingDINOHeadDetector,
    HuggingFaceGroundingDINOHeadDetector,
    MMDetectionGroundingDINOHeadDetector,
    build_head_detector,
    extract_head_detections,
    normalize_hf_head_detections,
    resolve_hf_model_source,
)

__all__ = [
    "GroundingDINOHeadDetector",
    "HuggingFaceGroundingDINOHeadDetector",
    "MMDetectionGroundingDINOHeadDetector",
    "build_head_detector",
    "extract_head_detections",
    "normalize_hf_head_detections",
    "resolve_hf_model_source",
]
