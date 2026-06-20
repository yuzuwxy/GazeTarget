from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass
class MaskCandidate:
    bbox: BBox
    mask_area: int
    predicted_iou: float = 0.0
    stability_score: float = 0.0
    mask: Optional[np.ndarray] = None

    @property
    def score(self) -> float:
        return float(self.predicted_iou)


@dataclass
class BBoxFilterConfig:
    min_width: float = 2.0
    min_height: float = 2.0
    min_area: float = 16.0
    min_area_ratio: float = 0.0001
    max_area_ratio: float = 0.95
    duplicate_iou_threshold: float = 0.85
    containment_threshold: float = 0.95
    containment_area_ratio: float = 0.5

    @classmethod
    def from_dict(cls, values):
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in (values or {}).items() if key in fields})


def mask_to_bbox(mask: np.ndarray) -> Optional[BBox]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W), got {mask.shape}")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[BBox]:
    if len(bbox) != 4:
        raise ValueError(f"BBox must contain four coordinates, got {bbox}")
    values = np.asarray(bbox, dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = values.tolist()
    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _intersection_area(first, second)
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_coverage(container: Sequence[float], candidate: Sequence[float]) -> float:
    candidate_area = bbox_area(candidate)
    if candidate_area <= 0:
        return 0.0
    return _intersection_area(container, candidate) / candidate_area


def _valid_candidate(
    candidate: MaskCandidate,
    image_size: Tuple[int, int],
    config: BBoxFilterConfig,
) -> Optional[MaskCandidate]:
    width, height = image_size
    clipped = clip_bbox(candidate.bbox, width, height)
    if clipped is None:
        return None
    box_width = clipped[2] - clipped[0]
    box_height = clipped[3] - clipped[1]
    area = bbox_area(clipped)
    image_area = float(width * height)
    area_ratio = area / image_area if image_area > 0 else 0.0
    if box_width < config.min_width or box_height < config.min_height:
        return None
    if area < config.min_area:
        return None
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return None
    return MaskCandidate(
        bbox=clipped,
        mask_area=int(candidate.mask_area),
        predicted_iou=float(candidate.predicted_iou),
        stability_score=float(candidate.stability_score),
        mask=candidate.mask,
    )


def filter_object_candidates(
    candidates: Sequence[MaskCandidate],
    image_size: Tuple[int, int],
    config: BBoxFilterConfig,
):
    valid = []
    for candidate in candidates:
        normalized = _valid_candidate(candidate, image_size, config)
        if normalized is not None:
            valid.append(normalized)

    valid.sort(
        key=lambda item: (
            -item.mask_area,
            -item.predicted_iou,
            -item.stability_score,
            item.bbox,
        )
    )
    kept = []
    for candidate in valid:
        candidate_area = bbox_area(candidate.bbox)
        remove = False
        for existing in kept:
            if bbox_iou(existing.bbox, candidate.bbox) >= config.duplicate_iou_threshold:
                remove = True
                break
            existing_area = bbox_area(existing.bbox)
            if existing_area <= 0:
                continue
            area_ratio = candidate_area / existing_area
            if (
                area_ratio <= config.containment_area_ratio
                and bbox_coverage(existing.bbox, candidate.bbox)
                >= config.containment_threshold
            ):
                remove = True
                break
        if not remove:
            kept.append(candidate)
    return kept


def nms_bboxes(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
):
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores, kind="stable")
    kept = []
    for index in order:
        if all(bbox_iou(boxes[index], boxes[other]) < iou_threshold for other in kept):
            kept.append(int(index))
    return boxes[kept], scores[kept]
