import numpy as np

from preprocessing.bbox import (
    BBoxFilterConfig,
    MaskCandidate,
    bbox_coverage,
    bbox_iou,
    clip_bbox,
    filter_object_candidates,
    mask_to_bbox,
)


def test_mask_to_bbox_uses_half_open_xyxy_coordinates():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    assert mask_to_bbox(mask) == (3.0, 2.0, 7.0, 5.0)


def test_clip_bbox_rejects_degenerate_boxes():
    assert clip_bbox((-2, 1, 12, 9), width=10, height=8) == (0.0, 1.0, 10.0, 8.0)
    assert clip_bbox((3, 3, 3, 5), width=10, height=8) is None


def test_bbox_overlap_metrics():
    outer = (0.0, 0.0, 10.0, 10.0)
    inner = (2.0, 2.0, 5.0, 5.0)

    assert bbox_iou(outer, inner) == 0.09
    assert bbox_coverage(outer, inner) == 1.0


def test_filter_removes_duplicate_and_contained_candidates():
    config = BBoxFilterConfig(
        min_width=2,
        min_height=2,
        min_area=4,
        min_area_ratio=0.0,
        max_area_ratio=1.0,
        duplicate_iou_threshold=0.8,
        containment_threshold=0.95,
        containment_area_ratio=0.5,
    )
    candidates = [
        MaskCandidate((0, 0, 10, 10), 100, 0.95, 0.95, None),
        MaskCandidate((0, 0, 9.8, 9.8), 96, 0.90, 0.90, None),
        MaskCandidate((2, 2, 5, 5), 9, 0.99, 0.99, None),
        MaskCandidate((12, 1, 16, 6), 20, 0.80, 0.80, None),
    ]

    kept = filter_object_candidates(candidates, (20, 20), config)

    assert [candidate.bbox for candidate in kept] == [
        (0.0, 0.0, 10.0, 10.0),
        (12.0, 1.0, 16.0, 6.0),
    ]


def test_filter_rejects_tiny_and_nearly_full_image_boxes():
    config = BBoxFilterConfig(
        min_width=3,
        min_height=3,
        min_area=9,
        min_area_ratio=0.01,
        max_area_ratio=0.8,
    )
    candidates = [
        MaskCandidate((0, 0, 2, 2), 4, 1.0, 1.0, None),
        MaskCandidate((0, 0, 10, 10), 100, 1.0, 1.0, None),
        MaskCandidate((2, 2, 7, 7), 25, 1.0, 1.0, None),
    ]

    kept = filter_object_candidates(candidates, (10, 10), config)

    assert [candidate.bbox for candidate in kept] == [(2.0, 2.0, 7.0, 7.0)]
