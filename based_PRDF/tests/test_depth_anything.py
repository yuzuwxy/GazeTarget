import numpy as np

from preprocessing.depth_anything import (
    DEPTH_STAT_NAMES,
    extract_depth_statistics,
    normalize_depth_map,
)


def test_normalize_depth_map_uses_per_image_minmax():
    depth = np.asarray([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    normalized = normalize_depth_map(depth)
    np.testing.assert_allclose(
        normalized,
        np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
    )


def test_extract_depth_statistics_prefers_object_mask_and_uses_boxes():
    depth = np.arange(20, dtype=np.float32).reshape(4, 5)
    object_masks = np.zeros((1, 4, 5), dtype=np.uint8)
    object_masks[0, 1:3, 2:4] = 1

    object_stats = extract_depth_statistics(
        depth,
        bboxes=np.asarray([[0, 0, 5, 4]], dtype=np.float32),
        masks=object_masks,
    )
    head_stats = extract_depth_statistics(
        depth,
        bboxes=np.asarray([[1, 0, 3, 2]], dtype=np.float32),
    )

    values = np.asarray([7, 8, 12, 13], dtype=np.float32)
    assert set(object_stats) == set(DEPTH_STAT_NAMES)
    assert object_stats["mean"][0] == np.mean(values)
    assert object_stats["median"][0] == np.median(values)
    assert object_stats["min"][0] == np.min(values)
    assert object_stats["max"][0] == np.max(values)
    assert object_stats["std"][0] == np.std(values)
    np.testing.assert_allclose(
        [head_stats[name][0] for name in DEPTH_STAT_NAMES],
        [4.0, 4.0, 1.0, 7.0, np.std([1, 2, 6, 7])],
    )


def test_extract_depth_statistics_returns_nan_for_invalid_regions():
    depth = np.ones((4, 5), dtype=np.float32)
    stats = extract_depth_statistics(
        depth,
        bboxes=np.asarray([[3, 2, 3, 4], [20, 20, 30, 30]], dtype=np.float32),
    )

    for name in DEPTH_STAT_NAMES:
        assert stats[name].shape == (2,)
        assert np.isnan(stats[name]).all()
