import pytest

from scripts.sanity_check_head_detector import validate_detections


def test_validate_detections_accepts_empty_and_in_bounds_results():
    validate_detections([], image_size=(20, 10))
    validate_detections(
        [{"bbox": [0, 1, 20, 10], "score": 0.8, "label": "head"}],
        image_size=(20, 10),
    )


def test_validate_detections_rejects_out_of_bounds_box():
    with pytest.raises(ValueError, match="invalid bbox"):
        validate_detections(
            [{"bbox": [-1, 0, 5, 5], "score": 0.8, "label": "head"}],
            image_size=(20, 10),
        )
