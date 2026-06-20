import numpy as np

from preprocessing.describe_anything import (
    DescribeAnythingCaptioner,
    boxes_to_masks,
)


class FakeDAM:
    def __init__(self):
        self.calls = []

    def get_description(self, image, mask, query, **kwargs):
        mask_array = np.asarray(mask) > 0
        self.calls.append((image.size, mask_array.copy(), query, kwargs))
        if "fail" in query:
            raise RuntimeError("expected failure")
        return f"pixels={int(mask_array.sum())}"


def test_boxes_to_masks_clips_and_preserves_alignment():
    masks = boxes_to_masks(
        np.asarray([[-2, 1, 4, 5], [3, 3, 3, 4]], dtype=np.float32),
        image_size=(6, 5),
    )

    assert masks.shape == (2, 5, 6)
    assert masks[0].sum() == 16
    assert masks[1].sum() == 0


def test_captioner_uses_object_masks_and_head_bbox_masks():
    model = FakeDAM()
    captioner = DescribeAnythingCaptioner(
        source_path="unused",
        checkpoint="unused",
        device="cuda:0",
        object_prompt="Describe object.",
        head_prompt="Describe head.",
        model=model,
    )
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    object_masks = np.zeros((1, 5, 6), dtype=np.uint8)
    object_masks[0, 1:3, 2:5] = 1

    result = captioner.describe(
        image,
        object_bboxes=np.asarray([[0, 0, 6, 5]], dtype=np.float32),
        object_masks=object_masks,
        head_bboxes=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
    )

    assert result == {
        "object_descriptions": ["pixels=6"],
        "head_descriptions": ["pixels=4"],
    }
    assert model.calls[0][2] == "<image>\nDescribe object."
    assert model.calls[1][2] == "<image>\nDescribe head."


def test_captioner_keeps_empty_string_for_failed_region():
    captioner = DescribeAnythingCaptioner(
        source_path="unused",
        checkpoint="unused",
        device="cuda:0",
        object_prompt="fail object",
        head_prompt="Describe head.",
        model=FakeDAM(),
    )
    result = captioner.describe(
        np.zeros((4, 4, 3), dtype=np.uint8),
        object_bboxes=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
        object_masks=None,
        head_bboxes=np.empty((0, 4), dtype=np.float32),
    )

    assert result["object_descriptions"] == [""]
    assert result["head_descriptions"] == []
