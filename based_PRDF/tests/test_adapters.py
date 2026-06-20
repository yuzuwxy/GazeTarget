import numpy as np
import pytest

from preprocessing.head_detector import (
    GroundingDINOHeadDetector,
    HuggingFaceGroundingDINOHeadDetector,
    MMDetectionGroundingDINOHeadDetector,
    build_head_detector,
    extract_head_detections,
    normalize_hf_head_detections,
    resolve_hf_model_source,
)
from preprocessing.sam2_segmenter import normalize_sam2_masks


def test_extract_head_detections_returns_typed_empty_arrays():
    boxes, scores = extract_head_detections(
        detection=None,
        image_size=(20, 10),
        score_threshold=0.3,
        nms_threshold=0.5,
    )

    assert boxes.shape == (0, 4)
    assert boxes.dtype == np.float32
    assert scores.shape == (0,)


def test_normalize_hf_head_detections_filters_and_orders_results():
    result = {
        "boxes": np.asarray(
            [
                [-2, 1, 12, 9],
                [0, 0, 9, 8],
                [1, 1, 2, 2],
                [2, 2, 7, 7],
                [3, 3, 8, 8],
            ],
            dtype=np.float32,
        ),
        "scores": np.asarray([0.90, 0.80, 0.99, 0.95, 0.70], dtype=np.float32),
        "text_labels": [
            "person head",
            "person",
            "head",
            "human head",
            "head",
        ],
    }

    detections = normalize_hf_head_detections(
        result,
        image_size=(10, 8),
        allowed_labels=["person head", "human head", "head"],
        score_threshold=0.5,
        nms_threshold=0.4,
        max_detections=2,
        min_width=2,
        min_height=2,
        min_area=4,
    )

    assert [item["label"] for item in detections] == ["human head", "person head"]
    assert detections[0]["bbox"] == [2.0, 2.0, 7.0, 7.0]
    assert detections[0]["score"] == pytest.approx(0.95)
    assert detections[1]["bbox"] == [0.0, 1.0, 10.0, 8.0]


def test_normalize_hf_head_detections_returns_empty_list():
    detections = normalize_hf_head_detections(
        {"boxes": [], "scores": [], "text_labels": []},
        image_size=(20, 10),
        allowed_labels=["head"],
        score_threshold=0.3,
        nms_threshold=0.5,
        max_detections=10,
        min_width=2,
        min_height=2,
        min_area=4,
    )

    assert detections == []


class FakeBatch(dict):
    def to(self, device):
        self.device = str(device)
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, text, return_tensors):
        self.calls.append((images.size, text, return_tensors))
        return FakeBatch(input_ids=np.asarray([[1, 2, 3]]))

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids=None,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=None,
        text_labels=None,
    ):
        self.calls.append(
            (threshold, text_threshold, target_sizes, text_labels, input_ids)
        )
        return [
            {
                "boxes": np.asarray([[1, 2, 9, 8]], dtype=np.float32),
                "scores": np.asarray([0.88], dtype=np.float32),
                "text_labels": ["person head"],
            }
        ]


class FakeModel:
    def __init__(self):
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        return {"inputs": inputs}


def test_hf_detector_preserves_pipeline_interface():
    processor = FakeProcessor()
    model = FakeModel()
    detector = HuggingFaceGroundingDINOHeadDetector(
        model_id="test/model",
        device="cpu",
        prompts=["person head", "human head"],
        allowed_labels=["person head", "human head"],
        score_threshold=0.3,
        box_threshold=0.4,
        text_threshold=0.2,
        nms_threshold=0.5,
        max_detections=5,
        min_width=2,
        min_height=2,
        min_area=4,
        processor=processor,
        model=model,
    ).load()

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    metadata = detector.detect_with_metadata(image)
    boxes, scores = detector.detect(image)

    assert metadata == [
        {"bbox": [1.0, 2.0, 9.0, 8.0], "score": pytest.approx(0.88), "label": "person head"}
    ]
    np.testing.assert_array_equal(
        boxes, np.asarray([[1, 2, 9, 8]], dtype=np.float32)
    )
    np.testing.assert_allclose(scores, np.asarray([0.88], dtype=np.float32))
    assert processor.calls[0][1] == ["person head", "human head"]
    assert processor.calls[1][0:3] == (0.4, 0.2, [(10, 12)])
    assert model.device == "cpu"
    assert model.eval_called


def test_build_head_detector_selects_backend():
    hf = build_head_detector(
        {
            "backend": "hf_grounding_dino",
            "model_id": "test/model",
            "prompts": ["head"],
        },
        device="cpu",
    )
    legacy = build_head_detector(
        {
            "backend": "mmdetection",
            "source_path": "vendor/mmdet",
            "checkpoint": "head.pth",
            "config": "head.py",
        },
        device="cpu",
    )

    assert isinstance(hf, HuggingFaceGroundingDINOHeadDetector)
    assert isinstance(legacy, MMDetectionGroundingDINOHeadDetector)
    assert GroundingDINOHeadDetector is MMDetectionGroundingDINOHeadDetector


def test_build_head_detector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported head detector backend"):
        build_head_detector({"backend": "unknown"}, device="cpu")


def test_resolve_hf_model_source_uses_local_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    captured = {}

    def fake_snapshot_download(repo_id, *, cache_dir, local_files_only):
        captured.update(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )

    source = resolve_hf_model_source(
        "organization/model",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    assert source == str(snapshot)
    assert captured == {
        "repo_id": "organization/model",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }


def test_normalize_sam2_masks_recomputes_mask_area():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 1

    normalized = normalize_sam2_masks(
        [
            {
                "segmentation": mask.astype(bool),
                "area": 999,
                "predicted_iou": 0.8,
                "stability_score": 0.9,
            }
        ],
        image_shape=(5, 6),
    )

    assert normalized[0]["area"] == 9
    assert normalized[0]["segmentation"].dtype == np.uint8
