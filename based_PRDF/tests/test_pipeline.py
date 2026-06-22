import h5py
import numpy as np
from PIL import Image

import main
from preprocessing.bbox import BBoxFilterConfig
from preprocessing.image_source import ImageRecord
from preprocessing.pipeline import GazePreprocessingPipeline
from utils.h5_writer import GazeH5Writer


class FakeSegmenter:
    def segment(self, image):
        mask_a = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_a[1:6, 2:8] = 1
        mask_duplicate = mask_a.copy()
        mask_b = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_b[2:5, 9:12] = 1
        return [
            {
                "segmentation": mask_a,
                "area": 30,
                "predicted_iou": 0.95,
                "stability_score": 0.95,
            },
            {
                "segmentation": mask_duplicate,
                "area": 30,
                "predicted_iou": 0.80,
                "stability_score": 0.80,
            },
            {
                "segmentation": mask_b,
                "area": 9,
                "predicted_iou": 0.90,
                "stability_score": 0.90,
            },
        ]


class FakeDepthEstimator:
    save_depth_map = True

    def __init__(self):
        self.calls = []

    def extract(self, image, *, object_bboxes, object_masks, head_bboxes):
        self.calls.append((image.copy(), object_bboxes.copy(), object_masks.copy(), head_bboxes.copy()))
        object_count = len(object_bboxes)
        head_count = len(head_bboxes)
        return {
            "depth_map": np.full(image.shape[:2], 0.5, dtype=np.float32),
            "object_depth": {
                name: np.full(object_count, 0.4, dtype=np.float32)
                for name in ("mean", "median", "min", "max", "std")
            },
            "head_depth": {
                name: np.full(head_count, 0.3, dtype=np.float32)
                for name in ("mean", "median", "min", "max", "std")
            },
        }


class FakeCaptioner:
    def __init__(self):
        self.calls = []

    def describe(self, image, *, object_bboxes, object_masks, head_bboxes):
        self.calls.append((image.copy(), object_bboxes.copy(), object_masks.copy(), head_bboxes.copy()))
        return {
            "object_descriptions": [f"object {i}" for i in range(len(object_bboxes))],
            "head_descriptions": [f"head {i}" for i in range(len(head_bboxes))],
        }


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeStack:
    def enter_context(self, context):
        return context


def test_build_pipeline_does_not_require_head_detector(monkeypatch):
    monkeypatch.setattr(main, "SAM2Segmenter", lambda **kwargs: FakeContext())
    monkeypatch.setattr(main, "_require_runtime_device", lambda device: None)
    config = {
        "runtime": {"device": "cpu"},
        "sam2": {
            "source_path": "sam2",
            "checkpoint": "sam.pt",
            "config": "sam.yaml",
        },
        "bbox_filter": {},
        "output": {"save_masks": False},
    }

    pipeline = main.build_pipeline(config, writer=object(), stack=FakeStack())

    assert not hasattr(pipeline, "head_detector")


def test_pipeline_runs_end_to_end_with_replaceable_models(tmp_path):
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (14, 8), color=(100, 120, 140)).save(image_path)
    output_path = tmp_path / "output.h5"
    record = ImageRecord(
        "sample",
        "input.jpg",
        image_path,
        head_bboxes=((0.0, 0.0, 4.0, 4.0),),
        head_scores=(1.0,),
    )

    with GazeH5Writer(output_path, metadata={}, overwrite=True) as writer:
        pipeline = GazePreprocessingPipeline(
            segmenter=FakeSegmenter(),
            bbox_config=BBoxFilterConfig(
                min_width=2,
                min_height=2,
                min_area=4,
                min_area_ratio=0,
                max_area_ratio=1,
                duplicate_iou_threshold=0.8,
            ),
            writer=writer,
            save_masks=False,
        )
        assert pipeline.run([record]) == 1

    with h5py.File(output_path, "r") as h5_file:
        group = h5_file["images"]["sample"]
        np.testing.assert_array_equal(
            group["object_bboxes"][:],
            np.asarray(
                [[2, 1, 8, 6], [2, 1, 8, 6], [9, 2, 12, 5]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            group["head_bboxes"][:],
            np.asarray([[0, 0, 4, 4]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            group["head_scores"][:], np.asarray([1.0], dtype=np.float32)
        )
        assert "object_masks" not in group


def test_pipeline_enriches_live_regions_before_h5_write(tmp_path):
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (14, 8), color=(100, 120, 140)).save(image_path)
    output_path = tmp_path / "output.h5"
    record = ImageRecord(
        "sample",
        "input.jpg",
        image_path,
        head_bboxes=((0.0, 0.0, 4.0, 4.0),),
        head_scores=(1.0,),
    )
    depth_estimator = FakeDepthEstimator()
    captioner = FakeCaptioner()

    with GazeH5Writer(
        output_path,
        metadata={},
        overwrite=True,
        save_masks=True,
        save_depth_map=True,
    ) as writer:
        pipeline = GazePreprocessingPipeline(
            segmenter=FakeSegmenter(),
            depth_estimator=depth_estimator,
            captioner=captioner,
            bbox_config=BBoxFilterConfig(
                min_width=2,
                min_height=2,
                min_area=4,
                min_area_ratio=0,
                max_area_ratio=1,
                duplicate_iou_threshold=0.8,
            ),
            writer=writer,
            save_masks=True,
        )
        pipeline.process_image(record)

    assert depth_estimator.calls[0][2].shape == (3, 8, 14)
    assert captioner.calls[0][2].shape == (3, 8, 14)
    with h5py.File(output_path, "r") as h5_file:
        group = h5_file["images"]["sample"]
        assert group["object_descriptions"].asstr()[:].tolist() == [
            "object 0",
            "object 1",
            "object 2",
        ]
        assert group["head_descriptions"].asstr()[:].tolist() == ["head 0"]
        np.testing.assert_allclose(
            group["object_depth"]["mean"][:], [0.4, 0.4, 0.4]
        )
        assert group["normalized_depth_map"].shape == (8, 14)
