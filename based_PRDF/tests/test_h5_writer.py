import json

import h5py
import numpy as np

from utils.h5_writer import GazeH5Writer


def test_writer_persists_schema_and_empty_detection_datasets(tmp_path):
    output_path = tmp_path / "gaze.h5"
    metadata = {
        "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2_checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "sam2_parameters": {"points_per_side": 32},
        "bbox_filter_parameters": {"min_area": 16},
        "head_detector": "grounding_dino",
        "head_detector_parameters": {"score_threshold": 0.3},
    }

    with GazeH5Writer(output_path, metadata=metadata, overwrite=True) as writer:
        writer.write_image(
            image_id="image_a",
            image_path="train/a.jpg",
            image_size=(12, 10),
            object_bboxes=np.empty((0, 4), dtype=np.float32),
            object_scores=np.empty((0,), dtype=np.float32),
            object_mask_areas=np.empty((0,), dtype=np.int64),
            head_bboxes=np.empty((0, 4), dtype=np.float32),
            head_scores=np.empty((0,), dtype=np.float32),
        )

    with h5py.File(output_path, "r") as h5_file:
        attrs = h5_file["metadata"].attrs
        assert attrs["schema_version"] == "2.1"
        assert attrs["bbox_format"] == "xyxy"
        assert attrs["bbox_coordinate_mode"] == "half_open"
        assert attrs["num_images"] == 1
        assert json.loads(attrs["sam2_parameters_json"]) == {"points_per_side": 32}
        image_group = h5_file["images"]["image_a"]
        assert image_group.attrs["image_path"] == "train/a.jpg"
        assert image_group["object_bboxes"].shape == (0, 4)
        assert image_group["head_bboxes"].shape == (0, 4)
        assert image_group["object_scores"].shape == (0,)
        assert image_group["head_scores"].shape == (0,)


def test_writer_optionally_persists_masks(tmp_path):
    output_path = tmp_path / "gaze.h5"
    masks = np.zeros((1, 5, 6), dtype=np.uint8)
    masks[0, 1:4, 2:5] = 1

    with GazeH5Writer(
        output_path,
        metadata={},
        overwrite=True,
        save_masks=True,
    ) as writer:
        writer.write_image(
            image_id="image_b",
            image_path="image_b.jpg",
            image_size=(6, 5),
            object_bboxes=np.asarray([[2, 1, 5, 4]], dtype=np.float32),
            object_scores=np.asarray([0.9], dtype=np.float32),
            object_mask_areas=np.asarray([9], dtype=np.int64),
            head_bboxes=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
            head_scores=np.asarray([0.8], dtype=np.float32),
            object_masks=masks,
        )

    with h5py.File(output_path, "r") as h5_file:
        group = h5_file["images"]["image_b"]
        assert group["object_masks"].shape == (1, 5, 6)
        np.testing.assert_array_equal(group["object_masks"][:], masks)


def test_writer_persists_aligned_descriptions_and_depth(tmp_path):
    output_path = tmp_path / "enriched.h5"
    object_depth = {
        name: np.asarray([value], dtype=np.float32)
        for name, value in {
            "mean": 0.4,
            "median": 0.5,
            "min": 0.1,
            "max": 0.8,
            "std": 0.2,
        }.items()
    }
    head_depth = {
        name: np.asarray([value], dtype=np.float32)
        for name, value in {
            "mean": 0.3,
            "median": 0.3,
            "min": 0.2,
            "max": 0.4,
            "std": 0.1,
        }.items()
    }
    depth_map = np.linspace(0, 1, 30, dtype=np.float32).reshape(5, 6)

    with GazeH5Writer(
        output_path, metadata={}, overwrite=True, save_depth_map=True
    ) as writer:
        writer.write_image(
            image_id="image_c",
            image_path="image_c.jpg",
            image_size=(6, 5),
            object_bboxes=np.asarray([[1, 1, 4, 4]], dtype=np.float32),
            object_scores=np.asarray([0.9], dtype=np.float32),
            object_mask_areas=np.asarray([9], dtype=np.int64),
            head_bboxes=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
            head_scores=np.asarray([0.8], dtype=np.float32),
            object_descriptions=["红色杯子"],
            head_descriptions=["visible head"],
            object_depth=object_depth,
            head_depth=head_depth,
            depth_map=depth_map,
        )

    with h5py.File(output_path, "r") as h5_file:
        assert h5_file["metadata"].attrs["schema_version"] == "2.1"
        group = h5_file["images"]["image_c"]
        assert group["object_descriptions"].asstr()[0] == "红色杯子"
        assert group["head_descriptions"].asstr()[0] == "visible head"
        np.testing.assert_allclose(group["object_depth"]["mean"][:], [0.4])
        np.testing.assert_allclose(group["head_depth"]["std"][:], [0.1])
        np.testing.assert_allclose(group["normalized_depth_map"][:], depth_map)
