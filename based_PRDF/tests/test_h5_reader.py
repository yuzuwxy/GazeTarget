import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from h5_reader import (
    bbox_label,
    collect_image_samples,
    draw_bboxes,
    draw_masks,
    load_image_from_sample,
    main,
    read_sample_from_h5,
    resolve_image_root,
    select_sample_indices,
    visualize_sample,
)


def _write_sample(path, *, include_masks=True):
    with h5py.File(path, "w") as h5_file:
        h5_file.create_group("metadata").attrs["bbox_format"] = "xyxy"
        group = h5_file.create_group("images").create_group("image/a")
        group.attrs["image_path"] = np.bytes_("train/a.jpg")
        group.attrs["width"] = 8
        group.attrs["height"] = 6
        group.create_dataset(
            "object_bboxes",
            data=np.asarray([[-2, 1, 5, 5]], dtype=np.float32),
        )
        group.create_dataset(
            "object_scores", data=np.asarray([0.91], dtype=np.float32)
        )
        group.create_dataset(
            "object_mask_areas", data=np.asarray([12], dtype=np.int64)
        )
        group.create_dataset(
            "head_bboxes",
            data=np.asarray([[5, 0, 10, 3]], dtype=np.float32),
        )
        group.create_dataset(
            "head_scores", data=np.asarray([0.82], dtype=np.float32)
        )
        if include_masks:
            mask = np.zeros((1, 6, 8), dtype=np.uint8)
            mask[0, 1:5, 1:4] = 1
            group.create_dataset("object_masks", data=mask)


def test_read_sample_decodes_actual_schema(tmp_path):
    h5_path = tmp_path / "sample.h5"
    _write_sample(h5_path)

    with h5py.File(h5_path, "r") as h5_file:
        sample = read_sample_from_h5(
            h5_file["images"]["image/a"], image_id="image/a", index=3
        )

    assert sample["image_id"] == "image/a"
    assert sample["index"] == 3
    assert sample["image_path"] == "train/a.jpg"
    assert sample["image_size"] == (8, 6)
    assert sample["object_bboxes"].shape == (1, 4)
    assert sample["head_bboxes"].shape == (1, 4)
    assert sample["object_masks"].shape == (1, 6, 8)


def test_read_sample_warns_and_defaults_missing_datasets(tmp_path):
    h5_path = tmp_path / "missing.h5"
    with h5py.File(h5_path, "w") as h5_file:
        group = h5_file.create_group("images").create_group("empty")
        group.attrs["image_path"] = "empty.jpg"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample = read_sample_from_h5(group, image_id="empty", index=0)

    assert sample["object_bboxes"].shape == (0, 4)
    assert sample["head_bboxes"].shape == (0, 4)
    assert sample["object_scores"].shape == (0,)
    assert sample["object_masks"] is None
    assert caught


def test_resolve_image_root_uses_input_type(tmp_path):
    gaze_root = resolve_image_root(
        {"input": {"type": "gazefollow", "path": str(tmp_path / "gaze")}}
    )
    directory_root = resolve_image_root(
        {"input": {"type": "directory", "path": str(tmp_path / "images")}}
    )

    assert gaze_root == tmp_path / "gaze" / "data_extended"
    assert directory_root == tmp_path / "images"


def test_load_image_supports_path_and_embedded_array(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "train" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)

    path_image = load_image_from_sample(
        {"image_path": "train/a.jpg"}, image_root=image_root, h5_path=tmp_path / "x.h5"
    )
    embedded = load_image_from_sample(
        {"image": np.full((4, 5, 3), 100, dtype=np.uint8)},
        image_root=None,
        h5_path=tmp_path / "x.h5",
    )

    assert path_image.size == (8, 6)
    assert embedded.size == (5, 4)


def test_draw_functions_clip_boxes_and_overlay_masks():
    image = Image.new("RGB", (8, 6), (255, 255, 255))
    boxed = draw_bboxes(
        image,
        np.asarray([[-2, 1, 10, 5]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        label="object",
        color=(0, 255, 0),
    )
    mask = np.zeros((1, 6, 8), dtype=np.uint8)
    mask[0, 1:4, 2:5] = 1
    overlaid = draw_masks(boxed, mask, alpha=0.5)

    assert boxed.size == (8, 6)
    assert np.asarray(overlaid)[2, 3].tolist() != [255, 255, 255]


def test_bbox_label_uses_only_region_number():
    assert bbox_label("object", 0) == "object_0"
    assert bbox_label("person", 12) == "person_12"


def test_select_sample_indices_applies_explicit_selection_and_limit():
    assert select_sample_indices(5, indices=[3, 1, 9], limit=2) == [3, 1]
    assert select_sample_indices(5, index=2, limit=None) == [2]
    assert select_sample_indices(5, limit=3) == [0, 1, 2]


def test_visualize_sample_overwrites_existing_output(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "train" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (255, 255, 255)).save(image_path)
    sample = {
        "image_id": "image/a",
        "index": 1,
        "image_path": "train/a.jpg",
        "object_bboxes": np.asarray([[0, 0, 4, 4]], dtype=np.float32),
        "object_scores": np.asarray([0.9], dtype=np.float32),
        "head_bboxes": np.empty((0, 4), dtype=np.float32),
        "head_scores": np.empty((0,), dtype=np.float32),
        "object_masks": None,
    }

    first = visualize_sample(
        sample,
        image_root=image_root,
        h5_path=tmp_path / "sample.h5",
        output_dir=tmp_path / "out",
        overwrite=False,
        draw_mask=True,
        draw_bbox=True,
        draw_head=True,
        mask_alpha=0.35,
    )
    first.write_bytes(b"stale")
    second = visualize_sample(
        sample,
        image_root=image_root,
        h5_path=tmp_path / "sample.h5",
        output_dir=tmp_path / "out",
        overwrite=False,
        draw_mask=True,
        draw_bbox=True,
        draw_head=True,
        mask_alpha=0.35,
    )

    assert first == tmp_path / "out" / "000001_image_a_vis.jpg"
    assert first.is_file()
    assert second == first
    assert first.read_bytes() != b"stale"


def test_visualize_sample_writes_region_json_without_descriptions_on_image(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "train" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 10), (255, 255, 255)).save(image_path)
    sample = {
        "image_id": "image/a",
        "index": 2,
        "image_path": "train/a.jpg",
        "object_bboxes": np.asarray([[1, 1, 5, 6]], dtype=np.float32),
        "object_scores": np.asarray([0.9], dtype=np.float32),
        "object_mask_areas": np.asarray([20], dtype=np.int64),
        "object_descriptions": ["red cup on the table"],
        "object_depth": {
            "mean": np.asarray([0.42], dtype=np.float32),
            "median": np.asarray([0.41], dtype=np.float32),
            "min": np.asarray([0.2], dtype=np.float32),
            "max": np.asarray([0.8], dtype=np.float32),
            "std": np.asarray([0.1], dtype=np.float32),
        },
        "head_bboxes": np.asarray([[6, 1, 10, 5]], dtype=np.float32),
        "head_scores": np.asarray([1.0], dtype=np.float32),
        "head_descriptions": ["visible person head"],
        "head_depth": {
            "mean": np.asarray([0.33], dtype=np.float32),
            "median": np.asarray([0.32], dtype=np.float32),
            "min": np.asarray([0.1], dtype=np.float32),
            "max": np.asarray([0.6], dtype=np.float32),
            "std": np.asarray([0.05], dtype=np.float32),
        },
        "object_masks": None,
    }

    image_output = visualize_sample(
        sample,
        image_root=image_root,
        h5_path=tmp_path / "sample.h5",
        output_dir=tmp_path / "out",
        draw_mask=False,
        draw_bbox=True,
        draw_head=True,
        mask_alpha=0.35,
        show_description=True,
        show_depth=True,
    )

    json_output = image_output.with_suffix(".json")
    assert json_output.is_file()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["image_id"] == "image/a"
    assert payload["object"] == [
        {
            "id": "object_0",
            "bbox": [1.0, 1.0, 5.0, 6.0],
            "score": 0.9,
            "depth": {
                "mean": 0.42,
                "median": 0.41,
                "min": 0.2,
                "max": 0.8,
                "std": 0.1,
            },
            "description": "red cup on the table",
        }
    ]
    assert payload["person"] == [
        {
            "id": "person_0",
            "bbox": [6.0, 1.0, 10.0, 5.0],
            "score": 1.0,
            "depth": {
                "mean": 0.33,
                "median": 0.32,
                "min": 0.1,
                "max": 0.6,
                "std": 0.05,
            },
            "description": "visible person head",
        }
    ]
    image_array = np.asarray(Image.open(image_output).convert("RGB"))
    assert (image_array[:, :, 0] > 200).any()


def test_main_uses_config_paths_and_indices(tmp_path):
    data_root = tmp_path / "dataset"
    image_path = data_root / "data_extended" / "train" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (255, 255, 255)).save(image_path)
    h5_path = tmp_path / "outputs" / "sample.h5"
    h5_path.parent.mkdir()
    _write_sample(h5_path, include_masks=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  type: gazefollow
  path: dataset
output:
  h5_path: outputs/sample.h5
visualization:
  output_dir: output/images
  draw_mask: true
  draw_bbox: true
  draw_head: true
  mask_alpha: 0.35
""",
        encoding="utf-8",
    )

    saved = main(
        [
            "--config",
            str(config_path),
            "--indices",
            "0",
        ]
    )

    assert saved == 1
    assert (tmp_path / "output/images/000000_image_a_vis.jpg").is_file()


def test_collect_image_samples_supports_nested_image_ids(tmp_path):
    h5_path = tmp_path / "nested.h5"
    _write_sample(h5_path, include_masks=False)

    with h5py.File(h5_path, "r") as h5_file:
        samples = collect_image_samples(h5_file["images"])

    assert [image_id for image_id, _ in samples] == ["image/a"]


def test_reader_loads_enrichment_fields_and_draws_text(tmp_path):
    h5_path = tmp_path / "enriched.h5"
    with h5py.File(h5_path, "w") as h5_file:
        group = h5_file.create_group("images").create_group("sample")
        group.attrs["image_path"] = "a.jpg"
        group.attrs["width"] = 8
        group.attrs["height"] = 6
        group.create_dataset(
            "object_bboxes", data=np.asarray([[0, 0, 4, 4]], dtype=np.float32)
        )
        group.create_dataset("object_scores", data=np.asarray([0.9], dtype=np.float32))
        group.create_dataset("object_mask_areas", data=np.asarray([16]))
        group.create_dataset("head_bboxes", data=np.empty((0, 4), dtype=np.float32))
        group.create_dataset("head_scores", data=np.empty((0,), dtype=np.float32))
        string_dtype = h5py.string_dtype("utf-8")
        group.create_dataset(
            "object_descriptions",
            data=np.asarray(["a long red cup on the table"], dtype=object),
            dtype=string_dtype,
        )
        group.create_dataset(
            "head_descriptions",
            data=np.asarray([], dtype=object),
            dtype=string_dtype,
        )
        depth_group = group.create_group("object_depth")
        head_depth_group = group.create_group("head_depth")
        for name in ("mean", "median", "min", "max", "std"):
            depth_group.create_dataset(name, data=np.asarray([0.42], dtype=np.float32))
            head_depth_group.create_dataset(name, data=np.empty((0,), dtype=np.float32))

    with h5py.File(h5_path, "r") as h5_file:
        sample = read_sample_from_h5(
            h5_file["images"]["sample"], image_id="sample", index=0
        )

    assert sample["object_descriptions"] == ["a long red cup on the table"]
    np.testing.assert_allclose(sample["object_depth"]["mean"], [0.42])
    image = draw_bboxes(
        Image.new("RGB", (8, 6), "white"),
        sample["object_bboxes"],
        scores=sample["object_scores"],
        descriptions=sample["object_descriptions"],
        mean_depth=sample["object_depth"]["mean"],
        show_description=True,
        show_depth=True,
        max_description_chars=12,
        label="object",
        color=(0, 255, 0),
    )
    assert image.size == (8, 6)
