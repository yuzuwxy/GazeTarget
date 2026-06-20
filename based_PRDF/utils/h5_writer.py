import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = "2.1"
DEPTH_STAT_NAMES = ("mean", "median", "min", "max", "std")


def _json_value(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _boxes(value):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    return array.reshape(-1, 4)


def _vector(value, dtype):
    return np.asarray(value, dtype=dtype).reshape(-1)


def _validate_boxes(boxes, image_size, name):
    width, height = image_size
    if boxes.size == 0:
        return
    valid = (
        (boxes[:, 0] >= 0)
        & (boxes[:, 1] >= 0)
        & (boxes[:, 2] <= width)
        & (boxes[:, 3] <= height)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    if not bool(valid.all()):
        raise ValueError(f"{name} contains invalid or out-of-bounds xyxy boxes")


class GazeH5Writer:
    """Atomically write gaze preprocessing results image by image."""

    def __init__(
        self,
        output_path,
        *,
        metadata,
        overwrite=False,
        save_masks=False,
        compression="gzip",
        compression_opts=4,
        save_depth_map=False,
    ):
        self.output_path = Path(output_path)
        self.metadata = dict(metadata or {})
        self.overwrite = bool(overwrite)
        self.save_masks = bool(save_masks)
        self.compression = compression
        self.compression_opts = compression_opts
        self.save_depth_map = bool(save_depth_map)
        self._file = None
        self._temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.tmp-{os.getpid()}"
        )

    def __enter__(self):
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}. Enable output.overwrite to replace it."
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._temporary_path.exists():
            self._temporary_path.unlink()
        self._file = h5py.File(self._temporary_path, "w")
        metadata_group = self._file.create_group("metadata")
        metadata_group.attrs["schema_version"] = SCHEMA_VERSION
        metadata_group.attrs["bbox_format"] = "xyxy"
        metadata_group.attrs["bbox_coordinate_mode"] = "half_open"
        metadata_group.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        metadata_group.attrs["num_images"] = 0
        for key, value in self.metadata.items():
            attribute = f"{key}_json" if isinstance(value, (dict, list, tuple)) else key
            metadata_group.attrs[attribute] = (
                _json_value(value) if isinstance(value, (dict, list, tuple)) else str(value)
            )
        self._file.create_group("images")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._file is not None:
                self._file["metadata"].attrs["num_images"] = len(self._file["images"])
                self._file.flush()
                self._file.close()
                self._file = None
            if exc_type is None:
                os.replace(self._temporary_path, self.output_path)
            elif self._temporary_path.exists():
                self._temporary_path.unlink()
        finally:
            self._file = None
        return False

    def write_image(
        self,
        *,
        image_id,
        image_path,
        image_size,
        object_bboxes,
        object_scores,
        object_mask_areas,
        head_bboxes,
        head_scores,
        object_masks=None,
        object_descriptions=None,
        head_descriptions=None,
        object_depth=None,
        head_depth=None,
        depth_map=None,
    ):
        if self._file is None:
            raise RuntimeError("GazeH5Writer must be used as a context manager")
        width, height = (int(image_size[0]), int(image_size[1]))
        objects = _boxes(object_bboxes)
        object_scores = _vector(object_scores, np.float32)
        object_mask_areas = _vector(object_mask_areas, np.int64)
        heads = _boxes(head_bboxes)
        head_scores = _vector(head_scores, np.float32)
        if len(objects) != len(object_scores) or len(objects) != len(object_mask_areas):
            raise ValueError("Object bbox, score, and mask-area counts must match")
        if len(heads) != len(head_scores):
            raise ValueError("Head bbox and score counts must match")
        object_descriptions = list(
            object_descriptions if object_descriptions is not None else [""] * len(objects)
        )
        head_descriptions = list(
            head_descriptions if head_descriptions is not None else [""] * len(heads)
        )
        if len(object_descriptions) != len(objects):
            raise ValueError("Object description and bbox counts must match")
        if len(head_descriptions) != len(heads):
            raise ValueError("Head description and bbox counts must match")

        def normalized_depth(values, count, label):
            values = values or {}
            result = {}
            for name in DEPTH_STAT_NAMES:
                vector = _vector(
                    values.get(name, np.full((count,), np.nan, dtype=np.float32)),
                    np.float32,
                )
                if len(vector) != count:
                    raise ValueError(
                        f"{label} depth '{name}' and bbox counts must match"
                    )
                result[name] = vector
            return result

        object_depth = normalized_depth(object_depth, len(objects), "Object")
        head_depth = normalized_depth(head_depth, len(heads), "Head")
        _validate_boxes(objects, (width, height), "object_bboxes")
        _validate_boxes(heads, (width, height), "head_bboxes")

        images = self._file["images"]
        image_id = str(image_id)
        if image_id in images:
            raise ValueError(f"Duplicate image_id: {image_id}")
        group = images.create_group(image_id)
        group.attrs["image_path"] = str(image_path)
        group.attrs["width"] = width
        group.attrs["height"] = height
        group.attrs["num_object_bboxes"] = len(objects)
        group.attrs["num_head_bboxes"] = len(heads)
        group.create_dataset("object_bboxes", data=objects)
        group.create_dataset("object_scores", data=object_scores)
        group.create_dataset("object_mask_areas", data=object_mask_areas)
        group.create_dataset("head_bboxes", data=heads)
        group.create_dataset("head_scores", data=head_scores)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            "object_descriptions",
            data=np.asarray([str(value or "") for value in object_descriptions], dtype=object),
            dtype=string_dtype,
        )
        group.create_dataset(
            "head_descriptions",
            data=np.asarray([str(value or "") for value in head_descriptions], dtype=object),
            dtype=string_dtype,
        )
        object_depth_group = group.create_group("object_depth")
        head_depth_group = group.create_group("head_depth")
        for name in DEPTH_STAT_NAMES:
            object_depth_group.create_dataset(name, data=object_depth[name])
            head_depth_group.create_dataset(name, data=head_depth[name])

        if self.save_masks:
            if object_masks is None:
                masks = np.zeros((len(objects), height, width), dtype=np.uint8)
            else:
                masks = np.asarray(object_masks, dtype=np.uint8)
            if masks.shape != (len(objects), height, width):
                raise ValueError(
                    "object_masks must have shape "
                    f"({len(objects)}, {height}, {width}), got {masks.shape}"
                )
            kwargs = {}
            if len(masks) and self.compression:
                kwargs["compression"] = self.compression
                kwargs["compression_opts"] = self.compression_opts
            group.create_dataset("object_masks", data=masks, **kwargs)

        if self.save_depth_map and depth_map is not None:
            depth = np.asarray(depth_map, dtype=np.float32)
            if depth.shape != (height, width):
                raise ValueError(
                    f"depth_map must have shape ({height}, {width}), got {depth.shape}"
                )
            kwargs = {}
            if self.compression:
                kwargs["compression"] = self.compression
                kwargs["compression_opts"] = self.compression_opts
            group.create_dataset("normalized_depth_map", data=depth, **kwargs)


# Kept as an explicit migration error for callers of the old schema.
class DetectionH5Writer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DetectionH5Writer schema 1.0 was replaced by GazeH5Writer schema 2.1"
        )
