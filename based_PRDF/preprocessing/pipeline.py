from typing import Iterable

import numpy as np
from PIL import Image

from .bbox import MaskCandidate, clip_bbox, mask_to_bbox


class GazePreprocessingPipeline:
    def __init__(
        self,
        *,
        segmenter,
        bbox_config,
        writer,
        save_masks=False,
        depth_estimator=None,
        captioner=None,
    ):
        self.segmenter = segmenter
        self.bbox_config = bbox_config
        self.writer = writer
        self.save_masks = bool(save_masks)
        self.depth_estimator = depth_estimator
        self.captioner = captioner

    def process_image(self, record):
        try:
            with Image.open(record.absolute_path) as image_file:
                image = np.asarray(image_file.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {record.absolute_path}: {exc}") from exc

        height, width = image.shape[:2]
        candidates = []
        for item in self.segmenter.segment(image):
            mask = np.asarray(item["segmentation"], dtype=np.uint8)
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            candidates.append(
                MaskCandidate(
                    bbox=bbox,
                    mask_area=int(mask.sum()),
                    predicted_iou=float(item.get("predicted_iou", 0.0)),
                    stability_score=float(item.get("stability_score", 0.0)),
                    mask=mask,
                )
            )
        # SAM2 bbox post-processing is intentionally disabled for inspection.
        objects = candidates
        head_bboxes, head_scores = self._record_head_annotations(
            record, image_size=(width, height)
        )
        object_bboxes = np.asarray(
            [item.bbox for item in objects], dtype=np.float32
        ).reshape(-1, 4)
        object_scores = np.asarray(
            [item.score for item in objects], dtype=np.float32
        )
        object_areas = np.asarray(
            [item.mask_area for item in objects], dtype=np.int64
        )
        live_object_masks = (
            np.stack([item.mask for item in objects])
            if objects
            else np.zeros((0, height, width), dtype=np.uint8)
        )
        depth_result = {
            "depth_map": None,
            "object_depth": None,
            "head_depth": None,
        }
        if self.depth_estimator is not None:
            depth_result = self.depth_estimator.extract(
                image,
                object_bboxes=object_bboxes,
                object_masks=live_object_masks,
                head_bboxes=head_bboxes,
            )
        description_result = {
            "object_descriptions": [""] * len(object_bboxes),
            "head_descriptions": [""] * len(head_bboxes),
        }
        if self.captioner is not None:
            description_result = self.captioner.describe(
                image,
                object_bboxes=object_bboxes,
                object_masks=live_object_masks,
                head_bboxes=head_bboxes,
            )
        self.writer.write_image(
            image_id=record.image_id,
            image_path=record.relative_path,
            image_size=(width, height),
            object_bboxes=object_bboxes,
            object_scores=object_scores,
            object_mask_areas=object_areas,
            head_bboxes=head_bboxes,
            head_scores=head_scores,
            object_masks=live_object_masks if self.save_masks else None,
            object_descriptions=description_result["object_descriptions"],
            head_descriptions=description_result["head_descriptions"],
            object_depth=depth_result["object_depth"],
            head_depth=depth_result["head_depth"],
            depth_map=depth_result["depth_map"],
        )

    def run(self, records: Iterable):
        processed = 0
        for record in records:
            try:
                self.process_image(record)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed preprocessing {record.relative_path}: {exc}"
                ) from exc
            processed += 1
        return processed

    @staticmethod
    def _record_head_annotations(record, *, image_size):
        width, height = image_size
        raw_boxes = getattr(record, "head_bboxes", ()) or ()
        raw_scores = getattr(record, "head_scores", ()) or ()
        boxes = []
        scores = []
        for index, box in enumerate(raw_boxes):
            clipped = clip_bbox(box, width, height)
            if clipped is None:
                continue
            boxes.append(clipped)
            score = float(raw_scores[index]) if index < len(raw_scores) else 1.0
            scores.append(score)
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32),
        )
