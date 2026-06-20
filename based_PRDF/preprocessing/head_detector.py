import inspect
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .bbox import bbox_area, bbox_iou, clip_bbox, nms_bboxes


def extract_head_detections(
    detection,
    image_size,
    score_threshold,
    nms_threshold,
):
    instances = getattr(detection, "pred_instances", None)
    if instances is None:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    def as_numpy(value, dtype):
        if value is None:
            return np.empty((0,), dtype=dtype)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    raw_boxes = as_numpy(getattr(instances, "bboxes", None), np.float32)
    raw_scores = as_numpy(getattr(instances, "scores", None), np.float32).reshape(-1)
    if raw_boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    raw_boxes = raw_boxes.reshape(-1, 4)
    boxes = []
    scores = []
    for index, box in enumerate(raw_boxes):
        score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        if score < score_threshold:
            continue
        clipped = clip_bbox(box, *image_size)
        if clipped is not None:
            boxes.append(clipped)
            scores.append(score)
    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return nms_bboxes(
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        nms_threshold,
    )


class MMDetectionGroundingDINOHeadDetector:
    def __init__(
        self,
        *,
        source_path,
        checkpoint,
        config,
        device="cuda:0",
        prompt="person head",
        score_threshold=0.3,
        nms_threshold=0.5,
    ):
        self.source_path = Path(source_path)
        self.checkpoint = Path(checkpoint)
        self.config = Path(config)
        self.device = str(device)
        self.prompt = str(prompt)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.model = None

    def load(self):
        for label, path in (
            ("MMDetection source directory", self.source_path),
            ("GroundingDINO checkpoint", self.checkpoint),
            ("GroundingDINO config", self.config),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        source = str(self.source_path.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                f"Unable to import MMDetection from {self.source_path}: {exc}"
            ) from exc
        self.model = init_detector(
            str(self.config.resolve()),
            str(self.checkpoint.resolve()),
            device=self.device,
        )
        return self

    def detect(self, image):
        if self.model is None:
            self.load()
        from mmdet.apis import inference_detector

        rgb = np.asarray(image, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        detection = inference_detector(
            self.model,
            bgr,
            text_prompt=self.prompt,
            custom_entities=True,
        )
        return extract_head_detections(
            detection,
            image_size=(rgb.shape[1], rgb.shape[0]),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

    def close(self):
        self.model = None

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _normalized_label(label):
    return re.sub(r"\s+", " ", str(label).strip().lower().strip(" .,:;"))


def _as_numpy(value, dtype):
    if value is None:
        return np.empty((0,), dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def normalize_hf_head_detections(
    result,
    *,
    image_size,
    allowed_labels,
    score_threshold,
    nms_threshold,
    max_detections,
    min_width,
    min_height,
    min_area,
):
    width, height = int(image_size[0]), int(image_size[1])
    boxes = _as_numpy(result.get("boxes"), np.float32)
    scores = _as_numpy(result.get("scores"), np.float32).reshape(-1)
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels", [])
    labels = list(labels)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape(-1, 4)
    allowed = {_normalized_label(label) for label in allowed_labels}

    candidates = []
    for index, box in enumerate(boxes):
        score = float(scores[index]) if index < len(scores) else 0.0
        label = str(labels[index]) if index < len(labels) else ""
        if score < float(score_threshold):
            continue
        if _normalized_label(label) not in allowed:
            continue
        clipped = clip_bbox(box, width, height)
        if clipped is None:
            continue
        box_width = clipped[2] - clipped[0]
        box_height = clipped[3] - clipped[1]
        if box_width < float(min_width) or box_height < float(min_height):
            continue
        if bbox_area(clipped) < float(min_area):
            continue
        candidates.append(
            {
                "bbox": [float(value) for value in clipped],
                "score": score,
                "label": label,
            }
        )

    candidates.sort(key=lambda item: -item["score"])
    kept = []
    for candidate in candidates:
        if all(
            bbox_iou(candidate["bbox"], existing["bbox"]) < float(nms_threshold)
            for existing in kept
        ):
            kept.append(candidate)
        if len(kept) >= int(max_detections):
            break
    return kept


def resolve_hf_model_source(model_id, *, cache_dir=None, local_files_only=False):
    source = Path(model_id).expanduser()
    if source.exists():
        return str(source.resolve())
    if not local_files_only:
        return str(model_id)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        str(model_id),
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=True,
    )


class HuggingFaceGroundingDINOHeadDetector:
    def __init__(
        self,
        *,
        model_id,
        device="cuda:0",
        prompts=None,
        allowed_labels=None,
        score_threshold=0.3,
        box_threshold=0.3,
        text_threshold=0.25,
        nms_threshold=0.5,
        max_detections=20,
        min_width=2,
        min_height=2,
        min_area=16,
        cache_dir=None,
        local_files_only=False,
        processor=None,
        model=None,
    ):
        self.model_id = str(model_id)
        self.device = str(device)
        self.prompts = list(prompts or ["person head", "human head", "head"])
        self.allowed_labels = list(allowed_labels or self.prompts)
        self.score_threshold = float(score_threshold)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
        self.min_width = float(min_width)
        self.min_height = float(min_height)
        self.min_area = float(min_area)
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.local_files_only = bool(local_files_only)
        self.processor = processor
        self.model = model

    def load(self):
        if self.processor is None or self.model is None:
            try:
                import transformers
                from transformers import (
                    AutoModelForZeroShotObjectDetection,
                    AutoProcessor,
                )

                kwargs = {
                    "cache_dir": self.cache_dir,
                    "local_files_only": self.local_files_only,
                }
                model_source = resolve_hf_model_source(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_source, **kwargs
                )
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_source, **kwargs
                )
            except Exception as exc:
                version = getattr(
                    sys.modules.get("transformers"), "__version__", "unknown"
                )
                raise RuntimeError(
                    "Unable to load Hugging Face Grounding DINO model "
                    f"'{self.model_id}' with transformers {version}. "
                    "Verify transformers>=4.50,<5, the model id, network/cache "
                    f"availability, and model compatibility. Original error: {exc}"
                ) from exc
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    def _post_process(self, outputs, inputs, image_size):
        method = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(method).parameters
        kwargs = {
            "text_threshold": self.text_threshold,
            "target_sizes": [(int(image_size[1]), int(image_size[0]))],
        }
        if "threshold" in parameters:
            kwargs["threshold"] = self.box_threshold
        elif "box_threshold" in parameters:
            kwargs["box_threshold"] = self.box_threshold
        else:
            raise RuntimeError(
                "Installed Transformers Grounding DINO processor supports "
                "neither 'threshold' nor 'box_threshold'."
            )
        if "input_ids" in parameters:
            kwargs["input_ids"] = inputs.get("input_ids")
        if "text_labels" in parameters:
            kwargs["text_labels"] = [self.prompts]
        return method(outputs, **kwargs)[0]

    def detect_with_metadata(self, image):
        if self.processor is None or self.model is None:
            self.load()
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
        pil_image = Image.fromarray(rgb, mode="RGB")
        inputs = self.processor(
            images=pil_image,
            text=self.prompts,
            return_tensors="pt",
        ).to(self.device)
        import torch

        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self._post_process(
            outputs,
            inputs,
            image_size=(rgb.shape[1], rgb.shape[0]),
        )
        return normalize_hf_head_detections(
            result,
            image_size=(rgb.shape[1], rgb.shape[0]),
            allowed_labels=self.allowed_labels,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            max_detections=self.max_detections,
            min_width=self.min_width,
            min_height=self.min_height,
            min_area=self.min_area,
        )

    def detect(self, image):
        detections = self.detect_with_metadata(image)
        boxes = np.asarray(
            [item["bbox"] for item in detections], dtype=np.float32
        ).reshape(-1, 4)
        scores = np.asarray(
            [item["score"] for item in detections], dtype=np.float32
        )
        return boxes, scores

    def close(self):
        self.model = None
        self.processor = None

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_head_detector(config, *, device):
    backend = str(config.get("backend", config.get("type", ""))).lower()
    if backend in {"hf_grounding_dino", "huggingface", "hf"}:
        return HuggingFaceGroundingDINOHeadDetector(
            model_id=config["model_id"],
            device=device,
            prompts=config.get("prompts"),
            allowed_labels=config.get("allowed_labels"),
            score_threshold=config.get("score_threshold", 0.3),
            box_threshold=config.get("box_threshold", 0.3),
            text_threshold=config.get("text_threshold", 0.25),
            nms_threshold=config.get("nms_threshold", 0.5),
            max_detections=config.get("max_detections", 20),
            min_width=config.get("min_width", 2),
            min_height=config.get("min_height", 2),
            min_area=config.get("min_area", 16),
            cache_dir=config.get("cache_dir"),
            local_files_only=config.get("local_files_only", False),
        )
    if backend in {"mmdetection", "grounding_dino"}:
        return MMDetectionGroundingDINOHeadDetector(
            source_path=config["source_path"],
            checkpoint=config["checkpoint"],
            config=config["config"],
            device=device,
            prompt=config.get("prompt", "person head"),
            score_threshold=config.get("score_threshold", 0.3),
            nms_threshold=config.get("nms_threshold", 0.5),
        )
    raise ValueError(f"Unsupported head detector backend: {backend!r}")


GroundingDINOHeadDetector = MMDetectionGroundingDINOHeadDetector
