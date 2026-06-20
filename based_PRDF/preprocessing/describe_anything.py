import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .bbox import clip_bbox


def boxes_to_masks(bboxes, *, image_size):
    width, height = int(image_size[0]), int(image_size[1])
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    masks = np.zeros((len(boxes), height, width), dtype=np.uint8)
    for index, box in enumerate(boxes):
        clipped = clip_bbox(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        masks[
            index,
            int(np.floor(y1)) : int(np.ceil(y2)),
            int(np.floor(x1)) : int(np.ceil(x2)),
        ] = 1
    return masks


class DescribeAnythingCaptioner:
    def __init__(
        self,
        *,
        source_path,
        checkpoint,
        device="cuda:0",
        object_prompt="Describe the highlighted gaze-target candidate object, including its category, appearance, and position.",
        head_prompt="Describe the highlighted person's head, including visibility, orientation, occlusion, and position.",
        conv_mode="v1",
        prompt_mode="full+focal_crop",
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=256,
        batch_size=1,
        model=None,
    ):
        self.source_path = Path(source_path)
        self.checkpoint = Path(checkpoint)
        self.device = str(device)
        self.object_prompt = str(object_prompt)
        self.head_prompt = str(head_prompt)
        self.conv_mode = str(conv_mode)
        self.prompt_mode = str(prompt_mode)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.model = model

    def load(self):
        import torch

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                f"Describe Anything requires an available CUDA device, got {self.device}"
            )
        if self.device not in {"cuda", "cuda:0"}:
            raise RuntimeError(
                "The bundled Describe Anything implementation places token "
                "tensors on cuda:0 and therefore requires device cuda:0"
            )
        if not self.source_path.is_dir():
            raise FileNotFoundError(
                f"Describe Anything source directory not found: {self.source_path}"
            )
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"Describe Anything checkpoint directory not found: {self.checkpoint}"
            )
        if self.model is None:
            source = str(self.source_path.resolve())
            if source not in sys.path:
                sys.path.insert(0, source)
            try:
                from dam import DescribeAnythingModel, disable_torch_init
            except ImportError as exc:
                raise ImportError(
                    f"Unable to import Describe Anything from {self.source_path}: {exc}"
                ) from exc
            disable_torch_init()
            self.model = DescribeAnythingModel(
                model_path=str(self.checkpoint.resolve()),
                conv_mode=self.conv_mode,
                prompt_mode=self.prompt_mode,
            ).to(self.device)
        return self

    def _query(self, prompt):
        return prompt if "<image>" in prompt else f"<image>\n{prompt}"

    def _describe_masks(self, image, masks, prompt):
        image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        descriptions = []
        for index, mask in enumerate(np.asarray(masks, dtype=np.uint8)):
            if not bool(mask.any()):
                warnings.warn(
                    f"Describe Anything region {index} is empty",
                    RuntimeWarning,
                    stacklevel=2,
                )
                descriptions.append("")
                continue
            try:
                description = self.model.get_description(
                    image_pil,
                    Image.fromarray(mask * 255, mode="L"),
                    self._query(prompt),
                    streaming=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )
                descriptions.append(str(description or "").strip())
            except Exception as exc:
                warnings.warn(
                    f"Describe Anything failed for region {index}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                descriptions.append("")
        return descriptions

    def describe(
        self,
        image,
        *,
        object_bboxes,
        object_masks,
        head_bboxes,
    ):
        if self.model is None:
            self.load()
        rgb = np.asarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        object_boxes = np.asarray(object_bboxes, dtype=np.float32).reshape(-1, 4)
        masks = None if object_masks is None else np.asarray(object_masks, dtype=np.uint8)
        if masks is None or masks.shape != (len(object_boxes), height, width):
            masks = boxes_to_masks(object_boxes, image_size=(width, height))
        head_masks = boxes_to_masks(head_bboxes, image_size=(width, height))
        return {
            "object_descriptions": self._describe_masks(
                rgb, masks, self.object_prompt
            ),
            "head_descriptions": self._describe_masks(
                rgb, head_masks, self.head_prompt
            ),
        }

    def close(self):
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
