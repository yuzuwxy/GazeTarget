import sys
from pathlib import Path

import numpy as np


def normalize_sam2_masks(raw_masks, image_shape):
    height, width = image_shape
    normalized = []
    for item in raw_masks or []:
        mask = np.asarray(item.get("segmentation"), dtype=np.uint8)
        if mask.shape != (height, width):
            raise ValueError(
                f"SAM2 mask shape must be {(height, width)}, got {mask.shape}"
            )
        mask = (mask > 0).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        normalized.append(
            {
                "segmentation": mask,
                "area": area,
                "predicted_iou": float(item.get("predicted_iou", 0.0)),
                "stability_score": float(item.get("stability_score", 0.0)),
            }
        )
    return normalized


class SAM2Segmenter:
    def __init__(
        self,
        *,
        source_path,
        checkpoint,
        config,
        device="cuda:0",
        dtype="bfloat16",
        generator=None,
    ):
        self.source_path = Path(source_path)
        self.checkpoint = Path(checkpoint)
        self.config = str(config)
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.generator_config = dict(generator or {})
        self.model = None
        self.mask_generator = None

    def load(self):
        if not self.source_path.is_dir():
            raise FileNotFoundError(f"SAM2 source directory not found: {self.source_path}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")
        source = str(self.source_path.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(
                f"Unable to import SAM2 from {self.source_path}: {exc}"
            ) from exc
        self.model = build_sam2(
            self.config,
            str(self.checkpoint.resolve()),
            device=self.device,
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model, **self.generator_config
        )
        return self

    def segment(self, image):
        if self.mask_generator is None:
            self.load()
        image = np.asarray(image, dtype=np.uint8)
        import torch

        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:
            raise ValueError(
                f"Unsupported SAM2 dtype '{self.dtype_name}'. "
                "Use a torch dtype name such as float16, bfloat16, or float32."
            )
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            raw_masks = self.mask_generator.generate(image)
        return normalize_sam2_masks(raw_masks, image.shape[:2])

    def close(self):
        self.mask_generator = None
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
