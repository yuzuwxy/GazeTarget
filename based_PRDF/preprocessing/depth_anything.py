import sys
import warnings
from pathlib import Path

import numpy as np

from .bbox import clip_bbox


DEPTH_STAT_NAMES = ("mean", "median", "min", "max", "std")
ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def normalize_depth_map(depth_map):
    depth = np.asarray(depth_map, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.full(depth.shape, np.nan, dtype=np.float32)
    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    if maximum - minimum <= 1e-8:
        result = np.zeros(depth.shape, dtype=np.float32)
        result[~finite] = np.nan
        return result
    result = (depth - minimum) / (maximum - minimum)
    return result.astype(np.float32)


def _bbox_mask(bbox, image_size):
    width, height = image_size
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    x1, y1 = int(np.floor(x1)), int(np.floor(y1))
    x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def extract_depth_statistics(depth_map, *, bboxes, masks=None):
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must have shape (H, W), got {depth.shape}")
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    mask_array = None if masks is None else np.asarray(masks)
    result = {
        name: np.full((len(boxes),), np.nan, dtype=np.float32)
        for name in DEPTH_STAT_NAMES
    }
    height, width = depth.shape
    for index, box in enumerate(boxes):
        region = None
        if (
            mask_array is not None
            and mask_array.ndim == 3
            and index < len(mask_array)
            and mask_array[index].shape == depth.shape
            and bool(np.asarray(mask_array[index]).any())
        ):
            region = np.asarray(mask_array[index], dtype=bool)
        if region is None:
            region = _bbox_mask(box, (width, height))
        if region is None:
            warnings.warn(
                f"Depth region {index} has an invalid or empty bbox",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        values = depth[region]
        values = values[np.isfinite(values)]
        if values.size == 0:
            warnings.warn(
                f"Depth region {index} contains no finite pixels",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        result["mean"][index] = values.mean()
        result["median"][index] = np.median(values)
        result["min"][index] = values.min()
        result["max"][index] = values.max()
        result["std"][index] = values.std()
    return result


class DepthAnythingEstimator:
    def __init__(
        self,
        *,
        source_path,
        checkpoint,
        device="cuda:0",
        encoder="vitl",
        input_size=518,
        normalization="per_image_minmax",
        save_depth_map=False,
        model=None,
    ):
        self.source_path = Path(source_path)
        self.checkpoint = Path(checkpoint)
        self.device = str(device)
        self.encoder = str(encoder)
        self.input_size = int(input_size)
        self.normalization = str(normalization)
        self.save_depth_map = bool(save_depth_map)
        self.model = model

    def load(self):
        import torch

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                f"Depth Anything requires an available CUDA device, got {self.device}"
            )
        if self.encoder not in ENCODER_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything encoder: {self.encoder}")
        if not self.source_path.is_dir():
            raise FileNotFoundError(
                f"Depth Anything source directory not found: {self.source_path}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Depth Anything checkpoint not found: {self.checkpoint}"
            )
        if self.model is None:
            source = str(self.source_path.resolve())
            if source not in sys.path:
                sys.path.insert(0, source)
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError as exc:
                raise ImportError(
                    f"Unable to import Depth Anything V2 from {self.source_path}: {exc}"
                ) from exc
            self.model = DepthAnythingV2(**ENCODER_CONFIGS[self.encoder])
            state = torch.load(
                str(self.checkpoint.resolve()), map_location="cpu", weights_only=False
            )
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        return self

    def predict(self, image):
        if self.model is None:
            self.load()
        import torch
        import torch.nn.functional as functional

        rgb = np.asarray(image, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        tensor, (height, width) = self.model.image2tensor(
            bgr, self.input_size
        )
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            raw_depth = self.model(tensor)
            raw_depth = functional.interpolate(
                raw_depth[:, None],
                (height, width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
        raw_depth = raw_depth.detach().cpu().numpy()
        if self.normalization == "per_image_minmax":
            return normalize_depth_map(raw_depth)
        if self.normalization == "none":
            return np.asarray(raw_depth, dtype=np.float32)
        raise ValueError(f"Unsupported depth normalization: {self.normalization}")

    def extract(self, image, *, object_bboxes, object_masks, head_bboxes):
        depth_map = self.predict(image)
        return {
            "depth_map": depth_map,
            "object_depth": extract_depth_statistics(
                depth_map, bboxes=object_bboxes, masks=object_masks
            ),
            "head_depth": extract_depth_statistics(
                depth_map, bboxes=head_bboxes
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
