#!/usr/bin/env python
import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from preprocessing.bbox import BBoxFilterConfig
from preprocessing.depth_anything import DepthAnythingEstimator
from preprocessing.describe_anything import DescribeAnythingCaptioner
from preprocessing.head_detector import build_head_detector
from preprocessing.image_source import discover_images
from preprocessing.pipeline import GazePreprocessingPipeline
from preprocessing.sam2_segmenter import SAM2Segmenter
from utils.h5_writer import GazeH5Writer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build gaze-target regions, descriptions, and depth attributes "
            "in HDF5 format."
        )
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--enable-description",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _require_runtime_device(device):
    if not str(device).startswith("cuda"):
        raise ValueError(f"Only CUDA devices are supported, got {device}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"runtime.device is {device}, but CUDA is not available in this environment"
        )


def build_pipeline(config, writer, stack):
    device = config["runtime"].get("device", "cuda:0")
    _require_runtime_device(device)
    sam2_config = config["sam2"]
    head_config = config["head_detector"]
    segmenter = stack.enter_context(
        SAM2Segmenter(
            source_path=sam2_config["source_path"],
            checkpoint=sam2_config["checkpoint"],
            config=sam2_config["config"],
            device=device,
            dtype=sam2_config.get("dtype", "bfloat16"),
            generator=sam2_config.get("generator", {}),
        )
    )
    head_detector = stack.enter_context(
        build_head_detector(head_config, device=device)
    )
    depth_estimator = None
    depth_config = config.get("depth", {})
    if depth_config.get("enabled", False):
        depth_device = depth_config.get("device", device)
        _require_runtime_device(depth_device)
        depth_estimator = stack.enter_context(
            DepthAnythingEstimator(
                source_path=depth_config["source_path"],
                checkpoint=depth_config["checkpoint"],
                device=depth_device,
                encoder=depth_config.get("encoder", "vitl"),
                input_size=depth_config.get("input_size", 518),
                normalization=depth_config.get(
                    "normalization", "per_image_minmax"
                ),
                save_depth_map=depth_config.get("save_depth_map", False),
            )
        )
    captioner = None
    description_config = config.get("description", {})
    if description_config.get("enabled", False):
        description_device = description_config.get("device", device)
        _require_runtime_device(description_device)
        captioner = stack.enter_context(
            DescribeAnythingCaptioner(
                source_path=description_config["source_path"],
                checkpoint=description_config["checkpoint"],
                device=description_device,
                object_prompt=description_config["object_prompt"],
                head_prompt=description_config["head_prompt"],
                conv_mode=description_config.get("conv_mode", "v1"),
                prompt_mode=description_config.get(
                    "prompt_mode", "full+focal_crop"
                ),
                temperature=description_config.get("temperature", 0.2),
                top_p=description_config.get("top_p", 0.5),
                num_beams=description_config.get("num_beams", 1),
                max_new_tokens=description_config.get("max_new_tokens", 256),
                batch_size=description_config.get("batch_size", 1),
            )
        )
    return GazePreprocessingPipeline(
        segmenter=segmenter,
        head_detector=head_detector,
        bbox_config=BBoxFilterConfig.from_dict(config["bbox_filter"]),
        writer=writer,
        save_masks=config["output"].get("save_masks", False),
        depth_estimator=depth_estimator,
        captioner=captioner,
    )


def run(config):
    records = discover_images(config["input"])
    if not records:
        raise ValueError("No input images were discovered")

    metadata = {
        "source_config": config["_config_path"],
        "sam2_config": config["sam2"]["config"],
        "sam2_checkpoint": config["sam2"]["checkpoint"],
        "sam2_parameters": config["sam2"].get("generator", {}),
        "bbox_filter_parameters": config["bbox_filter"],
        "head_detector": config["head_detector"].get(
            "backend",
            config["head_detector"].get("type", "grounding_dino"),
        ),
        "head_detector_parameters": {
            key: value
            for key, value in config["head_detector"].items()
            if key not in {"checkpoint"}
        },
        "description_enabled": config.get("description", {}).get(
            "enabled", False
        ),
        "description_parameters": config.get("description", {}),
        "depth_enabled": config.get("depth", {}).get("enabled", False),
        "depth_parameters": config.get("depth", {}),
        "depth_normalization": config.get("depth", {}).get(
            "normalization", "disabled"
        ),
        "object_depth_region_source": "mask_then_bbox",
        "head_depth_region_source": "bbox",
    }
    output = config["output"]
    with ExitStack() as stack:
        writer = stack.enter_context(
            GazeH5Writer(
                output["h5_path"],
                metadata=metadata,
                overwrite=output.get("overwrite", False),
                save_masks=output.get("save_masks", False),
                compression=output.get("compression", "gzip"),
                compression_opts=output.get("compression_opts", 4),
                save_depth_map=config.get("depth", {}).get(
                    "enabled", False
                )
                and config.get("depth", {}).get("save_depth_map", False),
            )
        )
        pipeline = build_pipeline(config, writer, stack)
        processed = pipeline.run(
            tqdm(records, desc="Building gaze H5", unit="image")
        )
    return processed


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.enable_description is not None:
        config.setdefault("description", {})["enabled"] = args.enable_description
    if args.enable_depth is not None:
        config.setdefault("depth", {})["enabled"] = args.enable_depth
    processed = run(config)
    print(f"Wrote {processed} images to {config['output']['h5_path']}")


if __name__ == "__main__":
    main()
