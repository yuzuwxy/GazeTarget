#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.head_detector import HuggingFaceGroundingDINOHeadDetector


DEFAULT_MODEL_ID = (
    "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
)


def validate_detections(detections, *, image_size):
    width, height = int(image_size[0]), int(image_size[1])
    for detection in detections:
        bbox = detection.get("bbox", [])
        valid = (
            len(bbox) == 4
            and all(math.isfinite(float(value)) for value in bbox)
            and 0 <= float(bbox[0]) < float(bbox[2]) <= width
            and 0 <= float(bbox[1]) < float(bbox[3]) <= height
        )
        if not valid:
            raise ValueError(f"Detection contains invalid bbox: {bbox}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check the Hugging Face person-head detector."
    )
    parser.add_argument("image", nargs="?")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["person head", "human head", "head"],
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "model_id": args.model_id,
                    "processor_class": AutoProcessor.__name__,
                    "model_class": AutoModelForZeroShotObjectDetection.__name__,
                },
                sort_keys=True,
            )
        )
        return
    if not args.image:
        raise ValueError("An image path is required unless --dry-run is used")

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        image_size = image.size
        with HuggingFaceGroundingDINOHeadDetector(
            model_id=args.model_id,
            device=args.device,
            prompts=args.prompts,
            allowed_labels=args.prompts,
            score_threshold=args.score_threshold,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ) as detector:
            detections = detector.detect_with_metadata(image)
    validate_detections(detections, image_size=image_size)
    print(
        json.dumps(
            {
                "image": str(image_path),
                "image_size": list(image_size),
                "detections": detections,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
