#!/usr/bin/env python
import argparse
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "images"


def _warning(message):
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_h5_structure(h5_file):
    def print_attrs(obj, indent):
        for name, value in obj.attrs.items():
            print(f"{indent}@{name}: {decode_h5_value(value)!r}")

    print("/")
    print_attrs(h5_file, "  ")

    def visitor(name, obj):
        depth = name.count("/") + 1
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            print(f"{indent}/{name}/")
            print_attrs(obj, indent + "  ")
        else:
            print(
                f"{indent}/{name} "
                f"shape={obj.shape} dtype={obj.dtype}"
            )
            print_attrs(obj, indent + "  ")

    h5_file.visititems(visitor)


def _read_dataset(group, name, default):
    if name not in group:
        _warning(f"Sample {group.name} is missing dataset '{name}'")
        return default
    try:
        return np.asarray(group[name][...])
    except Exception as exc:
        _warning(f"Unable to read {group.name}/{name}: {exc}")
        return default


def _normalize_boxes(value, field_name, sample_name):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if array.size % 4:
        _warning(
            f"Sample {sample_name} has malformed {field_name} shape "
            f"{array.shape}; ignoring it"
        )
        return np.empty((0, 4), dtype=np.float32)
    return array.reshape(-1, 4)


def _normalize_vector(value, dtype):
    return np.asarray(value, dtype=dtype).reshape(-1)


def read_sample_from_h5(group, *, image_id=None, index=0):
    sample_name = group.name
    image_id = str(image_id if image_id is not None else group.name.rsplit("/", 1)[-1])
    attrs = {name: decode_h5_value(value) for name, value in group.attrs.items()}

    image = None
    for name in ("image", "image_array", "rgb"):
        if name in group:
            image = _read_dataset(group, name, None)
            break

    object_bboxes = _normalize_boxes(
        _read_dataset(
            group, "object_bboxes", np.empty((0, 4), dtype=np.float32)
        ),
        "object_bboxes",
        sample_name,
    )
    head_bboxes = _normalize_boxes(
        _read_dataset(
            group, "head_bboxes", np.empty((0, 4), dtype=np.float32)
        ),
        "head_bboxes",
        sample_name,
    )
    object_scores = _normalize_vector(
        _read_dataset(
            group, "object_scores", np.empty((0,), dtype=np.float32)
        ),
        np.float32,
    )
    head_scores = _normalize_vector(
        _read_dataset(
            group, "head_scores", np.empty((0,), dtype=np.float32)
        ),
        np.float32,
    )
    object_mask_areas = _normalize_vector(
        _read_dataset(
            group, "object_mask_areas", np.empty((0,), dtype=np.int64)
        ),
        np.int64,
    )
    object_masks = (
        _read_dataset(group, "object_masks", None)
        if "object_masks" in group
        else None
    )
    def read_strings(name, count):
        if name not in group:
            return [""] * count
        values = group[name].asstr()[...]
        return [str(value or "") for value in np.asarray(values).reshape(-1)]

    def read_depth(name, count):
        depth_group = group.get(name)
        result = {}
        for stat in ("mean", "median", "min", "max", "std"):
            if isinstance(depth_group, h5py.Group) and stat in depth_group:
                result[stat] = np.asarray(
                    depth_group[stat][...], dtype=np.float32
                ).reshape(-1)
            else:
                result[stat] = np.full((count,), np.nan, dtype=np.float32)
        return result

    object_descriptions = read_strings(
        "object_descriptions", len(object_bboxes)
    )
    head_descriptions = read_strings("head_descriptions", len(head_bboxes))

    width = attrs.get("width")
    height = attrs.get("height")
    image_size = (
        (int(width), int(height))
        if width is not None and height is not None
        else None
    )
    return {
        "image_id": image_id,
        "index": int(index),
        "image_path": str(attrs.get("image_path", "")),
        "image_size": image_size,
        "attrs": attrs,
        "image": image,
        "object_bboxes": object_bboxes,
        "object_scores": object_scores,
        "object_mask_areas": object_mask_areas,
        "head_bboxes": head_bboxes,
        "head_scores": head_scores,
        "object_masks": object_masks,
        "object_descriptions": object_descriptions,
        "head_descriptions": head_descriptions,
        "object_depth": read_depth("object_depth", len(object_bboxes)),
        "head_depth": read_depth("head_depth", len(head_bboxes)),
        "depth_map": (
            _read_dataset(group, "normalized_depth_map", None)
            if "normalized_depth_map" in group
            else None
        ),
    }


def resolve_image_root(config, explicit_root=None):
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    input_config = (config or {}).get("input", {})
    path = input_config.get("path")
    if not path:
        return None
    root = Path(path).expanduser()
    if not root.is_absolute():
        config_path = Path((config or {}).get("_config_path", DEFAULT_CONFIG_PATH))
        root = config_path.parent / root
    if str(input_config.get("type", "directory")).lower() == "gazefollow":
        root = root / "data_extended"
    return root.resolve()


def _embedded_image(value):
    array = np.asarray(value)
    if array.ndim == 2:
        return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] in (3, 4):
        return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported embedded image shape: {array.shape}")


def load_image_from_sample(sample, *, image_root, h5_path):
    if sample.get("image") is not None:
        try:
            return _embedded_image(sample["image"])
        except Exception as exc:
            _warning(f"Unable to decode embedded image for {sample['image_id']}: {exc}")

    stored = str(sample.get("image_path", "")).strip()
    if not stored:
        _warning(f"Sample {sample['image_id']} has no image_path or embedded image")
        return None
    stored_path = Path(stored).expanduser()
    candidates = []
    if stored_path.is_absolute():
        candidates.append(stored_path)
    else:
        if image_root is not None:
            candidates.append(Path(image_root) / stored_path)
        candidates.append(Path(h5_path).resolve().parent / stored_path)
    for candidate in candidates:
        if candidate.is_file():
            try:
                with Image.open(candidate) as image_file:
                    return image_file.convert("RGB")
            except Exception as exc:
                _warning(f"Unable to load image {candidate}: {exc}")
                return None
    _warning(
        f"Image for sample {sample['image_id']} was not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )
    return None


def _clip_draw_box(box, image_size):
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if len(values) != 4 or not np.isfinite(values).all():
        return None
    width, height = image_size
    x1, y1, x2, y2 = values.tolist()
    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, min(x2, width - 1), min(y2, height - 1)


def draw_bboxes(
    image,
    bboxes,
    *,
    scores=None,
    descriptions=None,
    mean_depth=None,
    show_description=False,
    show_depth=False,
    max_description_chars=48,
    label,
    color,
    width=3,
):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    boxes = _normalize_boxes(bboxes, "bboxes", label)
    score_values = _normalize_vector(
        scores if scores is not None else [], np.float32
    )
    descriptions = list(descriptions or [])
    depth_values = _normalize_vector(
        mean_depth if mean_depth is not None else [], np.float32
    )
    for index, box in enumerate(boxes):
        clipped = _clip_draw_box(box, result.size)
        if clipped is None:
            _warning(f"Ignoring invalid {label} bbox: {box.tolist()}")
            continue
        draw.rectangle(clipped, outline=color, width=int(width))
        text = label
        if index < len(score_values) and np.isfinite(score_values[index]):
            text += f" {float(score_values[index]):.3f}"
        if show_depth and index < len(depth_values) and np.isfinite(depth_values[index]):
            text += f" d={float(depth_values[index]):.3f}"
        if show_description and index < len(descriptions):
            description = " ".join(str(descriptions[index]).split())
            limit = max(int(max_description_chars), 1)
            if len(description) > limit:
                description = description[: max(limit - 3, 1)] + "..."
            if description:
                text += f" {description}"
        text_box = draw.textbbox((clipped[0], clipped[1]), text)
        background = (
            text_box[0] - 1,
            text_box[1] - 1,
            text_box[2] + 1,
            text_box[3] + 1,
        )
        draw.rectangle(background, fill=color)
        draw.text((clipped[0], clipped[1]), text, fill=(0, 0, 0))
    return result


def draw_masks(image, masks, *, alpha=0.35):
    if masks is None:
        return image.convert("RGB").copy()
    array = np.asarray(masks)
    if array.size == 0:
        return image.convert("RGB").copy()
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        _warning(f"Ignoring masks with unsupported shape {array.shape}")
        return image.convert("RGB").copy()
    alpha = float(np.clip(alpha, 0, 1))
    result = image.convert("RGBA")
    colors = (
        (0, 180, 255),
        (255, 180, 0),
        (180, 0, 255),
        (0, 220, 120),
        (255, 80, 120),
    )
    for index, mask in enumerate(array):
        mask_image = Image.fromarray(
            (np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L"
        )
        if mask_image.size != result.size:
            _warning(
                f"Resizing mask {index} from {mask_image.size} to {result.size}"
            )
            mask_image = mask_image.resize(result.size, Image.Resampling.NEAREST)
        color = colors[index % len(colors)]
        overlay = Image.new("RGBA", result.size, color + (0,))
        overlay.putalpha(mask_image.point(lambda value: int(value * alpha)))
        result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def _safe_image_id(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "sample"


def visualize_sample(
    sample,
    *,
    image_root,
    h5_path,
    output_dir,
    overwrite=False,
    draw_mask=True,
    draw_bbox=True,
    draw_head=True,
    mask_alpha=0.35,
    show_description=True,
    show_depth=True,
    max_description_chars=48,
):
    image = load_image_from_sample(
        sample, image_root=image_root, h5_path=h5_path
    )
    if image is None:
        return None
    if draw_mask:
        image = draw_masks(
            image, sample.get("object_masks"), alpha=mask_alpha
        )
    if draw_bbox:
        image = draw_bboxes(
            image,
            sample.get("object_bboxes", []),
            scores=sample.get("object_scores"),
            descriptions=sample.get("object_descriptions"),
            mean_depth=sample.get("object_depth", {}).get("mean"),
            show_description=show_description,
            show_depth=show_depth,
            max_description_chars=max_description_chars,
            label="object",
            color=(0, 220, 0),
        )
    if draw_head:
        image = draw_bboxes(
            image,
            sample.get("head_bboxes", []),
            scores=sample.get("head_scores"),
            descriptions=sample.get("head_descriptions"),
            mean_depth=sample.get("head_depth", {}).get("mean"),
            show_description=show_description,
            show_depth=show_depth,
            max_description_chars=max_description_chars,
            label="head",
            color=(255, 50, 50),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{int(sample['index']):06d}_{_safe_image_id(sample['image_id'])}_vis.jpg"
    )
    if output_path.exists() and not overwrite:
        _warning(f"Output exists, skipping: {output_path}")
        return None
    image.save(output_path, format="JPEG", quality=95)
    print(f"Saved {output_path}")
    return output_path


def select_sample_indices(count, *, index=None, indices=None, limit=None):
    if index is not None:
        selected = [int(index)]
    elif indices is not None:
        selected = [int(value) for value in indices]
    else:
        selected = list(range(int(count)))
    valid = []
    for value in selected:
        if 0 <= value < count:
            valid.append(value)
        else:
            _warning(f"Sample index {value} is outside [0, {count})")
    if limit is not None:
        if int(limit) < 0:
            raise ValueError("--limit must be >= 0")
        valid = valid[: int(limit)]
    return valid


def collect_image_samples(images_group):
    samples = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        is_sample = (
            "image_path" in obj.attrs
            or any(
                field in obj
                for field in (
                    "image",
                    "image_array",
                    "rgb",
                    "object_bboxes",
                    "head_bboxes",
                )
            )
        )
        if is_sample:
            samples.append((name, obj))

    images_group.visititems(visitor)
    return samples


def _load_yaml(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def _bool_default(config, key, fallback):
    value = (config or {}).get("visualization", {}).get(key, fallback)
    return bool(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize preprocessing H5 object/head boxes and masks."
    )
    parser.add_argument("h5_path", nargs="?")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int)
    selection.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-structure", action="store_true")
    parser.add_argument(
        "--draw-mask", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--draw-bbox", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--draw-head", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--show-description",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--show-depth", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = _load_yaml(args.config)
    h5_value = args.h5_path or config.get("output", {}).get("h5_path")
    if not h5_value:
        raise ValueError("H5 path is required via argument or output.h5_path")
    h5_path = Path(h5_value).expanduser()
    if not h5_path.is_absolute():
        h5_path = Path(config["_config_path"]).parent / h5_path
    h5_path = h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    visual_config = config.get("visualization", {})
    output_value = (
        args.output_dir
        or visual_config.get("output_dir")
        or DEFAULT_OUTPUT_DIR
    )
    output_dir = Path(output_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(config["_config_path"]).parent / output_dir
    image_root = resolve_image_root(config, args.image_root)
    draw_mask = (
        args.draw_mask
        if args.draw_mask is not None
        else _bool_default(config, "draw_mask", True)
    )
    draw_bbox = (
        args.draw_bbox
        if args.draw_bbox is not None
        else _bool_default(config, "draw_bbox", True)
    )
    draw_head = (
        args.draw_head
        if args.draw_head is not None
        else _bool_default(config, "draw_head", True)
    )
    mask_alpha = float(visual_config.get("mask_alpha", 0.35))
    show_description = (
        args.show_description
        if args.show_description is not None
        else _bool_default(config, "show_description", True)
    )
    show_depth = (
        args.show_depth
        if args.show_depth is not None
        else _bool_default(config, "show_depth", True)
    )
    max_description_chars = int(
        visual_config.get("max_description_chars", 48)
    )

    saved = 0
    with h5py.File(h5_path, "r") as h5_file:
        if args.show_structure:
            print_h5_structure(h5_file)
        if "images" not in h5_file:
            raise ValueError("H5 file is missing required '/images' group")
        image_items = collect_image_samples(h5_file["images"])
        selected = select_sample_indices(
            len(image_items),
            index=args.index,
            indices=args.indices,
            limit=args.limit,
        )
        for index in selected:
            image_id, group = image_items[index]
            try:
                sample = read_sample_from_h5(
                    group, image_id=image_id, index=index
                )
                path = visualize_sample(
                    sample,
                    image_root=image_root,
                    h5_path=h5_path,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    draw_mask=draw_mask,
                    draw_bbox=draw_bbox,
                    draw_head=draw_head,
                    mask_alpha=mask_alpha,
                    show_description=show_description,
                    show_depth=show_depth,
                    max_description_chars=max_description_chars,
                )
                saved += path is not None
            except Exception as exc:
                _warning(f"Unable to visualize sample {index} ({image_id}): {exc}")
    print(f"Saved {saved} visualization image(s) to {output_dir.resolve()}")
    return saved


if __name__ == "__main__":
    main()
