import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    absolute_path: Path


def _image_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"img_{digest}"


def _records(root: Path, relative_paths: Iterable[str]) -> List[ImageRecord]:
    result = []
    for relative_path in relative_paths:
        normalized = Path(relative_path).as_posix()
        absolute_path = root / normalized
        if not absolute_path.is_file():
            raise FileNotFoundError(f"Image not found: {absolute_path}")
        result.append(ImageRecord(_image_id(normalized), normalized, absolute_path))
    return result


def _discover_directory(config):
    root = Path(config["path"])
    if not root.is_dir():
        raise NotADirectoryError(f"Input image directory not found: {root}")
    extensions = {
        extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
        for extension in config.get("extensions", DEFAULT_EXTENSIONS)
    }
    relative_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    return _records(root, relative_paths)


def _discover_gazefollow(config):
    data_root = Path(config["path"])
    image_root = data_root / "data_extended"
    split = str(config.get("split", "train")).lower()
    if split not in {"train", "test"}:
        raise ValueError("GazeFollow input.split must be 'train' or 'test'")
    annotation_path = image_root / f"{split}_annotations_release.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"GazeFollow annotation not found: {annotation_path}")
    relative_paths = []
    seen = set()
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            path = Path(row[0].strip()).as_posix()
            if path and path not in seen:
                seen.add(path)
                relative_paths.append(path)
    return _records(image_root, relative_paths)


def discover_images(config):
    source_type = str(config.get("type", "directory")).lower()
    if source_type == "directory":
        records = _discover_directory(config)
    elif source_type == "gazefollow":
        records = _discover_gazefollow(config)
    else:
        raise ValueError(f"Unsupported input.type: {source_type}")
    limit = config.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("input.limit must be >= 0 or null")
        records = records[:limit]
    return records
