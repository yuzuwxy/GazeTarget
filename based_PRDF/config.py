from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
REQUIRED_SECTIONS = (
    "input",
    "output",
    "sam2",
    "bbox_filter",
    "runtime",
)
PATH_FIELDS = {
    ("input", "path"),
    ("output", "h5_path"),
    ("sam2", "source_path"),
    ("sam2", "checkpoint"),
    ("description", "source_path"),
    ("description", "checkpoint"),
    ("depth", "source_path"),
    ("depth", "checkpoint"),
}


def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _validate_config(config, path):
    for section in REQUIRED_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config section '{section}' in {path}")

    source_type = str(config["input"].get("type", "")).lower()
    if source_type not in {"directory", "gazefollow"}:
        raise ValueError("input.type must be 'directory' or 'gazefollow'")
    if not config["input"].get("path"):
        raise ValueError("input.path is required")
    if not config["output"].get("h5_path"):
        raise ValueError("output.h5_path is required")
    for section, key in (
        ("sam2", "source_path"),
        ("sam2", "checkpoint"),
        ("sam2", "config"),
    ):
        if not config[section].get(key):
            raise ValueError(f"{section}.{key} is required")

    limit = config["input"].get("limit")
    if limit is not None and int(limit) < 0:
        raise ValueError("input.limit must be >= 0 or null")

    runtime_device = config["runtime"].get("device")
    if not runtime_device or not str(runtime_device).startswith("cuda"):
        raise ValueError("runtime.device must be a CUDA device")

    for section in ("description", "depth"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        if not values.get("enabled", False):
            continue
        for key in ("source_path", "checkpoint", "device"):
            if not values.get(key):
                raise ValueError(f"{section}.{key} is required when enabled")
        if not str(values["device"]).startswith("cuda"):
            raise ValueError(f"{section}.device must be a CUDA device")
        if int(values.get("batch_size", 1)) <= 0:
            raise ValueError(f"{section}.batch_size must be > 0")

    description = config.get("description", {})
    if description.get("enabled", False):
        for key in ("object_prompt", "head_prompt"):
            if not str(description.get(key, "")).strip():
                raise ValueError(
                    f"description.{key} is required when enabled"
                )
        if int(description.get("max_new_tokens", 256)) <= 0:
            raise ValueError("description.max_new_tokens must be > 0")

    depth = config.get("depth", {})
    if depth.get("enabled", False):
        if depth.get("encoder", "vitl") not in {"vits", "vitb", "vitl", "vitg"}:
            raise ValueError("depth.encoder must be vits, vitb, vitl, or vitg")
        if int(depth.get("input_size", 518)) <= 0:
            raise ValueError("depth.input_size must be > 0")
        if depth.get("normalization", "per_image_minmax") not in {
            "per_image_minmax",
            "none",
        }:
            raise ValueError(
                "depth.normalization must be 'per_image_minmax' or 'none'"
            )


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load, validate, and resolve paths relative to the YAML file."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    _validate_config(config, path)

    for section, key in PATH_FIELDS:
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = _resolve_path(value, path.parent)

    config["_config_path"] = str(path)
    return config
