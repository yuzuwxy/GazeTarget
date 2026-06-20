from pathlib import Path

import pytest

from config import load_config


def test_load_config_resolves_project_relative_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  type: directory
  path: images
output:
  h5_path: outputs/gaze.h5
  overwrite: true
sam2:
  source_path: vendor/SAM2
  checkpoint: checkpoints/sam.pt
  config: configs/sam.yaml
bbox_filter: {}
head_detector:
  type: grounding_dino
  source_path: vendor/mmdet
  checkpoint: checkpoints/head.pth
  config: configs/head.py
runtime:
  device: cuda:0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["input"]["path"] == str(tmp_path / "images")
    assert config["output"]["h5_path"] == str(tmp_path / "outputs/gaze.h5")
    assert config["sam2"]["checkpoint"] == str(tmp_path / "checkpoints/sam.pt")
    assert config["sam2"]["config"] == "configs/sam.yaml"


def test_load_config_requires_pipeline_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing config section"):
        load_config(config_path)


def test_load_config_accepts_hf_head_detector(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  type: directory
  path: images
output:
  h5_path: outputs/gaze.h5
sam2:
  source_path: vendor/SAM2
  checkpoint: checkpoints/sam.pt
  config: configs/sam.yaml
bbox_filter: {}
head_detector:
  backend: hf_grounding_dino
  model_id: openmmlab-community/test-model
  cache_dir: model-cache
  prompts: [person head, human head, head]
  allowed_labels: [person head, human head, head]
  score_threshold: 0.3
  box_threshold: 0.3
  text_threshold: 0.25
  nms_threshold: 0.5
  max_detections: 20
runtime:
  device: cuda:0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["head_detector"]["backend"] == "hf_grounding_dino"
    assert config["head_detector"]["model_id"] == "openmmlab-community/test-model"
    assert config["head_detector"]["cache_dir"] == str(tmp_path / "model-cache")


def test_load_config_rejects_unknown_head_detector_backend(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input: {type: directory, path: images}
output: {h5_path: outputs/gaze.h5}
sam2:
  source_path: vendor/SAM2
  checkpoint: checkpoints/sam.pt
  config: configs/sam.yaml
bbox_filter: {}
head_detector: {backend: unknown}
runtime: {device: cpu}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="head_detector.backend"):
        load_config(config_path)


def test_load_config_rejects_invalid_detection_threshold(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input: {type: directory, path: images}
output: {h5_path: outputs/gaze.h5}
sam2:
  source_path: vendor/SAM2
  checkpoint: checkpoints/sam.pt
  config: configs/sam.yaml
bbox_filter: {}
head_detector:
  backend: hf_grounding_dino
  model_id: test/model
  prompts: [head]
  score_threshold: 1.5
runtime: {device: cpu}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score_threshold"):
        load_config(config_path)


def test_load_config_resolves_enabled_enrichment_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input: {type: directory, path: images}
output: {h5_path: outputs/gaze.h5}
sam2:
  source_path: vendor/SAM2
  checkpoint: checkpoints/sam.pt
  config: configs/sam.yaml
bbox_filter: {}
head_detector:
  backend: hf_grounding_dino
  model_id: test/model
  prompts: [head]
description:
  enabled: true
  source_path: vendor/DAM
  checkpoint: checkpoints/DAM
  device: cuda:0
  object_prompt: describe the highlighted object
  head_prompt: describe the highlighted head
depth:
  enabled: true
  source_path: vendor/Depth
  checkpoint: checkpoints/depth.pth
  device: cuda:0
runtime: {device: cuda:0}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["description"]["checkpoint"] == str(tmp_path / "checkpoints/DAM")
    assert config["depth"]["checkpoint"] == str(tmp_path / "checkpoints/depth.pth")
