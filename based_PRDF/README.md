# Training-Free Gaze Target Preprocessing

This project builds object and head bounding-box proposals for gaze target
detection. It follows the staged organization used by PRDF while keeping the
implementation local and task-specific.

## Pipeline

```text
unique image loading
-> SAM2 automatic object masks
-> mask-to-bbox conversion
-> duplicate and containment filtering
-> Hugging Face MM Grounding DINO head detection
-> Depth Anything V2 full-image depth and region statistics
-> Describe Anything object/head region descriptions
-> HDF5 output
```

Object boxes are proposals derived from SAM2 masks. Head boxes are detections
from a Transformers Grounding DINO/MM Grounding DINO model using configurable
head prompts. GazeFollow ground-truth head annotations are not written as
detections.

Depth Anything and Describe Anything consume the live arrays produced for the
current image. They do not read bbox or mask inputs back from HDF5. Retained
SAM2 masks are passed directly to both enrichment steps and are also persisted
because `output.save_masks` defaults to `true`.

All boxes use image-pixel, half-open `xyxy` coordinates:

```text
[x1, y1, x2, y2]
0 <= x1 < x2 <= image_width
0 <= y1 < y2 <= image_height
```

## SAM2 Setup

The repository contains local SAM2 source code under `third_party/SAM2`. Put a
compatible checkpoint under `checkpoints` and configure it in `config.yaml`.
The supplied default expects:

```text
checkpoints/sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

The config name is resolved by Hydra inside the SAM2 package; it is not a
filesystem path relative to this project.

Install a PyTorch and torchvision build compatible with the machine's CUDA
driver. Then install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

The default head detector uses Hugging Face Transformers. The legacy local
MMDetection implementation is still available through the backend setting.

The recommended environment is the existing `mmgrounding` Conda environment:

```bash
conda activate mmgrounding
pip install -r requirements.txt
```

Install the PyTorch build for the machine separately. Transformers 4.50 or
newer is required for the configured Hugging Face MM Grounding DINO model.

Describe Anything and Depth Anything are configured from local sources and
weights:

```text
third_party/Describe-Anything
checkpoints/DAM-3B
third_party/Depth-Anything-V2
checkpoints/depth-anything-v2/depth_anything_v2_vitl.pth
```

All model inference is CUDA-only. The bundled Describe Anything implementation
places token tensors on `cuda:0`, so its configured device must be `cuda:0`.
Depth Anything uses the configured CUDA device explicitly.

## Configuration

Edit `config.yaml`:

- `input.type`: `gazefollow` or `directory`
- `input.path`: dataset root or image directory
- `input.split`: GazeFollow `train` or `test`
- `input.limit`: small integer for debugging, `null` for all images
- `output.h5_path`: output file
- `output.overwrite`: whether an existing file may be replaced; the default
  configuration enables this so each `python main.py` run replaces the
  previous HDF5 output
- `output.save_masks`: whether retained pixel masks are stored
- `sam2`: source, checkpoint, Hydra config, and generator parameters
- `bbox_filter`: size, duplicate-IoU, and containment thresholds
- `head_detector`: backend, model id, prompts, thresholds, NMS, and limits
- `description`: local DAM paths, object/head prompts, and generation settings
- `depth`: local Depth Anything paths, encoder, normalization, and map saving
- `runtime.device`: for example `cuda:0`

Relative filesystem paths are resolved relative to the YAML file.

The default Hugging Face configuration is:

```yaml
head_detector:
  backend: hf_grounding_dino
  model_id: openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det
  cache_dir: ./checkpoints/huggingface
  local_files_only: true
  prompts: ["person head", "human head", "head"]
  allowed_labels: ["person head", "human head", "head"]
  score_threshold: 0.3
  box_threshold: 0.3
  text_threshold: 0.25
  nms_threshold: 0.5
  max_detections: 20
  min_width: 2
  min_height: 2
  min_area: 16
```

`box_threshold` is passed to the model processor's grounded object detection
post-processing. `text_threshold` controls phrase extraction.
`score_threshold` is a final project-level confidence filter. Results are then
clipped, checked for minimum geometry, filtered by label, processed with NMS,
and truncated to `max_detections`.

To use the retained MMDetection implementation:

```yaml
head_detector:
  backend: mmdetection
  source_path: ./third_party/mmdetection
  checkpoint: ./checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.pth
  config: ./configs/grounding_dino_swin-t_obj365_v3det_inference.py
  prompt: person head
  score_threshold: 0.3
  nms_threshold: 0.5
```

Description and depth are enabled by default:

```yaml
description:
  enabled: true
  source_path: ./third_party/Describe-Anything
  checkpoint: ./checkpoints/DAM-3B
  device: cuda:0
  object_prompt: Describe the highlighted gaze-target candidate object...
  head_prompt: Describe the highlighted person's head...
  conv_mode: v1
  prompt_mode: full+focal_crop
  max_new_tokens: 256

depth:
  enabled: true
  source_path: ./third_party/Depth-Anything-V2
  checkpoint: ./checkpoints/depth-anything-v2/depth_anything_v2_vitl.pth
  device: cuda:0
  encoder: vitl
  input_size: 518
  normalization: per_image_minmax
  save_depth_map: false
```

Object descriptions focus on category, appearance, and image position. Head
descriptions focus on visibility, orientation, occlusion, and position. These
fields provide semantic and spatial cues for later gaze-target matching.

## Run

```bash
conda run -n mmgrounding python main.py --config config.yaml
```

Temporarily disable either enrichment stage without editing YAML:

```bash
conda run -n mmgrounding python main.py --config config.yaml \
  --no-enable-description

conda run -n mmgrounding python main.py --config config.yaml \
  --no-enable-depth
```

The output is written through a temporary file and atomically moved into place
after a successful run. Existing output is rejected unless
`output.overwrite: true`.

## Head Detector Sanity Check

Validate the installed API without downloading model weights:

```bash
conda run -n mmgrounding python scripts/sanity_check_head_detector.py --dry-run
```

Run one image on CPU:

```bash
conda run -n mmgrounding python scripts/sanity_check_head_detector.py \
  path/to/image.jpg \
  --device cpu
```

The script prints JSON records containing:

```json
{
  "bbox": [10.0, 20.0, 80.0, 100.0],
  "score": 0.87,
  "label": "person head"
}
```

It also rejects non-finite, degenerate, or out-of-image boxes. The main
pipeline stores the same boxes and scores in `head_bboxes` and `head_scores`;
labels are retained by the detector metadata interface for debugging but are
not added to the existing HDF5 schema.

## HDF5 Structure

```text
/metadata  # schema_version = "2.1"
  attrs:
    schema_version = "2.1"
    bbox_format = "xyxy"
    bbox_coordinate_mode = "half_open"
    num_images
    sam2_config
    sam2_checkpoint
    sam2_parameters_json
    bbox_filter_parameters_json
    head_detector
    head_detector_parameters_json

/images/{image_id}
  attrs:
    image_path
    width
    height
    num_object_bboxes
    num_head_bboxes

  object_bboxes       float32 [N, 4]
  object_scores       float32 [N]
  object_mask_areas   int64   [N]
  head_bboxes         float32 [M, 4]
  head_scores         float32 [M]
  object_descriptions UTF-8 [N]
  head_descriptions   UTF-8 [M]
  object_depth/
    mean              float32 [N]
    median            float32 [N]
    min               float32 [N]
    max               float32 [N]
    std               float32 [N]
  head_depth/
    mean              float32 [M]
    median            float32 [M]
    min               float32 [M]
    max               float32 [M]
    std               float32 [M]
  normalized_depth_map float32 [H, W]  # only when enabled
  object_masks        uint8   [N, H, W]  # only when enabled
```

Empty object or head results are still saved with shapes `(0, 4)` and `(0,)`.
This keeps downstream readers simple and makes detector failures distinguishable
from missing fields.

Descriptions and every depth statistic are index-aligned with their matching
bbox stream. Failed descriptions are empty strings. Invalid or empty depth
regions are NaN. Depth values are relative monocular depth normalized
independently per image to `[0,1]`; they are useful for comparing heads and
candidate targets within one image, not as metric distances across images.
Object depth uses the SAM2 mask when available and otherwise falls back to its
bbox. Head depth uses the head bbox.

## HDF5 Visualization

Use the standalone reader to inspect source images, SAM2 object proposals, head
detections, scores, and saved masks:

```bash
conda run -n mmgrounding python h5_reader.py --config config.yaml
```

By default it reads `output.h5_path` from `config.yaml` and saves JPEG files to
`visualization.output_dir`, currently:

```text
./output/images
```

Useful debug options:

```bash
# Print the complete H5 group, dataset, shape, dtype, and attribute summary.
conda run -n mmgrounding python h5_reader.py \
  --config config.yaml --show-structure --limit 1

# Visualize zero-based sample indices 0, 3, and 10.
conda run -n mmgrounding python h5_reader.py \
  --config config.yaml --indices 0 3 10

# Override paths or disable individual layers.
conda run -n mmgrounding python h5_reader.py outputs/gaze_preprocessing.h5 \
  --image-root /path/to/image/root \
  --output-dir ./output/images \
  --no-draw-mask \
  --overwrite
```

For `input.type: gazefollow`, the image root is inferred as
`input.path/data_extended`. For `directory`, it is `input.path`.
`--image-root` overrides that inference. The reader also supports H5 samples
that directly contain an `image`, `image_array`, or `rgb` dataset.

Visualization colors:

- green: SAM2-derived object bbox and object score;
- red: Grounding DINO head bbox and head score;
- translucent rotating colors: `object_masks`, when present.

By default labels also include a truncated description and mean depth. Control
them with:

```bash
conda run -n mmgrounding python h5_reader.py --config config.yaml \
  --show-description --show-depth

conda run -n mmgrounding python h5_reader.py --config config.yaml \
  --no-show-description --no-show-depth
```

The current schema does not save detection labels, so the reader uses the known
stream names `object` and `head`. Masks are only available when the H5 file was
generated with `output.save_masks: true`. Missing images or datasets produce
warnings and do not stop processing of later samples.

## Filtering

Mask boxes are clipped to the image and filtered by width, height, pixel area,
and image-area ratio. Lower-priority boxes are removed when they highly overlap
an already retained box or are nearly contained by a substantially larger box.
All thresholds are stored in `config.yaml` and copied into HDF5 metadata.

## Tests

```bash
conda run -n mmgrounding python -m pytest -v
```

Tests cover bbox geometry, filtering, image discovery, HDF5 schema, empty
detections, and a complete pipeline using replaceable lightweight model
adapters.

## Troubleshooting

- `CUDA is not available`: run on a node where the configured CUDA device is
  visible. The preprocessing pipeline is CUDA-only.
- `SAM2 checkpoint not found`: update `sam2.checkpoint`.
- `Unable to load Hugging Face Grounding DINO model`: verify
  `transformers>=4.50,<5`, the model id, network access, and `cache_dir`.
- Offline inference: keep the same `model_id` and `cache_dir`, then set
  `local_files_only: true`. The detector resolves the repository id to its
  local snapshot before calling Transformers, so loading does not request the
  network. `model_id` may also point directly to a local Transformers model
  directory containing `config.json`, `model.safetensors`, processor config,
  and tokenizer files.
- Legacy `Unable to import MMDetection`: verify `head_detector.source_path`
  and the local MMDetection dependencies.
- `Describe Anything requires an available CUDA device`: run on a CUDA-visible
  node and keep `description.device: cuda:0`.
- `Depth Anything requires an available CUDA device`: verify the configured
  CUDA device and checkpoint.
- CUDA out of memory: reduce SAM2 proposal density and DAM
  `max_new_tokens`, process fewer images for debugging, or disable one
  enrichment stage through its CLI flag.
- Existing or corrupt HDF5 output: set a new output path or explicitly enable
  overwrite. The writer does not append schema 2.1 data to old files.
