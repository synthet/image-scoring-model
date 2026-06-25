# Training

Fine-tune a YOLO11n-pose model on CUB-200-2011 bird part annotations mapped to our 6-keypoint head schema.

## Prerequisites

- Python ≥ 3.10, package installed: `pip install -e ".[dev]"`
- CUDA-enabled PyTorch for GPU training:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Download pretrained weights

Skip training by downloading the published checkpoint from Hugging Face: **[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
pip install huggingface_hub
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

See also [models/README.md](../models/README.md). To reproduce or improve weights, follow the CUB-200 bootstrap below.

## CUB-200 bootstrap

### 1. Download dataset

[CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) or the fast.ai mirror:

```bash
curl -L -o CUB_200_2011.tgz https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
```

### 2. Convert to YOLO pose format

`training/convert_cub200.py` maps CUB's 15 part landmarks to our schema. CUB provides separate `left eye` and `right eye` parts; bilateral eyes and shoulders are inferred when only one side is annotated.

```bash
python -m training.convert_cub200 \
  --cub-root D:/Datasets/CUB_200_2011 \
  --output data/wildlife_bird \
  --val-ratio 0.15
```

Expected output: ~11,770 train+val image/label pairs (images without visible eye or head are skipped).

Dataset layout:

```text
data/wildlife_bird/
  images/train/   labels/train/
  images/val/     labels/val/
```

Ultralytics config: `training/configs/wildlife_bird.yaml`.

### 3. Fine-tune

**Smoke test** (3 epochs, verifies GPU and data paths):

```bash
python training/train_pose.py --epochs 3 --batch 8 --device 0
```

**Full run** (copies best weights to `models/eye_pose_v0.pt` on completion):

```bash
python training/train_pose.py --epochs 100 --batch 16 --device 0 --output models/eye_pose_v0.pt
```

Training artifacts are written under `runs/pose/`. Weights are not committed to git; the release checkpoint is on [Hugging Face](https://huggingface.co/synthet/eye-pose-v0/tree/main).

### 4. Resume interrupted training

```bash
python training/train_pose.py \
  --resume runs/pose/runs/pose/wildlife_bird-2/weights/last.pt \
  --epochs 100 --batch 16 --device 0 \
  --output models/eye_pose_v0.pt
```

Ultralytics restores optimizer state and epoch counter from `last.pt`. The run continues in the same `save_dir` as the checkpoint.

## Validation

After fine-tuning, score sample images and inspect debug crops:

```bash
python -m eye_quality score \
  data/wildlife_bird/images/val/003.Sooty_Albatross__Sooty_Albatross_0001_1071.jpg \
  --debug-dir debug_crops
```

Check that:

- Eye keypoints land on the iris/pupil region
- Crops are sharp enough to distinguish focus differences within a burst
- Turned-away and tiny subjects get appropriate visibility/failure flags

CUB images are Flickr/studio-heavy. Validate on your own field wildlife bursts before setting culling thresholds.

## Keypoint label format

Each YOLO pose label line:

```text
<class> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <kpt1_v> ... <kpt6_x> <kpt6_y> <kpt6_v>
```

Coordinates are normalized 0–1. Visibility `v`: 0 = absent, 1 = occluded, 2 = visible.

The leading `<class> <cx> <cy> <w> <h>` fields are the **bird bounding box**. The same CUB labels can train a detection-only YOLO model (box without keypoints). See [BIRD_DETECTION.md](BIRD_DETECTION.md).

## Other data sources

| Source | Use |
|--------|-----|
| **3D-POP** | Supplemental pigeon keypoints |
| **Your wildlife library** | Target 500–2,000 labeled eye crops for MVP 2 learned classifier |
| **Custom COCO JSON** | `training/prepare_dataset.py --format coco` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Invalid CUDA device` | Install CUDA PyTorch (see Prerequisites) |
| All images skipped in conversion | Ensure CUB `parts/parts.txt` lists `left eye` / `right eye` (not a single `eye` part) |
| Generic COCO keypoints at inference | Download [eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main) to `models/eye_pose_v0.pt`; check CLI logs for weights path |
| OpenCV errors on Windows | Heuristic scoring falls back to NumPy; localization is unaffected |
