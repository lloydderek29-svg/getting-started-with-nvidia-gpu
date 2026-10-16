# Chapter 6: Computer Vision and GPU-Accelerated Object Detection

This directory prepares and audits image datasets, fine-tunes a classifier, trains a YOLO detector, optimizes inference, and builds a webcam application with latency measurements. Shared code is in `common/python/gpubook/vision/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `python/download_pets.py` | Downloads Oxford-IIIT Pet into a folder layout |
| `python/audit_dataset.py` | Finds corrupt files, duplicates, leakage, and imbalance |
| `python/bench_decode.py` | PIL, CPU, and GPU JPEG decoding speed |
| `python/finetune.py` | Two-phase ResNet-50 fine-tuning |
| `python/voc_to_yolo.py` | Converts Pascal VOC XML to YOLO labels |
| `python/check_yolo_dataset.py` | Validates YOLO labels before training |
| `python/yolo_train.py` | Trains and evaluates a YOLO model from a config |
| `python/bench_inference.py` | Batch, precision, and preprocessing sweep |
| `python/webcam_detect.py` | Live detection with per-stage latency |
| `python/accuracy_latency.py` | Accuracy and latency by size and precision |
| `configs/yolo_items.yaml` | YOLO training settings |

## Prerequisites

```bash
pip install -r requirements/torch.txt
pip install -r requirements/vision.txt
pip install -e .
```

Install either `opencv-python` for display windows or `opencv-python-headless` for servers, never both. Set `pythonpath` in `pyproject.toml` to include `chapters/ch06_vision/python` so the tests can import the scripts.

The Ultralytics library and its weights have a copyleft license with a separate commercial option. Read the current terms before building a product on them.

## Classification Workflow

```bash
python chapters/ch06_vision/python/download_pets.py
python chapters/ch06_vision/python/audit_dataset.py data/pets
python chapters/ch06_vision/python/finetune.py --data data/pets
```

The audit prints corrupt, duplicate, and cross-split counts and writes `data/pets/manifest.csv`. It exits nonzero on corrupt files or leakage. The fine-tuning run trains the head for 3 epochs and the full network for 7, prints the most confused class pairs, and saves `results/ch06/pets.pt`, which Chapters 9 and 10 depend on. Expect validation accuracy in the low nineties.

```bash
python chapters/ch06_vision/python/bench_decode.py data/pets
```

## Detection Workflow

Prepare your own data in this layout.

```text
data/items/
  images/train  images/val
  labels/train  labels/val
  items.yaml
```

Convert and validate labels, then smoke test the toolchain on the small sample dataset before a long run.

```bash
python chapters/ch06_vision/python/voc_to_yolo.py \
  xml_dir labels_dir classes.txt

python chapters/ch06_vision/python/check_yolo_dataset.py data/items 4

python - <<'EOF'
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data="coco8.yaml", epochs=3, imgsz=640, device=0)
EOF

python chapters/ch06_vision/python/yolo_train.py \
  chapters/ch06_vision/configs/yolo_items.yaml
```

The training run writes curves and a confusion matrix under `results/ch06/yolo/items_s/` and a `summary.json` with precision, recall, mAP50, and mAP50-95. Split detection data by video or session, never by frame.

## Inference and Application

```bash
python chapters/ch06_vision/python/bench_inference.py \
  --model results/ch06/yolo/items_s/weights/best.pt \
  --video data/clip.mp4

python chapters/ch06_vision/python/webcam_detect.py \
  --model results/ch06/yolo/items_s/weights/best.pt \
  --half --gpu-preprocess
```

Press `q` or Escape to quit the window. To measure the model and not the camera, run against a recorded file with the display off.

```bash
python chapters/ch06_vision/python/webcam_detect.py \
  --model results/ch06/yolo/items_s/weights/best.pt \
  --source data/clip.mp4 --half --gpu-preprocess \
  --no-display --save-json results/ch06/clip_latency.json
```

Expect a summary with FPS and p50, p95, and p99 latency for queue wait, detect, draw, and total. For cameras, queue wait should stay near zero, and a growing value means the detector cannot keep up. The total excludes the camera's internal delay and display time, so quote it as capture-to-result software latency.

```bash
python chapters/ch06_vision/python/accuracy_latency.py \
  --model results/ch06/yolo/items_s/weights/best.pt \
  --data data/items/items.yaml --video data/clip.mp4
```

The table marks configurations on the Pareto front of latency against mAP.

## Tests

```bash
pytest chapters/ch06_vision -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Webcam does not open | Try `--source 1` and check device permissions |
| GPU decode falls back to CPU | Expected for non-JPEG files and some builds |
| Live accuracy much lower than mAP | Domain shift, so add frames from your camera |
| Lag grows during live video | Use a smaller model, lower `--imgsz`, and add `--half` |
| `ImportError` for cv2 windows | Use the non-headless OpenCV package |