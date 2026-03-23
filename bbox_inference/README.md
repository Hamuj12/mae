# MAE BBox Inference

Standalone YOLO inference package for bbox + segmentation, intended for use outside ROS and as a dependency for `nav_ros`.

## What This Package Provides

- Detector object that supports `onnx` and `tensorrt` backends
- Model tasks: `detect` and `segment` (plus pass-through task labels for export)
- Warm-up and timing helpers for latency evaluation
- TensorRT export helper from `.pt` and supported `.onnx`
- Offline test runner that logs detections to CSV/JSON
- Visualization utility that reads CSV and writes overlay images (`bbox_viz/` or `seg_viz/`)

## Install

```bash
# from repo root of this package
cd ~/agnc_repos/mae/bbox_inference

# editable install for rapid iteration
python -m pip install -e .

# install with TensorRT extras
python -m pip install -e ".[tensorrt]"

# without dependencies
python -m pip install -e . --no-deps
```

## CLI entry points

This package exposes command aliases:
- `boxinfer-export-trt`
- `boxinfer-offline-test`
- `boxinfer-visualize-csv`

## Export TensorRT Engine

Detect example:

```bash
boxinfer-export-trt \
  --model-path /path/to/model.pt \
  --engine-out /path/to/model.engine \
  --task detect \
  --imgsz 1024 \
  --device 0 \
  --half
```

Segmentation example (non-square image size):

```bash
boxinfer-export-trt \
  --model-path /path/to/seg_model.pt \
  --engine-out /path/to/seg_model.engine \
  --task segment \
  --imgsz 3000 4096 \
  --device 0 \
  --half
```

## Standalone Offline Test

ONNX detect example:

```bash
boxinfer-offline-test \
  --backend onnx \
  --model-task detect \
  --dataset-dir /path/to/dataset \
  --model-path /path/to/model.onnx \
  --output-dir /path/to/output
```

TensorRT segmentation example:

```bash
boxinfer-offline-test \
  --backend tensorrt \
  --model-task segment \
  --dataset-dir /path/to/dataset \
  --model-path /path/to/model.engine \
  --input-size 3000 4096 \
  --device cuda:0 \
  --output-dir /path/to/output
```

If you get a segmentation fault for ONNX, try the following at `~/path2repo/mae/bbox_inference`:

```bash
python -m pip install -e .
hash -r
```

## Visualize From CSV

```bash
boxinfer-visualize-csv \
  --csv-path /path/to/detections.csv \
  --output-dir /path/to/output \
  --top-k 3
```
