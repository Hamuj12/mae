# MAE BBox Inference

Standalone inference-only YOLO bbox package intended for use outside ROS and as a dependency for `nav_ros`

## What This Package Provides

- Detector object that supports `onnx` and `tensorrt` backends
- Warm-up and timing helpers for latency evaluation
- TensorRT export helper from `.pt` and supported `.onnx`
- Offline test runner that logs detections to CSV/JSON
- Visualization utility that reads CSV and writes overlay images into `bbox_viz/`

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

## Runtime uses CLI entry points for export/test/visualization
This means we use aliases for running specifics scripts. 
Example:  `boxinfer-export-trt ...` instead of `python -m boxinfer.offline_test ...`.

## Export TensorRT Engine

From `.pt`:

```bash
# export TRT engine from Ultralytics .pt
boxinfer-export-trt \
  --model-path /path/to/model.pt \
  --engine-out /path/to/model.engine \
  --imgsz 1024 \
  --device 0 \
  --half
```

## Standalone Offline Test

```bash
# ONNX backend
boxinfer-offline-test \
  --backend onnx \
  --dataset-dir /path/to/dataset \
  --model-path /path/to/model.onnx \
  --output-dir /path/to/output
```
If you get a segmentation fault for onnx, try the following at the root `~/path2repo/mae/bbox_inference`:
```bash
python -m pip install -e . # reinstall package entry point
hash -r # clears bash's command lookup
```

TensorRT backend:

```bash
# TensorRT backend with engine file
boxinfer-offline-test \
  --backend tensorrt \
  --dataset-dir /path/to/dataset \
  --model-path /path/to/model.engine \
  --device cuda:0 \
  --output-dir /path/to/output
```

```

## Visualize From CSV

```bash
# generate overlay images and top-k panel from csv log
boxinfer-visualize-csv \
  --csv-path /path/to/bbox_detections.csv \
  --output-dir /path/to/bbox_viz \
  --top-k 3
```
