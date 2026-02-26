# theo_perception

ROS2 perception package containing `yolov8_fda_node`, a low-latency pub/sub YOLOv8 inference node.

## Features
- Subscribes to `sensor_msgs/msg/Image` (`rgb8` or `bgr8`) at any input resolution.
- Performs square letterbox preprocessing to `imgsz` (default `1024`) with pad color `[114,114,114]`.
- Runs YOLOv8 using Ultralytics model backends (`.pt` and `.engine`).
- Publishes detections as `vision_msgs/msg/Detection2DArray` in **original image pixel coordinates**.
- Optional overlay image publishing with rendered boxes.
- Uses a single-slot latest-frame queue to drop stale frames and reduce callback latency.

## Dependencies
### ROS dependencies
- `rclpy`
- `sensor_msgs`
- `vision_msgs`
- `std_msgs`
- `builtin_interfaces`

Install ROS dependencies (example for Humble):
```bash
sudo apt update
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge
```

> `cv_bridge` is optional in this package; image conversion is performed manually for `rgb8`/`bgr8`.

### Python runtime dependencies
Install into your ROS Python environment:
```bash
python3 -m pip install --upgrade numpy opencv-python ultralytics
```

If running `.pt` models, `torch` is required by Ultralytics. For `.engine` TensorRT models, use a compatible Ultralytics/TensorRT runtime.

## Build
From workspace root:
```bash
colcon build --packages-select theo_perception
source install/setup.bash
```

## Run
Example launch:
```bash
ros2 launch theo_perception yolov8_fda.launch.py \
  model_path:=/path/to/best.pt \
  topic_in:=/camera/color/image_raw \
  namespace:=eowyn
```

For TensorRT engine:
```bash
ros2 launch theo_perception yolov8_fda.launch.py \
  model_path:=/path/to/best.engine \
  namespace:=eowyn
```
Set `use_trt_engine:=true` if you want explicit config signaling (not strictly required when model path ends in `.engine`).

## Topics
- Subscribed
  - `topic_in` (`sensor_msgs/msg/Image`): default `/camera/color/image_raw`
- Published
  - `topic_out` (`vision_msgs/msg/Detection2DArray`): default `/perception/yolov8_fda/detections`
  - `overlay_topic` (`sensor_msgs/msg/Image`): default `/perception/yolov8_fda/overlay` when `publish_overlay=true`

Quick verify:
```bash
ros2 topic echo /perception/yolov8_fda/detections
```

## Parameters
- `model_path` (string, required)
- `device` (string, default `cuda:0`)
- `imgsz` (int, default `1024`)
- `conf` (float, default `0.25`)
- `iou` (float, default `0.45`)
- `max_det` (int, default `300`)
- `topic_in` (string)
- `topic_out` (string)
- `overlay_topic` (string)
- `publish_overlay` (bool)
- `use_trt_engine` (bool)
- `qos_reliability` (`best_effort|reliable`)
- `qos_history_depth` (int)
- `qos_durability` (`volatile|transient_local`)
- `frame_id_override` (string; if set, overrides outgoing frame_id)

## Coordinate mapping (letterbox -> original image)
Given original image size `(w0, h0)` and letterbox size `S`:
1. Scale: `r = min(S/w0, S/h0)`
2. Resized dimensions: `(w1, h1) = (round(w0*r), round(h0*r))`
3. Padding: `dw = S - w1`, `dh = S - h1`, split to left/right and top/bottom.
4. For model outputs in letterbox pixels `(x1_l, y1_l, x2_l, y2_l)`:
   - `x1_o = (x1_l - pad_left) / r`
   - `y1_o = (y1_l - pad_top) / r`
   - `x2_o = (x2_l - pad_left) / r`
   - `y2_o = (y2_l - pad_top) / r`
   - then clip to `[0, w0-1]` and `[0, h0-1]`.

This ensures `Detection2DArray` bounding boxes are reported in the original image pixel frame.

## Integration note
To include this in an existing system launch (for example `src/theo_core/launch/drive_test.launch.py`), add an `IncludeLaunchDescription` to `theo_perception/launch/yolov8_fda.launch.py` and provide `model_path` plus any topic remaps.
