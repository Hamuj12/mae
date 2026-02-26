from typing import Optional

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

from theo_perception.letterbox_utils import letterbox, map_xyxy_to_original


class YoloV8FDANode(Node):
    def __init__(self) -> None:
        super().__init__('yolov8_fda_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('max_det', 300)
        self.declare_parameter('topic_in', '/camera/color/image_raw')
        self.declare_parameter('topic_out', '/perception/yolov8_fda/detections')
        self.declare_parameter('overlay_topic', '/perception/yolov8_fda/overlay')
        self.declare_parameter('publish_overlay', False)
        self.declare_parameter('use_trt_engine', False)
        self.declare_parameter('qos_reliability', 'best_effort')
        self.declare_parameter('qos_history_depth', 5)
        self.declare_parameter('qos_durability', 'volatile')
        self.declare_parameter('frame_id_override', '')

        self.model_path = str(self.get_parameter('model_path').value)
        self.device = str(self.get_parameter('device').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.max_det = int(self.get_parameter('max_det').value)
        self.topic_in = str(self.get_parameter('topic_in').value)
        self.topic_out = str(self.get_parameter('topic_out').value)
        self.overlay_topic = str(self.get_parameter('overlay_topic').value)
        self.publish_overlay = bool(self.get_parameter('publish_overlay').value)
        self.use_trt_engine = bool(self.get_parameter('use_trt_engine').value)
        self.frame_id_override = str(self.get_parameter('frame_id_override').value)

        if not self.model_path:
            raise RuntimeError('Parameter "model_path" is required and cannot be empty.')

        qos = self._build_qos_profile()

        self.det_pub = self.create_publisher(Detection2DArray, self.topic_out, qos)
        self.overlay_pub = None
        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, self.overlay_topic, qos)

        self.latest_msg: Optional[Image] = None
        self.drop_count = 0
        self._processing = False

        self.sub = self.create_subscription(Image, self.topic_in, self._image_callback, qos)
        self.timer = self.create_timer(0.01, self._process_latest)

        self.class_names = {}
        self.model = self._load_model()
        self._warmup()

        self.get_logger().info(
            f'YOLOv8 FDA node ready. in={self.topic_in} out={self.topic_out} overlay={self.publish_overlay}'
        )

    def _build_qos_profile(self) -> QoSProfile:
        reliability_str = str(self.get_parameter('qos_reliability').value).lower()
        durability_str = str(self.get_parameter('qos_durability').value).lower()
        depth = int(self.get_parameter('qos_history_depth').value)

        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if reliability_str == 'best_effort'
            else ReliabilityPolicy.RELIABLE
        )
        durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if durability_str == 'transient_local'
            else DurabilityPolicy.VOLATILE
        )

        return QoSProfile(
            reliability=reliability,
            durability=durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, depth),
        )

    def _load_model(self):
        from ultralytics import YOLO

        model = YOLO(self.model_path)
        self.class_names = model.names if hasattr(model, 'names') else {}

        if self.use_trt_engine and not self.model_path.endswith('.engine'):
            self.get_logger().warn('use_trt_engine=true but model_path is not .engine; continuing anyway.')

        self.get_logger().info(f'Loaded model: {self.model_path} on {self.device}')
        return model

    def _warmup(self) -> None:
        dummy = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        try:
            _ = self.model.predict(
                source=dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )
            self.get_logger().info('Model warmup complete.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Model warmup failed: {exc}')

    def _image_callback(self, msg: Image) -> None:
        if self.latest_msg is not None:
            self.drop_count += 1
        self.latest_msg = msg

    def _process_latest(self) -> None:
        if self._processing:
            return
        if self.latest_msg is None:
            return

        msg = self.latest_msg
        self.latest_msg = None
        self._processing = True
        try:
            self._run_inference(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Inference loop failure: {exc}')
        finally:
            self._processing = False

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'Unsupported encoding {msg.encoding}. Expected rgb8 or bgr8.')

        channels = 3
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        expected = msg.height * msg.step
        if arr.size < expected:
            raise ValueError(f'Image data size {arr.size} < expected {expected}.')

        arr = arr[:expected].reshape((msg.height, msg.step // channels, channels))
        img = arr[:, : msg.width, :]

        if msg.encoding == 'rgb8':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    def _bgr_to_msg(self, img: np.ndarray, header_stamp: Time, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = header_stamp
        msg.header.frame_id = frame_id
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 3)
        msg.data = img.tobytes()
        return msg

    def _run_inference(self, msg: Image) -> None:
        bgr = self._image_msg_to_bgr(msg)
        letterboxed, meta = letterbox(bgr, self.imgsz)

        results = self.model.predict(
            source=letterboxed,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        det_array = Detection2DArray()
        det_array.header = msg.header
        if self.frame_id_override:
            det_array.header.frame_id = self.frame_id_override

        overlay = bgr.copy() if self.publish_overlay else None

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy_l = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32)
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

            xyxy_o = map_xyxy_to_original(xyxy_l, meta)

            for i in range(xyxy_o.shape[0]):
                x1, y1, x2, y2 = xyxy_o[i].tolist()
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                detection = Detection2D()
                detection.header = det_array.header
                bbox = BoundingBox2D()
                bbox.center.position.x = float(cx)
                bbox.center.position.y = float(cy)
                bbox.size_x = float(w)
                bbox.size_y = float(h)
                detection.bbox = bbox

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(cls_ids[i]))
                hypothesis.hypothesis.score = float(confs[i])
                detection.results.append(hypothesis)
                det_array.detections.append(detection)

                if overlay is not None:
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    cls_name = self.class_names.get(int(cls_ids[i]), str(int(cls_ids[i])))
                    label = f'{cls_name} {confs[i]:.2f}'
                    cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        label,
                        (p1[0], max(0, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        self.det_pub.publish(det_array)

        if overlay is not None and self.overlay_pub is not None:
            frame_id = self.frame_id_override if self.frame_id_override else msg.header.frame_id
            overlay_msg = self._bgr_to_msg(overlay, msg.header.stamp, frame_id)
            self.overlay_pub.publish(overlay_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloV8FDANode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
