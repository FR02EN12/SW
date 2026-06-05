#!/usr/bin/env python3
import json
import math
import os
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class LEDSignalPerceptionNode(Node):
    """Detect the opponent vehicle 5-LED panel and decode V2V patterns."""

    PATTERN_LABELS: Dict[str, str] = {
        '00000': 'normal',
        '11111': 'car detected',
        '10101': 'game',
        '01010': 'no game',
        '11000': 'rock',
        '10100': 'scissor',
        '10010': 'paper',
    }

    SIGN_BY_MASK: Dict[str, str] = {
        '11111': 'APPROACH',
        '10101': 'RPS_READY',
        '01010': 'WEIGHT_READY',
        '11000': 'RPS_ROCK',
        '10100': 'RPS_SCISSORS',
        '10010': 'RPS_PAPER',
    }

    def __init__(self):
        super().__init__('led_signal')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('process_fps', 15.0)
        self.declare_parameter('temporal_window', 5)
        self.declare_parameter('led_count', 5)

        # Legacy parameters kept so existing launch/config files remain valid.
        self.declare_parameter('brightness_threshold', 180)
        self.declare_parameter('saturation_threshold', 35)
        self.declare_parameter('min_pixel_threshold', 40)
        self.declare_parameter('min_blob_area_px', 8)
        self.declare_parameter('max_blob_area_px', 2500)
        self.declare_parameter('slot_sample_radius_px', 7)
        self.declare_parameter('slot_on_ratio_threshold', 0.08)

        # Parameters copied from the verified OpenCV classifier.
        self.declare_parameter('use_fisheye_undistort', True)
        self.declare_parameter(
            'camera_info_yaml',
            os.path.expanduser('~/fsd_ws/src/perception_py/config/fisheye.yaml'),
        )
        self.declare_parameter('fisheye_calibration_path', '')
        self.declare_parameter('fisheye_balance', 0.0)
        self.declare_parameter('dark_box_thresh', 70)
        self.declare_parameter('min_box_area', 10000)
        self.declare_parameter('min_box_aspect', 1.2)
        self.declare_parameter('max_box_aspect', 6.8)
        self.declare_parameter('min_box_fill_ratio', 0.42)
        self.declare_parameter('led_thresh', 170)
        self.declare_parameter('min_led_area', 5)
        self.declare_parameter('max_led_area', 2500)
        self.declare_parameter('max_led_y_diff', 25)
        self.declare_parameter('slot_tol_pixels', 18)
        self.declare_parameter('slot_tol_ratio', 0.38)
        self.declare_parameter('min_leds_for_box', 2)

        self.declare_parameter('opponent_latch_sec', 2.0)
        self.declare_parameter('led_panel_width_m', 0.12)
        self.declare_parameter('focal_length_px', 1400.14208)

        self._image_topic = str(self.get_parameter('image_topic').value)
        self._process_fps = float(self.get_parameter('process_fps').value)
        self._temporal_n = int(self.get_parameter('temporal_window').value)
        self._led_count = int(self.get_parameter('led_count').value)

        self._use_fisheye = bool(self.get_parameter('use_fisheye_undistort').value)
        self._fisheye_balance = float(self.get_parameter('fisheye_balance').value)
        self._dark_box_thresh = int(self.get_parameter('dark_box_thresh').value)
        self._min_box_area = int(self.get_parameter('min_box_area').value)
        self._min_box_aspect = float(self.get_parameter('min_box_aspect').value)
        self._max_box_aspect = float(self.get_parameter('max_box_aspect').value)
        self._min_box_fill_ratio = float(self.get_parameter('min_box_fill_ratio').value)
        self._led_thresh = int(self.get_parameter('led_thresh').value)
        self._min_led_area = int(self.get_parameter('min_led_area').value)
        self._max_led_area = int(self.get_parameter('max_led_area').value)
        self._max_led_y_diff = float(self.get_parameter('max_led_y_diff').value)
        self._slot_tol_pixels = int(self.get_parameter('slot_tol_pixels').value)
        self._slot_tol_ratio = float(self.get_parameter('slot_tol_ratio').value)
        self._min_leds_for_box = int(self.get_parameter('min_leds_for_box').value)

        self._opponent_latch = float(self.get_parameter('opponent_latch_sec').value)
        self._panel_width_m = float(self.get_parameter('led_panel_width_m').value)
        self._focal_px = float(self.get_parameter('focal_length_px').value)

        self._bridge = CvBridge()
        self._latest_image: Optional[np.ndarray] = None
        self._mask_votes: deque = deque(maxlen=max(self._temporal_n, 1))
        self._slot_locked = False
        self._fixed_strip_rect: Optional[Tuple[int, int, int, int]] = None
        self._slot_x_positions: Optional[np.ndarray] = None
        self._slot_y_position: Optional[float] = None
        self._opponent_latch_until = 0.0
        self._last_distance_m = 1.0
        self._last_angle_rad = 0.0
        self._last_bbox: List[float] = []
        self._camera_calibration = self._load_camera_calibration()
        self._undistort_maps: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, self._image_topic, self._image_cb, sensor_qos)

        self.pub_led_state = self.create_publisher(
            String, '/perception/opponent_led_state', 10)
        self.pub_v2v_sign = self.create_publisher(
            String, '/perception/opponent_v2v_sign', 10)
        self.pub_opponent = self.create_publisher(
            String, '/perception/opponent_vehicle', 10)

        self._fps_window_start = time.monotonic()
        self._fps_window_count = 0

        period = 1.0 / max(self._process_fps, 0.1)
        self.create_timer(period, self._process_cb)

        self.get_logger().info(
            'LEDSignalPerceptionNode initialised with black-box 5-LED classifier')

    def _image_cb(self, msg: Image) -> None:
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    def _process_cb(self) -> None:
        now = time.monotonic()

        if self._latest_image is None:
            self._publish('00000', 'UNKNOWN', 'no image', 0.0, False, now)
            return

        frame = self._latest_image.copy()
        frame = self._fisheye_undistort(frame)

        raw_mask, raw_label, box_detected, led_count, geometry_box = self._read_pattern(frame)

        self._mask_votes.append(raw_mask)
        mask, confidence = self._stable_mask()
        sign = self._classify_mask(mask)
        label = self.PATTERN_LABELS.get(mask, raw_label)

        if box_detected:
            self._opponent_latch_until = now + self._opponent_latch

        self._update_geometry(frame, geometry_box)
        self._publish(mask, sign, label, confidence, box_detected, now, led_count)

    def _load_camera_calibration(self) -> Optional[Dict[str, object]]:
        if not self._use_fisheye:
            return None

        yaml_configured = str(self.get_parameter('camera_info_yaml').value).strip()
        npz_configured = str(self.get_parameter('fisheye_calibration_path').value).strip()

        yaml_candidates = [
            yaml_configured,
            os.path.expanduser('~/fsd_ws/src/perception_py/config/fisheye.yaml'),
        ]
        for path in [p for p in yaml_candidates if p]:
            expanded = os.path.abspath(os.path.expanduser(path))
            if not os.path.isfile(expanded):
                continue
            try:
                calibration = self._load_camera_yaml(expanded)
                self.get_logger().info(
                    f'Loaded camera calibration yaml: {expanded} '
                    f'({calibration["model"]})')
                return calibration
            except Exception as exc:
                self.get_logger().warn(
                    f'Failed to load camera calibration yaml {expanded}: {exc}')
                return None

        npz_candidates = [
            npz_configured,
            'fisheye_calibration.npz',
            os.path.expanduser('~/fsd_ws/fisheye_calibration.npz'),
            os.path.expanduser('~/fsd_ws/src/perception_py/config/fisheye_calibration.npz'),
        ]
        for path in [p for p in npz_candidates if p]:
            expanded = os.path.abspath(os.path.expanduser(path))
            if not os.path.isfile(expanded):
                continue
            try:
                calibration = self._load_camera_npz(expanded)
                self.get_logger().info(f'Loaded fisheye npz calibration: {expanded}')
                return calibration
            except Exception as exc:
                self.get_logger().warn(
                    f'Failed to load fisheye npz calibration {expanded}: {exc}')
                return None

        self.get_logger().warn(
            'camera calibration yaml/npz not found; LED classifier will use raw frames')
        return None

    def _load_camera_yaml(self, path: str) -> Dict[str, object]:
        with open(path, 'r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)

        width = int(data['image_width'])
        height = int(data['image_height'])
        k = np.asarray(data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
        d = np.asarray(data['distortion_coefficients']['data'], dtype=np.float64)
        r = np.asarray(
            data.get('rectification_matrix', {}).get('data', np.eye(3).reshape(-1)),
            dtype=np.float64,
        ).reshape(3, 3)

        projection = data.get('projection_matrix', {})
        p = None
        if projection and 'data' in projection:
            p = np.asarray(projection['data'], dtype=np.float64).reshape(3, 4)

        return {
            'model': str(data.get('distortion_model', 'plumb_bob')),
            'k': k,
            'd': d,
            'r': r,
            'p': p,
            'dim': (width, height),
            'source': 'yaml',
        }

    def _load_camera_npz(self, path: str) -> Dict[str, object]:
        data = np.load(path)
        k = np.asarray(data['K'], dtype=np.float64)
        d = np.asarray(data['D'], dtype=np.float64)
        dim_raw = data['DIM']
        return {
            'model': 'fisheye',
            'k': k,
            'd': d,
            'r': np.eye(3, dtype=np.float64),
            'p': None,
            'dim': (int(dim_raw[0]), int(dim_raw[1])),
            'source': 'npz',
        }

    def _fisheye_undistort(self, frame: np.ndarray) -> np.ndarray:
        if self._camera_calibration is None:
            return frame

        calibration = self._camera_calibration
        k = calibration['k']
        d = calibration['d']
        dim = calibration['dim']
        model = str(calibration['model']).lower()
        h, w = frame.shape[:2]
        key = (w, h)

        if key not in self._undistort_maps:
            scaled_k = self._scale_camera_matrix(k, dim, w, h)
            if model in ('fisheye', 'equidistant'):
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    scaled_k, d, (w, h), np.eye(3), balance=self._fisheye_balance)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    scaled_k, d, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
            else:
                r = calibration['r']
                p = calibration['p']
                new_k = self._scale_projection_matrix(p, dim, w, h) if p is not None else scaled_k
                map1, map2 = cv2.initUndistortRectifyMap(
                    scaled_k,
                    d,
                    r,
                    new_k,
                    (w, h),
                    cv2.CV_16SC2,
                )
            self._undistort_maps[key] = (map1, map2)

        map1, map2 = self._undistort_maps[key]
        undistorted = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        uh, uw = undistorted.shape[:2]
        crop_x1 = int(uw * 0.12)
        crop_x2 = int(uw * 0.88)
        crop_y1 = int(uh * 0.18)
        crop_y2 = int(uh * 0.82)
        return undistorted[crop_y1:crop_y2, crop_x1:crop_x2]

    @staticmethod
    def _scale_camera_matrix(
        k: np.ndarray,
        dim: Tuple[int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        scaled = k.copy()
        sx = width / float(dim[0])
        sy = height / float(dim[1])
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] *= sx
        scaled[1, 2] *= sy
        return scaled

    @staticmethod
    def _scale_projection_matrix(
        p: np.ndarray,
        dim: Tuple[int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        scaled = p[:3, :3].copy()
        sx = width / float(dim[0])
        sy = height / float(dim[1])
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] *= sx
        scaled[1, 2] *= sy
        return scaled

    def _read_pattern(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[str, str, bool, int, Optional[Tuple[int, int, int, int]]]:
        if self._slots_are_valid():
            return self._read_locked_strip(frame_bgr)

        box, _dark_mask = self._find_black_box(frame_bgr, self._dark_box_thresh)
        if box is None:
            return '00000', 'waiting box', False, 0, None

        x, y, w, h = box
        big_roi = frame_bgr[y:y + h, x:x + w].copy()
        if big_roi.size == 0:
            return '00000', 'waiting box', False, 0, box

        gray_big = cv2.cvtColor(big_roi, cv2.COLOR_BGR2GRAY)
        gray_big = cv2.GaussianBlur(gray_big, (5, 5), 0)

        big_leds, _led_mask = self._detect_leds_in_box(gray_big, self._led_thresh)
        big_leds = self._refine_led_candidates(big_leds)
        if len(big_leds) != self._led_count:
            return '00000', f'waiting all-on, detected {len(big_leds)}', True, len(big_leds), box

        strip_rect = self._make_tight_strip_from_leds(big_roi.shape, big_leds, x, y)
        strip_rect = self._clamp_strip_rect(strip_rect, frame_bgr.shape)
        if strip_rect is None:
            return '00000', 'waiting strip ROI', True, len(big_leds), box

        sx1, sy1, sx2, sy2 = strip_rect
        strip_roi = frame_bgr[sy1:sy2, sx1:sx2].copy()
        if strip_roi.size == 0:
            return '00000', 'waiting strip ROI', True, len(big_leds), box

        gray_strip = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
        gray_strip = cv2.GaussianBlur(gray_strip, (5, 5), 0)
        strip_leds, _strip_mask = self._detect_leds_in_box(gray_strip, self._led_thresh)
        strip_leds = self._refine_led_candidates(strip_leds)

        if len(strip_leds) != self._led_count:
            label = f'waiting clean 5 LEDs in strip, detected {len(strip_leds)}'
            return '00000', label, True, len(strip_leds), (sx1, sy1, sx2 - sx1, sy2 - sy1)

        self._fixed_strip_rect = strip_rect
        self._slot_x_positions = np.asarray(
            sorted([led[0] for led in strip_leds]), dtype=np.float32)
        self._slot_y_position = float(np.median([led[1] for led in strip_leds]))
        self._slot_locked = True
        self.get_logger().info(
            f'LED fixed strip locked: rect={self._fixed_strip_rect}, '
            f'slots={self._slot_x_positions.tolist()}')
        return '11111', 'car detected', True, len(strip_leds), (sx1, sy1, sx2 - sx1, sy2 - sy1)

    def _read_locked_strip(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[str, str, bool, int, Optional[Tuple[int, int, int, int]]]:
        strip_rect = self._clamp_strip_rect(self._fixed_strip_rect, frame_bgr.shape)
        if strip_rect is None:
            self._slot_locked = False
            self._fixed_strip_rect = None
            self._slot_x_positions = None
            return '00000', 'waiting box', False, 0, None

        x1, y1, x2, y2 = strip_rect
        strip_roi = frame_bgr[y1:y2, x1:x2].copy()
        geometry_box = (x1, y1, x2 - x1, y2 - y1)
        if strip_roi.size == 0:
            return '00000', 'normal', True, 0, geometry_box

        gray_strip = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2GRAY)
        gray_strip = cv2.GaussianBlur(gray_strip, (5, 5), 0)
        strip_leds, _led_mask = self._detect_leds_in_box(gray_strip, self._led_thresh)
        strip_leds = self._refine_led_candidates(strip_leds)

        led_centers_x = sorted([led[0] for led in strip_leds])
        pattern = self._pattern_from_detected_leds(led_centers_x, self._slot_x_positions)
        mask = ''.join(str(bit) for bit in pattern)
        return mask, self.PATTERN_LABELS.get(mask, 'unknown'), True, len(strip_leds), geometry_box

    def _make_tight_strip_from_leds(
        self,
        big_box_shape: Tuple[int, ...],
        leds: List[Tuple[int, int, int, int, int, int, float]],
        box_x: int,
        box_y: int,
    ) -> Tuple[int, int, int, int]:
        bh_big, bw_big = big_box_shape[:2]

        led_x1 = min(led[2] for led in leds)
        led_y1 = min(led[3] for led in leds)
        led_x2 = max(led[2] + led[4] for led in leds)
        led_y2 = max(led[3] + led[5] for led in leds)

        led_span_x = max(1, led_x2 - led_x1)
        led_span_y = max(1, led_y2 - led_y1)

        pad_x = int(led_span_x * 0.18)
        pad_top = int(led_span_y * 1.10)
        pad_bottom = int(led_span_y * 0.45)

        sx1 = max(0, led_x1 - pad_x)
        sx2 = min(bw_big, led_x2 + pad_x)
        sy1 = max(0, led_y1 - pad_top)
        sy2 = min(bh_big, led_y2 + pad_bottom)

        return box_x + sx1, box_y + sy1, box_x + sx2, box_y + sy2

    def _clamp_strip_rect(
        self,
        rect: Optional[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        if rect is None:
            return None

        height, width = frame_shape[:2]
        x1, y1, x2, y2 = rect
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(1, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        y2 = max(1, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _slots_are_valid(self) -> bool:
        return (
            self._slot_locked and
            self._fixed_strip_rect is not None and
            self._slot_x_positions is not None
        )

    def _detect_leds_in_box(
        self,
        gray_box: np.ndarray,
        thresh_val: int,
    ) -> Tuple[List[Tuple[int, int, int, int, int, int, float]], np.ndarray]:
        _, led_mask = cv2.threshold(gray_box, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            led_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leds: List[Tuple[int, int, int, int, int, int, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_led_area or area > self._max_led_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            leds.append((cx, cy, x, y, w, h, area))

        return sorted(leds, key=lambda item: item[0]), led_mask

    def _refine_led_candidates(
        self,
        box_leds: List[Tuple[int, int, int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, int, int, float]]:
        if not box_leds:
            return []

        ys = np.asarray([led[1] for led in box_leds], dtype=np.float32)
        y_med = float(np.median(ys))

        filtered = [led for led in box_leds if abs(led[1] - y_med) <= self._max_led_y_diff]
        filtered = sorted(filtered, key=lambda item: item[0])

        if len(filtered) <= self._led_count:
            return filtered

        best_group = None
        best_score = 1e18
        for idx in range(len(filtered) - self._led_count + 1):
            group = filtered[idx:idx + self._led_count]
            xs = np.asarray([led[0] for led in group], dtype=np.float32)
            diffs = np.diff(xs)
            if np.any(diffs <= 0):
                continue

            score = float(np.std(diffs))
            if score < best_score:
                best_score = score
                best_group = group

        return best_group if best_group is not None else filtered[:self._led_count]

    def _find_black_box(
        self,
        frame_bgr: np.ndarray,
        thresh_val: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel_close = np.ones((9, 9), np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = frame_bgr.shape[:2]
        best_box = None
        best_score = -1e18

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_box_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(max(h, 1))
            if aspect < self._min_box_aspect or aspect > self._max_box_aspect:
                continue

            rect_area = w * h
            fill_ratio = area / float(max(rect_area, 1))
            if fill_ratio < self._min_box_fill_ratio:
                continue

            roi = frame_bgr[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            leds_in_box, _ = self._detect_leds_in_box(gray_roi, self._led_thresh)
            leds_in_box = self._refine_led_candidates(leds_in_box)

            led_count = len(leds_in_box)
            if led_count < self._min_leds_for_box:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            center_penalty = abs(cx - width / 2.0) + abs(cy - height / 2.0)
            score = led_count * 100000.0 + area - 2.0 * center_penalty

            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        return best_box, dark_mask

    def _pattern_from_detected_leds(
        self,
        led_centers_x: List[int],
        slot_x_positions: np.ndarray,
    ) -> Tuple[int, int, int, int, int]:
        slots = [0, 0, 0, 0, 0]
        if slot_x_positions is None:
            return tuple(slots)

        slot_xs = np.sort(np.asarray(slot_x_positions, dtype=np.float32))
        led_centers = sorted([float(x) for x in led_centers_x])

        if len(slot_xs) >= 2:
            step = float(np.median(np.diff(slot_xs)))
        else:
            step = 30.0
        tol = max(self._slot_tol_pixels, int(step * self._slot_tol_ratio))

        used_slots = set()
        for lx in led_centers:
            dists = np.abs(slot_xs - lx)
            nearest_slot = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_slot])

            if nearest_dist <= tol and nearest_slot not in used_slots:
                slots[nearest_slot] = 1
                used_slots.add(nearest_slot)

        return tuple(slots)

    def _stable_mask(self) -> Tuple[str, float]:
        if not self._mask_votes:
            return '00000', 0.0
        counts = Counter(self._mask_votes)
        mask, count = counts.most_common(1)[0]
        return mask, float(count) / float(len(self._mask_votes))

    def _classify_mask(self, mask: str) -> str:
        if mask in self.SIGN_BY_MASK:
            return self.SIGN_BY_MASK[mask]
        if '1' in mask:
            return 'YIELD_VALUE'
        return 'UNKNOWN'

    def _update_geometry(
        self,
        frame: np.ndarray,
        box: Optional[Tuple[int, int, int, int]],
    ) -> None:
        if box is None:
            return

        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]
        self._last_bbox = [
            round(float(x), 1),
            round(float(y), 1),
            round(float(x + w), 1),
            round(float(y + h), 1),
        ]

        if self._slot_x_positions is not None:
            xs = self._slot_x_positions.astype(np.float32)
            span_px = float(np.max(xs) - np.min(xs))
            center_x = float(x) + float(np.mean(xs))
            if span_px > 1.0:
                self._last_distance_m = (self._panel_width_m * self._focal_px) / span_px
        else:
            center_x = float(x) + w / 2.0

        self._last_angle_rad = math.atan2(center_x - frame_w / 2.0, self._focal_px)

    def _publish(
        self,
        mask: str,
        sign: str,
        label: str,
        confidence: float,
        box_detected: bool,
        now: float,
        led_count: int = 0,
    ) -> None:
        detected = box_detected or now < self._opponent_latch_until
        value = int(mask, 2) if len(mask) == self._led_count else 0
        legacy = String()
        legacy.data = sign
        self.pub_led_state.publish(legacy)

        sign_msg = String()
        sign_msg.data = json.dumps({
            'detected': detected,
            'sign': sign,
            'label': label,
            'mask': mask,
            'value': value,
            'confidence': round(confidence, 3),
            'distance_m': round(self._last_distance_m, 3),
            'angle_rad': round(self._last_angle_rad, 4),
            'box_detected': box_detected,
            'slot_locked': self._slots_are_valid(),
            'led_count': led_count,
        })
        self.pub_v2v_sign.publish(sign_msg)

        opp_msg = String()
        opp_msg.data = json.dumps({
            'detected': detected,
            'distance_m': round(self._last_distance_m, 3) if detected else 0.0,
            'angle_rad': round(self._last_angle_rad, 4) if detected else 0.0,
            'bbox': self._last_bbox if detected else [],
            'confidence': round(confidence, 3) if detected else 0.0,
            'source': 'led_panel',
            'led_mask': mask,
            'led_label': label,
            'v2v_sign': sign,
            'box_detected': box_detected,
            'slot_locked': self._slots_are_valid(),
            'led_count': led_count,
        })
        self.pub_opponent.publish(opp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LEDSignalPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except BaseException:
                pass


if __name__ == '__main__':
    main()
