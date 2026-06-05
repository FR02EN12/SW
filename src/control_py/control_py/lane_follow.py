#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class LaneFollowControlNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_follow')

        self.declare_parameter('center_error_m_topic', '/lane_error_center_m')
        self.declare_parameter('right_error_m_topic', '/lane_error_right_m')
        self.declare_parameter('heading_error_topic', '/lane_heading_error')
        self.declare_parameter('lane_status_topic', '/lane_status')
        self.declare_parameter('lane_width_m_topic', '/fused/lane_width_m')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_lane')
        self.declare_parameter('centerline_path_topic', '/fused/centerline_path')
        self.declare_parameter('scan_topic', '/scan')

        self.declare_parameter('control_hz', 20.0)

        self.declare_parameter('kp_m', 0.22)
        self.declare_parameter('k_heading', 0.03)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('max_ang_z', 0.08)

        self.declare_parameter('nominal_speed', 0.028)
        self.declare_parameter('min_speed', 0.018)
        self.declare_parameter('slow_error_m', 0.10)
        self.declare_parameter('hard_stop_error_m', 0.25)
        self.declare_parameter('rotate_in_place_on_large_error', False)

        self.declare_parameter('lane_timeout_sec', 0.35)
        self.declare_parameter('heading_timeout_sec', 0.35)
        self.declare_parameter('stop_on_invalid_lane', True)
        self.declare_parameter('require_heading', False)
        self.declare_parameter('hold_last_heading_sec', 0.20)
        self.declare_parameter('default_follow_mode', 'center')
        self.declare_parameter('robot_width_m', 0.19)
        self.declare_parameter('width_margin_m', 0.00)
        self.declare_parameter('width_hysteresis_m', 0.01)
        self.declare_parameter('width_timeout_sec', 0.4)
        self.declare_parameter('hold_last_width_sec', 1.0)
        self.declare_parameter('min_valid_width_m', 0.05)

        self.declare_parameter('camera_forward_offset_m', 0.0)
        self.declare_parameter('lookahead_m', 0.20)
        self.declare_parameter('lookahead_gain', 0.0)
        self.declare_parameter('use_lookahead', False)

        self.declare_parameter('live_center_deadband_m', 0.003)
        self.declare_parameter('live_heading_deadband_rad', 0.010)

        # motion_node-style multi-point pursuit over the OpenCV centerline path.
        self.declare_parameter('use_path_pursuit', True)
        self.declare_parameter('path_timeout_sec', 0.35)
        self.declare_parameter('path_min_points', 3)
        self.declare_parameter('path_lookahead_min_m', 0.05)
        self.declare_parameter('path_lookahead_max_m', 1.20)
        self.declare_parameter('path_reanchor_far_m', 0.12)
        self.declare_parameter('path_target_y_limit_m', 0.45)
        self.declare_parameter('path_focus_m', 0.42)
        self.declare_parameter('path_focus_sigma_m', 0.25)
        self.declare_parameter('path_gain_distance_m', 0.42)
        self.declare_parameter('path_preview_distance_m', 0.90)
        self.declare_parameter('straight_pursuit_gain', 0.08)
        self.declare_parameter('straight_heading_gain', 0.08)
        self.declare_parameter('straight_boost_pursuit_gain', 0.40)
        self.declare_parameter('straight_boost_heading_gain', 0.40)
        self.declare_parameter('curve_pursuit_gain', 1.62)
        self.declare_parameter('curve_heading_gain', 2.22)
        self.declare_parameter('curve_switch_ratio', 0.50)
        self.declare_parameter('heading_start_deg', 4.0)
        self.declare_parameter('heading_full_deg', 20.0)
        self.declare_parameter('straight_boost_on_deg', 5.0)
        self.declare_parameter('straight_boost_full_deg', 10.0)
        self.declare_parameter('straight_boost_off_deg', 0.0)
        self.declare_parameter('straight_boost_max_level', 1.8)
        self.declare_parameter('straight_boost_blend_alpha', 0.45)
        self.declare_parameter('preview_curv_start', 0.018)
        self.declare_parameter('preview_curv_full', 0.080)
        self.declare_parameter('preview_lateral_start_m', 0.04)
        self.declare_parameter('preview_lateral_full_m', 0.18)
        self.declare_parameter('angle_smooth_alpha', 0.55)
        self.declare_parameter('angle_max_step_deg', 12.0)
        self.declare_parameter('angle_deadband_deg', 0.6)
        self.declare_parameter('angle_to_ang_z_gain', 0.017453292519943295)
        self.declare_parameter('path_angular_sign', -1.0)
        self.declare_parameter('use_lidar_guard', True)
        self.declare_parameter('scan_timeout_sec', 0.30)
        self.declare_parameter('lidar_range_min_m', 0.16)
        self.declare_parameter('lidar_range_max_m', 3.50)
        self.declare_parameter('scan_yaw_offset_rad', 0.0)
        self.declare_parameter('obstacle_direction', 'forward')
        self.declare_parameter('scan_ignore_angle_min_deg', -30.0)
        self.declare_parameter('scan_ignore_angle_max_deg', 30.0)
        self.declare_parameter('scan_ignore_range_max_m', 0.35)
        self.declare_parameter('obstacle_x_min_m', 0.05)
        self.declare_parameter('obstacle_stop_x_m', 0.32)
        self.declare_parameter('obstacle_slow_x_m', 0.75)
        self.declare_parameter('obstacle_corridor_half_width_m', 0.16)
        self.declare_parameter('obstacle_corridor_padding_m', 0.06)
        self.declare_parameter('obstacle_min_points', 2)
        self.declare_parameter('obstacle_slow_speed_scale', 0.45)

        self.center_error_m_topic = str(self.get_parameter('center_error_m_topic').value)
        self.right_error_m_topic = str(self.get_parameter('right_error_m_topic').value)
        self.heading_error_topic = str(self.get_parameter('heading_error_topic').value)
        self.lane_status_topic = str(self.get_parameter('lane_status_topic').value)
        self.lane_width_m_topic = str(self.get_parameter('lane_width_m_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.centerline_path_topic = str(self.get_parameter('centerline_path_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)

        self.control_hz = float(self.get_parameter('control_hz').value)
        self.kp_m = float(self.get_parameter('kp_m').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.steering_sign = float(self.get_parameter('steering_sign').value)
        self.max_ang_z = float(self.get_parameter('max_ang_z').value)

        self.nominal_speed = float(self.get_parameter('nominal_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.slow_error_m = float(self.get_parameter('slow_error_m').value)
        self.hard_stop_error_m = float(self.get_parameter('hard_stop_error_m').value)
        self.rotate_in_place_on_large_error = bool(self.get_parameter('rotate_in_place_on_large_error').value)

        self.lane_timeout_sec = float(self.get_parameter('lane_timeout_sec').value)
        self.heading_timeout_sec = float(self.get_parameter('heading_timeout_sec').value)
        self.stop_on_invalid_lane = bool(self.get_parameter('stop_on_invalid_lane').value)
        self.require_heading = bool(self.get_parameter('require_heading').value)
        self.hold_last_heading_sec = float(self.get_parameter('hold_last_heading_sec').value)
        self.default_follow_mode = str(self.get_parameter('default_follow_mode').value).strip().lower()
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.width_margin_m = float(self.get_parameter('width_margin_m').value)
        self.width_hysteresis_m = float(self.get_parameter('width_hysteresis_m').value)
        self.width_timeout_sec = float(self.get_parameter('width_timeout_sec').value)
        self.hold_last_width_sec = float(self.get_parameter('hold_last_width_sec').value)
        self.min_valid_width_m = float(self.get_parameter('min_valid_width_m').value)
        self.min_passable_width_m = self.robot_width_m + self.width_margin_m

        self.camera_forward_offset_m = float(self.get_parameter('camera_forward_offset_m').value)
        self.lookahead_m = float(self.get_parameter('lookahead_m').value)
        self.lookahead_gain = float(self.get_parameter('lookahead_gain').value)
        self.use_lookahead = bool(self.get_parameter('use_lookahead').value)

        self.live_center_deadband_m = float(self.get_parameter('live_center_deadband_m').value)
        self.live_heading_deadband_rad = float(self.get_parameter('live_heading_deadband_rad').value)

        self.use_path_pursuit = bool(self.get_parameter('use_path_pursuit').value)
        self.path_timeout_sec = float(self.get_parameter('path_timeout_sec').value)
        self.path_min_points = max(2, int(self.get_parameter('path_min_points').value))
        self.path_lookahead_min_m = float(self.get_parameter('path_lookahead_min_m').value)
        self.path_lookahead_max_m = float(self.get_parameter('path_lookahead_max_m').value)
        self.path_reanchor_far_m = float(self.get_parameter('path_reanchor_far_m').value)
        self.path_target_y_limit_m = float(self.get_parameter('path_target_y_limit_m').value)
        self.path_focus_m = float(self.get_parameter('path_focus_m').value)
        self.path_focus_sigma_m = float(self.get_parameter('path_focus_sigma_m').value)
        self.path_gain_distance_m = float(self.get_parameter('path_gain_distance_m').value)
        self.path_preview_distance_m = float(self.get_parameter('path_preview_distance_m').value)
        self.straight_pursuit_gain = float(self.get_parameter('straight_pursuit_gain').value)
        self.straight_heading_gain = float(self.get_parameter('straight_heading_gain').value)
        self.straight_boost_pursuit_gain = float(self.get_parameter('straight_boost_pursuit_gain').value)
        self.straight_boost_heading_gain = float(self.get_parameter('straight_boost_heading_gain').value)
        self.curve_pursuit_gain = float(self.get_parameter('curve_pursuit_gain').value)
        self.curve_heading_gain = float(self.get_parameter('curve_heading_gain').value)
        self.curve_switch_ratio = float(self.get_parameter('curve_switch_ratio').value)
        self.heading_start_deg = float(self.get_parameter('heading_start_deg').value)
        self.heading_full_deg = float(self.get_parameter('heading_full_deg').value)
        self.straight_boost_on_deg = float(self.get_parameter('straight_boost_on_deg').value)
        self.straight_boost_full_deg = float(self.get_parameter('straight_boost_full_deg').value)
        self.straight_boost_off_deg = float(self.get_parameter('straight_boost_off_deg').value)
        self.straight_boost_max_level = float(self.get_parameter('straight_boost_max_level').value)
        self.straight_boost_blend_alpha = float(self.get_parameter('straight_boost_blend_alpha').value)
        self.preview_curv_start = float(self.get_parameter('preview_curv_start').value)
        self.preview_curv_full = float(self.get_parameter('preview_curv_full').value)
        self.preview_lateral_start_m = float(self.get_parameter('preview_lateral_start_m').value)
        self.preview_lateral_full_m = float(self.get_parameter('preview_lateral_full_m').value)
        self.angle_smooth_alpha = float(self.get_parameter('angle_smooth_alpha').value)
        self.angle_max_step_deg = float(self.get_parameter('angle_max_step_deg').value)
        self.angle_deadband_deg = float(self.get_parameter('angle_deadband_deg').value)
        self.angle_to_ang_z_gain = float(self.get_parameter('angle_to_ang_z_gain').value)
        self.path_angular_sign = float(self.get_parameter('path_angular_sign').value)
        self.use_lidar_guard = bool(self.get_parameter('use_lidar_guard').value)
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)
        self.lidar_range_min_m = float(self.get_parameter('lidar_range_min_m').value)
        self.lidar_range_max_m = float(self.get_parameter('lidar_range_max_m').value)
        self.scan_yaw_offset_rad = float(self.get_parameter('scan_yaw_offset_rad').value)
        self.obstacle_direction = str(self.get_parameter('obstacle_direction').value).strip().lower()
        self.scan_ignore_angle_min_deg = float(self.get_parameter('scan_ignore_angle_min_deg').value)
        self.scan_ignore_angle_max_deg = float(self.get_parameter('scan_ignore_angle_max_deg').value)
        self.scan_ignore_range_max_m = float(self.get_parameter('scan_ignore_range_max_m').value)
        self.obstacle_x_min_m = float(self.get_parameter('obstacle_x_min_m').value)
        self.obstacle_stop_x_m = float(self.get_parameter('obstacle_stop_x_m').value)
        self.obstacle_slow_x_m = float(self.get_parameter('obstacle_slow_x_m').value)
        self.obstacle_corridor_half_width_m = float(self.get_parameter('obstacle_corridor_half_width_m').value)
        self.obstacle_corridor_padding_m = float(self.get_parameter('obstacle_corridor_padding_m').value)
        self.obstacle_min_points = max(1, int(self.get_parameter('obstacle_min_points').value))
        self.obstacle_slow_speed_scale = float(self.get_parameter('obstacle_slow_speed_scale').value)

        self.center_err_m: Optional[float] = None
        self.right_err_m: Optional[float] = None
        self.heading_err: Optional[float] = None
        self.path_xs: Optional[np.ndarray] = None
        self.path_ys: Optional[np.ndarray] = None
        self.last_path_stamp: Optional[float] = None
        self.lane_width_m: Optional[float] = None
        self.last_width_stamp: Optional[float] = None
        self.last_good_width_m: Optional[float] = None
        self.last_good_width_stamp: Optional[float] = None
        self.prev_path_angle_deg: float = 0.0
        self.straight_boost_level: float = 0.0
        self.last_scan_stamp: Optional[float] = None
        self.lidar_stop: bool = False
        self.lidar_slow: bool = False
        self.nearest_obstacle_x_m: Optional[float] = None

        self.last_good_heading_err: float = 0.0
        self.last_good_heading_stamp: Optional[float] = None

        self.lane_status: str = 'unknown'
        self.control_mode: str = 'PASS_BLOCKED'
        self.safe_stop: bool = True

        self.last_lane_stamp: Optional[float] = None
        self.last_heading_stamp: Optional[float] = None

        self.create_subscription(Float32, self.center_error_m_topic, self.center_cb, 10)
        self.create_subscription(Float32, self.right_error_m_topic, self.right_cb, 10)
        self.create_subscription(Float32, self.heading_error_topic, self.heading_cb, 10)
        self.create_subscription(String, self.lane_status_topic, self.status_cb, 10)
        self.create_subscription(Float32, self.lane_width_m_topic, self.lane_width_cb, 10)
        self.create_subscription(Path, self.centerline_path_topic, self.path_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.05
        self.timer = self.create_timer(dt, self.control_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def center_cb(self, msg: Float32) -> None:
        self.center_err_m = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def right_cb(self, msg: Float32) -> None:
        self.right_err_m = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def heading_cb(self, msg: Float32) -> None:
        v = float(msg.data)
        if math.isfinite(v):
            self.heading_err = v
            self.last_heading_stamp = self.now_sec()
            self.last_good_heading_err = v
            self.last_good_heading_stamp = self.last_heading_stamp
        else:
            self.heading_err = None
            self.last_heading_stamp = None

    def status_cb(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()
        self.last_lane_stamp = self.now_sec()

    def lane_width_cb(self, msg: Float32) -> None:
        width = float(msg.data)
        self.lane_width_m = width
        self.last_width_stamp = self.now_sec()
        if math.isfinite(width) and width >= self.min_valid_width_m:
            self.last_good_width_m = width
            self.last_good_width_stamp = self.last_width_stamp

    def path_cb(self, msg: Path) -> None:
        xs = np.array([p.pose.position.x for p in msg.poses], dtype=np.float64)
        ys = np.array([p.pose.position.y for p in msg.poses], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < self.path_min_points:
            return
        order = np.argsort(xs)
        self.path_xs = xs[order]
        self.path_ys = ys[order]
        self.last_path_stamp = self.now_sec()

    def scan_cb(self, msg: LaserScan) -> None:
        self.last_scan_stamp = self.now_sec()
        self.lidar_stop = False
        self.lidar_slow = False
        self.nearest_obstacle_x_m = None

        if not self.use_lidar_guard:
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
        valid = (
            np.isfinite(ranges) &
            (ranges >= self.lidar_range_min_m) &
            (ranges <= min(self.lidar_range_max_m, msg.range_max - 1e-3))
        )
        if not np.any(valid):
            return

        angle_deg = np.degrees(angles)
        ignore_near_mount = (
            (angle_deg >= self.scan_ignore_angle_min_deg) &
            (angle_deg <= self.scan_ignore_angle_max_deg) &
            (ranges <= self.scan_ignore_range_max_m)
        )
        valid &= ~ignore_near_mount
        if not np.any(valid):
            return

        raw_xs = ranges[valid] * np.cos(angles[valid])
        raw_ys = ranges[valid] * np.sin(angles[valid])
        c = math.cos(self.scan_yaw_offset_rad)
        s = math.sin(self.scan_yaw_offset_rad)
        xs = c * raw_xs - s * raw_ys
        ys = s * raw_xs + c * raw_ys
        longitudinal = -xs if self.obstacle_direction == 'reverse' else xs
        roi = (longitudinal >= self.obstacle_x_min_m) & (longitudinal <= self.obstacle_slow_x_m)
        xs = xs[roi]
        ys = ys[roi]
        longitudinal = longitudinal[roi]
        if xs.size == 0:
            return

        center_y = (
            np.zeros_like(ys, dtype=np.float64)
            if self.obstacle_direction == 'reverse'
            else self.center_y_at_x(xs)
        )
        half_width = self.obstacle_corridor_half_width_m + self.obstacle_corridor_padding_m
        on_path = np.abs(ys - center_y) <= half_width
        if int(np.count_nonzero(on_path)) < self.obstacle_min_points:
            return

        hit_longitudinal = longitudinal[on_path]
        self.nearest_obstacle_x_m = float(np.min(hit_longitudinal))
        self.lidar_stop = int(np.count_nonzero(hit_longitudinal <= self.obstacle_stop_x_m)) >= self.obstacle_min_points
        self.lidar_slow = True

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def lane_fresh(self) -> bool:
        if self.last_lane_stamp is None:
            return False
        return (self.now_sec() - self.last_lane_stamp) <= self.lane_timeout_sec

    def heading_fresh(self) -> bool:
        if self.last_heading_stamp is None:
            return False
        if self.heading_err is None:
            return False
        return (self.now_sec() - self.last_heading_stamp) <= self.heading_timeout_sec

    def path_fresh(self) -> bool:
        if self.last_path_stamp is None:
            return False
        return (self.now_sec() - self.last_path_stamp) <= self.path_timeout_sec

    def scan_fresh(self) -> bool:
        if self.last_scan_stamp is None:
            return False
        return (self.now_sec() - self.last_scan_stamp) <= self.scan_timeout_sec

    def center_y_at_x(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64)
        if self.path_xs is None or self.path_ys is None or self.path_xs.size < 2 or not self.path_fresh():
            return np.zeros_like(xs, dtype=np.float64)

        path_xs = np.asarray(self.path_xs, dtype=np.float64)
        path_ys = np.asarray(self.path_ys, dtype=np.float64)
        valid = np.isfinite(path_xs) & np.isfinite(path_ys)
        path_xs = path_xs[valid]
        path_ys = path_ys[valid]
        if path_xs.size < 2:
            return np.zeros_like(xs, dtype=np.float64)

        order = np.argsort(path_xs)
        path_xs = path_xs[order]
        path_ys = path_ys[order]
        keep = np.concatenate(([True], np.diff(path_xs) > 1e-4))
        path_xs = path_xs[keep]
        path_ys = path_ys[keep]
        if path_xs.size < 2:
            return np.zeros_like(xs, dtype=np.float64)
        return np.interp(xs, path_xs, path_ys, left=path_ys[0], right=path_ys[-1])

    def apply_lidar_guard(self, tw: Twist) -> Twist:
        if not self.use_lidar_guard or not self.scan_fresh():
            return tw
        if self.lidar_stop:
            return Twist()
        if self.lidar_slow:
            tw.linear.x *= clamp(self.obstacle_slow_speed_scale, 0.0, 1.0)
        return tw

    def lane_valid_for_follow(self) -> bool:
        return self.lane_status == 'ok'

    def width_fresh(self) -> bool:
        if self.last_width_stamp is None:
            return False
        return (self.now_sec() - self.last_width_stamp) <= self.width_timeout_sec

    def held_width_available(self) -> bool:
        if self.last_good_width_stamp is None or self.last_good_width_m is None:
            return False
        return (self.now_sec() - self.last_good_width_stamp) <= self.hold_last_width_sec

    def resolve_width(self) -> Optional[float]:
        if (
            self.lane_width_m is not None and self.width_fresh() and
            math.isfinite(self.lane_width_m) and self.lane_width_m >= self.min_valid_width_m
        ):
            return self.lane_width_m
        if self.held_width_available():
            return self.last_good_width_m
        return None

    def update_lane_passability(self) -> None:
        if not self.lane_fresh():
            self.control_mode = 'PASS_BLOCKED'
            self.safe_stop = True
            return

        if self.stop_on_invalid_lane and not self.lane_valid_for_follow():
            self.control_mode = 'PASS_BLOCKED'
            self.safe_stop = True
            return

        width = self.resolve_width()
        if width is None:
            self.control_mode = 'PASS_BLOCKED'
            self.safe_stop = True
            return

        open_threshold = self.min_passable_width_m + self.width_hysteresis_m
        close_threshold = self.min_passable_width_m
        if self.control_mode == 'NORMAL_CENTER_DRIVE':
            passable = width > close_threshold
        else:
            passable = width >= open_threshold

        if passable:
            self.control_mode = 'NORMAL_CENTER_DRIVE'
            self.safe_stop = False
        else:
            self.control_mode = 'PASS_BLOCKED'
            self.safe_stop = True

    def select_error_m(self) -> Optional[float]:
        mode = self.control_mode.strip().upper()
        if mode in ('STOP', 'SAFE_STOP', 'WAIT_PASS', 'PASS_BLOCKED'):
            return None
        if mode == 'KEEP_RIGHT_APPROACH':
            return self.right_err_m
        if mode == 'NORMAL_CENTER_DRIVE':
            return self.center_err_m
        if self.default_follow_mode == 'right':
            return self.right_err_m
        return self.center_err_m

    def compute_speed(self, abs_error_m: float) -> float:
        if abs_error_m >= self.hard_stop_error_m:
            return 0.0 if self.rotate_in_place_on_large_error else self.min_speed
        alpha = clamp(abs_error_m / max(self.slow_error_m, 1e-6), 0.0, 1.0)
        speed = self.nominal_speed - (self.nominal_speed - self.min_speed) * alpha
        return clamp(speed, self.min_speed, self.nominal_speed)

    def compensate_error(self, e_cam: float, psi: float) -> tuple[float, float]:
        e_base = e_cam - self.camera_forward_offset_m * psi
        if self.use_lookahead:
            e_ctrl = e_base + self.lookahead_gain * self.lookahead_m * psi
        else:
            e_ctrl = e_base
        return e_base, e_ctrl

    def resolve_heading(self) -> Optional[float]:
        if self.heading_fresh() and self.heading_err is not None:
            return float(self.heading_err)
        can_hold = (
            self.last_good_heading_stamp is not None and
            (self.now_sec() - self.last_good_heading_stamp) <= self.hold_last_heading_sec
        )
        if can_hold:
            return float(self.last_good_heading_err)
        if self.require_heading:
            return None
        return 0.0

    def source_params(self):
        return (
            self.live_center_deadband_m,
            self.live_heading_deadband_rad,
            self.kp_m,
            self.k_heading,
            self.max_ang_z,
            self.nominal_speed,
        )

    @staticmethod
    def path_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if xs.size == 0:
            return np.array([], dtype=np.float64)
        seg = np.hypot(np.diff(xs), np.diff(ys))
        return np.concatenate(([0.0], np.cumsum(seg)))

    def reanchored_segment(self, xs: np.ndarray, ys: np.ndarray):
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0)
        if int(np.count_nonzero(valid)) < self.path_min_points:
            valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < self.path_min_points:
            return xs, ys, np.zeros_like(xs)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < self.path_min_points:
            return xs, ys, np.zeros_like(xs)

        anchor_idx = int(np.argmin(xs * xs + ys * ys))
        anchor_dist = math.hypot(float(xs[anchor_idx]), float(ys[anchor_idx]))
        dist = self.path_distance(xs, ys)
        rel_s = dist - float(dist[anchor_idx])

        min_s = 0.0 if anchor_dist > self.path_reanchor_far_m else self.path_lookahead_min_m
        mask = (rel_s >= min_s) & (rel_s <= self.path_lookahead_max_m)
        if int(np.count_nonzero(mask)) < self.path_min_points:
            mask = rel_s >= 0.0
        return xs[mask], ys[mask], rel_s[mask]

    def steer_gain_ratio(self, xs: np.ndarray, ys: np.ndarray, rel_s: np.ndarray, distance_m: float) -> tuple[float, float]:
        if xs.size < self.path_min_points:
            return 0.0, 0.0

        valid = (
            np.isfinite(xs) & np.isfinite(ys) & np.isfinite(rel_s) &
            (xs >= 0.0) & (rel_s >= 0.0) & (rel_s <= distance_m)
        )
        xs = xs[valid]
        ys = ys[valid]
        rel_s = rel_s[valid]
        if xs.size < self.path_min_points:
            return 0.0, 0.0

        order = np.argsort(rel_s)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < self.path_min_points:
            return 0.0, 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
            headings = np.degrees(np.arctan(dy))
            bend_delta = float(np.percentile(np.abs(headings - headings[0]), 85))
            line_angle_error = float(abs(np.average(headings)))
        except (FloatingPointError, ValueError):
            kappa = 0.0
            bend_delta = 0.0
            line_angle_error = 0.0

        curv_ratio = clamp(
            (kappa - self.preview_curv_start) /
            max(self.preview_curv_full - self.preview_curv_start, 1e-6),
            0.0, 1.0)
        bend_ratio = clamp(
            (bend_delta - self.heading_start_deg) /
            max(self.heading_full_deg - self.heading_start_deg, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, bend_ratio)), line_angle_error

    def preview_slow_ratio(self, xs: np.ndarray, ys: np.ndarray) -> float:
        if xs.size < self.path_min_points:
            return 0.0

        valid = (
            np.isfinite(xs) & np.isfinite(ys) &
            (xs >= 0.0) & (xs <= self.path_preview_distance_m)
        )
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < self.path_min_points:
            return 0.0

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        keep = np.concatenate(([True], np.diff(xs) > 1e-4))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < self.path_min_points:
            return 0.0

        try:
            dy = np.gradient(ys, xs)
            ddy = np.gradient(dy, xs)
            curv = np.abs(ddy) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-6)
            curv = curv[np.isfinite(curv)]
            kappa = float(np.percentile(curv, 85)) if curv.size else 0.0
        except (FloatingPointError, ValueError):
            kappa = 0.0

        lateral_delta = float(np.max(np.abs(ys - ys[0])))
        curv_ratio = clamp(
            (kappa - self.preview_curv_start) /
            max(self.preview_curv_full - self.preview_curv_start, 1e-6),
            0.0, 1.0)
        lateral_ratio = clamp(
            (lateral_delta - self.preview_lateral_start_m) /
            max(self.preview_lateral_full_m - self.preview_lateral_start_m, 1e-6),
            0.0, 1.0)
        return float(max(curv_ratio, lateral_ratio))

    def compute_path_angle(self, xs: np.ndarray, ys: np.ndarray, rel_s: np.ndarray, steer_ratio: float) -> float:
        if xs.size < self.path_min_points:
            return 0.0

        ys = np.clip(ys, -self.path_target_y_limit_m, self.path_target_y_limit_m)
        pursuit_angles = np.degrees(np.arctan2(-ys, np.maximum(xs, 1e-3)))

        slopes = np.gradient(ys, xs) if xs.size >= 2 else np.zeros_like(xs)
        heading_angles = np.degrees(np.arctan(-slopes))

        if steer_ratio >= self.curve_switch_ratio:
            pursuit_gain = self.curve_pursuit_gain
            heading_gain = self.curve_heading_gain
        else:
            boost_mix = clamp(self.straight_boost_level, 0.0, self.straight_boost_max_level)
            pursuit_gain = self.straight_pursuit_gain + (
                self.straight_boost_pursuit_gain - self.straight_pursuit_gain
            ) * boost_mix
            heading_gain = self.straight_heading_gain + (
                self.straight_boost_heading_gain - self.straight_heading_gain
            ) * boost_mix

        sample_angles = pursuit_gain * pursuit_angles + heading_gain * heading_angles
        focus = rel_s if rel_s.size == xs.size else xs
        sigma = max(self.path_focus_sigma_m, 1e-6)
        weights = np.exp(-((focus - self.path_focus_m) / sigma) ** 2)
        if not np.any(np.isfinite(weights)) or float(np.sum(weights)) <= 1e-9:
            return float(np.average(sample_angles))
        return float(np.average(sample_angles, weights=weights))

    def smooth_path_angle(self, angle_deg: float) -> float:
        delta = float(angle_deg - self.prev_path_angle_deg)
        if abs(delta) < self.angle_deadband_deg:
            angle_deg = self.prev_path_angle_deg
        else:
            angle_deg = self.prev_path_angle_deg + self.angle_smooth_alpha * delta
        step = clamp(
            angle_deg - self.prev_path_angle_deg,
            -self.angle_max_step_deg,
            self.angle_max_step_deg,
        )
        angle_deg = self.prev_path_angle_deg + step
        self.prev_path_angle_deg = angle_deg
        return angle_deg

    def path_pursuit_command(self, max_ang_use: float, speed_limit: float) -> Optional[Twist]:
        if not self.use_path_pursuit or not self.path_fresh():
            return None
        if self.path_xs is None or self.path_ys is None or self.path_xs.size < self.path_min_points:
            return None

        seg_xs, seg_ys, seg_s = self.reanchored_segment(self.path_xs, self.path_ys)
        if seg_xs.size < self.path_min_points:
            return None

        preview_ratio = self.preview_slow_ratio(seg_xs, seg_ys)
        near_ratio, line_angle_error = self.steer_gain_ratio(
            seg_xs, seg_ys, seg_s, self.path_gain_distance_m)
        preview_steer_ratio, _ = self.steer_gain_ratio(
            seg_xs, seg_ys, seg_s, self.path_preview_distance_m)
        steer_ratio = max(near_ratio, preview_steer_ratio)

        if steer_ratio <= self.curve_switch_ratio:
            boost_target = clamp(
                (line_angle_error - self.straight_boost_off_deg) /
                max(self.straight_boost_on_deg - self.straight_boost_off_deg, 1e-6),
                0.0, 1.0)
            boost_target = boost_target * boost_target
            if line_angle_error > self.straight_boost_on_deg:
                extra_boost = clamp(
                    (line_angle_error - self.straight_boost_on_deg) /
                    max(self.straight_boost_full_deg - self.straight_boost_on_deg, 1e-6),
                    0.0, 1.0)
                boost_target = 1.0 + ((self.straight_boost_max_level - 1.0) * extra_boost)
        else:
            boost_target = 0.0
        self.straight_boost_level += (
            self.straight_boost_blend_alpha *
            (float(boost_target) - self.straight_boost_level)
        )

        angle_deg = self.compute_path_angle(seg_xs, seg_ys, seg_s, steer_ratio)
        angle_deg = self.smooth_path_angle(angle_deg)
        angular = self.path_angular_sign * angle_deg * self.angle_to_ang_z_gain
        angular = clamp(angular, -max_ang_use, max_ang_use)

        speed_ratio = max(preview_ratio, min(1.0, abs(angular) / max(max_ang_use, 1e-6)))
        speed = self.nominal_speed - (self.nominal_speed - self.min_speed) * speed_ratio
        speed = clamp(speed, self.min_speed, min(self.nominal_speed, speed_limit))

        tw = Twist()
        tw.linear.x = speed
        tw.angular.z = angular
        return tw

    def control_step(self) -> None:
        self.update_lane_passability()

        if self.safe_stop:
            self.publish_stop()
            return

        if self.use_lidar_guard and self.scan_fresh() and self.lidar_stop:
            self.publish_stop()
            return

        if self.control_mode.strip().upper() == 'PASS_BLOCKED':
            self.publish_stop()
            return

        e_cam = self.select_error_m()
        if e_cam is None:
            self.publish_stop()
            return

        if not self.lane_fresh():
            self.publish_stop()
            return

        if self.stop_on_invalid_lane and not self.lane_valid_for_follow():
            self.publish_stop()
            return

        psi = self.resolve_heading()
        if psi is None:
            self.publish_stop()
            return

        e_base, e_ctrl = self.compensate_error(e_cam, psi)
        dead_e, dead_h, kp_use, k_heading_use, max_ang_use, speed_limit = self.source_params()

        if abs(e_ctrl) < dead_e:
            e_ctrl = 0.0
        if abs(psi) < dead_h:
            psi = 0.0

        path_twist = self.path_pursuit_command(max_ang_use, speed_limit)
        if path_twist is not None:
            self.cmd_pub.publish(self.apply_lidar_guard(path_twist))
            return

        abs_err = abs(e_ctrl)
        speed = min(self.compute_speed(abs_err), speed_limit)

        ang = self.steering_sign * (kp_use * e_ctrl + k_heading_use * psi)
        ang = clamp(ang, -max_ang_use, max_ang_use)

        tw = Twist()
        if self.rotate_in_place_on_large_error and abs_err >= self.hard_stop_error_m:
            tw.linear.x = 0.0
            tw.angular.z = ang
        else:
            tw.linear.x = speed
            tw.angular.z = ang

        self.cmd_pub.publish(self.apply_lidar_guard(tw))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneFollowControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.publish_stop()
            except BaseException:
                pass
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
