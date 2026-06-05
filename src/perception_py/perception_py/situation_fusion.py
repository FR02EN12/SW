#!/usr/bin/env python3
import json
import math
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String


class SituationFusionNode(Node):
    def __init__(self):
        super().__init__('situation_fusion')

        # Parameters
        self.declare_parameter('fusion_hz', 10.0)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('critical_distance_m', 0.3)
        self.declare_parameter('high_risk_distance_m', 0.8)
        # R2 fix: first-encounter narrow-passage fallback.
        # When space_memory has not yet learned any narrow passages,
        # fall back to live lane-width measurements from the FSD lane front-end.
        # A lane width below this threshold is treated as a narrow passage.
        self.declare_parameter('narrow_live_width_threshold_m', 0.40)
        # Temporary road-class thresholds until the final camera/LiDAR fused
        # width estimate is available: A=two robots pass, B=marginal, C=single-file.
        self.declare_parameter('road_class_a_width_m', 0.65)
        self.declare_parameter('road_class_b_width_m', 0.45)
        self.declare_parameter('battery_pct', 80.0)
        self.declare_parameter('rear_scan_topic', '/scan')
        self.declare_parameter('rear_obstacle_distance_m', 0.35)
        self.declare_parameter('rear_obstacle_half_angle_deg', 45.0)

        self._fusion_hz = self.get_parameter('fusion_hz').value
        self._stale_timeout = self.get_parameter('stale_timeout_sec').value
        self._critical_dist = self.get_parameter('critical_distance_m').value
        self._high_risk_dist = self.get_parameter('high_risk_distance_m').value
        self._narrow_live_thresh = self.get_parameter('narrow_live_width_threshold_m').value
        self._road_class_a_width = self.get_parameter('road_class_a_width_m').value
        self._road_class_b_width = self.get_parameter('road_class_b_width_m').value
        self._battery_pct = max(0.0, min(100.0, float(
            self.get_parameter('battery_pct').value)))
        self._rear_scan_topic = str(self.get_parameter('rear_scan_topic').value)
        self._rear_obstacle_distance_m = float(
            self.get_parameter('rear_obstacle_distance_m').value)
        self._rear_obstacle_half_angle_rad = math.radians(float(
            self.get_parameter('rear_obstacle_half_angle_deg').value))

        # Cached state and timestamps
        self._lane_status: str = ''
        self._lane_status_t: float = 0.0

        self._lane_error: float = 0.0
        self._lane_error_t: float = 0.0

        self._opponent: dict = {'detected': False}
        self._opponent_t: float = 0.0

        self._led_state: str = 'UNKNOWN'
        self._led_state_t: float = 0.0
        self._v2v_sign: dict = {'detected': False, 'sign': 'UNKNOWN', 'mask': '00000'}
        self._v2v_sign_t: float = 0.0

        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._pose_theta: float = 0.0
        self._pose_t: float = 0.0

        self._pullover_candidates: list = []
        self._pullover_t: float = 0.0

        self._narrow_passages: list = []
        self._narrow_t: float = 0.0

        # R2 fix: live lane width for fallback
        # narrow-passage detection before space memory is populated.
        self._live_lane_width_m: float = float('inf')
        self._live_lane_width_t: float = 0.0

        self._rear_obstacle_detected: bool = False
        self._rear_obstacle_distance_m_latest: float = float('inf')
        self._rear_obstacle_t: float = 0.0

        # Derived state tracking
        self._was_in_narrow: bool = False
        self._passed_narrow: bool = False

        # Opponent-cleared-narrow tracking.
        # _opp_last_detected_stamp records the last time the opponent was seen.
        # opponent_cleared_narrow becomes True once the opponent has not been
        # detected for > 0.8 s after having been detected at least once.
        # This confirms the opponent has passed through the narrow section
        # before the ego robot re-enters, preventing head-on re-collision.
        self._opp_last_detected_stamp: Optional[float] = None
        self._opponent_cleared_narrow: bool = False

        # Subscriptions
        self.sub_lane_status = self.create_subscription(
            String, '/lane_status', self._lane_status_cb, 10)
        self.sub_lane_error = self.create_subscription(
            Float32, '/lane_error_center_m', self._lane_error_cb, 10)
        self.sub_opponent = self.create_subscription(
            String, '/perception/opponent_vehicle', self._opponent_cb, 10)
        self.sub_led = self.create_subscription(
            String, '/perception/opponent_led_state', self._led_cb, 10)
        self.sub_v2v = self.create_subscription(
            String, '/perception/opponent_v2v_sign', self._v2v_cb, 10)
        self.sub_pose = self.create_subscription(
            PoseStamped, '/pose', self._pose_cb, 10)
        self.sub_pullover = self.create_subscription(
            String, '/memory/pull_over_candidates', self._pullover_cb, 10)
        self.sub_narrow = self.create_subscription(
            String, '/memory/narrow_passages', self._narrow_cb, 10)
        self.sub_lane_width = self.create_subscription(
            Float32, '/fused/lane_width_m', self._lane_width_cb, 10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_scan = self.create_subscription(
            LaserScan, self._rear_scan_topic, self._scan_cb, sensor_qos)

        # Publisher
        self.pub_scene = self.create_publisher(String, '/scene/understanding', 10)

        # Timer
        period = 1.0 / max(self._fusion_hz, 0.1)
        self.create_timer(period, self._fusion_cb)

        self.get_logger().info('SituationFusionNode initialised')

    # Subscription callbacks
    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _lane_status_cb(self, msg: String) -> None:
        self._lane_status = msg.data
        self._lane_status_t = self._now_sec()

    def _lane_error_cb(self, msg: Float32) -> None:
        self._lane_error = msg.data
        self._lane_error_t = self._now_sec()

    def _opponent_cb(self, msg: String) -> None:
        try:
            self._opponent = json.loads(msg.data)
        except json.JSONDecodeError:
            pass
        self._opponent_t = self._now_sec()

    def _led_cb(self, msg: String) -> None:
        self._led_state = msg.data
        self._led_state_t = self._now_sec()

    def _v2v_cb(self, msg: String) -> None:
        try:
            self._v2v_sign = json.loads(msg.data)
        except json.JSONDecodeError:
            self._v2v_sign = {'detected': False, 'sign': msg.data, 'mask': '00000'}
        self._v2v_sign_t = self._now_sec()

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._pose_x = msg.pose.position.x
        self._pose_y = msg.pose.position.y
        q = msg.pose.orientation
        self._pose_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose_t = self._now_sec()

    def _pullover_cb(self, msg: String) -> None:
        try:
            self._pullover_candidates = json.loads(msg.data)
        except json.JSONDecodeError:
            pass
        self._pullover_t = self._now_sec()

    def _narrow_cb(self, msg: String) -> None:
        try:
            self._narrow_passages = json.loads(msg.data)
        except json.JSONDecodeError:
            pass
        self._narrow_t = self._now_sec()

    def _lane_width_cb(self, msg: Float32) -> None:
        # R2 fix: store the most recent live lane width for fallback.
        if math.isfinite(msg.data) and msg.data > 0.0:
            self._live_lane_width_m = float(msg.data)
            self._live_lane_width_t = self._now_sec()

    def _scan_cb(self, msg: LaserScan) -> None:
        min_rear_range = float('inf')
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r):
                range_min = msg.range_min if msg.range_min > 0.0 else 0.0
                range_max = msg.range_max if msg.range_max > 0.0 else float('inf')
                if range_min <= r <= range_max:
                    norm_angle = math.atan2(math.sin(angle), math.cos(angle))
                    rear_delta = abs(math.pi - abs(norm_angle))
                    if rear_delta <= self._rear_obstacle_half_angle_rad:
                        min_rear_range = min(min_rear_range, float(r))
            angle += msg.angle_increment

        self._rear_obstacle_distance_m_latest = min_rear_range
        self._rear_obstacle_detected = (
            math.isfinite(min_rear_range) and
            min_rear_range <= self._rear_obstacle_distance_m
        )
        self._rear_obstacle_t = self._now_sec()

    # Fusion timer
    def _fusion_cb(self) -> None:
        now = self._now_sec()

        # Staleness check helper
        def freshness(t: float) -> str:
            return 'ok' if (now - t) < self._stale_timeout else 'stale'

        # In narrow passage?
        in_narrow = False
        nearest_narrow_m = float('inf')
        for p in self._narrow_passages:
            pcx = (p.get('x1', 0.0) + p.get('x2', 0.0)) / 2.0
            pcy = (p.get('y1', 0.0) + p.get('y2', 0.0)) / 2.0
            d = math.hypot(self._pose_x - pcx, self._pose_y - pcy)
            nearest_narrow_m = min(nearest_narrow_m, d)
            # Consider "in" if within half the passage span
            span = math.hypot(p.get('x2', 0.0) - p.get('x1', 0.0),
                              p.get('y2', 0.0) - p.get('y1', 0.0))
            if d < max(span / 2.0, 0.15):
                in_narrow = True

        if nearest_narrow_m == float('inf'):
            nearest_narrow_m = -1.0

        # R2 fix: fallback if no memorised narrow passages are known yet,
        # use the live lane width to decide whether we are *currently* in a
        # narrow passage. This enables narrow-passage logic on the very first
        # encounter, before space_memory has had a chance to learn.
        live_width_fresh = (now - self._live_lane_width_t) < self._stale_timeout
        if not self._narrow_passages and live_width_fresh:
            if self._live_lane_width_m < self._narrow_live_thresh:
                in_narrow = True
                # Use the live width as a proxy "distance" (0 means already in).
                nearest_narrow_m = 0.0

        # Derived: passed narrow passage (was in, now out)
        self._passed_narrow = self._was_in_narrow and not in_narrow
        self._was_in_narrow = in_narrow

        # Nearest pull-over
        nearest_pullover: Optional[dict] = None
        best_d = float('inf')
        if self._pullover_candidates:
            for c in self._pullover_candidates:
                d = math.hypot(self._pose_x - c.get('x', 0.0),
                               self._pose_y - c.get('y', 0.0))
                if d < best_d:
                    best_d = d
                    nearest_pullover = c

        # Opponent info
        opp_detected = self._opponent.get('detected', False)
        opp_dist = self._opponent.get('distance_m', 999.0)
        opp_angle = self._opponent.get('angle_rad', 0.0)

        # Track when opponent was last seen so we know when they've cleared.
        if opp_detected:
            self._opp_last_detected_stamp = now
            self._opponent_cleared_narrow = False  # still present
        elif self._opp_last_detected_stamp is not None:
            # Opponent was seen before; consider cleared after 0.8 s without detection.
            if (now - self._opp_last_detected_stamp) > 0.8:
                self._opponent_cleared_narrow = True
        # Reset everything when ego itself exits the narrow passage (new encounter).
        if self._passed_narrow:
            self._opp_last_detected_stamp = None
            self._opponent_cleared_narrow = False

        # Risk assessment
        risk = self._assess_risk(opp_detected, opp_dist, opp_angle, in_narrow,
                                 nearest_narrow_m)

        reached_pullover = nearest_pullover is not None and best_d < 0.3
        back_on_lane = self._lane_status == 'ok'
        road_width_m, road_class = self._classify_road(live_width_fresh)
        rear_obstacle_fresh = (now - self._rear_obstacle_t) < self._stale_timeout
        rear_obstacle_dist = (
            self._rear_obstacle_distance_m_latest
            if rear_obstacle_fresh and math.isfinite(self._rear_obstacle_distance_m_latest)
            else 999.0
        )
        rear_obstacle_detected = self._rear_obstacle_detected and rear_obstacle_fresh

        # Pullover geometry for cooperation decisions.
        # In a head-on narrow-passage deadlock the opponent faces us, so
        # their "behind" direction equals our "ahead" direction.
        pullover_behind_us = False
        pullover_behind_opponent = False
        pose_fresh = (now - self._pose_t) < self._stale_timeout
        if nearest_pullover is not None and pose_fresh:
            po_x = nearest_pullover.get('x', 0.0)
            po_y = nearest_pullover.get('y', 0.0)
            dx = po_x - self._pose_x
            dy = po_y - self._pose_y
            fwd_dot = dx * math.cos(self._pose_theta) + dy * math.sin(self._pose_theta)
            pullover_behind_us = fwd_dot < 0        # pullover requires us to reverse
            pullover_behind_opponent = opp_detected and fwd_dot > 0  # pullover is in opponent's backward half

        # Exit distances: how far each vehicle must travel to leave the narrow section.
        # Computed from the passage endpoints with respect to the current heading.
        our_exit_distance_m = float('inf')
        opponent_exit_distance_m = float('inf')
        if pose_fresh:
            fwd_x = math.cos(self._pose_theta)
            fwd_y = math.sin(self._pose_theta)
            for p in self._narrow_passages:
                px1, py1 = p.get('x1', 0.0), p.get('y1', 0.0)
                px2, py2 = p.get('x2', 0.0), p.get('y2', 0.0)
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                span = math.hypot(px2 - px1, py2 - py1)
                if math.hypot(self._pose_x - pcx, self._pose_y - pcy) > max(span / 2.0, 0.15):
                    continue  # not in this passage
                for ex, ey in ((px1, py1), (px2, py2)):
                    dot = (ex - self._pose_x) * fwd_x + (ey - self._pose_y) * fwd_y
                    d = math.hypot(self._pose_x - ex, self._pose_y - ey)
                    if dot > 0:
                        our_exit_distance_m = min(our_exit_distance_m, d)
                    else:
                        # Negative dot: exit is behind us = in opponent's forward direction
                        opponent_exit_distance_m = min(opponent_exit_distance_m, d)

        scene: Dict[str, Any] = {
            'timestamp': round(now, 3),
            'lane': {
                'status': self._lane_status,
                'status_freshness': freshness(self._lane_status_t),
                'center_error_m': round(self._lane_error, 4),
                'error_freshness': freshness(self._lane_error_t),
            },
            'opponent': {
                'detected': opp_detected,
                'distance_m': round(opp_dist, 3),
                'angle_rad': round(opp_angle, 4),
                'led_signal': self._led_state,  # fixed: was 'led_state' (key mismatch with cooperation)
                'v2v_sign': self._v2v_sign,
                'freshness': freshness(self._opponent_t),
                'v2v_freshness': freshness(self._v2v_sign_t),
            },
            'pose': {
                'x': round(self._pose_x, 4),
                'y': round(self._pose_y, 4),
                'theta': round(self._pose_theta, 4),
                'freshness': freshness(self._pose_t),
            },
            'environment': {
                'in_narrow_passage': in_narrow,
                'nearest_narrow_passage_m': round(nearest_narrow_m, 3),
                'pull_over_available': nearest_pullover is not None,
                'nearest_pull_over': nearest_pullover,
                'road_width_m': round(road_width_m, 3) if math.isfinite(road_width_m) else -1.0,
                'road_class': road_class,
                'freshness_narrow': freshness(self._narrow_t),
                'freshness_pullover': freshness(self._pullover_t),
            },
            # Flat aliases for easy access by downstream nodes
            'in_narrow_passage': in_narrow,
            'nearest_narrow_passage_m': round(nearest_narrow_m, 3),
            'passed_narrow_passage': self._passed_narrow,
            'reached_pullover': reached_pullover,
            'back_on_lane': back_on_lane,
            'nearest_pullover': nearest_pullover,
            'road_width_m': round(road_width_m, 3) if math.isfinite(road_width_m) else -1.0,
            'road_class': road_class,
            # Re-entry target: 0.5 m ahead in the robot's current heading so
            # the path tracker has a non-zero path to follow when leaving the pullover.
            'lane_reentry_point': {
                'x': round(self._pose_x + 0.5 * math.cos(self._pose_theta), 4),
                'y': round(self._pose_y + 0.5 * math.sin(self._pose_theta), 4),
                'theta': round(self._pose_theta, 4),
            },
            'risk_level': risk,
            # Cooperation geometry fields used by cooperation
            'pullover_behind_us': pullover_behind_us,
            'pullover_behind_opponent': pullover_behind_opponent,
            'our_exit_distance_m': round(our_exit_distance_m, 3) if math.isfinite(our_exit_distance_m) else 1e9,
            'opponent_exit_distance_m': round(opponent_exit_distance_m, 3) if math.isfinite(opponent_exit_distance_m) else 1e9,
            # True once the opponent has not been detected for > 0.8 s after
            # having been seen; signals it is safe for the ego robot to re-enter.
            'opponent_cleared_narrow': self._opponent_cleared_narrow,
            # Shells for future safety-mode integration. They are intentionally
            # passive for now; downstream scoring can read them without forcing stops.
            'vehicle_state': {
                'battery_pct': round(self._battery_pct, 1),
                'motion_ok': True,
                'fault': False,
                'source': 'manual_placeholder',
            },
            'rear_obstacle': {
                'detected': rear_obstacle_detected,
                'distance_m': round(rear_obstacle_dist, 3),
                'threshold_m': round(self._rear_obstacle_distance_m, 3),
                'sector_half_angle_deg': round(math.degrees(
                    self._rear_obstacle_half_angle_rad), 1),
                'freshness': 'ok' if rear_obstacle_fresh else 'stale',
                'source': self._rear_scan_topic,
            },
            'safety_mode': {
                'enabled': False,
                'state': 'TODO',
                'reason': 'placeholder',
            },
        }

        msg = String()
        msg.data = json.dumps(scene)
        self.pub_scene.publish(msg)

    def _assess_risk(self, opp_detected: bool, opp_dist: float,
                     opp_angle: float, in_narrow: bool,
                     nearest_narrow_m: float) -> str:
        # CRITICAL: opponent very close and roughly facing us
        if opp_detected and opp_dist < self._critical_dist:
            if abs(opp_angle) < math.radians(45):
                return 'CRITICAL'

        # HIGH: in narrow passage with opponent detected
        if in_narrow and opp_detected:
            return 'HIGH'

        # MEDIUM: approaching narrow passage or opponent visible at moderate range
        if opp_detected and opp_dist < self._high_risk_dist:
            return 'MEDIUM'
        if 0 < nearest_narrow_m < 0.5:
            return 'MEDIUM'

        return 'LOW'

    def _classify_road(self, live_width_fresh: bool) -> tuple[float, str]:
        # TODO(coord-fusion): replace this with fused camera/LiDAR road width
        # once the shared coordinate pipeline lands.
        if not live_width_fresh or not math.isfinite(self._live_lane_width_m):
            return float('inf'), 'UNKNOWN'
        width = float(self._live_lane_width_m)
        if width >= self._road_class_a_width:
            return width, 'A'
        if width >= self._road_class_b_width:
            return width, 'B'
        return width, 'C'


def main(args=None):
    rclpy.init(args=args)
    node = SituationFusionNode()
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
