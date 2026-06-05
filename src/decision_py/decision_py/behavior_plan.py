#!/usr/bin/env python3
# Performance target T5: reverse evasive path success rate >= 80 %
#   reverse_speed raised 0.015 -> 0.020 m/s for crisper path execution
from typing import Optional

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BehaviorPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('behavior_plan')

        self.declare_parameter('nominal_speed', 0.028)
        self.declare_parameter('slow_speed', 0.018)
        # T5: raised from 0.015 -> 0.020 m/s for improved reverse path execution.
        self.declare_parameter('reverse_speed', 0.020)
        self.declare_parameter('publish_hz', 10.0)

        self.nominal_speed = float(self.get_parameter('nominal_speed').value)
        self.slow_speed = float(self.get_parameter('slow_speed').value)
        self.reverse_speed = float(self.get_parameter('reverse_speed').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.driving_mode: str = 'LANE_FOLLOW'
        self.scene_data: dict = {}
        self.last_scene_stamp: Optional[float] = None

        self.create_subscription(String, '/planning/driving_mode', self.mode_cb, 10)
        self.create_subscription(String, '/scene/understanding', self.scene_cb, 10)

        self.goal_pub = self.create_publisher(String, '/planning/behavior_goal', 10)

        dt = 1.0 / self.publish_hz if self.publish_hz > 0.0 else 0.1
        self.timer = self.create_timer(dt, self.plan_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def mode_cb(self, msg: String) -> None:
        self.driving_mode = msg.data.strip()

    def scene_cb(self, msg: String) -> None:
        try:
            self.scene_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON in scene understanding')
        self.last_scene_stamp = self.now_sec()

    def plan_step(self) -> None:
        goal: dict = {}

        if self.driving_mode == 'LANE_FOLLOW':
            goal = {
                'type': 'lane_follow',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': self.nominal_speed,
                'reverse': False,
            }

        elif self.driving_mode == 'APPROACH_NARROW':
            goal = {
                'type': 'lane_follow',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': self.slow_speed,
                'reverse': False,
            }

        elif self.driving_mode == 'DEADLOCK_CHECK':
            goal = {
                'type': 'stop',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        elif self.driving_mode == 'NEGOTIATION':
            goal = {
                'type': 'stop',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        elif self.driving_mode == 'YIELD_REVERSE':
            pullover = (self.scene_data.get('nearest_pullover') or
                        self.scene_data.get('environment', {}).get('nearest_pull_over') or {})
            if pullover:
                goal = {
                    'type': 'reverse_to_pullover',
                    'target_x': pullover.get('x', 0.0),
                    'target_y': pullover.get('y', 0.0),
                    'target_theta': pullover.get('theta', 0.0),
                    'speed_limit': self.reverse_speed,
                    'reverse': True,
                }
            else:
                # The SLAM road-class/yield-route logic will provide a concrete
                # pull-over or lateral escape target later. Until then, do not
                # plan to the map origin as an accidental fallback.
                goal = {
                    'type': 'yield_wait_for_route',
                    'target_x': 0.0,
                    'target_y': 0.0,
                    'target_theta': 0.0,
                    'speed_limit': 0.0,
                    'reverse': False,
                }

        elif self.driving_mode == 'YIELD_SIDE':
            side_target = (self.scene_data.get('side_pull_over') or
                           self.scene_data.get('yield_side_target') or {})
            if side_target:
                goal = {
                    'type': 'side_pull_over',
                    'target_x': side_target.get('x', 0.0),
                    'target_y': side_target.get('y', 0.0),
                    'target_theta': side_target.get('theta', 0.0),
                    'speed_limit': self.slow_speed,
                    'reverse': False,
                }
            else:
                goal = {
                    'type': 'yield_wait_for_route',
                    'target_x': 0.0,
                    'target_y': 0.0,
                    'target_theta': 0.0,
                    'speed_limit': 0.0,
                    'reverse': False,
                }

        elif self.driving_mode == 'WAIT':
            goal = {
                'type': 'stop',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        elif self.driving_mode == 'WAIT_FOR_PASS':
            goal = {
                'type': 'wait_for_pass',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        elif self.driving_mode == 'YIELD_WAIT_CLEAR':
            goal = {
                'type': 'yield_wait_clear',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        elif self.driving_mode == 'REENTER':
            reentry = self.scene_data.get('lane_reentry_point') or self.scene_data.get('pose', {})
            goal = {
                'type': 'reenter_lane',
                'target_x': reentry.get('x', 0.0),
                'target_y': reentry.get('y', 0.0),
                'target_theta': reentry.get('theta', 0.0),
                'speed_limit': self.slow_speed,
                'reverse': False,
            }

        elif self.driving_mode == 'EMERGENCY_STOP':
            goal = {
                'type': 'emergency_stop',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        else:
            self.get_logger().warn(f'Unknown driving mode: {self.driving_mode}')
            goal = {
                'type': 'stop',
                'target_x': 0.0,
                'target_y': 0.0,
                'target_theta': 0.0,
                'speed_limit': 0.0,
                'reverse': False,
            }

        msg = String()
        msg.data = json.dumps(goal)
        self.goal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BehaviorPlannerNode()
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
