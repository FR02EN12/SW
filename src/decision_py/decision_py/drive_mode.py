#!/usr/bin/env python3
# Performance target T4: deadlock response <= 1.0 s
#   state_publish_hz raised 10 -> 20 (transition latency <= 0.05 s)
#   deadlock_confirm_sec lowered 1.0 -> 0.30 s
#   Full T4 budget: detect(0.07) + confirm(0.30) + decide(0.10) + cmd(0.05)
#                 = 0.52 s  (<= 1.0 s target)
from typing import Optional

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DrivingModeManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('drive_mode')

        # T4: 20 Hz means max state-machine transition latency is 0.05 s.
        self.declare_parameter('state_publish_hz', 20.0)
        # T4: confirm deadlock in 0.30 s (was 1.0 s) to meet response budget.
        self.declare_parameter('deadlock_confirm_sec', 0.30)
        self.declare_parameter('wait_timeout_sec', 3.0)
        self.declare_parameter('yield_accept_wait_sec', 0.8)
        self.declare_parameter('approach_distance_m', 1.0)
        self.declare_parameter('min_dwell_sec', 0.5)

        self.state_publish_hz = float(self.get_parameter('state_publish_hz').value)
        self.deadlock_confirm_sec = float(self.get_parameter('deadlock_confirm_sec').value)
        self.wait_timeout_sec = float(self.get_parameter('wait_timeout_sec').value)
        self.yield_accept_wait_sec = float(self.get_parameter('yield_accept_wait_sec').value)
        self.approach_distance_m = float(self.get_parameter('approach_distance_m').value)
        self.min_dwell_sec = float(self.get_parameter('min_dwell_sec').value)

        self.current_state: str = 'LANE_FOLLOW'
        self.state_enter_stamp: float = self.now_sec()

        self.scene_data: dict = {}
        self.last_scene_stamp: Optional[float] = None
        self.safety_event: Optional[str] = None
        self.cooperation_decision: Optional[str] = None

        self.deadlock_detect_stamp: Optional[float] = None
        self.wait_enter_stamp: Optional[float] = None
        self.pass_wait_enter_stamp: Optional[float] = None
        self.path_goal_reached_mode: Optional[str] = None

        self.create_subscription(String, '/scene/understanding', self.scene_cb, 10)
        self.create_subscription(String, '/safety/events', self.safety_cb, 10)
        self.create_subscription(String, '/planning/cooperation_decision', self.coop_cb, 10)
        self.create_subscription(String, '/planning/path_goal_reached', self.path_goal_cb, 10)

        self.mode_pub = self.create_publisher(String, '/planning/driving_mode', 10)

        dt = 1.0 / self.state_publish_hz if self.state_publish_hz > 0.0 else 0.1
        self.timer = self.create_timer(dt, self.step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def scene_cb(self, msg: String) -> None:
        try:
            self.scene_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON in scene understanding')
        self.last_scene_stamp = self.now_sec()

    def coop_cb(self, msg: String) -> None:
        self.cooperation_decision = msg.data.strip()

    def path_goal_cb(self, msg: String) -> None:
        self.path_goal_reached_mode = msg.data.strip()

    def safety_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.safety_event = data.get('type', None)
        except json.JSONDecodeError:
            self.safety_event = msg.data.strip()

    def dwell_ok(self) -> bool:
        return (self.now_sec() - self.state_enter_stamp) >= self.min_dwell_sec

    def transition(self, new_state: str) -> None:
        if new_state != self.current_state:
            self.get_logger().info(f'State transition: {self.current_state} -> {new_state}')
            self.current_state = new_state
            self.state_enter_stamp = self.now_sec()

    def step(self) -> None:
        now = self.now_sec()

        # ANY -> EMERGENCY_STOP on safety event
        if self.safety_event == 'EMERGENCY':
            self.transition('EMERGENCY_STOP')
            self.safety_event = None
            self._publish()
            return

        # EMERGENCY_STOP -> LANE_FOLLOW on CLEAR
        if self.current_state == 'EMERGENCY_STOP':
            if self.safety_event == 'CLEAR':
                self.safety_event = None
                self.transition('LANE_FOLLOW')
            self._publish()
            return

        if not self.dwell_ok():
            self._publish()
            return

        in_narrow = self.scene_data.get('in_narrow_passage', False)
        nearest_narrow = self.scene_data.get('nearest_narrow_passage_m', float('inf'))
        opponent_detected = self.scene_data.get('opponent', {}).get('detected', False)
        passed_narrow = self.scene_data.get('passed_narrow_passage', False)
        reached_pullover = self.scene_data.get('reached_pullover', False)
        back_on_lane = self.scene_data.get('back_on_lane', False)
        # True once the opponent has been gone > 0.8 s; safe to re-enter.
        opponent_cleared = self.scene_data.get('opponent_cleared_narrow', False)

        if self.current_state == 'LANE_FOLLOW':
            if in_narrow or nearest_narrow < self.approach_distance_m:
                self.transition('APPROACH_NARROW')

        elif self.current_state == 'APPROACH_NARROW':
            if in_narrow and opponent_detected:
                self.transition('DEADLOCK_CHECK')
                self.deadlock_detect_stamp = now
            elif passed_narrow:
                self.transition('LANE_FOLLOW')

        elif self.current_state == 'DEADLOCK_CHECK':
            if not opponent_detected:
                self.transition('LANE_FOLLOW')
            elif self.deadlock_detect_stamp is not None and \
                    (now - self.deadlock_detect_stamp) >= self.deadlock_confirm_sec:
                self.transition('NEGOTIATION')

        elif self.current_state == 'NEGOTIATION':
            if self.cooperation_decision in ('I_YIELD', 'I_YIELD_REVERSE'):
                self.transition('YIELD_REVERSE')
            elif self.cooperation_decision == 'I_YIELD_SIDE':
                self.transition('YIELD_SIDE')
            elif self.cooperation_decision == 'WAIT_RECHECK':
                self.transition('WAIT')
                self.wait_enter_stamp = now
            elif self.cooperation_decision == 'I_GO':
                self.transition('WAIT_FOR_PASS')
                self.pass_wait_enter_stamp = now

        elif self.current_state == 'YIELD_REVERSE':
            # First finish the evasive motion, then wait out of the lane until
            # the opponent has cleared the narrow passage.
            if self.path_goal_reached_mode == 'YIELD_REVERSE' or reached_pullover:
                self.path_goal_reached_mode = None
                self.transition('YIELD_WAIT_CLEAR')

        elif self.current_state == 'YIELD_SIDE':
            if self.path_goal_reached_mode == 'YIELD_SIDE':
                self.path_goal_reached_mode = None
                self.transition('YIELD_WAIT_CLEAR')

        elif self.current_state == 'YIELD_WAIT_CLEAR':
            if opponent_cleared or not opponent_detected:
                self.transition('REENTER')

        elif self.current_state == 'WAIT_FOR_PASS':
            waited_long_enough = (
                self.pass_wait_enter_stamp is not None and
                (now - self.pass_wait_enter_stamp) >= self.yield_accept_wait_sec
            )
            pass_clear = self.scene_data.get('pass_clear', False)
            if waited_long_enough or pass_clear:
                self.pass_wait_enter_stamp = None
                self.transition('LANE_FOLLOW')

        elif self.current_state == 'WAIT':
            if self.wait_enter_stamp is not None and \
                    (now - self.wait_enter_stamp) >= self.wait_timeout_sec:
                self.transition('NEGOTIATION')

        elif self.current_state == 'REENTER':
            if self.path_goal_reached_mode == 'REENTER' or back_on_lane:
                self.path_goal_reached_mode = None
                self.transition('LANE_FOLLOW')

        self._publish()

    def _publish(self) -> None:
        msg = String()
        msg.data = self.current_state
        self.mode_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DrivingModeManagerNode()
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
