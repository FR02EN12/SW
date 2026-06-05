#!/usr/bin/env python3
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LedBehaviorNode(Node):
    def __init__(self) -> None:
        super().__init__('led_behavior')

        self.declare_parameter('led_hz', 5.0)
        self.declare_parameter('min_hold_sec', 0.5)
        self.declare_parameter('v2v_pattern_timeout_sec', 1.0)

        self.led_hz = float(self.get_parameter('led_hz').value)
        self.min_hold_sec = float(self.get_parameter('min_hold_sec').value)
        self.v2v_pattern_timeout_sec = float(
            self.get_parameter('v2v_pattern_timeout_sec').value)

        self.driving_mode: str = ''
        self.cooperation_decision: str = ''
        self.v2v_led_pattern: str = ''
        self.v2v_led_pattern_stamp: Optional[float] = None
        self.current_led: str = 'MASK:11111'
        self.led_set_stamp: Optional[float] = None

        self.create_subscription(
            String, '/planning/driving_mode', self.mode_cb, 10
        )
        self.create_subscription(
            String, '/planning/cooperation_decision', self.coop_cb, 10
        )
        self.create_subscription(
            String, '/planning/v2v_led_pattern', self.v2v_pattern_cb, 10
        )

        self.led_pub = self.create_publisher(String, '/vehicle/led_cmd', 10)

        dt = 1.0 / self.led_hz if self.led_hz > 0 else 0.2
        self.timer = self.create_timer(dt, self.led_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def mode_cb(self, msg: String) -> None:
        self.driving_mode = str(msg.data).strip()

    def coop_cb(self, msg: String) -> None:
        self.cooperation_decision = str(msg.data).strip()

    def v2v_pattern_cb(self, msg: String) -> None:
        self.v2v_led_pattern = str(msg.data).strip()
        self.v2v_led_pattern_stamp = self.now_sec()

    def fresh_v2v_pattern(self) -> bool:
        return (
            self.v2v_led_pattern_stamp is not None and
            (self.now_sec() - self.v2v_led_pattern_stamp) <= self.v2v_pattern_timeout_sec
        )

    def resolve_led(self) -> str:
        mode = self.driving_mode.upper()

        if mode == 'EMERGENCY_STOP':
            return 'EMERGENCY'

        if mode in ('', 'LANE_FOLLOW', 'APPROACH_NARROW'):
            return 'MASK:11111'

        if self.fresh_v2v_pattern():
            return self.v2v_led_pattern

        return 'MASK:11111'

    def led_step(self) -> None:
        desired = self.resolve_led()
        now = self.now_sec()

        # Debounce: hold current LED for minimum duration
        if desired != self.current_led:
            if self.led_set_stamp is not None:
                elapsed = now - self.led_set_stamp
                if elapsed < self.min_hold_sec:
                    desired = self.current_led

        if desired != self.current_led:
            self.current_led = desired
            self.led_set_stamp = now

        out = String()
        out.data = self.current_led
        self.led_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LedBehaviorNode()
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
