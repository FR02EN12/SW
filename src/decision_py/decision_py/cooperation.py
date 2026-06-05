#!/usr/bin/env python3
import json
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CooperationDecisionNode(Node):
    RESERVED_WEIGHT_LED_VALUE = 10
    SCORE_TO_MASK = {value: f'{value:05b}' for value in range(32)}

    RPS_MASK = {
        'ROCK': '11000',
        'SCISSORS': '10100',
        'PAPER': '10010',
    }
    SIGN_TO_RPS = {
        'RPS_ROCK': 'ROCK',
        'RPS_SCISSORS': 'SCISSORS',
        'RPS_PAPER': 'PAPER',
    }
    RPS_BEATS = {
        'ROCK': 'SCISSORS',
        'SCISSORS': 'PAPER',
        'PAPER': 'ROCK',
    }

    def __init__(self) -> None:
        super().__init__('cooperation')

        self.declare_parameter('decision_hz', 10.0)
        self.declare_parameter('max_negotiation_sec', 5.0)
        self.declare_parameter('decision_hold_sec', 2.0)
        self.declare_parameter('v2v_use_rps', True)
        self.declare_parameter('prompt_protocol_choice', True)
        self.declare_parameter('manual_rps_input', True)
        self.declare_parameter('default_rps_choice', 'ROCK')
        self.declare_parameter('v2v_advertise_sec', 0.8)
        self.declare_parameter('v2v_sign_timeout_sec', 1.0)
        self.declare_parameter('rps_countdown_sec', 3.0)
        self.declare_parameter('deadlock_detection_hz', 20.0)
        self.declare_parameter('deadlock_distance_m', 1.5)
        self.declare_parameter('near_threshold_m', 0.8)
        self.declare_parameter('facing_angle_threshold_rad', 0.52)
        self.declare_parameter('distance_change_threshold_m', 0.05)

        self.decision_hz = float(self.get_parameter('decision_hz').value)
        self.max_negotiation_sec = float(self.get_parameter('max_negotiation_sec').value)
        self.decision_hold_sec = float(self.get_parameter('decision_hold_sec').value)
        self.v2v_use_rps = bool(self.get_parameter('v2v_use_rps').value)
        self.prompt_protocol_choice = bool(
            self.get_parameter('prompt_protocol_choice').value)
        self.manual_rps_input = bool(self.get_parameter('manual_rps_input').value)
        self.default_rps_choice = str(
            self.get_parameter('default_rps_choice').value).upper()
        if self.default_rps_choice not in self.RPS_MASK:
            self.default_rps_choice = 'ROCK'
        self.v2v_advertise_sec = float(self.get_parameter('v2v_advertise_sec').value)
        self.v2v_sign_timeout_sec = float(
            self.get_parameter('v2v_sign_timeout_sec').value)
        self.rps_countdown_sec = float(self.get_parameter('rps_countdown_sec').value)
        self.deadlock_detection_hz = float(
            self.get_parameter('deadlock_detection_hz').value)
        self.deadlock_distance_m = float(self.get_parameter('deadlock_distance_m').value)
        self.near_threshold_m = float(self.get_parameter('near_threshold_m').value)
        self.facing_angle_threshold_rad = float(
            self.get_parameter('facing_angle_threshold_rad').value)
        self.distance_change_threshold_m = float(
            self.get_parameter('distance_change_threshold_m').value)

        self.scene_data: dict = {}
        self.deadlock_state: dict = {}
        self.v2v_sign: dict = {'detected': False, 'sign': 'UNKNOWN', 'mask': '00000'}
        self.last_scene_stamp: Optional[float] = None
        self.last_v2v_stamp: Optional[float] = None
        self.deadlock_start_stamp: Optional[float] = None
        self.prev_opponent_distance: Optional[float] = None
        self.prev_distance_stamp: Optional[float] = None

        self.negotiation_start_stamp: Optional[float] = None
        self.current_decision: str = 'WAIT_RECHECK'
        self.current_led_pattern: str = 'MASK:11111'
        self.last_yield_score_detail: dict = {}
        self.decision_stamp: Optional[float] = None

        self.active_protocol: Optional[str] = None
        self.opponent_weight_ready_seen: bool = False
        self.local_rps_choice: Optional[str] = None
        self.rps_countdown_stamp: Optional[float] = None
        self.negotiation_logged: bool = False

        self.create_subscription(String, '/scene/understanding', self.scene_cb, 10)
        self.create_subscription(String, '/perception/opponent_v2v_sign', self.v2v_cb, 10)

        self.decision_pub = self.create_publisher(
            String, '/planning/cooperation_decision', 10)
        self.led_pattern_pub = self.create_publisher(
            String, '/planning/v2v_led_pattern', 10)

        detect_dt = (
            1.0 / self.deadlock_detection_hz
            if self.deadlock_detection_hz > 0.0 else 0.05
        )
        self.deadlock_timer = self.create_timer(detect_dt, self._detect_deadlock_step)
        dt = 1.0 / self.decision_hz if self.decision_hz > 0.0 else 0.2
        self.timer = self.create_timer(dt, self.decide_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def scene_cb(self, msg: String) -> None:
        try:
            self.scene_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON in scene understanding')
        self.last_scene_stamp = self.now_sec()

    def v2v_cb(self, msg: String) -> None:
        try:
            self.v2v_sign = json.loads(msg.data)
        except json.JSONDecodeError:
            self.v2v_sign = {'detected': False, 'sign': msg.data, 'mask': '00000'}
        if str(self.v2v_sign.get('sign', 'UNKNOWN')) == 'WEIGHT_READY':
            self.opponent_weight_ready_seen = True
        self.last_v2v_stamp = self.now_sec()

    def _detect_deadlock_step(self) -> None:
        now = self.now_sec()

        in_narrow = self.scene_data.get(
            'in_narrow_passage',
            self.scene_data.get('environment', {}).get('in_narrow_passage', False))
        nearest_narrow = self.scene_data.get(
            'nearest_narrow_passage_m',
            self.scene_data.get(
                'environment', {}).get('nearest_narrow_passage_m', float('inf')))
        opponent = self.scene_data.get('opponent', {})
        opponent_detected = bool(opponent.get('detected', False))
        opponent_angle = self._to_float(opponent.get('angle_rad', float('inf')), float('inf'))
        opponent_distance = self._to_float(
            opponent.get('distance_m', float('inf')), float('inf'))

        cond_narrow = bool(in_narrow) or (nearest_narrow < self.near_threshold_m)
        cond_opponent = opponent_detected
        cond_facing = abs(opponent_angle) < self.facing_angle_threshold_rad
        cond_close = opponent_distance < self.deadlock_distance_m

        cond_stopped = False
        if self.prev_opponent_distance is not None and self.prev_distance_stamp is not None:
            dt = now - self.prev_distance_stamp
            if dt > 0.0 and math.isfinite(opponent_distance):
                rate = abs(opponent_distance - self.prev_opponent_distance) / dt
                cond_stopped = rate < self.distance_change_threshold_m

        self.prev_opponent_distance = opponent_distance
        self.prev_distance_stamp = now

        detected = (cond_opponent and cond_facing and cond_close and
                    (cond_narrow or cond_stopped))
        confidence = (
            0.2 * float(cond_narrow) +
            0.2 * float(cond_opponent) +
            0.2 * float(cond_facing) +
            0.2 * float(cond_close) +
            0.2 * float(cond_stopped)
        )

        if detected:
            if self.deadlock_start_stamp is None:
                self.deadlock_start_stamp = now
            duration = now - self.deadlock_start_stamp
        else:
            self.deadlock_start_stamp = None
            duration = 0.0

        self.deadlock_state = {
            'detected': detected,
            'confidence': round(confidence, 3),
            'duration_sec': round(duration, 3),
            'opponent_distance_m': round(opponent_distance, 3),
        }

    def decide_step(self) -> None:
        now = self.now_sec()

        deadlock_detected = self.deadlock_state.get('detected', False)
        if not deadlock_detected:
            self._reset_negotiation()
            self.current_decision = 'WAIT_RECHECK'
            self.current_led_pattern = 'MASK:11111'
            self._publish(protocol='idle')
            return

        if self.negotiation_start_stamp is None:
            self.negotiation_start_stamp = now
            self.decision_stamp = None
            self.active_protocol = self._select_protocol()
            self.opponent_weight_ready_seen = False
            self.local_rps_choice = None
            self.rps_countdown_stamp = now
            self.negotiation_logged = False

        if not self.negotiation_logged:
            self.get_logger().info(
                f'협상 모드 진입: protocol={self.active_protocol}, '
                f'opponent_led={self.v2v_sign}')
            self.negotiation_logged = True

        if self.decision_stamp is not None:
            if (now - self.decision_stamp) < self.decision_hold_sec:
                self._publish(protocol='held')
                return

        elapsed = now - self.negotiation_start_stamp
        if elapsed >= self.max_negotiation_sec:
            self.current_decision = 'I_YIELD'
            self.current_led_pattern = 'MASK:11111'
            self.decision_stamp = now
            self.get_logger().info('Negotiation timeout, defaulting to I_YIELD')
            self._publish(protocol='timeout')
            return

        if self.active_protocol == 'rps':
            decision, status = self._decide_with_rps(now)
            self.current_decision = decision
            if decision in ('I_GO', 'I_YIELD'):
                self.decision_stamp = now
            self._publish(protocol=status)
            return

        decision, status = self._decide_with_weight()
        self.current_decision = decision
        if decision in ('I_GO', 'I_YIELD'):
            self.decision_stamp = now
        self._publish(protocol=status)

    def _reset_negotiation(self) -> None:
        self.negotiation_start_stamp = None
        self.decision_stamp = None
        self.active_protocol = None
        self.opponent_weight_ready_seen = False
        self.local_rps_choice = None
        self.rps_countdown_stamp = None
        self.negotiation_logged = False

    def _select_protocol(self) -> str:
        default = 'rps' if self.v2v_use_rps else 'weight'
        if not self.prompt_protocol_choice:
            return default

        try:
            answer = input(
                '\n[협상 모드] 가위바위보로 결정할까요? '
                'y=가위바위보 / n=양보점수: '
            ).strip().lower()
        except (EOFError, OSError):
            self.get_logger().warn(
                f'협상 프로토콜 입력을 받을 수 없어 기본값 사용: {default}')
            return default

        if answer in ('y', 'yes', '1', 'rps', 'rock', '가위바위보'):
            return 'rps'
        if answer in ('n', 'no', '0', 'weight', 'score', '점수', '양보점수'):
            return 'weight'
        self.get_logger().warn(f'알 수 없는 협상 입력 "{answer}", 기본값 사용: {default}')
        return default

    def _prompt_rps_choice(self) -> str:
        if not self.manual_rps_input:
            return self.default_rps_choice

        prompt = '[가위바위보] rock/r, scissors/s, paper/p 입력: '
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, OSError):
            self.get_logger().warn(
                f'가위바위보 입력을 받을 수 없어 기본값 사용: {self.default_rps_choice}')
            return self.default_rps_choice

        mapping = {
            'r': 'ROCK',
            'rock': 'ROCK',
            '바위': 'ROCK',
            's': 'SCISSORS',
            'scissor': 'SCISSORS',
            'scissors': 'SCISSORS',
            '가위': 'SCISSORS',
            'p': 'PAPER',
            'paper': 'PAPER',
            '보': 'PAPER',
        }
        choice = mapping.get(raw)
        if choice is None:
            self.get_logger().warn(
                f'알 수 없는 가위바위보 입력 "{raw}", 기본값 사용: {self.default_rps_choice}')
            return self.default_rps_choice
        return choice

    def _decide_with_rps(self, now: float) -> Tuple[str, str]:
        if self.rps_countdown_stamp is None:
            self.rps_countdown_stamp = now

        elapsed = now - self.rps_countdown_stamp
        if self.local_rps_choice is None:
            self.current_led_pattern = 'MASK:10101'
            if elapsed < self.rps_countdown_sec:
                remain = max(0.0, self.rps_countdown_sec - elapsed)
                return 'WAIT_RECHECK', f'rps_countdown remain={remain:.1f}s'
            self.local_rps_choice = self._prompt_rps_choice()
            self.get_logger().info(f'내 가위바위보 선택: {self.local_rps_choice}')

        opponent_choice = self.SIGN_TO_RPS.get(str(self.v2v_sign.get('sign', '')))
        self.current_led_pattern = f'MASK:{self.RPS_MASK[self.local_rps_choice]}'

        if opponent_choice is None:
            return 'WAIT_RECHECK', f'rps_wait local={self.local_rps_choice}'

        if opponent_choice == self.local_rps_choice:
            self.get_logger().info(
                f'가위바위보 비김: local={self.local_rps_choice}, opp={opponent_choice}; '
                f'{self.rps_countdown_sec:.1f}초 뒤 재입력')
            self.local_rps_choice = None
            self.rps_countdown_stamp = now
            self.current_led_pattern = 'MASK:10101'
            return 'WAIT_RECHECK', f'rps_tie local={opponent_choice}'

        local_wins = self.RPS_BEATS[self.local_rps_choice] == opponent_choice
        # Project rule: the RPS winner yields; the loser waits, then passes.
        if local_wins:
            return 'I_YIELD', f'rps_win_yield local={self.local_rps_choice} opp={opponent_choice}'
        return 'I_GO', f'rps_lose_go local={self.local_rps_choice} opp={opponent_choice}'

    def _decide_with_weight(self) -> Tuple[str, str]:
        score, value, detail = self._compute_yield_score()
        self.last_yield_score_detail = detail

        if self.negotiation_start_stamp is not None:
            elapsed = self.now_sec() - self.negotiation_start_stamp
            if elapsed < self.v2v_advertise_sec:
                self.current_led_pattern = 'MASK:01010'
                return 'WAIT_RECHECK', 'weight_ready'

        self.current_led_pattern = f'MASK:{value:05b}'

        if not self._v2v_fresh():
            return 'WAIT_RECHECK', f'weight_wait_no_sign local_score={score} value={value}'

        opponent_sign = str(self.v2v_sign.get('sign', 'UNKNOWN'))
        opponent_mask = str(self.v2v_sign.get('mask', '00000')).strip()
        opponent_detected = bool(self.v2v_sign.get('detected', False))
        if not opponent_detected:
            return 'WAIT_RECHECK', f'weight_wait_no_opponent local={value:05b}'

        if opponent_sign == 'WEIGHT_READY' or opponent_mask == '01010':
            self.opponent_weight_ready_seen = True
            return 'WAIT_RECHECK', f'weight_wait_opp_ready local={value:05b}'

        if opponent_sign in ('APPROACH', 'RPS_READY') and not self.opponent_weight_ready_seen:
            return 'WAIT_RECHECK', f'weight_wait_opp_sign={opponent_sign}'

        try:
            opponent_value = int(opponent_mask, 2)
        except (TypeError, ValueError):
            return 'WAIT_RECHECK', 'weight_wait_bad_value'

        if value > opponent_value:
            return 'I_YIELD', (
                f'weight_yield local={value:05b}({value}) '
                f'opp={opponent_value:05b}({opponent_value})')
        if value < opponent_value:
            return 'I_GO', (
                f'weight_go local={value:05b}({value}) '
                f'opp={opponent_value:05b}({opponent_value})')
        return 'WAIT_RECHECK', f'weight_tie value={value:05b}({value})'

    def _v2v_fresh(self) -> bool:
        if self.last_v2v_stamp is None:
            return False
        return (self.now_sec() - self.last_v2v_stamp) <= self.v2v_sign_timeout_sec

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _to_float(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _range_score(self, value: float, lo: float, hi: float) -> float:
        if not math.isfinite(value) or hi <= lo:
            return 0.0
        return self._clamp01((value - lo) / (hi - lo))

    def _distance_to(self, target: dict) -> float:
        pose = self.scene_data.get('pose', {})
        if not target or not pose:
            return float('inf')
        tx = float(target.get('x', float('nan')))
        ty = float(target.get('y', float('nan')))
        px = float(pose.get('x', float('nan')))
        py = float(pose.get('y', float('nan')))
        if not all(math.isfinite(v) for v in (tx, ty, px, py)):
            return float('inf')
        return math.hypot(tx - px, ty - py)

    @classmethod
    def _score_100_to_led_value(cls, score: int) -> int:
        score = max(0, min(100, int(score)))
        value = max(0, min(31, int(round(score * 31.0 / 100.0))))
        if value != cls.RESERVED_WEIGHT_LED_VALUE:
            return value
        return 9 if score < (cls.RESERVED_WEIGHT_LED_VALUE * 100.0 / 31.0) else 11

    def _compute_yield_score(self) -> Tuple[int, int, dict]:
        environment = self.scene_data.get('environment', {})

        road_class = str(
            self.scene_data.get('road_class',
                                environment.get('road_class', 'UNKNOWN'))).upper()
        road_width = self._to_float(
            self.scene_data.get('road_width_m', environment.get('road_width_m', -1.0)),
            -1.0,
        )

        rear = self.scene_data.get('rear_obstacle', {})
        rear_detected = bool(rear.get('detected', False))
        rear_dist = self._to_float(rear.get('distance_m', 999.0), 999.0)

        vehicle = self.scene_data.get('vehicle_state', {})
        battery_pct = self._to_float(vehicle.get('battery_pct', 80.0), 80.0)
        vehicle_fault = bool(vehicle.get('fault', False)) or not bool(vehicle.get('motion_ok', True))

        if road_class == 'A':
            road_rank = 2
            road_class_score = 1.0
        elif road_class == 'B':
            road_rank = 1
            road_class_score = 0.55
        elif road_class == 'C':
            road_rank = 0
            road_class_score = 0.10
        elif math.isfinite(road_width) and road_width > 0.0:
            if road_width >= 0.65:
                road_class = 'A'
                road_rank = 2
                road_class_score = 1.0
            elif road_width >= 0.45:
                road_class = 'B'
                road_rank = 1
                road_class_score = 0.55
            else:
                road_class = 'C'
                road_rank = 0
                road_class_score = 0.10
        else:
            road_rank = 1
            road_class_score = 0.35

        rear_clear_rank = 0 if rear_detected else 1
        rear_clearance_score = 1.0 if rear_clear_rank else 0.0

        if vehicle_fault:
            battery_score = 0.0
        elif math.isfinite(battery_pct):
            battery_score = self._clamp01(battery_pct / 100.0)
        else:
            battery_score = 0.5

        score_100 = max(0, min(100, int(round(
            (road_class_score * 50.0) +
            (rear_clearance_score * 30.0) +
            (battery_score * 20.0)
        ))))
        raw_value = max(0, min(31, int(round(score_100 * 31.0 / 100.0))))
        value = self._score_100_to_led_value(score_100)
        mask = self.SCORE_TO_MASK[value]

        terms = {
            'road_width_class': road_class_score,
            'rear_obstacle_clearance': rear_clearance_score,
            'battery_state': battery_score,
        }
        detail = {
            'score': score_100,
            'score_100': score_100,
            'score_0_31_raw': raw_value,
            'score_0_31': value,
            'score_binary': mask,
            'value': value,
            'formula': 'score_100=50*road_width_class + 30*rear_clearance + 20*battery_state; led_value=round(score_100*31/100), with value 10 remapped to 9/11',
            'priority': [
                'road_width_class',
                'rear_obstacle_clearance',
                'battery_state',
            ],
            'terms': {k: round(v, 3) for k, v in terms.items()},
            'weights': {
                'road_width_class': 50,
                'rear_obstacle_clearance': 30,
                'battery_state': 20,
            },
            'score_to_led_mapping': '0-100 -> 0-31, excluding 10',
            'reserved_led_value': self.RESERVED_WEIGHT_LED_VALUE,
            'road_class': road_class,
            'road_rank': road_rank,
            'rear_clear_rank': rear_clear_rank,
            'road_width_m': round(road_width, 3) if math.isfinite(road_width) else -1.0,
            'rear_obstacle': {
                'detected': rear_detected,
                'distance_m': round(rear_dist, 3) if math.isfinite(rear_dist) else -1.0,
            },
            'battery_pct': round(battery_pct, 1) if math.isfinite(battery_pct) else -1.0,
        }
        return score_100, value, detail

    def _publish(self, protocol: str) -> None:
        _ = protocol
        msg = String()
        msg.data = self.current_decision
        self.decision_pub.publish(msg)

        pattern_msg = String()
        pattern_msg.data = self.current_led_pattern
        self.led_pattern_pub.publish(pattern_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CooperationDecisionNode()
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
