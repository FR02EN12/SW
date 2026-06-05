#!/usr/bin/env python3
"""LED I/O node: physical LED control interface (GPIO, Arduino serial, or simulation)."""

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LedIoNode(Node):
    RESERVED_SCORE_VALUE = 10
    SCORE_TO_MASK = {value: f'{value:05b}' for value in range(32)}

    ARDUINO_CMD_BY_MASK = {
        '00000': '0',  # normal
        '11111': '1',  # all on / legacy car detected
        '10101': '2',  # game
        '01010': '3',  # no game / weight ready
        '11000': '4',  # rock
        '10100': '5',  # scissor
        '10010': '6',  # paper
    }

    def __init__(self) -> None:
        super().__init__('led_io')

        self.declare_parameter('backend', 'log_only')
        self.declare_parameter('arduino_port', '/dev/ttyACM1')
        self.declare_parameter('arduino_fallback_ports', ['/dev/ttyACM1', '/dev/ttyUSB1'])
        self.declare_parameter('arduino_baudrate', 9600)
        self.declare_parameter('arduino_timeout_sec', 0.1)
        self.declare_parameter('arduino_boot_wait_sec', 2.0)
        self.declare_parameter('arduino_command_newline', True)
        self.declare_parameter('arduino_raw_mask_enabled', True)
        self.declare_parameter('gpio_led_pins', [17, 27, 22, 5, 6])
        self.declare_parameter('blink_hz', 2.0)

        self.backend = str(self.get_parameter('backend').value)
        self.arduino_port = str(self.get_parameter('arduino_port').value)
        self.arduino_fallback_ports = [
            str(port) for port in self.get_parameter('arduino_fallback_ports').value
        ]
        self.arduino_baudrate = int(self.get_parameter('arduino_baudrate').value)
        self.arduino_timeout_sec = float(
            self.get_parameter('arduino_timeout_sec').value)
        self.arduino_boot_wait_sec = float(
            self.get_parameter('arduino_boot_wait_sec').value)
        self.arduino_command_newline = bool(
            self.get_parameter('arduino_command_newline').value)
        self.arduino_raw_mask_enabled = bool(
            self.get_parameter('arduino_raw_mask_enabled').value)
        self.gpio_led_pins = [
            int(pin) for pin in self.get_parameter('gpio_led_pins').value
        ]
        self.blink_hz = float(self.get_parameter('blink_hz').value)

        self.current_cmd: str = ''
        self.current_state: str = 'OFF'
        self.blink_on: bool = False
        self.gpio = None
        self.serial_conn = None
        self.serial_port_open: Optional[str] = None

        # Try to initialise GPIO backend
        if self.backend == 'gpio':
            try:
                import RPi.GPIO as GPIO
                self.gpio = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                if len(self.gpio_led_pins) < 5:
                    self.get_logger().warn(
                        'gpio_led_pins must contain five pins; falling back to log_only')
                    self.backend = 'log_only'
                else:
                    for pin in self.gpio_led_pins[:5]:
                        GPIO.setup(pin, GPIO.OUT)
                        GPIO.output(pin, GPIO.LOW)
                    self.get_logger().info('GPIO backend initialised')
            except ImportError:
                self.get_logger().warn('RPi.GPIO not available, falling back to log_only')
                self.backend = 'log_only'

        if self.backend in ('arduino_serial', 'serial'):
            self._init_arduino_serial()

        self.create_subscription(String, '/vehicle/led_cmd', self._cmd_cb, 10)

        # Blink timer (for EMERGENCY)
        blink_dt = 1.0 / (self.blink_hz * 2.0) if self.blink_hz > 0 else 0.25
        self.create_timer(blink_dt, self._blink_tick)
        self._apply_led('MASK:11111')
        self.current_cmd = 'MASK:11111'

        self.get_logger().info(f'LedIoNode started, backend={self.backend}')

    def _cmd_cb(self, msg: String) -> None:
        cmd = msg.data.strip()
        if cmd != self.current_cmd:
            self.current_cmd = cmd
            if cmd.upper() != 'EMERGENCY':
                self._apply_led(cmd)

    def _blink_tick(self) -> None:
        if self.current_cmd.upper() == 'EMERGENCY':
            self.blink_on = not self.blink_on
            if self.blink_on:
                self._set_mask('11111')
                self.current_state = 'MASK:11111'
            else:
                self._set_mask('00000')
                self.current_state = 'MASK:00000'

    def _apply_led(self, cmd: str) -> None:
        mask = self._extract_mask(cmd)
        if mask is not None:
            self._set_mask(mask)
            self.current_state = f'MASK:{mask}'
            return

        self.get_logger().warn(
            f'Invalid LED command "{cmd}". Use MASK:xxxxx or SCORE_0_31:n.')

    def _extract_mask(self, cmd: str) -> str | None:
        raw = cmd.strip()
        upper = raw.upper()
        if upper.startswith('MASK:'):
            mask = raw.split(':', 1)[1].strip()
            return mask if self._valid_mask(mask) else None
        if upper.startswith(('VALUE:', 'BINARY:', 'SCORE:', 'SCORE_0_31:')):
            try:
                value = int(raw.split(':', 1)[1].strip())
            except ValueError:
                return None
            return self._score_to_mask(value)
        if self._valid_mask(raw):
            return raw
        if raw.startswith('{'):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
            mask = str(data.get('mask', '')).strip()
            if self._valid_mask(mask):
                return mask
            if 'value' in data:
                try:
                    return self._score_to_mask(int(data['value']))
                except (TypeError, ValueError):
                    return None
            if 'score_0_31' in data:
                try:
                    return self._score_to_mask(int(data['score_0_31']))
                except (TypeError, ValueError):
                    return None
        return None

    def _valid_mask(self, mask: str) -> bool:
        return len(mask) == 5 and all(bit in ('0', '1') for bit in mask)

    def _score_to_mask(self, value: int) -> str:
        value = max(0, min(31, int(value)))
        if value == self.RESERVED_SCORE_VALUE:
            value = 11
        return self.SCORE_TO_MASK[value]

    def _set_mask(self, mask: str) -> None:
        if self.backend == 'gpio' and self.gpio is not None and len(self.gpio_led_pins) >= 5:
            for pin, bit in zip(self.gpio_led_pins[:5], mask):
                self.gpio.output(pin, self.gpio.HIGH if bit == '1' else self.gpio.LOW)
        elif self.backend == 'arduino_serial':
            self._set_arduino_mask(mask)
        else:
            self.get_logger().debug(f'LED [log_only]: MASK={mask}')

    def _init_arduino_serial(self) -> None:
        try:
            import serial
        except ImportError:
            self.get_logger().warn('pyserial not available, falling back to log_only')
            self.backend = 'log_only'
            return

        ports = self._candidate_arduino_ports()
        for port in ports:
            try:
                conn = serial.Serial(
                    port=port,
                    baudrate=self.arduino_baudrate,
                    timeout=self.arduino_timeout_sec,
                    write_timeout=self.arduino_timeout_sec,
                )
                if self.arduino_boot_wait_sec > 0.0:
                    time.sleep(self.arduino_boot_wait_sec)
                self.serial_conn = conn
                self.serial_port_open = port
                self.backend = 'arduino_serial'
                self.get_logger().info(
                    f'Arduino LED serial connected: {port} @ {self.arduino_baudrate}')
                self._set_arduino_mask('11111')
                return
            except Exception as exc:
                self.get_logger().warn(f'Failed to open Arduino LED serial {port}: {exc}')

        self.get_logger().warn(
            'No Arduino LED serial port opened; falling back to log_only')
        self.backend = 'log_only'

    def _candidate_arduino_ports(self) -> list[str]:
        ports = [self.arduino_port] + self.arduino_fallback_ports
        result: list[str] = []
        for port in ports:
            port = str(port).strip()
            if port and port not in result:
                result.append(port)
        return result

    def _set_arduino_mask(self, mask: str) -> None:
        command = self.ARDUINO_CMD_BY_MASK.get(mask)
        if command is None:
            if self.arduino_raw_mask_enabled:
                command = f'M{mask}'
            else:
                self.get_logger().warn(
                    f'Arduino code does not define MASK:{mask}; sending normal/off')
                command = self.ARDUINO_CMD_BY_MASK['00000']

        if self.serial_conn is None:
            self.get_logger().debug(f'Arduino serial not connected: cmd={command} mask={mask}')
            return

        payload = command + ('\n' if self.arduino_command_newline else '')
        try:
            self.serial_conn.write(payload.encode('ascii'))
            self.serial_conn.flush()
            self.get_logger().debug(
                f'Arduino LED command sent: cmd={command} mask={mask}')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to write Arduino LED command {command} for MASK:{mask}: {exc}')

    def destroy_node(self) -> None:
        if self.backend == 'gpio' and self.gpio is not None:
            for pin in self.gpio_led_pins[:5]:
                self.gpio.output(pin, self.gpio.LOW)
            self.gpio.cleanup()
        if self.serial_conn is not None:
            try:
                self._set_arduino_mask('00000')
                self.serial_conn.close()
            except BaseException:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LedIoNode()
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
