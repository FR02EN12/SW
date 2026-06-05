#!/usr/bin/env python3
"""Compatibility wrapper for the verified usb_cam camera driver.

The FSD stack keeps the camera_driver entry point in perception_py,
but the actual camera capture is delegated to usb_cam_node_exe, matching the
camera command that was validated on the robot.
"""

import os

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from rclpy.node import Node


class UsbCamWrapper(Node):
    def __init__(self) -> None:
        super().__init__('camera_driver')

        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('pixel_format', 'yuyv2rgb')
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('camera_name', 'fisheye')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')

    def usb_cam_command(self) -> list[str]:
        try:
            usb_cam_prefix = get_package_prefix('usb_cam')
        except PackageNotFoundError as exc:
            raise RuntimeError(
                'usb_cam package is required for camera_driver. '
                'Install/source the same environment used by: '
                'ros2 run usb_cam usb_cam_node_exe'
            ) from exc

        executable = os.path.join(usb_cam_prefix, 'lib', 'usb_cam', 'usb_cam_node_exe')
        if not os.path.exists(executable):
            raise RuntimeError(f'usb_cam executable not found: {executable}')

        framerate = self.get_parameter('framerate').value
        if framerate is None:
            framerate = self.get_parameter('fps').value

        image_topic = str(self.get_parameter('image_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)

        return [
            executable,
            '--ros-args',
            '-r', '__node:=camera_driver',
            '-r', f'image_raw:={image_topic}',
            '-r', f'camera_info:={camera_info_topic}',
            '-p', f'video_device:={self.get_parameter("video_device").value}',
            '-p', f'image_width:={int(self.get_parameter("image_width").value)}',
            '-p', f'image_height:={int(self.get_parameter("image_height").value)}',
            '-p', f'pixel_format:={self.get_parameter("pixel_format").value}',
            '-p', f'framerate:={float(framerate)}',
            '-p', f'camera_name:={self.get_parameter("camera_name").value}',
        ]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UsbCamWrapper()
    try:
        command = node.usb_cam_command()
        node.get_logger().info('Replacing camera_driver with usb_cam_node_exe')
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

    os.execv(command[0], command)


if __name__ == '__main__':
    main()
