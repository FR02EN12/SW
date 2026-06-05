# FSD Cooperative Driving Stack

## Build

```bash
cd ~/fsd_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to bringup --symlink-install
```

## Run

```bash
source /opt/ros/humble/setup.bash
source ~/fsd_ws/install/setup.bash
ros2 launch bringup fsd.launch.py
```

## Flow

```text
perception_py/camera_driver
  -> perception_py/lane_final
  -> perception_cpp/localization_slam + perception_py/led_signal, space_memory, situation_fusion
  -> decision_cpp/path_plan, safety_supervisor + decision_py/drive_mode, cooperation, behavior_plan
  -> control_py/lane_follow, led_behavior, led_io + control_cpp/path_track, cmd_mux
```

## Source Layout

```text
src/bringup/                        launch and shared FSD params
src/perception_cpp/                 localization and map
src/perception_py/                  camera, lane, LED, space, situation perception
src/decision_cpp/                   path plan and safety supervisor
src/decision_py/                    drive mode, cooperation, behavior planning
src/control_cpp/                    path tracking and command mux
src/control_py/                     lane follow, LED behavior, LED IO, Arduino firmware
```
