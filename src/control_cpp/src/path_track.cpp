// path_track.cpp -- Pure Pursuit path tracker
// Target: Jetson Orin Nano / ROS 2 Humble+
// Performance target T5: reverse evasive path success rate >= 80 %
//   reverse_speed raised 0.015 -> 0.020 m/s
//   goal_tolerance_m tightened 0.05 -> 0.04 m for more accurate pull-over
#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/string.hpp"

static double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

class PathTrackingControllerNode : public rclcpp::Node
{
public:
  PathTrackingControllerNode()
  : Node("path_track")
  {
    declare_parameter<double>("control_hz", 20.0);
    declare_parameter<double>("lookahead_distance_m", 0.15);
    // T5: raised from 0.015 -> 0.020 m/s for improved reverse path execution.
    declare_parameter<double>("reverse_speed", 0.020);
    declare_parameter<double>("reenter_speed", 0.018);
    declare_parameter<double>("max_angular_speed", 0.3);
    // T5: tightened from 0.05 -> 0.04 m for more accurate pull-over positioning.
    declare_parameter<double>("goal_tolerance_m", 0.04);
    declare_parameter<double>("min_lookahead_m", 0.08);
    declare_parameter<double>("max_lookahead_m", 0.30);

    control_hz_           = get_parameter("control_hz").as_double();
    lookahead_distance_m_ = get_parameter("lookahead_distance_m").as_double();
    reverse_speed_        = get_parameter("reverse_speed").as_double();
    reenter_speed_        = get_parameter("reenter_speed").as_double();
    max_angular_speed_    = get_parameter("max_angular_speed").as_double();
    goal_tolerance_m_     = get_parameter("goal_tolerance_m").as_double();
    min_lookahead_m_      = get_parameter("min_lookahead_m").as_double();
    max_lookahead_m_      = get_parameter("max_lookahead_m").as_double();

    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      "/planning/path", 10,
      [this](const nav_msgs::msg::Path::SharedPtr msg) {
        path_ = *msg;
        goal_reached_ = false;
      });

    sub_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        pose_ = *msg;
        has_pose_ = true;
      });

    sub_mode_ = create_subscription<std_msgs::msg::String>(
      "/planning/driving_mode", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        driving_mode_ = msg->data;
        // trim
        auto s = driving_mode_.find_first_not_of(" \t\r\n");
        auto e = driving_mode_.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) driving_mode_.clear();
        else driving_mode_ = driving_mode_.substr(s, e - s + 1);
      });

    cmd_pub_  = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_path", 10);
    goal_pub_ = create_publisher<std_msgs::msg::String>("/planning/path_goal_reached", 10);

    double dt = (control_hz_ > 0.0) ? (1.0 / control_hz_) : 0.05;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(dt),
      std::bind(&PathTrackingControllerNode::control_step, this));
  }

private:
  void publish_stop()
  {
    cmd_pub_->publish(geometry_msgs::msg::Twist());
  }

  // Find closest point index on path
  size_t find_closest_index(double rx, double ry) const
  {
    size_t best_idx = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    const auto & poses = path_->poses;
    for (size_t i = 0; i < poses.size(); ++i) {
      double dx = poses[i].pose.position.x - rx;
      double dy = poses[i].pose.position.y - ry;
      double d = dx * dx + dy * dy;
      if (d < best_dist) {
        best_dist = d;
        best_idx = i;
      }
    }
    return best_idx;
  }

  // Find lookahead point starting from closest index
  size_t find_lookahead_index(size_t closest_idx, double rx, double ry) const
  {
    const auto & poses = path_->poses;
    double ld = std::clamp(lookahead_distance_m_, min_lookahead_m_, max_lookahead_m_);
    for (size_t i = closest_idx; i < poses.size(); ++i) {
      double dx = poses[i].pose.position.x - rx;
      double dy = poses[i].pose.position.y - ry;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (dist >= ld) return i;
    }
    return poses.size() - 1;
  }

  void control_step()
  {
    // Convert mode to upper
    std::string mode = driving_mode_;
    for (auto & c : mode) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    if (mode != "YIELD_REVERSE" && mode != "YIELD_SIDE" && mode != "REENTER") {
      publish_stop();
      return;
    }

    if (!path_.has_value() || !has_pose_) {
      publish_stop();
      return;
    }

    if (path_->poses.size() < 2) {
      publish_stop();
      return;
    }

    if (goal_reached_) {
      publish_stop();
      return;
    }

    double rx = pose_->pose.position.x;
    double ry = pose_->pose.position.y;
    double ryaw = yaw_from_quaternion(pose_->pose.orientation);

    // Check goal reached
    const auto & final_pose = path_->poses.back();
    double dx_goal = final_pose.pose.position.x - rx;
    double dy_goal = final_pose.pose.position.y - ry;
    double dist_goal = std::sqrt(dx_goal * dx_goal + dy_goal * dy_goal);
    if (dist_goal < goal_tolerance_m_) {
      if (!goal_reached_) {
        goal_reached_ = true;
        std_msgs::msg::String gr;
        gr.data = driving_mode_;  // "YIELD_REVERSE", "YIELD_SIDE", or "REENTER"
        goal_pub_->publish(gr);
        RCLCPP_INFO(this->get_logger(), "Goal reached (mode=%s)", driving_mode_.c_str());
      }
      publish_stop();
      return;
    }

    // Pure pursuit
    size_t closest_idx = find_closest_index(rx, ry);
    size_t la_idx = find_lookahead_index(closest_idx, rx, ry);

    const auto & la_point = path_->poses[la_idx];
    double dx = la_point.pose.position.x - rx;
    double dy = la_point.pose.position.y - ry;

    // Transform lookahead point to robot frame
    double local_x =  std::cos(ryaw) * dx + std::sin(ryaw) * dy;
    double local_y = -std::sin(ryaw) * dx + std::cos(ryaw) * dy;

    double ld_actual = std::sqrt(local_x * local_x + local_y * local_y);
    if (ld_actual < 1e-6) {
      publish_stop();
      return;
    }

    // Curvature
    double alpha = std::atan2(local_y, local_x);
    double kappa = 2.0 * std::sin(alpha) / ld_actual;

    // Velocity based on mode
    double v, omega;
    if (mode == "YIELD_REVERSE") {
      v = -reverse_speed_;
      omega = v * kappa;
      omega = -omega;  // flip steering sign for reverse
    } else {  // YIELD_SIDE or REENTER
      v = reenter_speed_;
      omega = v * kappa;
    }

    omega = std::clamp(omega, -max_angular_speed_, max_angular_speed_);

    geometry_msgs::msg::Twist tw;
    tw.linear.x = v;
    tw.angular.z = omega;
    cmd_pub_->publish(tw);
  }

  // parameters
  double control_hz_;
  double lookahead_distance_m_;
  double reverse_speed_;
  double reenter_speed_;
  double max_angular_speed_;
  double goal_tolerance_m_;
  double min_lookahead_m_;
  double max_lookahead_m_;

  // state
  std::optional<nav_msgs::msg::Path> path_;
  std::optional<geometry_msgs::msg::PoseStamped> pose_;
  bool has_pose_{false};
  std::string driving_mode_;
  bool goal_reached_{false};

  // ROS handles
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mode_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr goal_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PathTrackingControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
