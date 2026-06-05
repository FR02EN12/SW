// cmd_mux.cpp -- Velocity multiplexer with integrated safety filtering
#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class CmdMuxNode : public rclcpp::Node
{
public:
  CmdMuxNode()
  : Node("cmd_mux")
  {
    declare_parameter<double>("mux_hz", 20.0);
    declare_parameter<double>("source_timeout_sec", 0.5);
    declare_parameter<double>("transition_duration_sec", 0.3);
    declare_parameter<double>("approach_speed_scale", 0.7);
    declare_parameter<double>("emergency_hold_sec", 1.0);
    declare_parameter<double>("warning_scale", 0.5);
    declare_parameter<double>("max_linear_accel", 0.03);
    declare_parameter<double>("max_angular_accel", 0.5);
    declare_parameter<double>("max_linear_speed", 0.05);
    declare_parameter<double>("max_angular_speed", 0.5);

    mux_hz_ = get_parameter("mux_hz").as_double();
    source_timeout_sec_ = get_parameter("source_timeout_sec").as_double();
    transition_duration_sec_ = get_parameter("transition_duration_sec").as_double();
    approach_speed_scale_ = get_parameter("approach_speed_scale").as_double();
    emergency_hold_sec_ = get_parameter("emergency_hold_sec").as_double();
    warning_scale_ = get_parameter("warning_scale").as_double();
    max_linear_accel_ = get_parameter("max_linear_accel").as_double();
    max_angular_accel_ = get_parameter("max_angular_accel").as_double();
    max_linear_speed_ = get_parameter("max_linear_speed").as_double();
    max_angular_speed_ = get_parameter("max_angular_speed").as_double();

    sub_mode_ = create_subscription<std_msgs::msg::String>(
      "/planning/driving_mode", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        driving_mode_ = trim(msg->data);
      });

    sub_lane_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_lane", 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        last_cmd_lane_ = *msg;
        last_cmd_lane_stamp_ = now_sec();
      });

    sub_path_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_path", 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        last_cmd_path_ = *msg;
        last_cmd_path_stamp_ = now_sec();
      });

    sub_safety_ = create_subscription<std_msgs::msg::String>(
      "/safety/events", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        safety_cb(msg->data);
      });

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    double dt = (mux_hz_ > 0.0) ? (1.0 / mux_hz_) : 0.05;
    timer_ = create_wall_timer(
      std::chrono::duration<double>(dt),
      std::bind(&CmdMuxNode::mux_step, this));
  }

private:
  double now_sec() const
  {
    return this->now().seconds();
  }

  static std::string trim(const std::string & s)
  {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
  }

  static std::string to_upper(const std::string & s)
  {
    std::string t = trim(s);
    for (auto & c : t) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return t;
  }

  static std::string extract_string_field(const std::string & data, const std::string & key)
  {
    std::string search = "\"" + key + "\"";
    auto pos = data.find(search);
    if (pos == std::string::npos) return "";
    auto colon = data.find(':', pos + search.size());
    if (colon == std::string::npos) return "";
    auto q1 = data.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = data.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return data.substr(q1 + 1, q2 - q1 - 1);
  }

  void safety_cb(const std::string & data)
  {
    std::string level = extract_string_field(data, "severity");
    if (level.empty()) {
      level = extract_string_field(data, "type");
    }
    if (level.empty()) {
      level = data;
    }
    level = to_upper(level);

    if (level == "EMERGENCY" || level == "CRITICAL") {
      safety_level_ = "EMERGENCY";
      emergency_until_ = now_sec() + emergency_hold_sec_;
    } else if (level == "WARNING") {
      safety_level_ = "WARNING";
    } else if (level == "INFO" || level == "CLEAR") {
      safety_level_ = "CLEAR";
    }
  }

  std::string source_for_mode(const std::string & mode) const
  {
    std::string mu = to_upper(mode);
    if (mu == "LANE_FOLLOW" || mu == "APPROACH_NARROW") return "lane";
    if (mu == "YIELD_REVERSE" || mu == "YIELD_SIDE" || mu == "REENTER") return "path";
    if (mu == "WAIT_FOR_PASS" || mu == "YIELD_WAIT_CLEAR") return "zero";
    return "zero";
  }

  bool source_fresh(const std::string & source) const
  {
    double now = now_sec();
    if (source == "lane") {
      return last_cmd_lane_.has_value() && last_cmd_lane_stamp_.has_value() &&
        (now - *last_cmd_lane_stamp_) <= source_timeout_sec_;
    }
    if (source == "path") {
      return last_cmd_path_.has_value() && last_cmd_path_stamp_.has_value() &&
        (now - *last_cmd_path_stamp_) <= source_timeout_sec_;
    }
    return true;
  }

  geometry_msgs::msg::Twist get_source_cmd(const std::string & source) const
  {
    if (source == "lane" && last_cmd_lane_.has_value()) return *last_cmd_lane_;
    if (source == "path" && last_cmd_path_.has_value()) return *last_cmd_path_;
    return geometry_msgs::msg::Twist();
  }

  static geometry_msgs::msg::Twist blend_twist(
    const geometry_msgs::msg::Twist & a,
    const geometry_msgs::msg::Twist & b,
    double alpha)
  {
    geometry_msgs::msg::Twist out;
    out.linear.x = a.linear.x * (1.0 - alpha) + b.linear.x * alpha;
    out.angular.z = a.angular.z * (1.0 - alpha) + b.angular.z * alpha;
    return out;
  }

  void publish_stop()
  {
    cmd_pub_->publish(geometry_msgs::msg::Twist());
    prev_linear_ = 0.0;
    prev_angular_ = 0.0;
    prev_stamp_ = now_sec();
  }

  geometry_msgs::msg::Twist apply_safety_limits(const geometry_msgs::msg::Twist & input)
  {
    double now = now_sec();
    double target_lin = input.linear.x;
    double target_ang = input.angular.z;

    if (safety_level_ == "WARNING") {
      target_lin *= warning_scale_;
      target_ang *= warning_scale_;
    }

    target_lin = std::clamp(target_lin, -max_linear_speed_, max_linear_speed_);
    target_ang = std::clamp(target_ang, -max_angular_speed_, max_angular_speed_);

    if (prev_stamp_.has_value()) {
      double dt = now - *prev_stamp_;
      if (dt > 0.0) {
        double max_dlin = max_linear_accel_ * dt;
        double max_dang = max_angular_accel_ * dt;
        target_lin = prev_linear_ +
          std::clamp(target_lin - prev_linear_, -max_dlin, max_dlin);
        target_ang = prev_angular_ +
          std::clamp(target_ang - prev_angular_, -max_dang, max_dang);
      }
    }

    prev_linear_ = target_lin;
    prev_angular_ = target_ang;
    prev_stamp_ = now;

    geometry_msgs::msg::Twist out;
    out.linear.x = target_lin;
    out.angular.z = target_ang;
    return out;
  }

  void mux_step()
  {
    double now = now_sec();

    if (emergency_until_.has_value()) {
      if (now < *emergency_until_) {
        publish_stop();
        return;
      }
      emergency_until_.reset();
      if (safety_level_ == "EMERGENCY") {
        safety_level_ = "CLEAR";
      }
    }

    std::string mode = driving_mode_;
    std::string source = source_for_mode(mode);
    if (!source_fresh(source)) {
      publish_stop();
      return;
    }

    if (prev_source_.has_value() && source != *prev_source_) {
      transition_start_ = now;
      prev_output_ = last_published_;
    }

    auto target_cmd = get_source_cmd(source);
    if (to_upper(mode) == "APPROACH_NARROW") {
      target_cmd.linear.x *= approach_speed_scale_;
    }

    geometry_msgs::msg::Twist selected;
    if (transition_start_.has_value()) {
      double elapsed = now - *transition_start_;
      if (elapsed < transition_duration_sec_ && transition_duration_sec_ > 0.0) {
        selected = blend_twist(prev_output_, target_cmd, elapsed / transition_duration_sec_);
      } else {
        selected = target_cmd;
        transition_start_.reset();
      }
    } else {
      selected = target_cmd;
    }

    geometry_msgs::msg::Twist output = apply_safety_limits(selected);
    prev_source_ = source;
    last_published_ = output;
    cmd_pub_->publish(output);
  }

  double mux_hz_{};
  double source_timeout_sec_{};
  double transition_duration_sec_{};
  double approach_speed_scale_{};
  double emergency_hold_sec_{};
  double warning_scale_{};
  double max_linear_accel_{};
  double max_angular_accel_{};
  double max_linear_speed_{};
  double max_angular_speed_{};

  std::string driving_mode_;
  std::optional<geometry_msgs::msg::Twist> last_cmd_lane_;
  std::optional<double> last_cmd_lane_stamp_;
  std::optional<geometry_msgs::msg::Twist> last_cmd_path_;
  std::optional<double> last_cmd_path_stamp_;

  std::string safety_level_{"CLEAR"};
  std::optional<double> emergency_until_;
  double prev_linear_{0.0};
  double prev_angular_{0.0};
  std::optional<double> prev_stamp_;

  std::optional<std::string> prev_source_;
  std::optional<double> transition_start_;
  geometry_msgs::msg::Twist prev_output_;
  geometry_msgs::msg::Twist last_published_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mode_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_lane_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_path_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_safety_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CmdMuxNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
