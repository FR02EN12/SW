// safety_supervisor.cpp -- Safety supervisor
// Target: Jetson Orin Nano / ROS 2 Humble+
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"

// ---------------------------------------------------------------------------
// Minimal inline JSON helpers
// ---------------------------------------------------------------------------
namespace json_util {

// Find value string for a top-level or nested key.
// For nested access use dot notation: "opponent.distance_m"
inline std::string find_value(const std::string &json, const std::string &key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return "";
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size()) return "";
  if (json[pos] == '"') {
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
  }
  if (json[pos] == '{' || json[pos] == '[') {
    // Return nested object/array as-is (find matching brace)
    char open = json[pos], close = (open == '{') ? '}' : ']';
    int depth = 1;
    std::size_t i = pos + 1;
    while (i < json.size() && depth > 0) {
      if (json[i] == open) ++depth;
      else if (json[i] == close) --depth;
      ++i;
    }
    return json.substr(pos, i - pos);
  }
  auto end = json.find_first_of(",} \t\n\r", pos);
  if (end == std::string::npos) end = json.size();
  return json.substr(pos, end - pos);
}

inline double get_double(const std::string &json, const std::string &key, double fb = 0.0) {
  auto v = find_value(json, key);
  if (v.empty()) return fb;
  try { return std::stod(v); } catch (...) { return fb; }
}

inline bool get_bool(const std::string &json, const std::string &key, bool fb = false) {
  auto v = find_value(json, key);
  return v == "true" ? true : (v == "false" ? false : fb);
}

inline std::string get_string(const std::string &json, const std::string &key,
                              const std::string &fb = "") {
  auto v = find_value(json, key);
  return v.empty() ? fb : v;
}

}  // namespace json_util

// ---------------------------------------------------------------------------
using SteadyClock = std::chrono::steady_clock;
using TimePoint   = std::chrono::steady_clock::time_point;

static double elapsed_sec(const TimePoint &from) {
  return std::chrono::duration<double>(SteadyClock::now() - from).count();
}

// ---------------------------------------------------------------------------
class SafetySupervisorNode : public rclcpp::Node {
public:
  SafetySupervisorNode() : Node("safety_supervisor") {
    // Parameters
    this->declare_parameter<double>("safety_hz", 20.0);
    this->declare_parameter<double>("emergency_stop_distance_m", 0.15);
    this->declare_parameter<double>("stuck_timeout_sec", 5.0);
    this->declare_parameter<double>("lane_lost_timeout_sec", 3.0);
    this->declare_parameter<double>("scene_timeout_sec", 2.0);

    safety_hz_                = this->get_parameter("safety_hz").as_double();
    emergency_stop_distance_m_ = this->get_parameter("emergency_stop_distance_m").as_double();
    stuck_timeout_sec_        = this->get_parameter("stuck_timeout_sec").as_double();
    lane_lost_timeout_sec_    = this->get_parameter("lane_lost_timeout_sec").as_double();
    scene_timeout_sec_        = this->get_parameter("scene_timeout_sec").as_double();

    // State
    last_pose_change_stamp_ = SteadyClock::now();

    // Subscribers
    scene_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/scene/understanding", 10,
        std::bind(&SafetySupervisorNode::scene_cb, this, std::placeholders::_1));
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
        std::bind(&SafetySupervisorNode::cmd_vel_cb, this, std::placeholders::_1));
    // Publishers
    event_pub_ = this->create_publisher<std_msgs::msg::String>("/safety/events", 10);

    // Timer
    double dt = 1.0 / safety_hz_;
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(dt),
        std::bind(&SafetySupervisorNode::check_safety, this));

    RCLCPP_INFO(this->get_logger(), "SafetySupervisorNode started at %.0f Hz", safety_hz_);
  }

private:
  // Parameters
  double safety_hz_{};
  double emergency_stop_distance_m_{};
  double stuck_timeout_sec_{};
  double lane_lost_timeout_sec_{};
  double scene_timeout_sec_{};

  // State
  std::string scene_json_;
  bool   has_scene_{false};
  TimePoint last_scene_stamp_;

  bool   has_cmd_vel_{false};
  geometry_msgs::msg::Twist last_cmd_vel_;

  double last_pose_x_{0.0};
  double last_pose_y_{0.0};
  bool   has_last_pose_{false};
  TimePoint last_pose_change_stamp_;

  bool   lane_lost_active_{false};
  TimePoint lane_lost_stamp_;

  std::string current_severity_{"CLEAR"};

  // ROS handles
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr scene_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ------------------------------------------------------------------
  void scene_cb(const std_msgs::msg::String::SharedPtr msg) {
    scene_json_ = msg->data;
    has_scene_ = true;
    last_scene_stamp_ = SteadyClock::now();
  }

  void cmd_vel_cb(const geometry_msgs::msg::Twist::SharedPtr msg) {
    last_cmd_vel_ = *msg;
    has_cmd_vel_ = true;
  }

  // ------------------------------------------------------------------
  void publish_event(const std::string &event_type, const std::string &severity,
                     const std::string &message) {
    std_msgs::msg::String msg;
    msg.data = "{\"type\":\"" + event_type +
               "\",\"severity\":\"" + severity +
               "\",\"message\":\"" + message + "\"}";
    event_pub_->publish(msg);
    current_severity_ = severity;
  }

  void safety_mode_shell(std::vector<std::string> &events_fired) {
    (void)events_fired;
    // TODO(safety-mode): wire rear obstacle, battery, and actuator-health inputs.
    // Intended future policy:
    // - rear obstacle close during reverse -> EMERGENCY
    // - low battery or degraded actuator state -> WARNING / yield-score penalty
    // - repeated controller fault -> CRITICAL stop and require operator clear
  }

  // ------------------------------------------------------------------
  void check_safety() {
    auto now = SteadyClock::now();
    std::vector<std::string> events_fired;

    // --- 1. Communication timeout ---
    if (has_scene_) {
      double age = elapsed_sec(last_scene_stamp_);
      if (age > scene_timeout_sec_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "No scene update for %.1fs", age);
        publish_event("COMM_FAIL", "CRITICAL", buf);
        events_fired.push_back("COMM_FAIL");
      }
    }

    // --- 2. Collision proximity ---
    if (has_scene_ && !scene_json_.empty()) {
      std::string opp = json_util::find_value(scene_json_, "opponent");
      if (!opp.empty()) {
        bool detected = json_util::get_bool(opp, "detected");
        if (detected) {
          double dist = json_util::get_double(opp, "distance_m", 999.0);
          if (dist < emergency_stop_distance_m_) {
            char buf[80];
            std::snprintf(buf, sizeof(buf), "Collision imminent: opponent at %.2fm", dist);
            publish_event("EMERGENCY", "CRITICAL", buf);
            events_fired.push_back("EMERGENCY");
          }
        }
      }
    }

    // --- 3. Lane lost timeout ---
    if (has_scene_ && !scene_json_.empty()) {
      std::string lane = json_util::find_value(scene_json_, "lane");
      std::string status = lane.empty() ? "ok" : json_util::get_string(lane, "status", "ok");
      if (status == "lost") {
        if (!lane_lost_active_) {
          lane_lost_active_ = true;
          lane_lost_stamp_ = now;
        } else {
          double dur = elapsed_sec(lane_lost_stamp_);
          if (dur > lane_lost_timeout_sec_) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "Lane lost for %.1fs", dur);
            publish_event("LANE_LOST", "WARNING", buf);
            events_fired.push_back("LANE_LOST");
          }
        }
      } else {
        lane_lost_active_ = false;
      }
    }

    // --- 4. Stuck detection ---
    if (has_scene_ && has_cmd_vel_) {
      std::string pose_json = json_util::find_value(scene_json_, "pose");
      double px = json_util::get_double(pose_json, "x");
      double py = json_util::get_double(pose_json, "y");

      bool cmd_moving = (std::fabs(last_cmd_vel_.linear.x) > 0.001 ||
                         std::fabs(last_cmd_vel_.angular.z) > 0.01);

      if (has_last_pose_) {
        double dist = std::hypot(px - last_pose_x_, py - last_pose_y_);
        if (dist > 0.005) {
          last_pose_change_stamp_ = now;
        }
      }
      last_pose_x_ = px;
      last_pose_y_ = py;
      has_last_pose_ = true;

      if (cmd_moving && elapsed_sec(last_pose_change_stamp_) > stuck_timeout_sec_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Robot stuck for %.1fs",
                      elapsed_sec(last_pose_change_stamp_));
        publish_event("STUCK", "WARNING", buf);
        events_fired.push_back("STUCK");
      }
    }

    safety_mode_shell(events_fired);

    // CLEAR
    if (events_fired.empty() && current_severity_ != "CLEAR") {
      publish_event("CLEAR", "INFO", "All safety checks passed");
    }
  }
};

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SafetySupervisorNode>());
  rclcpp::shutdown();
  return 0;
}
