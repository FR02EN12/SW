// path_plan.cpp -- A* path planner
// Target: Jetson Orin Nano / ROS 2 Humble+
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

static double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

// ---------------------------------------------------------------------------
// Minimal inline JSON helpers (no external deps)
// ---------------------------------------------------------------------------
namespace json_util {

// Extract the raw value (string, number, bool, or nested {}/[]) for a key.
// Handles nested objects and arrays so callers can pass a sub-object to another
// get_* call without a full JSON library.
inline std::string find_value(const std::string &json, const std::string &key,
                               const std::string &fallback = "") {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size()) return fallback;
  if (json[pos] == '"') {
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return fallback;
    return json.substr(pos + 1, end - pos - 1);
  }
  if (json[pos] == '{' || json[pos] == '[') {
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

// Return the string value for a given key, or fallback.
inline std::string get_string(const std::string &json, const std::string &key,
                              const std::string &fallback = "") {
  return find_value(json, key, fallback);
}

inline double get_double(const std::string &json, const std::string &key,
                         double fallback = 0.0) {
  std::string v = find_value(json, key, "");
  if (v.empty()) return fallback;
  try { return std::stod(v); } catch (...) { return fallback; }
}

inline bool get_bool(const std::string &json, const std::string &key,
                     bool fallback = false) {
  std::string v = find_value(json, key, "");
  return v == "true" ? true : (v == "false" ? false : fallback);
}

// Parse a JSON array of objects and return each object as a raw string.
inline std::vector<std::string> get_array_objects(const std::string &json) {
  std::vector<std::string> result;
  auto start = json.find('[');
  if (start == std::string::npos) return result;
  std::size_t pos = start + 1;
  while (pos < json.size()) {
    auto obj_start = json.find('{', pos);
    if (obj_start == std::string::npos) break;
    int depth = 1;
    std::size_t i = obj_start + 1;
    while (i < json.size() && depth > 0) {
      if (json[i] == '{') ++depth;
      else if (json[i] == '}') --depth;
      ++i;
    }
    result.push_back(json.substr(obj_start, i - obj_start));
    pos = i;
  }
  return result;
}

}  // namespace json_util

// ---------------------------------------------------------------------------
// Hash for std::pair<int,int>
// ---------------------------------------------------------------------------
struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    // Combine two ints using a fast mixer.
    auto h1 = std::hash<int>{}(p.first);
    auto h2 = std::hash<int>{}(p.second);
    return h1 ^ (h2 * 2654435761u + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
  }
};

// ---------------------------------------------------------------------------
// A* open-set element
// ---------------------------------------------------------------------------
struct AStarEntry {
  double f;
  double g;
  int x;
  int y;
  bool operator>(const AStarEntry &o) const { return f > o.f; }
};

// ---------------------------------------------------------------------------
// PathPlannerNode
// ---------------------------------------------------------------------------
class PathPlannerNode : public rclcpp::Node {
public:
  PathPlannerNode() : Node("path_plan") {
    // Parameters
    this->declare_parameter<double>("inflation_radius_m", 0.15);
    this->declare_parameter<int>("max_iterations", 0);
    this->declare_parameter<double>("replan_distance_m", 0.3);
    this->declare_parameter<double>("path_resolution_m", 0.05);
    this->declare_parameter<double>("plan_hz", 2.0);
    this->declare_parameter<int>("smoothing_window", 5);

    inflation_radius_m_ = this->get_parameter("inflation_radius_m").as_double();
    max_iterations_     = this->get_parameter("max_iterations").as_int();
    replan_distance_m_  = this->get_parameter("replan_distance_m").as_double();
    plan_hz_            = this->get_parameter("plan_hz").as_double();
    smoothing_window_   = static_cast<int>(this->get_parameter("smoothing_window").as_int());

    // Subscribers
    goal_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/planning/behavior_goal", 10,
        std::bind(&PathPlannerNode::goal_cb, this, std::placeholders::_1));
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 10,
        std::bind(&PathPlannerNode::map_cb, this, std::placeholders::_1));
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pose", 10,
        std::bind(&PathPlannerNode::pose_cb, this, std::placeholders::_1));
    pullover_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/memory/pull_over_candidates", 10,
        std::bind(&PathPlannerNode::pullover_cb, this, std::placeholders::_1));
    scene_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/scene/understanding", 10,
        std::bind(&PathPlannerNode::scene_cb, this, std::placeholders::_1));

    // Publisher
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planning/path", 10);

    // Timer
    double dt = (plan_hz_ > 0.0) ? (1.0 / plan_hz_) : 0.5;
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(dt),
        std::bind(&PathPlannerNode::plan_step, this));
  }

private:
  // Parameters
  double inflation_radius_m_{};
  int    max_iterations_{};
  double replan_distance_m_{};
  double plan_hz_{};
  int    smoothing_window_{};

  // Map state
  int    map_width_{0};
  int    map_height_{0};
  double map_resolution_{0.05};
  double map_origin_x_{0.0};
  double map_origin_y_{0.0};
  std::vector<int> cost_map_;

  // Latest messages
  std::string behavior_goal_json_;
  std::string last_goal_json_;
  geometry_msgs::msg::PoseStamped::SharedPtr current_pose_;
  nav_msgs::msg::Path::SharedPtr last_published_path_;

  // Cached goal fields for replan check
  std::string last_goal_type_;
  double last_goal_x_{0.0};
  double last_goal_y_{0.0};

  // Scene data for dynamic obstacle injection (opponent vehicle position)
  std::string scene_json_;
  bool has_scene_{false};

  // Stored pullover candidates: list of (world_x, world_y) pairs
  std::vector<std::pair<double, double>> pullover_candidates_;

  // Path deviation monitoring: force replan when robot drifts > replan_distance_m_ from path
  static constexpr double kPathDevThreshold = 0.25;  // metres

  // ROS handles
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pullover_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr scene_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ------------------------------------------------------------------
  // Callbacks
  // ------------------------------------------------------------------
  void goal_cb(const std_msgs::msg::String::SharedPtr msg) {
    behavior_goal_json_ = msg->data;
  }

  void map_cb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    map_width_      = static_cast<int>(msg->info.width);
    map_height_     = static_cast<int>(msg->info.height);
    map_resolution_ = msg->info.resolution;
    map_origin_x_   = msg->info.origin.position.x;
    map_origin_y_   = msg->info.origin.position.y;
    build_cost_map(msg->data);
  }

  void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    current_pose_ = msg;
  }

  void pullover_cb(const std_msgs::msg::String::SharedPtr msg) {
    pullover_candidates_.clear();
    for (const auto &obj : json_util::get_array_objects(msg->data)) {
      double px = json_util::get_double(obj, "x");
      double py = json_util::get_double(obj, "y");
      pullover_candidates_.emplace_back(px, py);
    }
  }

  void scene_cb(const std_msgs::msg::String::SharedPtr msg) {
    scene_json_ = msg->data;
    has_scene_ = true;
  }

  // ------------------------------------------------------------------
  // Cost map with obstacle inflation
  // ------------------------------------------------------------------
  void build_cost_map(const std::vector<int8_t> &raw) {
    const int w = map_width_;
    const int h = map_height_;
    const int sz = w * h;
    const int inflate_cells = std::max(1, static_cast<int>(inflation_radius_m_ / map_resolution_));

    cost_map_.assign(sz, 0);

    // Mark occupied
    for (int i = 0; i < sz; ++i) {
      if (raw[i] > 50 || raw[i] < 0) {
        cost_map_[i] = 100;
      }
    }

    // Inflate on a copy to avoid cascading.
    std::vector<int> inflated(cost_map_);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (cost_map_[y * w + x] < 100) continue;
        for (int dy = -inflate_cells; dy <= inflate_cells; ++dy) {
          int ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (int dx = -inflate_cells; dx <= inflate_cells; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            double dist = std::sqrt(static_cast<double>(dx * dx + dy * dy));
            if (dist <= static_cast<double>(inflate_cells)) {
              int idx = ny * w + nx;
              if (inflated[idx] < 100) inflated[idx] = 100;
            }
          }
        }
      }
    }
    cost_map_ = std::move(inflated);
  }

  // ------------------------------------------------------------------
  // Coordinate transforms
  // ------------------------------------------------------------------
  std::pair<int, int> world_to_grid(double wx, double wy) const {
    int gx = static_cast<int>(std::floor((wx - map_origin_x_) / map_resolution_));
    int gy = static_cast<int>(std::floor((wy - map_origin_y_) / map_resolution_));
    return {gx, gy};
  }

  std::pair<double, double> grid_to_world(int gx, int gy) const {
    double wx = gx * map_resolution_ + map_origin_x_ + map_resolution_ * 0.5;
    double wy = gy * map_resolution_ + map_origin_y_ + map_resolution_ * 0.5;
    return {wx, wy};
  }

  // ------------------------------------------------------------------
  // Replan heuristic
  // ------------------------------------------------------------------
  bool need_replan() const {
    std::string goal_type = json_util::get_string(behavior_goal_json_, "type");
    if (goal_type != last_goal_type_) return true;
    double tx = json_util::get_double(behavior_goal_json_, "target_x");
    double ty = json_util::get_double(behavior_goal_json_, "target_y");
    double dx = tx - last_goal_x_;
    double dy = ty - last_goal_y_;
    return std::sqrt(dx * dx + dy * dy) > replan_distance_m_;
  }

  // ------------------------------------------------------------------
  // Planning timer callback
  // ------------------------------------------------------------------
  void plan_step() {
    std::string goal_type = json_util::get_string(behavior_goal_json_, "type");

    if (goal_type != "reverse_to_pullover" &&
        goal_type != "side_pull_over" &&
        goal_type != "reenter_lane") {
      // Cache goal type so need_replan() doesn't trigger on next same non-planning goal
      last_goal_type_ = goal_type;
      auto empty = std::make_unique<nav_msgs::msg::Path>();
      empty->header.stamp = this->now();
      empty->header.frame_id = "map";
      path_pub_->publish(std::move(*empty));
      return;
    }

    if (!current_pose_ || cost_map_.empty()) return;

    // Force replan when robot has drifted off the current path
    if (path_deviation_exceeded()) {
      RCLCPP_INFO(this->get_logger(), "Path deviation exceeded %.2fm; replanning", kPathDevThreshold);
      last_published_path_.reset();
      last_goal_type_.clear();
    }

    if (!need_replan() && last_published_path_) {
      path_pub_->publish(*last_published_path_);
      return;
    }

    double start_x = current_pose_->pose.position.x;
    double start_y = current_pose_->pose.position.y;
    double goal_x  = json_util::get_double(behavior_goal_json_, "target_x");
    double goal_y  = json_util::get_double(behavior_goal_json_, "target_y");
    double goal_theta = json_util::get_double(behavior_goal_json_, "target_theta");
    bool   is_reverse = json_util::get_bool(behavior_goal_json_, "reverse");

    auto [sx, sy] = world_to_grid(start_x, start_y);
    auto [gx, gy] = world_to_grid(goal_x, goal_y);

    // Build a working cost map with the opponent vehicle injected as a
    // dynamic obstacle so A* avoids their current position
    std::vector<int> working_cost = cost_map_;
    inject_opponent_cost(working_cost);

    auto path_opt = astar_with_cost(sx, sy, gx, gy, working_cost);
    if (!path_opt.has_value()) {
      RCLCPP_WARN(this->get_logger(), "A* failed to find path; publishing fallback reverse path");
      // Fallback: publish a short in-place reverse path so the controller
      // knows to back up slightly and retry rather than stall indefinitely
      auto fallback = std::make_shared<nav_msgs::msg::Path>();
      fallback->header.stamp = this->now();
      fallback->header.frame_id = "map";
      double start_yaw = yaw_from_quaternion(current_pose_->pose.orientation);
      for (int step = 1; step <= 5; ++step) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = fallback->header;
        double backoff = step * map_resolution_ * 2.0;
        ps.pose.position.x = start_x - backoff * std::cos(start_yaw);
        ps.pose.position.y = start_y - backoff * std::sin(start_yaw);
        ps.pose.position.z = 0.0;
        ps.pose.orientation = current_pose_->pose.orientation;
        fallback->poses.push_back(ps);
      }
      path_pub_->publish(*fallback);
      return;
    }

    auto path_indices = smooth_path(path_opt.value());

    // Post-smoothing validity: if the smoothed path clips through an obstacle
    // (moving-average can cut corners), fall back to the raw A* path
    if (!smooth_path_valid(path_indices, working_cost)) {
      RCLCPP_WARN(this->get_logger(), "Smoothed path clips obstacle; using raw A* path");
      path_indices = path_opt.value();
    }

    // Convert to world points
    std::vector<std::pair<double, double>> world_pts;
    world_pts.reserve(path_indices.size());
    for (auto &[px, py] : path_indices) {
      world_pts.push_back(grid_to_world(px, py));
    }

    // Build Path message
    auto path_msg = std::make_shared<nav_msgs::msg::Path>();
    path_msg->header.stamp = this->now();
    path_msg->header.frame_id = "map";
    path_msg->poses.reserve(world_pts.size());

    for (std::size_t i = 0; i < world_pts.size(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path_msg->header;
      pose.pose.position.x = world_pts[i].first;
      pose.pose.position.y = world_pts[i].second;
      pose.pose.position.z = 0.0;

      double yaw;
      if (i + 1 < world_pts.size()) {
        double dx = world_pts[i + 1].first  - world_pts[i].first;
        double dy = world_pts[i + 1].second - world_pts[i].second;
        yaw = std::atan2(dy, dx);
        if (is_reverse) yaw += M_PI;
      } else {
        yaw = goal_theta;
        if (is_reverse) yaw += M_PI;
      }

      pose.pose.orientation.z = std::sin(yaw * 0.5);
      pose.pose.orientation.w = std::cos(yaw * 0.5);
      path_msg->poses.push_back(pose);
    }

    path_pub_->publish(*path_msg);
    last_published_path_ = path_msg;
    last_goal_type_ = goal_type;
    last_goal_x_ = goal_x;
    last_goal_y_ = goal_y;

    RCLCPP_INFO(this->get_logger(),
                "Published path with %zu poses (type=%s, reverse=%s)",
                path_msg->poses.size(), goal_type.c_str(),
                is_reverse ? "true" : "false");
  }

  // ------------------------------------------------------------------
  // Dynamic opponent cost injection
  // Temporarily marks the opponent vehicle's estimated grid cells as
  // occupied so A* treats it as an obstacle during reverse planning.
  // Caller must pass a working copy of cost_map_; the base map is unchanged.
  // ------------------------------------------------------------------
  void inject_opponent_cost(std::vector<int> &tmp_cost) const {
    if (!has_scene_ || scene_json_.empty() || cost_map_.empty()) return;

    const std::string opp = json_util::find_value(scene_json_, "opponent");
    if (opp.empty() || !json_util::get_bool(opp, "detected")) return;

    double opp_dist  = json_util::get_double(opp, "distance_m", 999.0);
    double opp_angle = json_util::get_double(opp, "angle_rad",  0.0);

    // Derive opponent world position from our pose + polar measurement
    const std::string pose_json = json_util::find_value(scene_json_, "pose");
    double pose_x     = json_util::get_double(pose_json, "x",     0.0);
    double pose_y     = json_util::get_double(pose_json, "y",     0.0);
    double pose_theta = json_util::get_double(pose_json, "theta", 0.0);

    double opp_wx = pose_x + opp_dist * std::cos(pose_theta + opp_angle);
    double opp_wy = pose_y + opp_dist * std::sin(pose_theta + opp_angle);

    auto [gx, gy] = world_to_grid(opp_wx, opp_wy);

    // Inflate by robot body radius + safety margin (~25 cm)
    int inflate = std::max(1, static_cast<int>(0.25 / map_resolution_));
    for (int dy = -inflate; dy <= inflate; ++dy) {
      for (int dx = -inflate; dx <= inflate; ++dx) {
        int nx = gx + dx, ny = gy + dy;
        if (nx < 0 || nx >= map_width_ || ny < 0 || ny >= map_height_) continue;
        if (std::hypot(static_cast<double>(dx), static_cast<double>(dy)) <= inflate)
          tmp_cost[ny * map_width_ + nx] = 100;
      }
    }
  }

  // ------------------------------------------------------------------
  // Path deviation check
  // Returns true when the robot's current pose has drifted further than
  // kPathDevThreshold from every waypoint in the last published path,
  // signalling that a replan is needed.
  // ------------------------------------------------------------------
  bool path_deviation_exceeded() const {
    if (!last_published_path_ || !current_pose_) return false;
    if (last_published_path_->poses.empty()) return false;

    double px = current_pose_->pose.position.x;
    double py = current_pose_->pose.position.y;
    double min_dist = std::numeric_limits<double>::max();
    for (const auto &ps : last_published_path_->poses) {
      double d = std::hypot(px - ps.pose.position.x, py - ps.pose.position.y);
      if (d < min_dist) min_dist = d;
    }
    return min_dist > kPathDevThreshold;
  }

  // ------------------------------------------------------------------
  // Post-smoothing obstacle validity check
  // Ensures no cell in the smoothed path is marked occupied (smoothing
  // can cut corners through inflated obstacles).
  // ------------------------------------------------------------------
  bool smooth_path_valid(const std::vector<std::pair<int, int>> &path,
                         const std::vector<int> &cost) const {
    if (path.empty()) return false;
    for (auto &[px, py] : path) {
      if (px < 0 || px >= map_width_ || py < 0 || py >= map_height_) return false;
      if (cost[py * map_width_ + px] >= 100) return false;
    }
    for (std::size_t i = 1; i < path.size(); ++i) {
      if (!line_free(path[i - 1].first, path[i - 1].second,
                     path[i].first, path[i].second, cost)) {
        return false;
      }
    }
    return true;
  }

  bool cell_blocked(int x, int y, const std::vector<int> &cost) const {
    if (x < 0 || x >= map_width_ || y < 0 || y >= map_height_) return true;
    return cost[y * map_width_ + x] >= 100;
  }

  bool line_free(int x0, int y0, int x1, int y1, const std::vector<int> &cost) const {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int x = x0, y = y0;

    while (true) {
      if (cell_blocked(x, y, cost)) return false;
      if (x == x1 && y == y1) return true;
      int prev_x = x;
      int prev_y = y;
      int e2 = 2 * err;
      if (e2 > -dy) { err -= dy; x += sx; }
      if (e2 <  dx) { err += dx; y += sy; }

      if (x != prev_x && y != prev_y) {
        if (cell_blocked(x, prev_y, cost) || cell_blocked(prev_x, y, cost)) {
          return false;
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // A* search on an 8-connected grid.
  // astar_with_cost accepts an externally supplied cost map (e.g., with
  // dynamic obstacles already injected); astar() is a convenience wrapper.
  // ------------------------------------------------------------------
  std::optional<std::vector<std::pair<int, int>>>
  astar(int sx, int sy, int gx, int gy) const {
    return astar_with_cost(sx, sy, gx, gy, cost_map_);
  }

  std::optional<std::vector<std::pair<int, int>>>
  astar_with_cost(int sx, int sy, int gx, int gy,
                  const std::vector<int> &cost) const {
    const int w = map_width_;
    const int h = map_height_;

    if (sx < 0 || sx >= w || sy < 0 || sy >= h) return std::nullopt;
    if (gx < 0 || gx >= w || gy < 0 || gy >= h) return std::nullopt;
    if (cost[sy * w + sx] >= 100) return std::nullopt;
    if (cost[gy * w + gx] >= 100) return std::nullopt;

    // 8-connected neighbor offsets
    static constexpr int DX[8] = { 1, -1,  0,  0, 1,  1, -1, -1};
    static constexpr int DY[8] = { 0,  0,  1, -1, 1, -1,  1, -1};
    static constexpr double COST[8] = {1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414};

    auto heuristic = [gx, gy](int x, int y) -> double {
      double dx = x - gx;
      double dy = y - gy;
      return std::sqrt(dx * dx + dy * dy);
    };

    // Min-heap
    std::priority_queue<AStarEntry, std::vector<AStarEntry>, std::greater<AStarEntry>> open;
    open.push({heuristic(sx, sy), 0.0, sx, sy});

    std::unordered_map<std::pair<int,int>, double, PairHash> g_cost;
    g_cost[{sx, sy}] = 0.0;

    std::unordered_map<std::pair<int,int>, std::pair<int,int>, PairHash> came_from;
    std::unordered_set<std::pair<int,int>, PairHash> closed;

    const int iteration_limit = max_iterations_ > 0 ? max_iterations_ : w * h;
    int iterations = 0;
    while (!open.empty() && iterations < iteration_limit) {
      ++iterations;
      auto cur = open.top();
      open.pop();

      std::pair<int,int> cp{cur.x, cur.y};
      if (closed.count(cp)) continue;
      closed.insert(cp);

      if (cur.x == gx && cur.y == gy) {
        // Reconstruct
        std::vector<std::pair<int,int>> path;
        std::pair<int,int> node{gx, gy};
        path.push_back(node);
        while (came_from.count(node)) {
          node = came_from[node];
          path.push_back(node);
        }
        std::reverse(path.begin(), path.end());
        return path;
      }

      for (int i = 0; i < 8; ++i) {
        int nx = cur.x + DX[i];
        int ny = cur.y + DY[i];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        std::pair<int,int> np{nx, ny};
        if (closed.count(np)) continue;
        if (cost[ny * w + nx] >= 100) continue;  // use injected cost map
        if (DX[i] != 0 && DY[i] != 0) {
          if (cost[cur.y * w + nx] >= 100 || cost[ny * w + cur.x] >= 100) continue;
        }

        double new_g = cur.g + COST[i];
        auto it = g_cost.find(np);
        if (it == g_cost.end() || new_g < it->second) {
          g_cost[np] = new_g;
          open.push({new_g + heuristic(nx, ny), new_g, nx, ny});
          came_from[np] = cp;
        }
      }
    }
    return std::nullopt;
  }

  // ------------------------------------------------------------------
  // Moving-average path smoothing
  // ------------------------------------------------------------------
  std::vector<std::pair<int,int>>
  smooth_path(const std::vector<std::pair<int,int>> &path) const {
    int n = static_cast<int>(path.size());
    if (n <= smoothing_window_) return path;

    int half = smoothing_window_ / 2;
    std::vector<std::pair<int,int>> smoothed;
    smoothed.reserve(n);

    for (int i = 0; i < n; ++i) {
      int start = std::max(0, i - half);
      int end   = std::min(n, i + half + 1);
      double sx = 0.0, sy = 0.0;
      for (int j = start; j < end; ++j) {
        sx += path[j].first;
        sy += path[j].second;
      }
      int cnt = end - start;
      smoothed.emplace_back(
          static_cast<int>(std::round(sx / cnt)),
          static_cast<int>(std::round(sy / cnt)));
    }
    return smoothed;
  }
};

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
