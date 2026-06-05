// localization_slam.cpp -- EKF-based localization with occupancy grid
// Fuses /scan + /odom + /imu, publishes /pose + /map + /tf.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

static double normalize_angle(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

static double yaw_from_quat(double qx, double qy, double qz, double qw) {
  return std::atan2(2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz));
}

// 3x3 matrix helpers (row-major)
struct Mat3 {
  double m[9]{};
  Mat3() { std::memset(m, 0, sizeof(m)); }
  double& at(int r, int c) { return m[r * 3 + c]; }
  double  at(int r, int c) const { return m[r * 3 + c]; }
  static Mat3 eye() { Mat3 I; I.at(0,0)=I.at(1,1)=I.at(2,2)=1.0; return I; }
};

static Mat3 mat_add(const Mat3& A, const Mat3& B) {
  Mat3 C; for (int i = 0; i < 9; ++i) C.m[i] = A.m[i] + B.m[i]; return C;
}

static Mat3 mat_mul(const Mat3& A, const Mat3& B) {
  Mat3 C;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        C.at(i,j) += A.at(i,k) * B.at(k,j);
  return C;
}

static Mat3 mat_transpose(const Mat3& A) {
  Mat3 T;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) T.at(i,j) = A.at(j,i);
  return T;
}

static bool mat_inv(const Mat3& A, Mat3& Ainv) {
  // 3x3 inverse via cofactors
  double det =
      A.at(0,0)*(A.at(1,1)*A.at(2,2)-A.at(1,2)*A.at(2,1))
    - A.at(0,1)*(A.at(1,0)*A.at(2,2)-A.at(1,2)*A.at(2,0))
    + A.at(0,2)*(A.at(1,0)*A.at(2,1)-A.at(1,1)*A.at(2,0));
  if (std::abs(det) < 1e-15) return false;
  double inv_det = 1.0 / det;
  Ainv.at(0,0) = (A.at(1,1)*A.at(2,2)-A.at(1,2)*A.at(2,1)) * inv_det;
  Ainv.at(0,1) = (A.at(0,2)*A.at(2,1)-A.at(0,1)*A.at(2,2)) * inv_det;
  Ainv.at(0,2) = (A.at(0,1)*A.at(1,2)-A.at(0,2)*A.at(1,1)) * inv_det;
  Ainv.at(1,0) = (A.at(1,2)*A.at(2,0)-A.at(1,0)*A.at(2,2)) * inv_det;
  Ainv.at(1,1) = (A.at(0,0)*A.at(2,2)-A.at(0,2)*A.at(2,0)) * inv_det;
  Ainv.at(1,2) = (A.at(0,2)*A.at(1,0)-A.at(0,0)*A.at(1,2)) * inv_det;
  Ainv.at(2,0) = (A.at(1,0)*A.at(2,1)-A.at(1,1)*A.at(2,0)) * inv_det;
  Ainv.at(2,1) = (A.at(0,1)*A.at(2,0)-A.at(0,0)*A.at(2,1)) * inv_det;
  Ainv.at(2,2) = (A.at(0,0)*A.at(1,1)-A.at(0,1)*A.at(1,0)) * inv_det;
  return true;
}

// ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
class LocalizationSlamNode : public rclcpp::Node {
public:
  LocalizationSlamNode() : Node("localization_slam") {
    // Parameters
    declare_parameter<double>("localization_hz", 20.0);
    declare_parameter<double>("map_publish_hz", 1.0);
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_footprint");
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("imu_topic", "/imu");
    declare_parameter<std::string>("scan_topic", "/scan");

    // EKF process noise
    declare_parameter<double>("process_noise_xy", 0.002);
    declare_parameter<double>("process_noise_theta", 0.005);
    // Observation noise (scan matching)
    declare_parameter<double>("obs_noise_xy", 0.05);
    declare_parameter<double>("obs_noise_theta", 0.02);

    // Map parameters
    declare_parameter<double>("map_resolution", 0.05);
    declare_parameter<int>("map_width", 200);
    declare_parameter<int>("map_height", 200);
    declare_parameter<double>("map_origin_x", -5.0);
    declare_parameter<double>("map_origin_y", -5.0);
    declare_parameter<double>("log_odds_hit", 0.9);
    declare_parameter<double>("log_odds_miss", -0.7);
    declare_parameter<double>("log_odds_max", 5.0);
    declare_parameter<double>("log_odds_min", -2.0);

    // Scan matching
    declare_parameter<double>("scan_match_max_range", 3.0);
    declare_parameter<int>("scan_match_skip", 4);

    loc_hz_     = get_parameter("localization_hz").as_double();
    map_pub_hz_ = get_parameter("map_publish_hz").as_double();
    map_frame_  = get_parameter("map_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    imu_topic_  = get_parameter("imu_topic").as_string();
    scan_topic_ = get_parameter("scan_topic").as_string();

    pn_xy_    = get_parameter("process_noise_xy").as_double();
    pn_theta_ = get_parameter("process_noise_theta").as_double();
    on_xy_    = get_parameter("obs_noise_xy").as_double();
    on_theta_ = get_parameter("obs_noise_theta").as_double();

    map_res_  = get_parameter("map_resolution").as_double();
    map_w_    = get_parameter("map_width").as_int();
    map_h_    = get_parameter("map_height").as_int();
    map_ox_   = get_parameter("map_origin_x").as_double();
    map_oy_   = get_parameter("map_origin_y").as_double();
    lo_hit_   = get_parameter("log_odds_hit").as_double();
    lo_miss_  = get_parameter("log_odds_miss").as_double();
    lo_max_   = get_parameter("log_odds_max").as_double();
    lo_min_   = get_parameter("log_odds_min").as_double();
    sm_range_ = get_parameter("scan_match_max_range").as_double();
    sm_skip_  = std::max(1, static_cast<int>(get_parameter("scan_match_skip").as_int()));

    // Init state
    P_ = Mat3::eye();
    P_.at(0,0) = 0.1; P_.at(1,1) = 0.1; P_.at(2,2) = 0.1;

    log_odds_.assign(map_w_ * map_h_, 0.0);

    // TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Subscribers
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 10,
        std::bind(&LocalizationSlamNode::odom_cb, this, std::placeholders::_1));
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, 10,
        std::bind(&LocalizationSlamNode::imu_cb, this, std::placeholders::_1));
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LocalizationSlamNode::scan_cb, this, std::placeholders::_1));

    // Publishers
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pose", 10);
    pub_map_  = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/map", rclcpp::QoS(1).transient_local().reliable());

    // Localization timer
    double dt = (loc_hz_ > 0.0) ? (1.0 / loc_hz_) : 0.05;
    loc_timer_ = create_wall_timer(
        std::chrono::duration<double>(dt),
        std::bind(&LocalizationSlamNode::localization_step, this));

    // Map publish timer
    double mdt = (map_pub_hz_ > 0.0) ? (1.0 / map_pub_hz_) : 1.0;
    map_timer_ = create_wall_timer(
        std::chrono::duration<double>(mdt),
        std::bind(&LocalizationSlamNode::publish_map, this));

    RCLCPP_INFO(get_logger(),
        "LocalizationSlamNode started: loc=%.0fHz map=%.1fHz grid=%dx%d@%.2fm",
        loc_hz_, map_pub_hz_, map_w_, map_h_, map_res_);
  }

private:
  // Parameters
  double loc_hz_, map_pub_hz_;
  std::string map_frame_, odom_frame_, base_frame_;
  std::string odom_topic_, imu_topic_, scan_topic_;
  double pn_xy_, pn_theta_, on_xy_, on_theta_;
  double map_res_;
  int map_w_, map_h_;
  double map_ox_, map_oy_;
  double lo_hit_, lo_miss_, lo_max_, lo_min_;
  double sm_range_;
  int sm_skip_;

  // EKF state: x, y, theta in map frame
  double x_{0.0}, y_{0.0}, theta_{0.0};
  Mat3 P_;

  // Odometry tracking
  double odom_x_{0.0}, odom_y_{0.0}, odom_theta_{0.0};
  double prev_odom_x_{0.0}, prev_odom_y_{0.0}, prev_odom_theta_{0.0};
  bool has_odom_{false};

  // IMU
  double imu_yaw_rate_{0.0};

  // Scan
  sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
  std::mutex scan_mutex_;

  // Occupancy grid (log-odds)
  std::vector<double> log_odds_;

  // ROS handles
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_;
  rclcpp::TimerBase::SharedPtr loc_timer_, map_timer_;

  // Callbacks
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    double ox = msg->pose.pose.position.x;
    double oy = msg->pose.pose.position.y;
    auto &q = msg->pose.pose.orientation;
    double ot = yaw_from_quat(q.x, q.y, q.z, q.w);

    if (!has_odom_) {
      prev_odom_x_ = ox; prev_odom_y_ = oy; prev_odom_theta_ = ot;
      has_odom_ = true;
    }
    odom_x_ = ox; odom_y_ = oy; odom_theta_ = ot;
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_yaw_rate_ = msg->angular_velocity.z;
  }

  void scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    last_scan_ = msg;
  }

  // Grid helpers
  std::pair<int,int> world_to_grid(double wx, double wy) const {
    int gx = static_cast<int>(std::floor((wx - map_ox_) / map_res_));
    int gy = static_cast<int>(std::floor((wy - map_oy_) / map_res_));
    return {gx, gy};
  }

  bool in_grid(int gx, int gy) const {
    return gx >= 0 && gx < map_w_ && gy >= 0 && gy < map_h_;
  }

  void update_cell(int gx, int gy, double delta) {
    if (!in_grid(gx, gy)) return;
    double &v = log_odds_[gy * map_w_ + gx];
    v = std::clamp(v + delta, lo_min_, lo_max_);
  }

  // Bresenham ray from (x0,y0) to (x1,y1). A finite return marks the endpoint
  // occupied; max-range/no-return rays only carve free space.
  void trace_ray(int x0, int y0, int x1, int y1, bool mark_hit) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int cx = x0, cy = y0;
    while (true) {
      if (cx == x1 && cy == y1) {
        update_cell(cx, cy, mark_hit ? lo_hit_ : lo_miss_);
        break;
      }
      update_cell(cx, cy, lo_miss_);
      int e2 = 2 * err;
      if (e2 > -dy) { err -= dy; cx += sx; }
      if (e2 <  dx) { err += dx; cy += sy; }
      if (!in_grid(cx, cy)) break;
    }
  }

  // EKF predict
  void ekf_predict() {
    if (!has_odom_) return;

    // Compute odometry delta
    double dx_odom = odom_x_ - prev_odom_x_;
    double dy_odom = odom_y_ - prev_odom_y_;
    double dtheta_odom = normalize_angle(odom_theta_ - prev_odom_theta_);

    prev_odom_x_ = odom_x_;
    prev_odom_y_ = odom_y_;
    prev_odom_theta_ = odom_theta_;

    // Transform delta to robot frame then to map frame
    double d_trans = std::sqrt(dx_odom * dx_odom + dy_odom * dy_odom);
    double d_rot1 = (d_trans > 1e-4) ?
        normalize_angle(std::atan2(dy_odom, dx_odom) - prev_odom_theta_ + dtheta_odom) : 0.0;

    // Apply in map frame
    x_ += d_trans * std::cos(theta_ + d_rot1);
    y_ += d_trans * std::sin(theta_ + d_rot1);
    theta_ = normalize_angle(theta_ + dtheta_odom);

    // Jacobian F = dg/dx
    Mat3 F = Mat3::eye();
    F.at(0, 2) = -d_trans * std::sin(theta_);
    F.at(1, 2) =  d_trans * std::cos(theta_);

    // Process noise
    Mat3 Q;
    double dt2 = d_trans * d_trans;
    Q.at(0, 0) = pn_xy_ * pn_xy_ + 0.01 * dt2;
    Q.at(1, 1) = pn_xy_ * pn_xy_ + 0.01 * dt2;
    Q.at(2, 2) = pn_theta_ * pn_theta_ + 0.1 * dtheta_odom * dtheta_odom;

    P_ = mat_add(mat_mul(F, mat_mul(P_, mat_transpose(F))), Q);
  }

  // Scan matching for EKF update
  // Simple scan-to-map correlation: evaluate score for small perturbations
  // and pick the best shift as the observation.
  struct Correction { double dx, dy, dtheta; double score; };

  Correction scan_match() {
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = last_scan_;
    }

    Correction best{0.0, 0.0, 0.0, -1.0};
    if (!scan || scan->ranges.empty()) return best;

    // Precompute scan endpoints in robot frame
    struct Pt { double x, y; };
    std::vector<Pt> endpoints;
    int n = static_cast<int>(scan->ranges.size());
    for (int i = 0; i < n; i += sm_skip_) {
      double r = scan->ranges[i];
      if (!std::isfinite(r) || r < scan->range_min || r > sm_range_) continue;
      double angle = scan->angle_min + i * scan->angle_increment;
      endpoints.push_back({r * std::cos(angle), r * std::sin(angle)});
    }
    if (endpoints.empty()) return best;

    // Search small perturbations around current pose
    static const double dxy_steps[] = {-0.02, 0.0, 0.02};
    static const double dth_steps[] = {-0.02, 0.0, 0.02};

    for (double dxx : dxy_steps) {
      for (double dyy : dxy_steps) {
        for (double dth : dth_steps) {
          double cx = x_ + dxx;
          double cy = y_ + dyy;
          double ct = theta_ + dth;
          double cosT = std::cos(ct), sinT = std::sin(ct);

          double score = 0.0;
          for (auto &ep : endpoints) {
            double wx = cx + cosT * ep.x - sinT * ep.y;
            double wy = cy + sinT * ep.x + cosT * ep.y;
            auto [gx, gy] = world_to_grid(wx, wy);
            if (in_grid(gx, gy) && log_odds_[gy * map_w_ + gx] > 0.5) {
              score += 1.0;
            }
          }
          if (score > best.score) {
            best = {dxx, dyy, dth, score};
          }
        }
      }
    }
    return best;
  }

  // EKF update
  void ekf_update(const Correction &corr) {
    if (corr.score <= 0.0) return;

    // Observation noise
    Mat3 R;
    R.at(0,0) = on_xy_ * on_xy_;
    R.at(1,1) = on_xy_ * on_xy_;
    R.at(2,2) = on_theta_ * on_theta_;

    // Innovation
    double z[3] = {corr.dx, corr.dy, corr.dtheta};

    // S = H P H^T + R = P + R
    Mat3 S = mat_add(P_, R);

    // K = P S^{-1}
    Mat3 Sinv;
    if (!mat_inv(S, Sinv)) return;
    Mat3 K = mat_mul(P_, Sinv);

    // State update
    x_     += K.at(0,0)*z[0] + K.at(0,1)*z[1] + K.at(0,2)*z[2];
    y_     += K.at(1,0)*z[0] + K.at(1,1)*z[1] + K.at(1,2)*z[2];
    theta_  = normalize_angle(
        theta_ + K.at(2,0)*z[0] + K.at(2,1)*z[1] + K.at(2,2)*z[2]);

    // P = (I - K H) P = (I - K) P
    Mat3 IKH = Mat3::eye();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        IKH.at(i,j) -= K.at(i,j);
    P_ = mat_mul(IKH, P_);
  }

  // Update occupancy grid from current scan and pose
  void update_occupancy_grid() {
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = last_scan_;
    }
    if (!scan || scan->ranges.empty()) return;

    auto [robot_gx, robot_gy] = world_to_grid(x_, y_);
    if (!in_grid(robot_gx, robot_gy)) return;
    int n = static_cast<int>(scan->ranges.size());

    for (int i = 0; i < n; i += sm_skip_) {
      double r = scan->ranges[i];
      if (std::isnan(r)) continue;
      if (std::isfinite(r) && r < scan->range_min) continue;

      bool mark_hit = false;
      double ray_len = sm_range_;
      if (std::isfinite(scan->range_max) && scan->range_max > 0.0) {
        ray_len = std::min(ray_len, static_cast<double>(scan->range_max));
      }

      if (std::isfinite(r)) {
        if (r <= sm_range_) {
          mark_hit = true;
          ray_len = r;
        }
      }

      double angle = scan->angle_min + i * scan->angle_increment + theta_;
      double hx = x_ + ray_len * std::cos(angle);
      double hy = y_ + ray_len * std::sin(angle);
      auto [hgx, hgy] = world_to_grid(hx, hy);

      trace_ray(robot_gx, robot_gy, hgx, hgy, mark_hit);
    }
  }

  // Main localization step
  void localization_step() {
    ekf_predict();

    auto corr = scan_match();
    ekf_update(corr);

    update_occupancy_grid();

    publish_pose();
    publish_tf();
  }

  // Publish pose
  void publish_pose() {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = now();
    pose.header.frame_id = map_frame_;
    pose.pose.position.x = x_;
    pose.pose.position.y = y_;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.z = std::sin(theta_ / 2.0);
    pose.pose.orientation.w = std::cos(theta_ / 2.0);
    pub_pose_->publish(pose);
  }

  // Publish TF: map -> odom
  void publish_tf() {
    if (!has_odom_) return;

    // map->odom = map->base * inv(odom->base)
    double tf_theta = normalize_angle(theta_ - odom_theta_);
    double cos_t = std::cos(tf_theta), sin_t = std::sin(tf_theta);
    double tf_x = x_ - (cos_t * odom_x_ - sin_t * odom_y_);
    double tf_y = y_ - (sin_t * odom_x_ + cos_t * odom_y_);

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now();
    t.header.frame_id = map_frame_;
    t.child_frame_id = odom_frame_;
    t.transform.translation.x = tf_x;
    t.transform.translation.y = tf_y;
    t.transform.translation.z = 0.0;
    t.transform.rotation.z = std::sin(tf_theta / 2.0);
    t.transform.rotation.w = std::cos(tf_theta / 2.0);
    tf_broadcaster_->sendTransform(t);
  }

  // Publish occupancy grid
  void publish_map() {
    auto msg = std::make_unique<nav_msgs::msg::OccupancyGrid>();
    msg->header.stamp = now();
    msg->header.frame_id = map_frame_;
    msg->info.resolution = static_cast<float>(map_res_);
    msg->info.width = static_cast<unsigned>(map_w_);
    msg->info.height = static_cast<unsigned>(map_h_);
    msg->info.origin.position.x = map_ox_;
    msg->info.origin.position.y = map_oy_;
    msg->info.origin.orientation.w = 1.0;

    msg->data.resize(map_w_ * map_h_);
    for (int i = 0; i < map_w_ * map_h_; ++i) {
      double lo = log_odds_[i];
      if (std::abs(lo) < 0.1) {
        msg->data[i] = -1;  // unknown
      } else {
        // Convert log-odds to probability [0, 100]
        double prob = 1.0 / (1.0 + std::exp(-lo));
        msg->data[i] = static_cast<int8_t>(std::clamp(prob * 100.0, 0.0, 100.0));
      }
    }

    pub_map_->publish(std::move(*msg));
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationSlamNode>());
  rclcpp::shutdown();
  return 0;
}
