#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/wireless_controller.hpp"

namespace {

constexpr int kActionDim = 12;
constexpr int kObsDim = 484;
constexpr float kPosStopF = 2.146e9f;
constexpr float kVelStopF = 16000.0f;

constexpr std::array<int, kActionDim> kCommonToUnitree = {
    3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};

constexpr std::array<float, kActionDim> kHardwareLow = {
    -1.0472f, -1.5708f, -2.7227f,
    -1.0472f, -1.5708f, -2.7227f,
    -1.0472f, -0.5236f, -2.7227f,
    -1.0472f, -0.5236f, -2.7227f};

constexpr std::array<float, kActionDim> kHardwareHigh = {
    1.0472f, 3.4907f, -0.83776f,
    1.0472f, 3.4907f, -0.83776f,
    1.0472f, 4.5379f, -0.83776f,
    1.0472f, 4.5379f, -0.83776f};

struct Vec3 {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};

  Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
  Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
  Vec3 operator*(float scale) const { return {x * scale, y * scale, z * scale}; }
};

Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
  return a * (1.0f - t) + b * t;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Quat {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float w{1.0f};
};

Quat normalize(Quat q) {
  const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-8f) {
    return {};
  }
  q.x /= n;
  q.y /= n;
  q.z /= n;
  q.w /= n;
  if (q.w < 0.0f) {
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;
    q.w = -q.w;
  }
  return q;
}

Quat quat_mul(const Quat& a, const Quat& b) {
  return normalize({
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z});
}

Quat axis_angle(const Vec3& axis, float angle) {
  const float s = std::sin(0.5f * angle);
  return normalize({axis.x * s, axis.y * s, axis.z * s, std::cos(0.5f * angle)});
}

Quat slerp(Quat a, Quat b, float t) {
  float cos_half = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (cos_half < 0.0f) {
    b.x = -b.x;
    b.y = -b.y;
    b.z = -b.z;
    b.w = -b.w;
    cos_half = -cos_half;
  }
  if (cos_half > 0.9995f) {
    return normalize({
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z),
        a.w + t * (b.w - a.w)});
  }
  const float half = std::acos(std::clamp(cos_half, -1.0f, 1.0f));
  const float sin_half = std::sin(half);
  const float ra = std::sin((1.0f - t) * half) / sin_half;
  const float rb = std::sin(t * half) / sin_half;
  return normalize({ra * a.x + rb * b.x, ra * a.y + rb * b.y, ra * a.z + rb * b.z, ra * a.w + rb * b.w});
}

Vec3 quat_rotate(const Quat& q, const Vec3& v) {
  const Vec3 qv{q.x, q.y, q.z};
  const Vec3 t = cross(qv, v) * 2.0f;
  return v + t * q.w + cross(qv, t);
}

std::array<float, 6> quat_to_tan_norm(const Quat& q) {
  const Vec3 tan = quat_rotate(q, {1.0f, 0.0f, 0.0f});
  const Vec3 norm = quat_rotate(q, {0.0f, 0.0f, 1.0f});
  return {tan.x, tan.y, tan.z, norm.x, norm.y, norm.z};
}

void append_vec3(std::vector<float>& obs, const Vec3& v) {
  obs.push_back(v.x);
  obs.push_back(v.y);
  obs.push_back(v.z);
}

void append_tan_norm(std::vector<float>& obs, const Quat& q) {
  const auto tn = quat_to_tan_norm(q);
  obs.insert(obs.end(), tn.begin(), tn.end());
}

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    out.push_back(cell);
  }
  return out;
}

struct ReferenceFrame {
  float time{0.0f};
  Vec3 root_pos;
  Quat root_rot;
  Vec3 root_vel;
  Vec3 root_ang_vel;
  std::array<float, kActionDim> dof_pos{};
  std::array<float, kActionDim> dof_vel{};
};

class ReferenceMotion {
 public:
  void load(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
      throw std::runtime_error("Failed to open reference motion: " + path);
    }
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      if (line.empty()) {
        continue;
      }
      const auto cells = split_csv(line);
      if (cells.size() != 38) {
        throw std::runtime_error("Expected 38 columns in reference CSV, got " + std::to_string(cells.size()));
      }
      std::vector<float> v;
      v.reserve(cells.size());
      for (const auto& cell : cells) {
        v.push_back(std::stof(cell));
      }
      ReferenceFrame frame;
      int i = 0;
      frame.time = v[i++];
      frame.root_pos = {v[i++], v[i++], v[i++]};
      frame.root_rot = normalize({v[i++], v[i++], v[i++], v[i++]});
      frame.root_vel = {v[i++], v[i++], v[i++]};
      frame.root_ang_vel = {v[i++], v[i++], v[i++]};
      for (int j = 0; j < kActionDim; ++j) frame.dof_pos[j] = v[i++];
      for (int j = 0; j < kActionDim; ++j) frame.dof_vel[j] = v[i++];
      frames_.push_back(frame);
    }
    if (frames_.size() < 2) {
      throw std::runtime_error("Reference motion needs at least two frames.");
    }
    length_ = frames_.back().time;
    wrap_delta_ = frames_.back().root_pos - frames_.front().root_pos;
    wrap_delta_.z = 0.0f;
  }

  ReferenceFrame sample(float time) const {
    const float cycles = std::floor(time / length_);
    float local = std::fmod(time, length_);
    if (local < 0.0f) {
      local += length_;
    }
    auto upper = std::lower_bound(
        frames_.begin(), frames_.end(), local,
        [](const ReferenceFrame& frame, float t) { return frame.time < t; });
    if (upper == frames_.begin()) {
      return add_wrap(*upper, cycles);
    }
    if (upper == frames_.end()) {
      return add_wrap(frames_.back(), cycles);
    }
    const auto& b = *upper;
    const auto& a = *(upper - 1);
    const float denom = std::max(b.time - a.time, 1e-6f);
    const float blend = (local - a.time) / denom;
    ReferenceFrame out;
    out.time = time;
    out.root_pos = lerp(a.root_pos, b.root_pos, blend) + wrap_delta_ * cycles;
    out.root_rot = slerp(a.root_rot, b.root_rot, blend);
    out.root_vel = lerp(a.root_vel, b.root_vel, blend);
    out.root_ang_vel = lerp(a.root_ang_vel, b.root_ang_vel, blend);
    for (int i = 0; i < kActionDim; ++i) {
      out.dof_pos[i] = a.dof_pos[i] * (1.0f - blend) + b.dof_pos[i] * blend;
      out.dof_vel[i] = a.dof_vel[i] * (1.0f - blend) + b.dof_vel[i] * blend;
    }
    return out;
  }

 private:
  ReferenceFrame add_wrap(ReferenceFrame frame, float cycles) const {
    frame.root_pos = frame.root_pos + wrap_delta_ * cycles;
    return frame;
  }

  std::vector<ReferenceFrame> frames_;
  float length_{1.0f};
  Vec3 wrap_delta_;
};

struct LegDef {
  Vec3 hip_offset;
  Vec3 thigh_offset;
  Vec3 calf_offset;
  Vec3 foot_offset;
  int dof_base;
};

class Go2Kinematics {
 public:
  std::array<Vec3, 4> key_feet_rel(const Vec3& root_pos, const Quat& root_rot, const std::array<float, kActionDim>& dof) const {
    std::array<Vec3, 4> by_common_leg{};
    for (int leg = 0; leg < 4; ++leg) {
      const auto& def = legs_[leg];
      const Quat hip_q = axis_angle({1.0f, 0.0f, 0.0f}, dof[def.dof_base + 0]);
      const Quat thigh_q = axis_angle({0.0f, 1.0f, 0.0f}, dof[def.dof_base + 1]);
      const Quat calf_q = axis_angle({0.0f, 1.0f, 0.0f}, dof[def.dof_base + 2]);
      const Vec3 hip_pos = root_pos + quat_rotate(root_rot, def.hip_offset);
      const Quat hip_rot = quat_mul(root_rot, hip_q);
      const Vec3 thigh_pos = hip_pos + quat_rotate(hip_rot, def.thigh_offset);
      const Quat thigh_rot = quat_mul(hip_rot, thigh_q);
      const Vec3 calf_pos = thigh_pos + quat_rotate(thigh_rot, def.calf_offset);
      const Quat calf_rot = quat_mul(thigh_rot, calf_q);
      const Vec3 foot_pos = calf_pos + quat_rotate(calf_rot, def.foot_offset);
      by_common_leg[leg] = foot_pos - root_pos;
    }
    return {by_common_leg[1], by_common_leg[0], by_common_leg[3], by_common_leg[2]};
  }

  void append_joint_tan_norm(std::vector<float>& obs, const std::array<float, kActionDim>& dof) const {
    for (int leg = 0; leg < 4; ++leg) {
      const int base = legs_[leg].dof_base;
      append_tan_norm(obs, axis_angle({1.0f, 0.0f, 0.0f}, dof[base + 0]));
      append_tan_norm(obs, axis_angle({0.0f, 1.0f, 0.0f}, dof[base + 1]));
      append_tan_norm(obs, axis_angle({0.0f, 1.0f, 0.0f}, dof[base + 2]));
      append_tan_norm(obs, Quat{});
    }
  }

 private:
  const std::array<LegDef, 4> legs_ = {
      LegDef{{0.1934f, 0.0465f, 0.0f}, {0.0f, 0.0955f, 0.0f}, {0.0f, 0.0f, -0.213f}, {0.0f, 0.0f, -0.213f}, 0},
      LegDef{{0.1934f, -0.0465f, 0.0f}, {0.0f, -0.0955f, 0.0f}, {0.0f, 0.0f, -0.213f}, {0.0f, 0.0f, -0.213f}, 3},
      LegDef{{-0.1934f, 0.0465f, 0.0f}, {0.0f, 0.0955f, 0.0f}, {0.0f, 0.0f, -0.213f}, {0.0f, 0.0f, -0.213f}, 6},
      LegDef{{-0.1934f, -0.0465f, 0.0f}, {0.0f, -0.0955f, 0.0f}, {0.0f, 0.0f, -0.213f}, {0.0f, 0.0f, -0.213f}, 9},
  };
};

class ObservationBuilder {
 public:
  explicit ObservationBuilder(bool imu_quat_wxyz, float policy_dt)
      : imu_quat_wxyz_(imu_quat_wxyz), policy_dt_(policy_dt) {}

  std::vector<float> build(const unitree_go::msg::LowState& low_state, const ReferenceMotion& reference, float phase_time, float phase_speed) {
    std::array<float, kActionDim> dof{};
    std::array<float, kActionDim> dof_vel{};
    for (int i = 0; i < kActionDim; ++i) {
      const int motor = kCommonToUnitree[i];
      dof[i] = low_state.motor_state[motor].q;
      dof_vel[i] = low_state.motor_state[motor].dq;
    }

    const ReferenceFrame current_ref = reference.sample(phase_time);
    Vec3 root_pos = current_ref.root_pos;
    root_pos.z = std::max(root_pos.z, 0.05f);
    const Quat root_rot = read_imu_quat(low_state);
    const Vec3 root_vel = current_ref.root_vel * phase_speed;
    const Vec3 root_ang_vel = {
        low_state.imu_state.gyroscope[0],
        low_state.imu_state.gyroscope[1],
        low_state.imu_state.gyroscope[2]};

    std::vector<float> obs;
    obs.reserve(kObsDim);
    obs.push_back(root_pos.z);
    append_tan_norm(obs, root_rot);
    append_vec3(obs, root_vel);
    append_vec3(obs, root_ang_vel);
    kin_.append_joint_tan_norm(obs, dof);
    obs.insert(obs.end(), dof_vel.begin(), dof_vel.end());
    const auto feet_rel = kin_.key_feet_rel(root_pos, root_rot, dof);
    for (const auto& foot : feet_rel) {
      append_vec3(obs, foot);
    }

    for (int step : {1, 2, 3}) {
      const ReferenceFrame target = reference.sample(phase_time + step * policy_dt_ * phase_speed);
      Vec3 target_root_delta = target.root_pos - root_pos;
      target_root_delta.z = target.root_pos.z;
      append_vec3(obs, target_root_delta);
      append_tan_norm(obs, target.root_rot);
      kin_.append_joint_tan_norm(obs, target.dof_pos);
      const auto target_feet_rel = kin_.key_feet_rel(target.root_pos, target.root_rot, target.dof_pos);
      for (const auto& foot : target_feet_rel) {
        append_vec3(obs, foot);
      }
    }

    if (obs.size() != kObsDim) {
      throw std::runtime_error("Built observation with " + std::to_string(obs.size()) + " values, expected " + std::to_string(kObsDim));
    }
    for (float value : obs) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Observation contains a non-finite value");
      }
    }
    return obs;
  }

 private:
  Quat read_imu_quat(const unitree_go::msg::LowState& low_state) const {
    const auto& q = low_state.imu_state.quaternion;
    if (imu_quat_wxyz_) {
      return normalize({q[1], q[2], q[3], q[0]});
    }
    return normalize({q[0], q[1], q[2], q[3]});
  }

  bool imu_quat_wxyz_;
  float policy_dt_;
  Go2Kinematics kin_;
};

class PolicyRunner {
 public:
  explicit PolicyRunner(const std::string& path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "go2_mimickit_policy"),
        session_(nullptr) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(env_, path.c_str(), options);
  }

  std::array<float, kActionDim> run(const std::vector<float>& obs) {
    if (obs.size() != kObsDim) {
      throw std::runtime_error("Policy input observation has invalid size");
    }
    std::array<int64_t, 2> input_shape{1, kObsDim};
    std::array<int64_t, 2> output_shape{1, kActionDim};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(obs.data()), obs.size(), input_shape.data(), input_shape.size());
    std::array<float, kActionDim> output{};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory, output.data(), output.size(), output_shape.data(), output_shape.size());
    const char* input_names[] = {"raw_obs"};
    const char* output_names[] = {"joint_position"};
    session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, &output_tensor, 1);
    for (float value : output) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Policy produced a non-finite action");
      }
    }
    return output;
  }

 private:
  Ort::Env env_;
  Ort::Session session_;
};

struct RawBmsCmd {
  uint8_t off;
  std::array<uint8_t, 3> reserve;
};

struct RawMotorCmd {
  uint8_t mode;
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  std::array<uint32_t, 3> reserve;
};

struct RawLowCmd {
  std::array<uint8_t, 2> head;
  uint8_t level_flag;
  uint8_t frame_reserve;
  std::array<uint32_t, 2> sn;
  std::array<uint32_t, 2> version;
  uint16_t bandwidth;
  std::array<RawMotorCmd, 20> motor_cmd;
  RawBmsCmd bms_cmd;
  std::array<uint8_t, 40> wireless_remote;
  std::array<uint8_t, 12> led;
  std::array<uint8_t, 2> fan;
  uint8_t gpio;
  uint32_t reserve;
  uint32_t crc;
};

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1u << 31;
    uint32_t data = ptr[i];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      if (crc & 0x80000000) {
        crc <<= 1;
        crc ^= polynomial;
      } else {
        crc <<= 1;
      }
      if (data & xbit) {
        crc ^= polynomial;
      }
      xbit >>= 1;
    }
  }
  return crc;
}

void fill_crc(unitree_go::msg::LowCmd& msg) {
  RawLowCmd raw{};
  std::memcpy(raw.head.data(), msg.head.data(), 2);
  raw.level_flag = msg.level_flag;
  raw.frame_reserve = msg.frame_reserve;
  std::memcpy(raw.sn.data(), msg.sn.data(), 8);
  std::memcpy(raw.version.data(), msg.version.data(), 8);
  raw.bandwidth = msg.bandwidth;
  for (int i = 0; i < 20; ++i) {
    raw.motor_cmd[i].mode = msg.motor_cmd[i].mode;
    raw.motor_cmd[i].q = msg.motor_cmd[i].q;
    raw.motor_cmd[i].dq = msg.motor_cmd[i].dq;
    raw.motor_cmd[i].tau = msg.motor_cmd[i].tau;
    raw.motor_cmd[i].kp = msg.motor_cmd[i].kp;
    raw.motor_cmd[i].kd = msg.motor_cmd[i].kd;
    std::memcpy(raw.motor_cmd[i].reserve.data(), msg.motor_cmd[i].reserve.data(), 12);
  }
  raw.bms_cmd.off = msg.bms_cmd.off;
  std::memcpy(raw.bms_cmd.reserve.data(), msg.bms_cmd.reserve.data(), 3);
  std::memcpy(raw.wireless_remote.data(), msg.wireless_remote.data(), 40);
  std::memcpy(raw.led.data(), msg.led.data(), 12);
  std::memcpy(raw.fan.data(), msg.fan.data(), 2);
  raw.gpio = msg.gpio;
  raw.reserve = msg.reserve;
  raw.crc = crc32_core(reinterpret_cast<uint32_t*>(&raw), (sizeof(RawLowCmd) >> 2) - 1);
  msg.crc = raw.crc;
}

class Go2PolicyNode : public rclcpp::Node {
 public:
  Go2PolicyNode()
      : Node("go2_policy_node") {
    load_params();
    reference_.load(reference_path_);
    policy_ = std::make_unique<PolicyRunner>(policy_path_);
    obs_builder_ = std::make_unique<ObservationBuilder>(imu_quat_wxyz_, 1.0f / policy_hz_);
    init_low_cmd();

    low_cmd_pub_ = create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
    low_state_sub_ = create_subscription<unitree_go::msg::LowState>(
        "/lowstate", 10, [this](unitree_go::msg::LowState::SharedPtr msg) {
          low_state_ = *msg;
          have_low_state_ = true;
        });
    wireless_sub_ = create_subscription<unitree_go::msg::WirelessController>(
        "/wirelesscontroller", 10, [this](unitree_go::msg::WirelessController::SharedPtr msg) {
          handle_wireless(*msg);
        });

    const auto period = std::chrono::duration<double>(1.0 / low_level_hz_);
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this]() {
      tick();
    });

    RCLCPP_WARN(
        get_logger(),
        "GO2 MimicKit deploy node started. dry_run=%s policy=%s reference=%s",
        dry_run_ ? "true" : "false",
        policy_path_.c_str(),
        reference_path_.c_str());
  }

 private:
  void load_params() {
    policy_path_ = declare_parameter<std::string>("policy_path", "output/go2_deploy/deepmimic_go2_pace/policy.onnx");
    reference_path_ = declare_parameter<std::string>("reference_path", "output/go2_deploy/deepmimic_go2_pace/pace_reference.csv");
    dry_run_ = declare_parameter<bool>("dry_run", true);
    publish_damping_when_disarmed_ = declare_parameter<bool>("publish_damping_when_disarmed", true);
    imu_quat_wxyz_ = declare_parameter<bool>("imu_quat_wxyz", true);
    low_level_hz_ = declare_parameter<double>("low_level_hz", 500.0);
    policy_hz_ = declare_parameter<double>("policy_hz", 30.0);
    kp_ = declare_parameter<double>("kp", 40.0);
    kd_ = declare_parameter<double>("kd", 3.0);
    damping_kd_ = declare_parameter<double>("damping_kd", 2.0);
    phase_speed_ = declare_parameter<double>("phase_speed_default", 1.0);
    phase_speed_min_ = declare_parameter<double>("phase_speed_min", 0.5);
    phase_speed_max_ = declare_parameter<double>("phase_speed_max", 1.3);
    phase_speed_step_ = declare_parameter<double>("phase_speed_step", 0.05);
    arm_button_mask_ = declare_parameter<int>("arm_button_mask", 1);
    disarm_button_mask_ = declare_parameter<int>("disarm_button_mask", 2);
    estop_button_mask_ = declare_parameter<int>("estop_button_mask", 4);
    enable_hold_button_mask_ = declare_parameter<int>("enable_hold_button_mask", 0);
    speed_up_button_mask_ = declare_parameter<int>("speed_up_button_mask", 0);
    speed_down_button_mask_ = declare_parameter<int>("speed_down_button_mask", 0);
    blend_seconds_ = declare_parameter<double>("blend_seconds", 1.0);
    log_every_n_policy_ticks_ = declare_parameter<int>("log_every_n_policy_ticks", 30);
    if (low_level_hz_ <= 0.0 || policy_hz_ <= 0.0) {
      throw std::runtime_error("low_level_hz and policy_hz must be positive");
    }
    phase_speed_ = std::clamp(phase_speed_, phase_speed_min_, phase_speed_max_);
    policy_stride_ = std::max(1, static_cast<int>(std::round(low_level_hz_ / policy_hz_)));
  }

  void init_low_cmd() {
    low_cmd_.head[0] = 0xFE;
    low_cmd_.head[1] = 0xEF;
    low_cmd_.level_flag = 0xFF;
    low_cmd_.gpio = 0;
    for (int i = 0; i < 20; ++i) {
      low_cmd_.motor_cmd[i].mode = 0x01;
      low_cmd_.motor_cmd[i].q = kPosStopF;
      low_cmd_.motor_cmd[i].dq = kVelStopF;
      low_cmd_.motor_cmd[i].kp = 0.0f;
      low_cmd_.motor_cmd[i].kd = 0.0f;
      low_cmd_.motor_cmd[i].tau = 0.0f;
    }
  }

  bool key_pressed(uint16_t keys, int mask) const {
    return mask != 0 && ((keys & static_cast<uint16_t>(mask)) != 0);
  }

  void handle_wireless(const unitree_go::msg::WirelessController& msg) {
    const uint16_t keys = msg.keys;
    if (key_pressed(keys, estop_button_mask_)) {
      estop_ = true;
      armed_ = false;
      RCLCPP_ERROR(get_logger(), "Emergency stop requested from wireless controller");
    }
    if (!estop_ && key_pressed(keys, arm_button_mask_)) {
      if (!armed_) {
        blend_ticks_remaining_ = static_cast<int>(std::round(blend_seconds_ * policy_hz_));
        RCLCPP_WARN(get_logger(), "Policy armed");
      }
      armed_ = true;
    }
    if (key_pressed(keys, disarm_button_mask_)) {
      armed_ = false;
      RCLCPP_WARN(get_logger(), "Policy disarmed");
    }
    if (key_pressed(keys, speed_up_button_mask_)) {
      phase_speed_ = std::min(phase_speed_max_, phase_speed_ + phase_speed_step_);
    }
    if (key_pressed(keys, speed_down_button_mask_)) {
      phase_speed_ = std::max(phase_speed_min_, phase_speed_ - phase_speed_step_);
    }
    policy_hold_enabled_ = enable_hold_button_mask_ == 0 || key_pressed(keys, enable_hold_button_mask_);
  }

  std::array<float, kActionDim> current_joint_pos_common() const {
    std::array<float, kActionDim> out{};
    for (int i = 0; i < kActionDim; ++i) {
      out[i] = low_state_.motor_state[kCommonToUnitree[i]].q;
    }
    return out;
  }

  void tick() {
    if (!have_low_state_) {
      return;
    }

    if (estop_ || !armed_ || !policy_hold_enabled_) {
      if (publish_damping_when_disarmed_) {
        fill_damping_cmd();
        publish_cmd();
      }
      return;
    }

    if ((low_level_tick_ % policy_stride_) == 0) {
      try {
        auto obs = obs_builder_->build(low_state_, reference_, phase_time_, static_cast<float>(phase_speed_));
        auto policy_target = policy_->run(obs);
        clip_target(policy_target);
        if (blend_ticks_remaining_ > 0) {
          const auto current = current_joint_pos_common();
          const float alpha = 1.0f - static_cast<float>(blend_ticks_remaining_) /
                                         std::max(1.0, blend_seconds_ * policy_hz_);
          for (int i = 0; i < kActionDim; ++i) {
            policy_target[i] = current[i] * (1.0f - alpha) + policy_target[i] * alpha;
          }
          --blend_ticks_remaining_;
        }
        target_common_ = policy_target;
        phase_time_ += static_cast<float>((1.0 / policy_hz_) * phase_speed_);
        ++policy_tick_;
        if (log_every_n_policy_ticks_ > 0 && (policy_tick_ % log_every_n_policy_ticks_) == 0) {
          RCLCPP_INFO(
              get_logger(),
              "policy tick=%d phase=%.3f speed=%.2f target[0]=%.3f dry_run=%s",
              policy_tick_,
              phase_time_,
              phase_speed_,
              target_common_[0],
              dry_run_ ? "true" : "false");
        }
      } catch (const std::exception& exc) {
        RCLCPP_ERROR(get_logger(), "Policy tick failed: %s", exc.what());
        armed_ = false;
        fill_damping_cmd();
        publish_cmd();
        return;
      }
    }

    fill_policy_cmd();
    publish_cmd();
    ++low_level_tick_;
  }

  void clip_target(std::array<float, kActionDim>& target) const {
    for (int i = 0; i < kActionDim; ++i) {
      target[i] = std::clamp(target[i], kHardwareLow[i], kHardwareHigh[i]);
    }
  }

  void fill_damping_cmd() {
    init_low_cmd();
    for (int i = 0; i < 12; ++i) {
      low_cmd_.motor_cmd[i].mode = 0x01;
      low_cmd_.motor_cmd[i].q = kPosStopF;
      low_cmd_.motor_cmd[i].dq = 0.0f;
      low_cmd_.motor_cmd[i].kp = 0.0f;
      low_cmd_.motor_cmd[i].kd = static_cast<float>(damping_kd_);
      low_cmd_.motor_cmd[i].tau = 0.0f;
    }
  }

  void fill_policy_cmd() {
    init_low_cmd();
    for (int common_i = 0; common_i < kActionDim; ++common_i) {
      const int motor_i = kCommonToUnitree[common_i];
      low_cmd_.motor_cmd[motor_i].mode = 0x01;
      low_cmd_.motor_cmd[motor_i].q = target_common_[common_i];
      low_cmd_.motor_cmd[motor_i].dq = 0.0f;
      low_cmd_.motor_cmd[motor_i].kp = static_cast<float>(kp_);
      low_cmd_.motor_cmd[motor_i].kd = static_cast<float>(kd_);
      low_cmd_.motor_cmd[motor_i].tau = 0.0f;
    }
  }

  void publish_cmd() {
    fill_crc(low_cmd_);
    if (!dry_run_) {
      low_cmd_pub_->publish(low_cmd_);
    }
  }

  std::string policy_path_;
  std::string reference_path_;
  bool dry_run_{true};
  bool publish_damping_when_disarmed_{true};
  bool imu_quat_wxyz_{true};
  double low_level_hz_{500.0};
  double policy_hz_{30.0};
  double kp_{40.0};
  double kd_{3.0};
  double damping_kd_{2.0};
  double phase_speed_{1.0};
  double phase_speed_min_{0.5};
  double phase_speed_max_{1.3};
  double phase_speed_step_{0.05};
  int arm_button_mask_{1};
  int disarm_button_mask_{2};
  int estop_button_mask_{4};
  int enable_hold_button_mask_{0};
  int speed_up_button_mask_{0};
  int speed_down_button_mask_{0};
  int policy_stride_{17};
  int log_every_n_policy_ticks_{30};
  double blend_seconds_{1.0};
  int blend_ticks_remaining_{0};

  bool have_low_state_{false};
  bool armed_{false};
  bool estop_{false};
  bool policy_hold_enabled_{true};
  uint64_t low_level_tick_{0};
  int policy_tick_{0};
  float phase_time_{0.0f};
  std::array<float, kActionDim> target_common_{};

  ReferenceMotion reference_;
  std::unique_ptr<PolicyRunner> policy_;
  std::unique_ptr<ObservationBuilder> obs_builder_;
  unitree_go::msg::LowState low_state_;
  unitree_go::msg::LowCmd low_cmd_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
  rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr wireless_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Go2PolicyNode>());
  } catch (const std::exception& exc) {
    std::cerr << "go2_policy_node failed: " << exc.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
