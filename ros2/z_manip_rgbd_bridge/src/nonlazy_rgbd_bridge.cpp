#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;

namespace z_manip_rgbd_bridge
{

class NonLazyRgbdBridge final : public rclcpp::Node
{
public:
  NonLazyRgbdBridge()
  : Node("z_manip_nonlazy_rgbd_bridge")
  {
    const auto color_input = declare_parameter<std::string>(
      "color_input_base_topic", "/nuc/camera/color/image_raw");
    const auto depth_input = declare_parameter<std::string>(
      "depth_input_base_topic", "/nuc/camera/aligned_depth_to_color/image_raw");
    const auto info_input = declare_parameter<std::string>(
      "camera_info_input_topic", "/nuc/camera/color/camera_info");
    const auto color_output = declare_parameter<std::string>(
      "color_output_topic", "/camera/color/image_raw");
    const auto depth_output = declare_parameter<std::string>(
      "depth_output_topic", "/camera/aligned_depth_to_color/image_raw");
    const auto info_output = declare_parameter<std::string>(
      "camera_info_output_topic", "/camera/color/camera_info");

    color_pub_ = create_publisher<sensor_msgs::msg::Image>(
      color_output, rclcpp::SensorDataQoS());
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
      depth_output, rclcpp::SensorDataQoS());
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      info_output, rclcpp::SensorDataQoS());

    color_sub_ = image_transport::create_subscription(
      this,
      color_input,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr & message) {
        color_pub_->publish(*message);
        ++color_count_;
      },
      "compressed",
      rmw_qos_profile_sensor_data);
    depth_sub_ = image_transport::create_subscription(
      this,
      depth_input,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr & message) {
        depth_pub_->publish(*message);
        ++depth_count_;
      },
      "compressedDepth",
      rmw_qos_profile_sensor_data);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_input,
      rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        info_pub_->publish(*message);
        ++info_count_;
      });

    status_timer_ = create_wall_timer(30s, [this]() {
      RCLCPP_INFO(
        get_logger(),
        "non-lazy RGB-D forwarding totals: color=%llu depth=%llu info=%llu",
        static_cast<unsigned long long>(color_count_.load()),
        static_cast<unsigned long long>(depth_count_.load()),
        static_cast<unsigned long long>(info_count_.load()));
    });
    RCLCPP_INFO(
      get_logger(),
      "ready: permanent compressed RGB/depth subscriptions -> raw RGB-D; read-only");
  }

private:
  image_transport::Subscriber color_sub_;
  image_transport::Subscriber depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  std::atomic<std::uint64_t> color_count_{0};
  std::atomic<std::uint64_t> depth_count_{0};
  std::atomic<std::uint64_t> info_count_{0};
};

}  // namespace z_manip_rgbd_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<z_manip_rgbd_bridge::NonLazyRgbdBridge>());
  rclcpp::shutdown();
  return 0;
}
