#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>

// 바퀴 간 거리 [m]
static const double WHEEL_BASE = 0.5;

ros::Publisher pub_left;
ros::Publisher pub_right;

void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
  double v = msg->linear.x;
  double w = msg->angular.z;
  // 차동 주행 속도 계산
  double v_left  = v - (w * WHEEL_BASE / 2.0);
  double v_right = v + (w * WHEEL_BASE / 2.0);

  std_msgs::Float32 left_msg, right_msg;
  left_msg.data  = v_left;
  right_msg.data = v_right;
  pub_left.publish(left_msg);
  pub_right.publish(right_msg);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "twist_to_diff_cpp");
  ros::NodeHandle nh;

  pub_left  = nh.advertise<std_msgs::Float32>("cmd_vel_left",  1);
  pub_right = nh.advertise<std_msgs::Float32>("cmd_vel_right", 1);
  ros::Subscriber sub = nh.subscribe("cmd_vel", 1, cmdVelCallback);

  ros::spin();
  return 0;
}
