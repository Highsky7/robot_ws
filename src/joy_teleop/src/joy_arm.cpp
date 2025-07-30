#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float64.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <vector> // std::vector를 사용하기 위해 추가
// 조인트 속도를 변경할 조인트 이름 설정 (예: "shoulder_pan_joint")
std::vector<std::string> joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

ros::Publisher joint_trajectory_publisher;
// 콜백 함수: 조이스틱 입력을 받는 함수
void joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
   std::vector<double> joint_speeds;
    joint_speeds.push_back(joy->axes[0] * 1.570796);  // joint1: 왼쪽/오른쪽 스틱 (X축)
    joint_speeds.push_back(joy->axes[1] * 1.570796);  // joint2: 왼쪽/오른쪽 스틱 (Y축)
    joint_speeds.push_back(joy->axes[7] * 1.570796);  // joint3: D-패드 (Y축) - 보통 -1, 0, 1
    
    // 버튼으로 제어하는 경우, 버튼이 눌리면 최대 속도, 아니면 0
    // 만약 버튼을 누르는 동안만 움직이고 싶다면 이렇게 설정할 수 있습니다.
    double joint4_speed = 0.0;
    if (joy->buttons[2]) 
    { // 예: 'X' 버튼 (또는 해당하는 인덱스)
        joint4_speed = 1.570796; // 음수 값으로 한 방향으로만 움직이게
    }
    else if(joy->buttons[0])
    {
        joint4_speed = -1.570796;
    }
    joint_speeds.push_back(joint4_speed);

    joint_speeds.push_back(joy->axes[4] * 0.392699);  // joint5: 오른쪽 스틱 (Y축)
    joint_speeds.push_back(joy->axes[6] * 0.392699);  // joint6: joint1과 동일한 축으로 제어하는 것이 의도된 것인지 확인 필요

    // 조인트 속도를 퍼블리시할 메시지 생성
    trajectory_msgs::JointTrajectory joint_trajectory_msg;
    trajectory_msgs::JointTrajectoryPoint point;
    
    // 조인트 이름 설정 (전역 변수 사용)
    joint_trajectory_msg.joint_names = joint_names;

    // 조인트 속도 설정
    point.velocities = joint_speeds;

    // 시간을 설정 (현재 시간으로부터의 상대적인 시간)
    // 0.1초는 이 명령이 컨트롤러에 의해 0.1초 후에 실행될 목표 지점임을 의미합니다.
    point.time_from_start = ros::Duration(0.1); 

    joint_trajectory_msg.points.push_back(point);

    // 전역 퍼블리셔 객체를 사용하여 메시지 퍼블리시
    joint_trajectory_publisher.publish(joint_trajectory_msg);
}

int main(int argc, char** argv)
{
   // ROS 노드 초기화
    ros::init(argc, argv, "joy_teleop_node");
    ros::NodeHandle nh; // 노드 핸들을 main 함수에서 생성

    // 퍼블리셔 객체를 main 함수에서 한 번만 생성
    // "/arm_controller/command" 토픽은 ROS 제어기(예: JointTrajectoryController)가 수신하는 표준 토픽입니다.
    joint_trajectory_publisher = nh.advertise<trajectory_msgs::JointTrajectory>("/arm_controller/command", 1);

    // joy 토픽을 구독하여 조이스틱 입력을 받음
    // 큐 사이즈는 10으로 유지 (들어오는 메시지를 처리하기 위한 버퍼)
    ros::Subscriber sub = nh.subscribe("joy", 10, joyCallback);

    // ROS 이벤트 루프 실행 (콜백 함수들이 호출되도록 함)
    ros::spin();
    
    return 0;
}
