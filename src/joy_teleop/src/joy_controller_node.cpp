#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <string>
#include <vector>
#include <cmath> // std::abs 사용

// 조이스틱 버튼 및 축 인덱스를 상수로 정의하여 가독성 향상
// (Logitech F710, Xbox 컨트롤러 등 표준 레이아웃 기준)
namespace joy_mapping
{
    // Axes
    const int AXIS_LEFT_STICK_X = 0; // L2 + joint1, R2 + joint5
    const int AXIS_LEFT_STICK_Y = 1; // L2 + joint2, R2 + joint6
    const int AXIS_RIGHT_STICK_X = 4; // L2 + joint3, R2 + grip
    const int AXIS_RIGHT_STICK_Y  = 3; // L2 + joint4, R2 
    
    // Buttons
    const int BUTTON_X = 2; // 'X' 버튼 (네모 버튼)
    const int BUTTON_A = 0; // 'A' 버튼 (엑스 버튼)
    const int BUTTON_B = 1;
    const int BUTTON_C = 3;
    const int BUTTON_ENABLE_L = 6; // L2 또는 LT 버튼 (활성화 버튼으로 사용)
    const int BUTTON_ENABLE_R = 7  ; // L2 또는 LT 버튼 (활성화 버튼으로 사용)
}

/**
 * @class JoyController
 * @brief 조이스틱 입력을 처리하고, 로봇 제어 명령을 발행하는 클래스
 */
class JoyController
{
public:
    // 생성자: 노드 핸들, 퍼블리셔, 서브스크라이버 등 초기화
    JoyController() : nh_("~") // Private NodeHandle 사용
    {
        // ROS 파라미터 서버에서 설정 값 불러오기 (없으면 기본값 사용)
        nh_.param<double>("deadzone", deadzone_, 0.9);
        nh_.param<int>("enable_button_l", enable_button_l, joy_mapping::BUTTON_ENABLE_L); 
        nh_.param<int>("enable_button_r", enable_button_r, joy_mapping::BUTTON_ENABLE_R);
        

        // 퍼블리셔와 서브스크라이버 초기화
        joy_pub_ = nh_.advertise<std_msgs::String>("joy_input", 10);
        joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &JoyController::joyCallback, this);
        

        ROS_INFO("JoyController node has been initialized.");
        ROS_INFO("Using deadzone: %f", deadzone_);
        ROS_INFO("Using enable button index: %d", enable_button_l);
        ROS_INFO("Using enable button index: %d", enable_button_r);
    
    }

private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
    {
    // L2 또는 R2 버튼이 눌린 경우에만 동작
    if (joy->buttons.size() <= enable_button_l || joy->buttons[enable_button_l] == 0)
    {
        if (joy->buttons.size() <= enable_button_r || joy->buttons[enable_button_r] == 0)
            return;
    }

    if (joy->buttons[enable_button_l] == 1) {
        processAxisInput(joy, joy_mapping::AXIS_LEFT_STICK_X, "joint1");
        processAxisInput(joy, joy_mapping::AXIS_LEFT_STICK_Y, "joint2");
        processAxisInput(joy, joy_mapping::AXIS_RIGHT_STICK_X, "joint3");
        processAxisInput(joy, joy_mapping::AXIS_RIGHT_STICK_Y, "joint4");
       
        
    }

    if (joy->buttons[enable_button_r] == 1) 
    {
        processAxisInput(joy, joy_mapping::AXIS_LEFT_STICK_X, "joint5");
        processAxisInput(joy, joy_mapping::AXIS_LEFT_STICK_Y, "joint6");
        processAxisInput(joy, joy_mapping::AXIS_RIGHT_STICK_Y, "grip");
    }
    }


    /**
     * @brief 아날로그 축(Axis) 입력을 처리하고 토픽을 발행하는 헬퍼 함수
     * @param joy 조이스틱 메시지
     * @param axis_index 확인할 축의 인덱스
     * @param joint_name 해당 축에 매핑된 관절 이름
     */
    void processAxisInput(const sensor_msgs::Joy::ConstPtr& joy, int axis_index, const std::string& joint_name)
    {
        // joy->axes 배열의 크기가 유효한지 확인
        if (joy->axes.size() <= axis_index) return;

        std_msgs::String msg;
        float axis_value = joy->axes[axis_index];

        if (std::abs(axis_value) > deadzone_)
        {
            if (axis_value > 0)
            {
                msg.data = joint_name;
            }
            else
            {
                msg.data = "-" + joint_name;
            }
            joy_pub_.publish(msg);
            ROS_INFO("Published: %s", msg.data.c_str());
        }
    }

    /**
     * @brief 버튼(Button) 입력을 처리하고 토픽을 발행하는 헬퍼 함수
     * @param joy 조이스틱 메시지
     * @param button_index 확인할 버튼의 인덱스
     * @param joint_name 해당 버튼에 매핑된 관절 이름
     */
   
    ros::NodeHandle nh_;
    ros::Publisher joy_pub_;
    ros::Subscriber joy_sub_;
    
    double deadzone_;
    int enable_button_l;
    int enable_button_r;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "joy_controller_node");
    JoyController joy_controller;
    ros::spin();
    return 0;
}