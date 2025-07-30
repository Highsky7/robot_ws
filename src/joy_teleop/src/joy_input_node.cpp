#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>

ros::Publisher joy_pub;

void joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    std_msgs::String msg;
    if (joy->buttons[6]==1)
    {
        if (joy->axes[0] > 0)  // joint1
        {
            msg.data = "joint1";
        }
        else if (joy->axes[0] < 0)  
        {
            msg.data = "-joint1";
        }

        else if (joy->axes[1] > 0) //joint2
        {
            msg.data = "joint2";
        }
        else if (joy->axes[1] < 0)
        {
            msg.data = "-joint2";
        }
        
        else if (joy->axes[7] == 1) //joint3
        {
            msg.data = "joint3";
        }
        else if (joy->axes[7] == -1)
        {
            msg.data = "-joint3";
        }

        else if (joy->buttons[2] == 1) //joint4
        {
            msg.data = "joint4";
        }
        else if (joy->buttons[0] == -1)
        {
            msg.data = "-joint4";
        }

        else if (joy->axes[4] > 0) //joint5
        {
            msg.data = "joint5";
        }
        else if (joy->axes[4] < 0)
        {
            msg.data = "-joint5";
        }

        else if (joy->axes[6] > 0) //joint6
        {
            msg.data = "joint6";
        }
        else if (joy->axes[6] < 0)
        {
            msg.data = "-joint6";
        }
        else
        {
            return; 
        }
        joy_pub.publish(msg);  // 토픽 발행
        ROS_INFO("Published: %s", msg.data.c_str());
    }
}
    

int main(int argc, char** argv)
{
    ros::init(argc, argv, "joy_input_node");
    ros::NodeHandle nh;

    joy_pub = nh.advertise<std_msgs::String>("joy_input", 10);
    ros::Subscriber sub = nh.subscribe("joy", 10, joyCallback);


    ros::spin();
    return 0;
}
