#!/usr/bin/env python
import rospy
import socket
from std_msgs.msg import Float32
from std_msgs.msg import String

class NetworkSender:
    def __init__(self):
        jetson_ip = rospy.get_param('~jetson_ip', '192.168.0.215')
        self.jetson_port = 9999
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jetson_address = (jetson_ip, self.jetson_port)

        # CAN ID 정의
        self.drive_can_id = 101
        self.manipulator_can_id_1 = 102
        self.manipulator_can_id_2 = 103

        # 구동부 토픽 구독
        self.left_speed_ms = 0.0
        self.right_speed_ms = 0.0
        rospy.Subscriber('cmd_vel_left', Float32, self.drive_left_callback)
        rospy.Subscriber('cmd_vel_right', Float32, self.drive_right_callback)

        # 매니퓰레이터 토픽 구독
        rospy.Subscriber('joy_input', String, self.manipulator_callback)

        rospy.loginfo("Unified Network Sender node started.")

    def drive_left_callback(self, msg):
        self.left_speed_ms = msg.data
        # 왼쪽 속도 값을 받을 때마다 양쪽 속도를 모두 전송
        self.send_drive_command()

    def drive_right_callback(self, msg):
        self.right_speed_ms = msg.data
        # 오른쪽 속도 값을 받을 때마다 양쪽 속도를 모두 전송
        self.send_drive_command()

    def manipulator_callback(self, msg):
        cmd = msg.data
        joint_name = cmd.replace('-', '')
        can_id = 0
        if joint_name in ["joint2", "joint3", "grip"]:
            can_id = self.manipulator_can_id_1
        elif joint_name in ["joint1", "joint4", "joint5", "joint6"]:
            can_id = self.manipulator_can_id_2

        if can_id != 0:
            command = f"{can_id},{cmd}"
            self.sock.sendto(command.encode(), self.jetson_address)

    def send_drive_command(self):
        command = f"{self.drive_can_id},{self.left_speed_ms},{self.right_speed_ms}"
        self.sock.sendto(command.encode(), self.jetson_address)

if __name__ == '__main__':
    rospy.init_node('network_sender')
    NetworkSender()
    rospy.spin()