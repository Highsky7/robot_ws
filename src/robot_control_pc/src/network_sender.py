import rospy
import socket
from std_msgs.msg import Float32

class NetworkSender:
    def __init__(self):
        jetson_ip = rospy.get_param('~jetson_ip', '192.168.0.215')
        self.jetson_port = 9999
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jetson_address = (jetson_ip, self.jetson_port)

        self.left_speed_ms = 0.0
        self.right_speed_ms = 0.0
        self.can_id = 101 # 구동부 아두이노의 CAN ID

        rospy.Subscriber('cmd_vel_left', Float32, self.left_callback)
        rospy.Subscriber('cmd_vel_right', Float32, self.right_callback)
        rospy.Timer(rospy.Duration(0.05), self.send_command)
        rospy.loginfo("Network Sender node started. Sending to " + jetson_ip)

    def left_callback(self, msg): self.left_speed_ms = msg.data
    def right_callback(self, msg): self.right_speed_ms = msg.data

    def send_command(self, event):
        # ID,왼쪽속도(float),오른쪽속도(float) 형태의 텍스트 명령 생성
        command = f"{self.can_id},{self.left_speed_ms},{self.right_speed_ms}"
        self.sock.sendto(command.encode(), self.jetson_address)

if __name__ == '__main__':
    rospy.init_node('network_sender')
    NetworkSender()
    rospy.spin()