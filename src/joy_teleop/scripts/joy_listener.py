import rospy
from sensor_msgs.msg import Joy

def callback(data):
    # axes (조이스틱 축 값)
    rospy.loginfo("Axes: %s", data.axes)

    # buttons (버튼 값)
    rospy.loginfo("Buttons: %s", data.buttons)

def listener():
    rospy.init_node('joy_listener', anonymous=True)
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
