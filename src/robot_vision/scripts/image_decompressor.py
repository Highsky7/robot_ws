#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class ImageDecompressor:
    """
    하나의 CompressedImage 토픽을 구독하여 CPU를 사용해 이미지를 디코딩한 후,
    여러 다른 노드들이 사용할 수 있도록 압축이 풀린 raw Image 토픽으로 재발행하는
    중앙 집중식 압축 해제 노드입니다.
    이를 통해 각 비전 노드에서 발생하는 중복된 CPU 부하를 제거합니다.
    """
    def __init__(self):
        rospy.init_node('image_decompressor_node', anonymous=True)
        
        self.bridge = CvBridge()

        # --- 파라미터 설정 ---
        # 구독할 압축 이미지 토픽
        compressed_topic = rospy.get_param('~compressed_topic', '/jetson/camera/image/compressed')
        # 발행할 Raw 이미지 토픽
        raw_topic = rospy.get_param('~raw_topic', '/camera/color/image_raw_uncompressed')

        # 압축 해제된 이미지를 발행할 퍼블리셔
        self.raw_image_pub = rospy.Publisher(raw_topic, Image, queue_size=5)

        # 원본 압축 이미지를 구독할 서브스크라이버
        self.compressed_image_sub = rospy.Subscriber(
            compressed_topic, 
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size=2**24  # 메시지 드랍 방지를 위한 충분한 버퍼
        )

        rospy.loginfo(f"Image Decompressor node started.")
        rospy.loginfo(f"Subscribing to: {compressed_topic}")
        rospy.loginfo(f"Publishing to: {raw_topic}")

    def callback(self, compressed_msg):
        """
        압축 이미지를 받아 디코딩하고 다시 발행하는 콜백 함수.
        """
        try:
            # 1. ROS CompressedImage 메시지를 OpenCV 이미지로 변환 (압축 해제)
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 2. OpenCV 이미지를 ROS raw Image 메시지로 변환
            # encoding='bgr8'은 일반적인 OpenCV 컬러 이미지 형식입니다.
            raw_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            # 3. 원본 메시지의 타임스탬프와 프레임 ID를 그대로 사용하여 동기화 문제 방지
            raw_image_msg.header = compressed_msg.header

            # 4. 압축 해제된 이미지 메시지 발행
            self.raw_image_pub.publish(raw_image_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Failed to decompress and republish image: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        decompressor = ImageDecompressor()
        decompressor.run()
    except rospy.ROSInterruptException:
        pass