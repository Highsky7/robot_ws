#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

class SupplyBoxTrackerROS:
    def __init__(self):
        """
        클래스 초기화 함수
        ROS 노드, 퍼블리셔, 서브스크라이버, YOLO 모델, CvBridge 등을 설정합니다.
        """
        rospy.init_node('supply_box_tracker_ros', anonymous=True)

        # 1. CvBridge 및 YOLO 모델 초기화
        self.bridge = CvBridge()
        self.model_path = './tracking1.pt'
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo(f"YOLOv8 모델 로딩 성공: {self.model_path}")
        except Exception as e:
            rospy.logerr(f"YOLOv8 모델 로딩 실패: {e}")
            rospy.signal_shutdown("YOLO 모델 로딩 실패")
            return

        # 2. 카메라 내부 파라미터 저장을 위한 변수
        self.intrinsics = None
        
        # 3. ROS 퍼블리셔 설정
        self.distance_publisher = rospy.Publisher('/supply_distance', Point, queue_size=10)

        # 4. ROS 서브스크라이버 설정 (메시지 필터 사용)
        # 구독할 토픽 이름을 확인하고 필요시 수정하세요.
        image_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)

        # 타임스탬프가 근사적으로 일치하는 메시지들을 동기화하여 콜백 함수로 전달
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, info_sub], 
            queue_size=10, 
            slop=0.2 # 메시지 간 허용 시간차 (초 단위)
        )
        ts.registerCallback(self.callback)

        rospy.loginfo("ROS 토픽 서브스크라이버 준비 완료. 메시지를 기다립니다...")

    def deproject_pixel_to_point(self, u, v, depth):
        """
        픽셀 좌표와 깊이 값을 이용해 3D 공간 좌표를 계산하는 함수
        :param u: 픽셀의 x 좌표
        :param v: 픽셀의 y 좌표
        :param depth: 해당 픽셀의 깊이 값 (미터 단위)
        :return: (x, y, z) 3D 좌표
        """
        if self.intrinsics is None:
            return None
        
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']

        x = (u - ppx) * depth / fx
        y = (v - ppy) * depth / fy
        z = depth
        
        return x, y, z

    def callback(self, compressed_image_msg, depth_msg, camera_info_msg):
        """
        동기화된 메시지를 수신했을 때 호출되는 메인 콜백 함수
        """
        # 1. 카메라 내부 파라미터(Intrinsics)를 한 번만 저장
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': camera_info_msg.K[0],
                'fy': camera_info_msg.K[4],
                'ppx': camera_info_msg.K[2],
                'ppy': camera_info_msg.K[5],
                'width': camera_info_msg.width,
                'height': camera_info_msg.height
            }
            rospy.loginfo(f"카메라 내부 파라미터 수신 완료: {self.intrinsics}")

        try:
            # 2. 압축된 컬러 이미지 디코딩
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 3. 뎁스 이미지 변환 (ROS Image -> OpenCV)
            # 16UC1 인코딩 (16-bit unsigned, 1 channel) -> mm 단위
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        except Exception as e:
            rospy.logerr(f"이미지 변환 실패: {e}")
            return

        # 4. YOLOv8 추론
        results = self.model(color_image, verbose=False)

        # 5. 결과 처리
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 뎁스 이미지에서 깊이 값 추출 (mm -> m)
                depth_in_meters = depth_image[cy, cx] / 1000.0

                if depth_in_meters > 0:
                    # 6. 3D 좌표 계산 (직접 구현한 함수 사용)
                    point_cam = self.deproject_pixel_to_point(cx, cy, depth_in_meters)
                    if point_cam is None:
                        continue
                    
                    # 7. 좌표계 변환
                    x_cam, y_cam, z_cam = point_cam
                    x_new = z_cam
                    y_new = -x_cam
                    z_new = -y_cam

                    # 8. ROS 토픽 발행
                    point_msg = Point()
                    point_msg.x = x_new
                    point_msg.y = y_new
                    point_msg.z = z_new
                    self.distance_publisher.publish(point_msg)

                    # 9. 시각화
                    label = f"X:{x_new:.2f} Y:{y_new:.2f} Z:{z_new:.2f}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

        # 10. 결과 이미지 창에 표시
        cv2.imshow("YOLOv8 ROS Tracker", color_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        tracker = SupplyBoxTrackerROS()
        # rospy.spin()은 노드가 종료될 때까지 대기하며 콜백 함수를 계속 호출합니다.
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()