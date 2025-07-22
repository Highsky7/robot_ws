#!/usr.bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf2_ros
from tf.transformations import quaternion_matrix
import threading
import ast

# ROS 메시지 타입
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class CompetitionPathPlanner:
    """
    대회용 최종 경로 생성 노드.
    Intel Realsense D435i의 데이터를 기반으로 '투영 중심선' 알고리즘을 사용하여
    로봇의 'base_link' 프레임 기준의 안정적인 3D 주행 경로를 생성하고 발행합니다.
    모든 주요 설정은 ROS 파라미터를 통해 튜닝할 수 있습니다.
    """
    def __init__(self):
        rospy.loginfo("--- Competition Path Planner [Hinton Edition] ---")
        rospy.loginfo("🚀 Initializing...")
        self.bridge = CvBridge()

        # --- ROS 파라미터 불러오기 (대회 현장 튜닝용) ---
        # 1. 기준 좌표계: 로봇의 움직이는 기준. 'base_link'가 표준.
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        
        # 2. 경로 생성 관련
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)  # 경로를 생성할 최대 전방 거리 (m)
        self.num_path_points = rospy.get_param('~num_path_points', 20) # 경로점 개수
        self.point_downsample_rate = rospy.get_param('~downsample', 4) # 포인트 클라우드 다운샘플링 비율

        # 3. 경로 안정화 필터
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.5) # EMA 필터 알파 값 (0~1)

        # 4. 주행 영역 검출 (HSV 색상)
        # rosparam으로 배열을 불러옵니다. launch 파일에서 쉽게 수정 가능!
        lower_hsv_param = rospy.get_param('~lower_hsv_bound', [80, 50, 100])
        upper_hsv_param = rospy.get_param('~upper_hsv_bound', [100, 255, 255])

        # launch 파일에서 읽은 파라미터가 문자열일 경우 안전하게 리스트로 변환
        if isinstance(lower_hsv_param, str):
            lower_hsv = ast.literal_eval(lower_hsv_param)
        else:
            lower_hsv = lower_hsv_param # 기본값 (리스트)

        if isinstance(upper_hsv_param, str):
            upper_hsv = ast.literal_eval(upper_hsv_param)
        else:
            upper_hsv = upper_hsv_param # 기본값 (리스트)

        self.lower_hsv_bound = np.array(lower_hsv)
        self.upper_hsv_bound = np.array(upper_hsv)
        
        rospy.loginfo(f"Target Frame: {self.robot_base_frame}")
        rospy.loginfo(f"Path Lookahead: {self.path_lookahead}m")
        rospy.loginfo(f"HSV Lower: {self.lower_hsv_bound}, Upper: {self.upper_hsv_bound}")

        # --- 내부 변수 초기화 ---
        self.kernel = np.ones((5, 5), np.uint8)
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- ROS Subscriber & Publisher ---
        color_topic = "/camera/color/image_raw/compressed"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"

        color_sub = message_filters.Subscriber(color_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)

        self.path_pub = rospy.Publisher("/competition_path", Path, queue_size=1)
        self.viz_pub = rospy.Publisher("/drivable_area/viz", Image, queue_size=1)
        
        rospy.loginfo("✅ Planner initialized successfully. Ready for the competition!")

    def synchronized_callback(self, compressed_color_msg, depth_msg, info_msg):
        """동기화된 센서 데이터를 받아 경로 생성을 총괄하는 메인 콜백 함수."""
        try:
            # 1. 데이터 디코딩 및 준비
            np_arr = np.frombuffer(compressed_color_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 2. 주행 가능 영역 마스크 생성
            mask_2d = self.create_drivable_mask(cv_color)
            if not np.any(mask_2d):
                rospy.logwarn_throttle(5, "Drivable area mask is empty. No path generated.")
                return

            # 3. 2D 마스크 -> 3D 포인트 클라우드 변환
            points_3d_camera = self.unproject_2d_to_3d(mask_2d, cv_depth, info_msg)
            if points_3d_camera.shape[0] < 100:
                rospy.logwarn_throttle(5, "Not enough valid 3D points to generate a path.")
                return

            # 4. '투영 중심선' 알고리즘으로 경로 생성 및 발행
            self.generate_and_publish_path(points_3d_camera, info_msg.header)

            # 5. 시각화 이미지 발행
            self.visualize(cv_color, mask_2d)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"An error occurred in synchronized_callback: {e}")

    def generate_and_publish_path(self, points_3d_cam_optical, header):
        """'투영 중심선' 알고리즘의 핵심. 3D 포인트를 기반으로 경로를 생성하고 발행합니다."""
        try:
            # 1. 좌표계 변환: camera_color_optical_frame -> base_link
            transform = self.tf_buffer.lookup_transform(
                self.robot_base_frame,       # Target Frame (e.g., 'base_link')
                header.frame_id,             # Source Frame (e.g., 'camera_color_optical_frame')
                header.stamp,
                rospy.Duration(0.2)
            )
            trans_matrix = self.transform_to_matrix(transform)
            
            points_homogeneous = np.hstack((points_3d_cam_optical, np.ones((points_3d_cam_optical.shape[0], 1))))
            points_3d_robot_base = (trans_matrix @ points_homogeneous.T).T[:, :3]

            # 2. Top-down 뷰에서 중심선 계산 (다항식 회귀)
            valid_indices = (points_3d_robot_base[:, 0] > 0.1) & (points_3d_robot_base[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20:
                return

            x_coords = points_3d_robot_base[valid_indices, 0]
            y_coords = points_3d_robot_base[valid_indices, 1]
            
            path_poly_coeffs = np.polyfit(x_coords, y_coords, 2)
            path_poly = np.poly1d(path_poly_coeffs)

            # 3. 경로점(Waypoints) 생성 및 Z값 복원
            path_x = np.linspace(0.0, x_coords.max(), self.num_path_points)
            path_y = path_poly(path_x)

            raw_path_points_3d = []
            for px, py in zip(path_x, path_y):
                search_radius = 0.15
                distances = np.linalg.norm(points_3d_robot_base[:, :2] - np.array([px, py]), axis=1)
                nearby_points = points_3d_robot_base[distances < search_radius]
                if nearby_points.shape[0] > 3:
                    z = np.median(nearby_points[:, 2])
                    raw_path_points_3d.append(np.array([px, py, z]))

            if len(raw_path_points_3d) < self.num_path_points / 2: return

            # 4. 시간적 안정화 필터(EMA) 적용
            alpha = self.smoothing_factor
            if self.smoothed_path_points_3d is None or len(self.smoothed_path_points_3d) != len(raw_path_points_3d):
                self.smoothed_path_points_3d = raw_path_points_3d
            else:
                for i in range(len(raw_path_points_3d)):
                    self.smoothed_path_points_3d[i] = alpha * raw_path_points_3d[i] + (1 - alpha) * self.smoothed_path_points_3d[i]
            
            # 5. 최종 경로 메시지 생성 및 발행
            path_msg = Path()
            path_msg.header.stamp = header.stamp
            path_msg.header.frame_id = self.robot_base_frame # 기준 프레임은 'base_link'
            
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
                
            self.path_pub.publish(path_msg)

        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, f"TF transform lookup failed from {header.frame_id} to {self.robot_base_frame}. Is TF running? Error: {e}")
        except Exception as e:
            rospy.logerr(f"Path generation failed: {e}")

    # --- 이하 헬퍼 함수들은 이전과 동일 ---
    def create_drivable_mask(self, cv_color):
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv_bound, self.upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def unproject_2d_to_3d(self, mask_2d, cv_depth, camera_info):
        fx, fy = camera_info.K[0], camera_info.K[4]
        cx, cy = camera_info.K[2], camera_info.K[5]
        v, u = np.where(mask_2d > 0)
        if self.point_downsample_rate > 1:
            indices = np.arange(len(u))
            downsampled_indices = indices[::self.point_downsample_rate]
            u, v = u[downsampled_indices], v[downsampled_indices]
        depths_mm = cv_depth[v, u]
        valid_indices = depths_mm > 0
        u, v, depths_mm = u[valid_indices], v[valid_indices], depths_mm[valid_indices]
        z_meters = depths_mm / 1000.0
        x_meters = (u - cx) * z_meters / fx
        y_meters = (v - cy) * z_meters / fy
        return np.vstack((x_meters, y_meters, z_meters)).T

    def transform_to_matrix(self, transform: TransformStamped):
        t = transform.transform.translation
        r = transform.transform.rotation
        trans_matrix = quaternion_matrix([r.x, r.y, r.z, r.w])
        trans_matrix[0, 3] = t.x
        trans_matrix[1, 3] = t.y
        trans_matrix[2, 3] = t.z
        return trans_matrix

    def visualize(self, cv_color, mask_2d):
        """시각화 이미지를 생성하고 발행합니다."""
        viz_image = cv_color.copy()
        
        # 마스크가 True인 영역 (주행 가능 영역)이 존재할 경우에만 오버레이를 적용합니다.
        if np.any(mask_2d):
            # 오버레이할 단색 컬러를 정의합니다. (BGR 순서)
            overlay_color = np.array([144, 255, 224], dtype=np.uint8) # 민트색 계열

            # NumPy의 배열 연산을 사용하여 픽셀 단위로 블렌딩합니다.
            # cv2.addWeighted와 동일한 연산: (원본 픽셀 * 0.5) + (오버레이 색상 * 0.5)
            # NumPy는 overlay_color(3,)를 viz_image[mask_2d > 0](N, 3)의 모든 행에 맞게 브로드캐스팅해줍니다.
            
            # 1. 블렌딩할 원본 픽셀들을 가져옵니다.
            original_pixels = viz_image[mask_2d > 0]
            
            # 2. NumPy를 이용해 가중 합을 계산합니다. 부동소수점 연산 후 uint8로 변환합니다.
            blended_pixels = (original_pixels * 0.5 + overlay_color * 0.5).astype(np.uint8)
            
            # 3. 계산된 결과를 다시 원본 이미지의 해당 위치에 할당합니다.
            viz_image[mask_2d > 0] = blended_pixels
        
        cv2.imshow("Original Image" , cv_color)
        cv2.imshow("Drivable Area Visualization", viz_image)
        
        cv2.waitKey(1)

        try:
            self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logwarn(f"Could not publish viz image: {e}")

def main():
    rospy.init_node('hsv_path_planner_node', anonymous=True)
    try:
        planner = CompetitionPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutting down.")

if __name__ == '__main__':
    main()