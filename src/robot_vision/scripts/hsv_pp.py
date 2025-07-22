#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf2_ros
from tf.transformations import quaternion_matrix
import threading
import ast
import math

# ROS 메시지 타입
from sensor_msgs.msg import Image, CameraInfo, CompressedImage # [수정] CompressedImage 임포트 확인
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class CompetitionPathPlanner:
    """
    HSV 색상 기반으로 주행 가능 영역을 인식하고,
    'Pure Pursuit' 알고리즘으로 생성된 경로를 따라 터틀봇3를 자율주행 시키는 노드.
    """
    def __init__(self):
        rospy.loginfo("--- Competition Path Planner & Controller [Hinton Edition] ---")
        rospy.loginfo("🚀 Initializing...")
        self.bridge = CvBridge()

        # --- 경로 생성 파라미터 ---
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 20)
        self.point_downsample_rate = rospy.get_param('~downsample', 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.5)
        
        lower_hsv_param = rospy.get_param('~lower_hsv_bound', [80, 50, 100])
        upper_hsv_param = rospy.get_param('~upper_hsv_bound', [100, 255, 255])
        self.lower_hsv_bound = np.array(ast.literal_eval(lower_hsv_param) if isinstance(lower_hsv_param, str) else lower_hsv_param)
        self.upper_hsv_bound = np.array(ast.literal_eval(upper_hsv_param) if isinstance(upper_hsv_param, str) else upper_hsv_param)

        # --- [추가] 주행 제어 파라미터 ---
        self.max_linear_velocity = rospy.get_param('~max_linear_velocity', 0.15)
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', 1.5)
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.4)
        
        rospy.loginfo(f"Target Frame: {self.robot_base_frame}")
        rospy.loginfo(f"Max Linear Vel: {self.max_linear_velocity} m/s, Max Angular Vel: {self.max_angular_velocity} rad/s")
        rospy.loginfo(f"Lookahead Distance: {self.lookahead_distance} m")

        # --- 내부 변수 초기화 ---
        self.kernel = np.ones((5, 5), np.uint8)
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- ROS Subscriber & Publisher ---
        # [수정] 라즈베리파이에서 오는 압축된 이미지 토픽을 구독하도록 변경
        color_topic = "/camera/color/image_raw/compressed"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"

        # [수정] 컬러 이미지 구독자의 메시지 타입을 CompressedImage로 변경
        color_sub = message_filters.Subscriber(color_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.synchronized_callback)

        self.path_pub = rospy.Publisher("/competition_path", Path, queue_size=1)
        self.viz_pub = rospy.Publisher("/drivable_area/viz", Image, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.loginfo("✅ Planner & Controller initialized successfully. Ready to drive!")

    def synchronized_callback(self, compressed_color_msg, depth_msg, info_msg):
        """메인 콜백: 센서 데이터 수신 -> 경로 생성 -> 로봇 구동"""
        try:
            # [수정] CvBridge를 사용하여 압축 이미지를 OpenCV 이미지로 변환 (더 안정적인 방법)
            cv_color = self.bridge.compressed_imgmsg_to_cv2(compressed_color_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            mask_2d = self.create_drivable_mask(cv_color)
            self.visualize(cv_color, mask_2d)

            if not np.any(mask_2d):
                rospy.logwarn_throttle(5, "Drivable area mask is empty. Stopping robot.")
                self.stop_robot()
                return

            points_3d_camera = self.unproject_2d_to_3d(mask_2d, cv_depth, info_msg)
            if points_3d_camera.shape[0] < 100:
                rospy.logwarn_throttle(5, "Not enough valid 3D points. Stopping robot.")
                self.stop_robot()
                return

            self.generate_and_publish_path(points_3d_camera, info_msg.header)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            self.stop_robot()
        except Exception as e:
            rospy.logerr(f"An error occurred in synchronized_callback: {e}")
            self.stop_robot()

    def generate_and_publish_path(self, points_3d_cam_optical, header):
        # ... (이하 로직은 이전과 거의 동일) ...
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, header.frame_id, header.stamp, rospy.Duration(0.2))
            trans_matrix = self.transform_to_matrix(transform)
            points_homogeneous = np.hstack((points_3d_cam_optical, np.ones((points_3d_cam_optical.shape[0], 1))))
            points_3d_robot_base = (trans_matrix @ points_homogeneous.T).T[:, :3]

            valid_indices = (points_3d_robot_base[:, 0] > 0.1) & (points_3d_robot_base[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20:
                self.stop_robot()
                return

            x_coords = points_3d_robot_base[valid_indices, 0]
            y_coords = points_3d_robot_base[valid_indices, 1]
            path_poly_coeffs = np.polyfit(x_coords, y_coords, 2)
            path_poly = np.poly1d(path_poly_coeffs)

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
                else:
                    if len(raw_path_points_3d) > 0:
                        raw_path_points_3d.append(np.array([px, py, raw_path_points_3d[-1][2]]))
                    else:
                        z = np.median(points_3d_robot_base[valid_indices, 2])
                        raw_path_points_3d.append(np.array([px, py, z]))

            if len(raw_path_points_3d) < self.num_path_points / 2:
                self.stop_robot()
                return

            current_path = np.array(raw_path_points_3d)
            if self.smoothed_path_points_3d is None or len(self.smoothed_path_points_3d) != len(current_path):
                self.smoothed_path_points_3d = current_path
            else:
                alpha = self.smoothing_factor
                self.smoothed_path_points_3d = alpha * current_path + (1 - alpha) * self.smoothed_path_points_3d
            
            path_msg = Path()
            path_msg.header.stamp = header.stamp
            path_msg.header.frame_id = self.robot_base_frame
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
            self.path_pub.publish(path_msg)
            self.drive_robot(path_msg)

        except Exception as e:
            rospy.logerr(f"Path generation/driving failed: {e}")
            self.stop_robot()

    def drive_robot(self, path_msg):
        # ... (이하 로직은 이전과 동일) ...
        if len(path_msg.poses) < 2:
            self.stop_robot()
            return
        
        target_point = None
        for point in path_msg.poses:
            dist = math.sqrt(point.pose.position.x**2 + point.pose.position.y**2)
            if dist > self.lookahead_distance:
                target_point = point
                break
        
        if target_point is None:
            target_point = path_msg.poses[-1]

        target_x = target_point.pose.position.x
        target_y = target_point.pose.position.y
        alpha = math.atan2(target_y, target_x)
        
        angular_vel = 2.0 * alpha
        linear_vel = self.max_linear_velocity * (1.0 - abs(alpha) / (math.pi / 2))

        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        linear_vel = np.clip(linear_vel, 0.0, self.max_linear_velocity)

        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)

    def create_drivable_mask(self, cv_color):
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv_bound, self.upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def unproject_2d_to_3d(self, mask_2d, cv_depth, camera_info):
        fx, fy = camera_info.K[0], camera_info.K[4]; cx, cy = camera_info.K[2], camera_info.K[5]
        v, u = np.where(mask_2d > 0)
        if self.point_downsample_rate > 1:
            indices = np.arange(len(u)); u, v = u[::self.point_downsample_rate], v[::self.point_downsample_rate]
        depths_mm = cv_depth[v, u]
        valid_indices = depths_mm > 0; u, v, depths_mm = u[valid_indices], v[valid_indices], depths_mm[valid_indices]
        z_meters = depths_mm / 1000.0; x_meters = (u - cx) * z_meters / fx; y_meters = (v - cy) * z_meters / fy
        return np.vstack((x_meters, y_meters, z_meters)).T

    def transform_to_matrix(self, transform: TransformStamped):
        t = transform.transform.translation; r = transform.transform.rotation
        mat = quaternion_matrix([r.x, r.y, r.z, r.w])
        mat[0, 3], mat[1, 3], mat[2, 3] = t.x, t.y, t.z
        return mat

    def visualize(self, cv_color, mask_2d):
        viz_image = cv_color.copy()
        if np.any(mask_2d):
            overlay_color = np.array([144, 255, 224], dtype=np.uint8)
            original_pixels = viz_image[mask_2d > 0]
            blended_pixels = (original_pixels * 0.5 + overlay_color * 0.5).astype(np.uint8)
            viz_image[mask_2d > 0] = blended_pixels
        try:
            self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logwarn(f"Could not publish viz image: {e}")

def main():
    rospy.init_node('hsv_path_planner_node', anonymous=True)
    planner = CompetitionPathPlanner()
    rospy.on_shutdown(planner.stop_robot)
    rospy.spin()

if __name__ == '__main__':
    main()