#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tf2_ros
from tf.transformations import quaternion_matrix
import threading

# ROS 메시지 타입
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class YoloPathPlanner:
    """
    대회용 YOLO 기반 최종 경로 생성 노드.
    YOLOv8 모델로 주행 가능 영역을 인식하고, '투영 중심선' 알고리즘을 사용하여
    로봇의 'base_link' 프레임 기준의 안정적인 3D 주행 경로를 생성하고 발행합니다.
    """
    def __init__(self):
        rospy.loginfo("--- YOLO Path Planner [Hinton Edition] ---")
        rospy.loginfo("🚀 Initializing...")
        self.bridge = CvBridge()

        # --- ROS 파라미터 불러오기 ---
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 20)
        self.point_downsample_rate = rospy.get_param('~downsample', 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.4)

        # --- YOLO 모델 관련 파라미터 ---
        model_path = rospy.get_param('~yolo_model_path', '')
        self.yolo_confidence = rospy.get_param('~yolo_confidence', 0.5)
        self.yolo_imgsz = rospy.get_param('~yolo_imgsz', 640)
        # ❗ 당신의 모델에서 '주행 가능 영역'에 해당하는 클래스 ID
        self.drivable_class_index = rospy.get_param('~drivable_class_index', 0)

        if not model_path:
            rospy.logerr("YOLO model path is not set! Please provide '~yolo_model_path' parameter.")
            rospy.signal_shutdown("YOLO model path not provided.")
            return

        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
                rospy.loginfo("YOLO model loaded to CUDA GPU.")
            else:
                rospy.loginfo("YOLO model loaded to CPU.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model from {model_path}: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        # --- 내부 변수, TF, Subscriber/Publisher 초기화 (이전과 동일) ---
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        color_topic = "/camera/color/image_raw/compressed"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        color_sub = message_filters.Subscriber(color_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], 5, 0.1)
        self.ts.registerCallback(self.synchronized_callback)

        self.path_pub = rospy.Publisher("/competition_path_yolo", Path, queue_size=1)
        self.viz_pub = rospy.Publisher("/drivable_area/viz_yolo", Image, queue_size=1)
        
        rospy.loginfo("✅ YOLO Planner initialized successfully. Ready to drive!")

    def synchronized_callback(self, compressed_color_msg, depth_msg, info_msg):
        """메인 콜백 함수"""
        try:
            np_arr = np.frombuffer(compressed_color_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 2. [변경점] YOLO를 사용하여 주행 가능 영역 마스크 생성
            mask_2d = self.create_mask_from_yolo(cv_color)

            if mask_2d is None or not np.any(mask_2d):
                rospy.logwarn_throttle(5, "YOLO did not detect a drivable area. No path generated.")
                return

            points_3d_camera = self.unproject_2d_to_3d(mask_2d, cv_depth, info_msg)
            if points_3d_camera.shape[0] < 100:
                rospy.logwarn_throttle(5, "Not enough valid 3D points from mask.")
                return

            self.generate_and_publish_path(points_3d_camera, info_msg.header)
            self.visualize(cv_color, mask_2d)

        except Exception as e:
            rospy.logerr(f"An error occurred in callback: {e}", exc_info=True)

    def create_mask_from_yolo(self, cv_color):
        """YOLO 추론 결과를 바탕으로 주행 가능 영역 마스크를 생성합니다."""
        # YOLO 모델로 추론 수행
        results = self.model.predict(cv_color, conf=self.yolo_confidence, imgsz=self.yolo_imgsz, verbose=False)
        
        result = results[0]
        if result.masks is None:
            return None # 마스크가 검출되지 않음

        # 최종 마스크를 생성할 빈 이미지 준비
        final_mask = np.zeros(result.orig_shape[:2], dtype=np.uint8)
        
        # 검출된 객체들의 클래스 ID와 마스크 데이터
        detected_classes = result.boxes.cls.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()

        # '주행 가능 영역' 클래스에 해당하는 마스크만 필터링
        drivable_indices = np.where(detected_classes == self.drivable_class_index)[0]

        if len(drivable_indices) == 0:
            return None # 주행 가능 영역이 검출되지 않음

        # 모든 주행 가능 영역 마스크를 하나로 합치기
        for idx in drivable_indices:
            mask = masks_data[idx]
            # YOLO 마스크는 이미지 크기에 맞게 리사이즈 필요
            resized_mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            final_mask = np.maximum(final_mask, (resized_mask * 255).astype(np.uint8))
            
        return final_mask

    # --- generate_and_publish_path 및 나머지 헬퍼 함수들은 이전 코드와 100% 동일합니다 ---
    # (모듈화의 장점: 인식부만 교체하고 계획부는 그대로 사용 가능)
    def generate_and_publish_path(self, points_3d_cam_optical, header):
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, header.frame_id, header.stamp, rospy.Duration(0.2))
            trans_matrix = self.transform_to_matrix(transform)
            points_homogeneous = np.hstack((points_3d_cam_optical, np.ones((points_3d_cam_optical.shape[0], 1))))
            points_3d_robot_base = (trans_matrix @ points_homogeneous.T).T[:, :3]
            valid_indices = (points_3d_robot_base[:, 0] > 0.1) & (points_3d_robot_base[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20: return
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
            if len(raw_path_points_3d) < self.num_path_points / 2: return
            alpha = self.smoothing_factor
            if self.smoothed_path_points_3d is None or len(self.smoothed_path_points_3d) != len(raw_path_points_3d):
                self.smoothed_path_points_3d = raw_path_points_3d
            else:
                for i in range(len(raw_path_points_3d)):
                    self.smoothed_path_points_3d[i] = alpha * raw_path_points_3d[i] + (1 - alpha) * self.smoothed_path_points_3d[i]
            path_msg = Path(header=rospy.Header(stamp=header.stamp, frame_id=self.robot_base_frame))
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, f"TF transform lookup failed: {e}")
        except Exception as e:
            rospy.logerr(f"Path generation failed: {e}")

    def unproject_2d_to_3d(self, mask_2d, cv_depth, camera_info):
        fx, fy = camera_info.K[0], camera_info.K[4]; cx, cy = camera_info.K[2], camera_info.K[5]
        v, u = np.where(mask_2d > 0)
        if self.point_downsample_rate > 1:
            v, u = v[::self.point_downsample_rate], u[::self.point_downsample_rate]
        depths = cv_depth[v, u]
        valid = depths > 0; u, v, depths = u[valid], v[valid], depths[valid]
        z = depths / 1000.0; x = (u - cx) * z / fx; y = (v - cy) * z / fy
        return np.vstack((x, y, z)).T

    def transform_to_matrix(self, t: TransformStamped):
        trans, rot = t.transform.translation, t.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[0, 3], mat[1, 3], mat[2, 3] = trans.x, trans.y, trans.z
        return mat

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
            
        # cv2.imshow("Drivable Area Visualization", viz_image)
        
        # cv2.waitKey(1)
        try:
            self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logwarn(f"Could not publish viz image: {e}")

def main():
    rospy.init_node('yolo_path_planner_node')
    try:
        planner = YoloPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutting down.")

if __name__ == '__main__':
    main()