#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tf2_ros
from tf.transformations import quaternion_matrix
import message_filters

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from cv_bridge import CvBridge

class UnifiedVisionNodeV2:
    def __init__(self):
        rospy.init_node('unified_vision_node_v2', anonymous=True)
        rospy.loginfo("--- Unified Vision Node v2 (Optimized Hinton Edition) ---")

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using compute device: {self.device}")

        # --- 모델 로딩 ---
        # ... (이 부분은 기존과 동일) ...
        try:
            supply_model_path = rospy.get_param('~supply_model_path', './tracking1.pt')
            self.supply_model = YOLO(supply_model_path).to(self.device)
            marker_model_path = rospy.get_param('~marker_model_path', './vision_marker2.pt')
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']
            path_model_path = rospy.get_param('~yolo_model_path', './weights.pt')
            self.path_model = YOLO(path_model_path).to(self.device)
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO models: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        # --- 파라미터 로딩 ---
        # ... (기존과 동일) ...
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 20)
        self.point_downsample_rate = rospy.get_param('~downsample', 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.4)
        self.yolo_confidence = rospy.get_param('~yolo_confidence', 0.5)
        self.drivable_class_index = rospy.get_param('~drivable_class_index', 0)
        
        # --- [최적화 1] 연산용 이미지 크기 설정 ---
        self.proc_width = rospy.get_param('~proc_width', 640)
        self.proc_height = rospy.get_param('~proc_height', 480)
        rospy.loginfo(f"Processing images at resolution: {self.proc_width}x{self.proc_height}")

        # --- [최적화 2] 연산 주기 조절을 위한 변수 ---
        self.path_planning_interval = rospy.Duration(1.0 / 5.0)  # 5 Hz
        self.last_path_planning_time = rospy.Time(0)

        # --- 내부 변수 및 TF 초기화 ---
        self.camera_intrinsics = None
        self.scaled_camera_intrinsics = None # 스케일된 파라미터 저장용
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 퍼블리셔 선언 ---
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=1)
        self.path_pub = rospy.Publisher('/competition_path_yolo', Path, queue_size=1)
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz/compressed', CompressedImage, queue_size=1)
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz/compressed', CompressedImage, queue_size=1)

        # --- 서브스크라이버 선언 ---
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        realsense_img_sub = message_filters.Subscriber(realsense_img_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=5, slop=0.2)
        self.ts.registerCallback(self.realsense_callback)
        rospy.loginfo(f"Synchronized subscribers for Realsense started on topic: {realsense_img_topic}")
        
        usb_cam_topic = '/usb_cam/image_raw/compressed'
        self.usb_cam_sub = rospy.Subscriber(usb_cam_topic, CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscriber for USB Cam started on topic: {usb_cam_topic}")
        rospy.loginfo("✅ Unified Vision Node v2 (Optimized) initialized successfully.")


    def realsense_callback(self, compressed_image_msg, depth_msg, info_msg):
        try:
            # 압축 해제
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_orig = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # --- [최적화 1] 이미지 리사이즈 ---
            cv_color = cv2.resize(cv_color_orig, (self.proc_width, self.proc_height))
            cv_depth = cv2.resize(cv_depth_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            
            # 카메라 파라미터를 한 번만, 리사이즈된 해상도에 맞게 스케일링
            if self.scaled_camera_intrinsics is None:
                scale_x = self.proc_width / info_msg.width
                scale_y = self.proc_height / info_msg.height
                self.scaled_camera_intrinsics = {
                    'fx': info_msg.K[0] * scale_x, 'fy': info_msg.K[4] * scale_y,
                    'ppx': info_msg.K[2] * scale_x, 'ppy': info_msg.K[5] * scale_y
                }
                rospy.loginfo(f"Original Cam Intrinsics: {info_msg.K}")
                rospy.loginfo(f"Scaled Cam Intrinsics for {self.proc_width}x{self.proc_height}: {self.scaled_camera_intrinsics}")

            # 가벼운 작업은 매번 실행
            self.run_supply_tracking(cv_color, cv_depth)
            
            # --- [최적화 2] 무거운 작업은 주기적으로 실행 ---
            current_time = rospy.Time.now()
            if (current_time - self.last_path_planning_time) >= self.path_planning_interval:
                self.run_yolo_path_planning(cv_color, cv_depth, depth_msg.header)
                self.last_path_planning_time = current_time

            # --- [버그 수정] 시각화 이미지를 다시 압축해서 발행 ---
            # 시각화를 위해 작은 이미지(cv_color)를 원본 크기로 다시 늘려서 그릴 수도 있지만,
            # 성능을 위해 그냥 작은 이미지를 발행합니다.
            output_msg = CompressedImage()
            output_msg.header.stamp = current_time
            output_msg.format = "jpeg"
            output_msg.data = np.array(cv2.imencode('.jpg', cv_color)[1]).tobytes()
            self.realsense_viz_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr(f"Error in Realsense callback: {e}", exc_info=True)


    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results)

            # --- [버그 수정] USB 캠 시각화도 압축해서 발행 ---
            output_msg = CompressedImage()
            output_msg.header.stamp = rospy.Time.now()
            output_msg.format = "jpeg"
            output_msg.data = np.array(cv2.imencode('.jpg', annotated_image)[1]).tobytes()
            self.usb_cam_viz_pub.publish(output_msg)
        except Exception as e:
            rospy.logerr(f"Error in USB Cam callback: {e}")
    
    # --- 이하 함수들은 'camera_intrinsics'를 'scaled_camera_intrinsics'로 변경해야 함 ---
    
    def run_supply_tracking(self, color_image, depth_image):
        if self.scaled_camera_intrinsics is None: return
        results = self.supply_model(color_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth_in_meters = depth_image[cy, cx] / 1000.0
                if depth_in_meters > 0:
                    x_cam, y_cam, z_cam = self.deproject_pixel_to_point(cx, cy, depth_in_meters)
                    point_msg = Point(x=z_cam, y=-x_cam, z=-y_cam)
                    self.distance_pub.publish(point_msg)
                    label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run_yolo_path_planning(self, color_image, depth_image, header):
        if self.scaled_camera_intrinsics is None: return
        mask = self.create_yolo_drivable_mask(color_image)
        if mask is None or not np.any(mask): 
            rospy.logwarn_throttle(2.0, "YOLO path planning: Drivable area not detected.")
            return
        
        # 시각화를 위해 주행 가능 영역을 컬러 이미지에 오버레이
        color_image[mask > 0] = cv2.addWeighted(color_image[mask > 0], 0.5, np.full_like(color_image[mask > 0], (0, 255, 0)), 0.5, 0)

        points_3d = self.unproject_2d_to_3d(mask, depth_image)
        if points_3d.shape[0] < 50: return # 포인트 개수 기준 하향 조정
        self.generate_and_publish_path(points_3d, header)

    def create_yolo_drivable_mask(self, color_image):
        # ... (기존과 동일) ...
        results = self.path_model(color_image, conf=self.yolo_confidence, verbose=False)
        result = results[0]
        if result.masks is None: return None
        final_mask = np.zeros((self.proc_height, self.proc_width), dtype=np.uint8) # 리사이즈된 크기로 마스크 생성
        drivable_indices = np.where(result.boxes.cls.cpu().numpy() == self.drivable_class_index)[0]
        if len(drivable_indices) == 0: return None
        for idx in drivable_indices:
            mask_data = result.masks.data.cpu().numpy()[idx]
            # YOLO 마스크는 보통 입력과 다른 크기이므로, 우리 처리 크기에 맞게 리사이즈
            resized_mask = cv2.resize(mask_data, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            final_mask = np.maximum(final_mask, (resized_mask * 255).astype(np.uint8))
        return final_mask

    def deproject_pixel_to_point(self, u, v, depth):
        # 스케일된 파라미터 사용
        fx, fy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy']
        ppx, ppy = self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
        x = (u - ppx) * depth / fx
        y = (v - ppy) * depth / fy
        return x, y, depth

    def unproject_2d_to_3d(self, mask_2d, cv_depth):
        # 스케일된 파라미터 사용
        fx, fy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy']
        cx, cy = self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
        v, u = np.where(mask_2d > 0)
        if self.point_downsample_rate > 1:
            v, u = v[::self.point_downsample_rate], u[::self.point_downsample_rate]
        depths = cv_depth[v, u]
        valid = depths > 0; u, v, depths = u[valid], v[valid], depths[valid]
        z = depths / 1000.0; x = (u - cx) * z / fx; y = (v - cy) * z / fy
        return np.vstack((x, y, z)).T

    # ... (generate_and_publish_path, transform_to_matrix, draw_marker_detections 함수는 기존과 동일) ...
    def generate_and_publish_path(self, points_3d_cam, header):
        # ... (기존과 동일)
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, header.frame_id, header.stamp, rospy.Duration(0.2))
            trans_matrix = self.transform_to_matrix(transform)
            points_hom = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))
            points_3d_robot = (trans_matrix @ points_hom.T).T[:, :3]
            valid_indices = (points_3d_robot[:, 0] > 0.1) & (points_3d_robot[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20: return
            x, y = points_3d_robot[valid_indices, 0], points_3d_robot[valid_indices, 1]
            coeffs = np.polyfit(x, y, 2); poly = np.poly1d(coeffs)
            path_x = np.linspace(0.0, x.max(), self.num_path_points)
            path_y = poly(path_x)
            raw_path = []
            for px, py in zip(path_x, path_y):
                dists = np.linalg.norm(points_3d_robot[:, :2] - np.array([px, py]), axis=1)
                nearby_pts = points_3d_robot[dists < 0.15]
                if nearby_pts.shape[0] > 3:
                    raw_path.append(np.array([px, py, np.median(nearby_pts[:, 2])]))
            if len(raw_path) < self.num_path_points / 2: return
            if self.smoothed_path_points_3d is None or len(self.smoothed_path_points_3d) != len(raw_path):
                self.smoothed_path_points_3d = raw_path
            else:
                for i in range(len(raw_path)):
                    self.smoothed_path_points_3d[i] = self.smoothing_factor * raw_path[i] + (1 - self.smoothing_factor) * self.smoothed_path_points_3d[i]

            path_msg = Path()
            path_msg.header.stamp = header.stamp
            path_msg.header.frame_id = self.robot_base_frame
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")

    def transform_to_matrix(self, t: TransformStamped):
        trans, rot = t.transform.translation, t.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[:3, 3] = [trans.x, trans.y, trans.z]
        return mat

    def draw_marker_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                text = f"{label}: {conf:.2f}"; cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

if __name__ == '__main__':
    try:
        node = UnifiedVisionNodeV2()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass