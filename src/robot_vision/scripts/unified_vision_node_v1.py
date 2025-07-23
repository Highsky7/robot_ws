#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tf2_ros
from tf.transformations import quaternion_matrix
import ast
import message_filters

# ROS 메시지 타입
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError

class UnifiedVisionNodeV1:
    """
    [수정됨] HSV 기반 경로 계획을 포함하는 통합 비전 노드.
    모든 시각화 결과는 OpenCV 창 대신 ROS 토픽으로 발행됩니다.
    """
    def __init__(self):
        rospy.init_node('unified_vision_node_v1', anonymous=True)
        rospy.loginfo("--- Unified Vision Node v1 (HSV Path) [Hinton Edition] ---")

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using compute device: {self.device}")

        # --- 1. 모든 YOLO 모델 한 번에 로딩 ---
        try:
            # 보급품 추적용 모델
            supply_model_path = rospy.get_param('~supply_model_path', './tracking1.pt')
            self.supply_model = YOLO(supply_model_path).to(self.device)
            rospy.loginfo(f"Supply tracking model loaded from: {supply_model_path}")

            # USB 캠 마커 인식용 모델
            marker_model_path = rospy.get_param('~marker_model_path', './vision_marker2.pt')
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']
            rospy.loginfo(f"Marker detection model loaded from: {marker_model_path}")

        except Exception as e:
            rospy.logerr(f"Failed to load YOLO models: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        # --- 2. 모든 파라미터 로딩 ---
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 20)
        self.point_downsample_rate = rospy.get_param('~downsample', 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.5)
        lower_hsv_param = rospy.get_param('~lower_hsv_bound', '[0, 0, 0]')
        upper_hsv_param = rospy.get_param('~upper_hsv_bound', '[100, 255, 255]')
        self.lower_hsv_bound = np.array(ast.literal_eval(lower_hsv_param))
        self.upper_hsv_bound = np.array(ast.literal_eval(upper_hsv_param))
        self.kernel = np.ones((5, 5), np.uint8)

        # --- 3. 내부 변수 및 TF 초기화 ---
        self.camera_intrinsics = None
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 4. 모든 퍼블리셔 선언 ---
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=10)
        self.path_pub = rospy.Publisher('/competition_path_hsv', Path, queue_size=1)
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz', Image, queue_size=1)
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz', Image, queue_size=1)

        # --- 5. 모든 서브스크라이버 선언 ---
        realsense_img_topic = '/camera/color/image_raw_uncompressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        realsense_img_sub = message_filters.Subscriber(realsense_img_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], 10, 0.1)
        self.ts.registerCallback(self.realsense_callback)
        rospy.loginfo(f"Synchronized subscribers for Realsense started on topic: {realsense_img_topic}")
        usb_cam_topic = '/usb_cam/image_raw/compressed'
        self.usb_cam_sub = rospy.Subscriber(usb_cam_topic, CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscriber for USB Cam started on topic: {usb_cam_topic}")
        rospy.loginfo("✅ Unified Vision Node v1 initialized successfully.")


    def realsense_callback(self, image_msg, depth_msg, info_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            if self.camera_intrinsics is None:
                self.camera_intrinsics = {'fx': info_msg.K[0], 'fy': info_msg.K[4], 'ppx': info_msg.K[2], 'ppy': info_msg.K[5]}
                rospy.loginfo(f"Realsense camera intrinsics received: {self.camera_intrinsics}")

            # --- START: 수정된 부분 ---
            # Task 수행: 보급품 추적(시각화 포함) + HSV 경로 계획
            # 이제 run_supply_tracking 함수가 직접 cv_color 이미지에 그림을 그립니다.
            self.run_supply_tracking(cv_color, cv_depth)
            
            # 경로 계획 함수는 시각화가 필요 없으므로 그대로 둡니다.
            self.run_hsv_path_planning(cv_color, cv_depth, info_msg.header)

            # 시각화 정보가 그려진 이미지를 발행합니다.
            self.realsense_viz_pub.publish(self.bridge.cv2_to_imgmsg(cv_color, 'bgr8'))
            # --- END: 수정된 부분 ---

        except Exception as e:
            rospy.logerr(f"Error in Realsense callback: {e}", exc_info=True)


    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            results = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results)
            self.usb_cam_viz_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8'))
        except Exception as e:
            rospy.logerr(f"Error in USB Cam callback: {e}")
    
    # --- Task Specific Methods ---

    # --- START: 수정된 부분 ---
    def run_supply_tracking(self, color_image, depth_image):
        """
        [수정됨] 보급품을 추적하고, 전달받은 color_image에 직접 시각화 결과를 그립니다.
        """
        results = self.supply_model(color_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                depth_in_meters = depth_image[cy, cx] / 1000.0

                if depth_in_meters > 0 and self.camera_intrinsics:
                    x_cam, y_cam, z_cam = self.deproject_pixel_to_point(cx, cy, depth_in_meters)
                    point_msg = Point(x=z_cam, y=-x_cam, z=-y_cam)
                    self.distance_pub.publish(point_msg)

                    # 시각화 로직 추가
                    label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2) # 노란색 상자
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # --- END: 수정된 부분 ---

    def run_hsv_path_planning(self, color_image, depth_image, header):
        mask = self.create_drivable_mask(color_image)
        if not np.any(mask): return
        points_3d = self.unproject_2d_to_3d(mask, depth_image)
        if points_3d.shape[0] < 100: return
        self.generate_and_publish_path(points_3d, header)

    # --- Helper Functions (이하 코드는 이전과 동일) ---
    def deproject_pixel_to_point(self, u, v, depth):
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        ppx, ppy = self.camera_intrinsics['ppx'], self.camera_intrinsics['ppy']
        x = (u - ppx) * depth / fx
        y = (v - ppy) * depth / fy
        return x, y, depth

    def create_drivable_mask(self, cv_color):
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv_bound, self.upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def unproject_2d_to_3d(self, mask_2d, cv_depth):
        if self.camera_intrinsics is None: return np.array([])
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['ppx'], self.camera_intrinsics['ppy']
        v, u = np.where(mask_2d > 0)
        if self.point_downsample_rate > 1:
            v, u = v[::self.point_downsample_rate], u[::self.point_downsample_rate]
        depths = cv_depth[v, u]
        valid = depths > 0
        u, v, depths = u[valid], v[valid], depths[valid]
        z = depths / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.vstack((x, y, z)).T

    def generate_and_publish_path(self, points_3d_cam, header):
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, header.frame_id, header.stamp, rospy.Duration(0.2))
            trans_matrix = self.transform_to_matrix(transform)
            points_hom = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))
            points_3d_robot = (trans_matrix @ points_hom.T).T[:, :3]
            
            valid_indices = (points_3d_robot[:, 0] > 0.1) & (points_3d_robot[:, 0] < self.path_lookahead)
            if np.sum(valid_indices) < 20: return

            x, y = points_3d_robot[valid_indices, 0], points_3d_robot[valid_indices, 1]
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            
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
        except Exception as e:
            rospy.logerr(f"Error in generate_and_publish_path: {e}", exc_info=True)

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
                text = f"{label}: {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

if __name__ == '__main__':
    try:
        node = UnifiedVisionNodeV1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass