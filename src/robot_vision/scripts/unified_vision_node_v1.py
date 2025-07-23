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

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge

class UnifiedVisionNodeV1Optimized:
    def __init__(self):
        rospy.init_node('unified_vision_node_v1_optimized', anonymous=True)
        rospy.loginfo("--- Unified Vision Node v1 (Optimized Hinton Edition) ---")

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using compute device: {self.device}")

        # --- [최적화 1] 연산 부하 감소를 위한 이미지 리사이즈 설정 ---
        # 원본 해상도(예: 640x480)에서 더 작은 크기로 줄여서 처리 속도를 높입니다.
        # (320, 240) 또는 (416, 416) 등 모델에 맞는 크기로 설정 가능합니다.
        self.proc_width = rospy.get_param('~proc_width', 320)
        self.proc_height = rospy.get_param('~proc_height', 320)
        rospy.loginfo(f"Processing images at resolution: {self.proc_width}x{self.proc_height}")

        # 모델 로딩 (기존과 동일)
        try:
            supply_model_path = rospy.get_param('~supply_model_path', './tracking1.pt')
            self.supply_model = YOLO(supply_model_path).to(self.device)
            marker_model_path = rospy.get_param('~marker_model_path', './vision_marker2.pt')
            self.marker_model = YOLO(marker_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO models: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        # 파라미터 로딩 (기존과 동일)
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
        
        self.camera_intrinsics = None
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 퍼블리셔 (기존과 동일)
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=10)
        self.path_pub = rospy.Publisher('/competition_path_hsv', Path, queue_size=1)
        
        # --- [최적화 2] 시각화 토픽도 압축해서 발행 ---
        # rqt_image_view가 원격 PC에 있을 경우 네트워크 부하를 크게 줄여줍니다.
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz/compressed', CompressedImage, queue_size=1)
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz/compressed', CompressedImage, queue_size=1)

        # --- [최적화 3] 압축된 이미지 토픽 구독 ---
        # 네트워크 대역폭 문제를 해결하기 위한 가장 중요한 변경입니다.
        # realsense-ros launch 파일에서 `enable_sync:=true`가 설정되어 있는지 확인하세요.
        # 토픽 이름은 `rostopic list`로 실제 발행되는 이름으로 맞춰야 합니다.
        realsense_img_topic = '/camera/color/image_raw/compressed' 
        depth_topic = "/camera/aligned_depth_to_color/image_raw" # 뎁스는 보통 압축하지 않음
        info_topic = "/camera/color/camera_info"
        
        realsense_img_sub = message_filters.Subscriber(realsense_img_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=10, slop=0.2) # slop 값을 약간 늘려 동기화 안정성 확보
        self.ts.registerCallback(self.realsense_callback)
        rospy.loginfo(f"Synchronized subscribers started for Realsense on topic: {realsense_img_topic}")
        
        usb_cam_topic = '/usb_cam/image_raw/compressed'
        self.usb_cam_sub = rospy.Subscriber(usb_cam_topic, CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscriber for USB Cam started on topic: {usb_cam_topic}")
        rospy.loginfo("✅ Optimized Unified Vision Node v1 initialized successfully.")

    def realsense_callback(self, compressed_img_msg, depth_msg, info_msg):
        try:
            # --- [최적화 3] 압축 해제 ---
            np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
            cv_color_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth_orig = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            
            # --- [최적화 1] 연산을 위해 이미지 리사이즈 ---
            cv_color = cv2.resize(cv_color_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
            # 뎁스 이미지도 동일한 비율로 리사이즈해야 좌표가 맞습니다.
            cv_depth = cv2.resize(cv_depth_orig, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)

            # 카메라 내부 파라미터도 리사이즈된 해상도에 맞게 스케일링해야 합니다.
            if self.camera_intrinsics is None:
                orig_w, orig_h = info_msg.width, info_msg.height
                scale_x = self.proc_width / orig_w
                scale_y = self.proc_height / orig_h
                
                self.camera_intrinsics = {
                    'fx': info_msg.K[0] * scale_x, 
                    'fy': info_msg.K[4] * scale_y, 
                    'ppx': info_msg.K[2] * scale_x, 
                    'ppy': info_msg.K[5] * scale_y
                }
                rospy.loginfo(f"Realsense camera intrinsics scaled to {self.proc_width}x{self.proc_height}: {self.camera_intrinsics}")

            # 리사이즈된 이미지로 모든 처리 수행
            self.run_supply_tracking(cv_color, cv_depth)
            self.run_hsv_path_planning(cv_color, cv_depth, depth_msg.header)

            # --- [최적화 2] 시각화는 원본 해상도 이미지에 그려서 발행 ---
            # 최종 결과는 선명하게 보기 위해 원본 이미지에 다시 그립니다.
            # (만약 이 과정도 느리다면 그냥 리사이즈된 cv_color를 사용해도 됩니다)
            # 이 예제에서는 처리된 작은 이미지(cv_color)를 바로 발행하여 부하를 최소화합니다.
            self.publish_compressed_viz(self.realsense_viz_pub, cv_color)

        except Exception as e:
            rospy.logerr(f"Error in Realsense callback: {e}", exc_info=True)
            
    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # USB 캠도 필요하다면 리사이즈하여 처리 속도를 높일 수 있습니다.
            # cv_image_resized = cv2.resize(cv_image, (self.proc_width, self.proc_height))
            results = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results)
            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            rospy.logerr(f"Error in USB Cam callback: {e}")

    # --- [최적화 2] 압축 발행 헬퍼 함수 ---
    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)

    # 나머지 함수들은 기존과 거의 동일하나, 리사이즈된 intrinsics를 사용합니다.
    # ... (run_supply_tracking, run_hsv_path_planning 등 나머지 함수들은 여기에 그대로 붙여넣기) ...
    # (단, 이 함수들 내부에서 camera_intrinsics를 사용하는지 확인해야 합니다. 현재 구조에서는 이미 self.camera_intrinsics를 사용하므로 OK)
    def run_supply_tracking(self, color_image, depth_image):
        # ... (기존 코드와 동일)
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

                    label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 이하 모든 도우미 함수(Helper Functions)는 기존과 동일하게 유지합니다.
    def run_hsv_path_planning(self, color_image, depth_image, header):
        # ... (이하 모든 함수 기존과 동일)
        mask = self.create_drivable_mask(color_image)
        if not np.any(mask): return
        points_3d = self.unproject_2d_to_3d(mask, depth_image)
        if points_3d.shape[0] < 100: return
        self.generate_and_publish_path(points_3d, header)

    def deproject_pixel_to_point(self, u, v, depth):
        # ...
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        ppx, ppy = self.camera_intrinsics['ppx'], self.camera_intrinsics['ppy']
        x = (u - ppx) * depth / fx
        y = (v - ppy) * depth / fy
        return x, y, depth

    def create_drivable_mask(self, cv_color):
        # ...
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv_bound, self.upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def unproject_2d_to_3d(self, mask_2d, cv_depth):
        # ...
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
        # ...
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

    def transform_to_matrix(self, t):
        # ...
        trans, rot = t.transform.translation, t.transform.rotation
        from tf.transformations import quaternion_matrix
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[:3, 3] = [trans.x, trans.y, trans.z]
        return mat

    def draw_marker_detections(self, image, results):
        # ...
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
        node = UnifiedVisionNodeV1Optimized()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass