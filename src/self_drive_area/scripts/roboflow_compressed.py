#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_matrix
import threading

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import message_filters
from cv_bridge import CvBridge

class AdvancedYoloPlanner:
    def __init__(self):
        rospy.loginfo("Initializing Advanced YOLO Path Planner...")
        self.bridge = CvBridge()
        self.latest_color_msg, self.latest_depth_msg, self.latest_info_msg = None, None, None
        self.data_lock = threading.Lock()
        
        self.cv_window_name = "Drivable Area Visualization (Advanced YOLO)"

        # 파라미터
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'camera_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 15)
        self.point_downsample_rate = rospy.get_param('~downsample', 5)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.4)
        self.yolo_img_size = rospy.get_param('~imgsz', 640)
        self.smoothed_path_points = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        model_path = './weights.pt' # ❗ 경로 수정 필수
        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available(): self.model.to('cuda')
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}"); rospy.signal_shutdown("Model loading failed")

        color_topic, depth_topic, info_topic = "/camera/color/image_raw/compressed", "/camera/aligned_depth_to_color/image_raw", "/camera/color/camera_info"
        color_sub, depth_sub, info_sub = message_filters.Subscriber(color_topic, CompressedImage), message_filters.Subscriber(depth_topic, Image), message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], 2, 0.5)
        self.ts.registerCallback(self.synchronized_callback)

        self.path_pub = rospy.Publisher("/path", Path, queue_size=1)
        self.viz_pub = rospy.Publisher("/drivable_area/viz_2d", Image, queue_size=1)
        
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        rospy.loginfo("Node initialization complete. Advanced YOLO Planner is running.")

    def synchronized_callback(self, compressed_color_msg, depth_msg, info_msg):
        with self.data_lock:
            self.latest_color_msg, self.latest_depth_msg, self.latest_info_msg = compressed_color_msg, depth_msg, info_msg

    def processing_loop(self):
        rate = rospy.Rate(30)
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_AUTOSIZE)
        while not rospy.is_shutdown():
            local_color_msg, local_depth_msg, local_info_msg = None, None, None
            with self.data_lock:
                if self.latest_color_msg:
                    local_color_msg, local_depth_msg, local_info_msg = self.latest_color_msg, self.latest_depth_msg, self.latest_info_msg
                    self.latest_color_msg = None
            if not local_color_msg:
                rate.sleep()
                continue
            
            try:
                np_arr = np.frombuffer(local_color_msg.data, np.uint8)
                cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv_depth = self.bridge.imgmsg_to_cv2(local_depth_msg, "16UC1")
            except Exception as e:
                rospy.logerr(f"Data decoding error: {e}")
                continue
            
            results = self.model.predict(cv_color, verbose=False, conf=0.5, imgsz=self.yolo_img_size)
            mask_2d = self.process_inference_results(results, cv_color.shape)

            if mask_2d is not None and np.any(mask_2d):
                points_in_camera = self.get_3d_points(mask_2d, cv_depth, local_info_msg)
                if points_in_camera is not None and points_in_camera.shape[0] > 100:
                    self.generate_and_publish_path(points_in_camera, local_info_msg)

            path_points_2d = self.generate_path_from_mask_2d(mask_2d if mask_2d is not None else np.array([]))
            viz_image = self.create_visualization_image(cv_color, mask_2d, path_points_2d)
            try:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
            except CvBridgeError: pass
            cv2.imshow(self.cv_window_name, viz_image)
            cv2.waitKey(1)
            rate.sleep()
    
    # [핵심 변경] '투영 중심선' 알고리즘을 적용한 새로운 경로 생성 함수
    def generate_and_publish_path(self, points_3d_cam, info_msg):
        try:
            # 1. 3D 포인트들을 로봇 기준 좌표계('camera_link')로 변환
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, info_msg.header.frame_id, info_msg.header.stamp, rospy.Duration(0.1))
            trans_matrix = self.transform_to_matrix(transform)
            points_homogeneous = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))
            points_3d_robot = (trans_matrix @ points_homogeneous.T).T[:, :3]

            # 2. 2D 평면 투영 (Top-down View) 및 중심선 계산
            valid_x_indices = (points_3d_robot[:, 0] > 0.1) & (points_3d_robot[:, 0] < self.path_lookahead)
            if np.sum(valid_x_indices) < 20: return

            x_coords = points_3d_robot[valid_x_indices, 0]
            y_coords = points_3d_robot[valid_x_indices, 1]
            
            path_poly_coeffs = np.polyfit(x_coords, y_coords, 2)
            path_poly = np.poly1d(path_poly_coeffs)

            # 3. 높이(Z)값 복원
            path_x = np.linspace(x_coords.min(), x_coords.max(), self.num_path_points)
            path_y = path_poly(path_x)

            raw_path_points = []
            for px, py in zip(path_x, path_y):
                search_radius = 0.15
                distances = np.linalg.norm(points_3d_robot[:, :2] - np.array([px, py]), axis=1)
                nearby_points = points_3d_robot[distances < search_radius]
                
                if nearby_points.shape[0] > 5:
                    z = np.median(nearby_points[:, 2])
                    raw_path_points.append(np.array([px, py, z]))

            if len(raw_path_points) < self.num_path_points / 2: return

            # 4. 시간적 안정화 필터(EMA) 적용
            alpha = self.smoothing_factor
            if self.smoothed_path_points is None or len(self.smoothed_path_points) != len(raw_path_points):
                self.smoothed_path_points = raw_path_points
            else:
                for i in range(len(raw_path_points)):
                    self.smoothed_path_points[i] = alpha * raw_path_points[i] + (1 - alpha) * self.smoothed_path_points[i]
            
            # 5. 최종 경로 메시지 생성 및 발행
            path_msg = Path(header=rospy.Header(stamp=info_msg.header.stamp, frame_id=self.robot_base_frame))
            for p in self.smoothed_path_points:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)

        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Advanced path generation failed: {e}")

    # --- 나머지 함수들은 이전과 동일 ---
    def process_inference_results(self, results, original_shape):
        if not results or results[0].masks is None or len(results[0].masks.data) == 0: return None
        final_mask = np.zeros(original_shape[:2], dtype=np.uint8)
        drivable_class_index = 0
        drivable_indices = np.where(results[0].boxes.cls.cpu().numpy() == drivable_class_index)[0]
        if len(drivable_indices) == 0: return None
        drivable_polygons = results[0].masks.xy
        for idx in drivable_indices:
            cv2.fillPoly(final_mask, [drivable_polygons[idx].astype(np.int32)], 255)
        return final_mask

    def get_3d_points(self, mask_2d, cv_depth, info_msg):
        fx, fy, cx, cy = info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5]
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

    def create_visualization_image(self, cv_color, mask_2d, path_points_2d):
        viz_image = cv_color.copy()
        if mask_2d is not None and mask_2d.shape == viz_image.shape[:2]:
            overlay_color = np.array([0, 255, 0], dtype=np.uint8)
            viz_image[mask_2d > 0] = (0.5 * viz_image[mask_2d > 0].astype(np.float32) + 0.5 * overlay_color.astype(np.float32)).astype(np.uint8)
        if path_points_2d:
            for x, y in path_points_2d:
                cv2.circle(viz_image, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        return viz_image

    def generate_path_from_mask_2d(self, mask):
        path_points = []
        if mask.size == 0: return path_points
        h, w = mask.shape
        for y in range(h - 1, int(h / 2), -20):
            points_x = np.where(mask[y, :] > 0)[0]
            if points_x.size > 0: path_points.append((int(np.mean(points_x)), y))
        return path_points

    def transform_to_matrix(self, transform):
        t, r = transform.transform.translation, transform.transform.rotation
        mat = quaternion_matrix([r.x, r.y, r.z, r.w])
        mat[0, 3], mat[1, 3], mat[2, 3] = t.x, t.y, t.z
        return mat

def main():
    rospy.init_node('advanced_yolo_planner', anonymous=True)
    try:
        planner = AdvancedYoloPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("Shutting down. OpenCV windows closed.")

if __name__ == '__main__':
    main()