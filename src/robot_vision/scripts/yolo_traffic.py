#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: yolo_vision_node.py (Corrected and Modified)

import rospy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import message_filters

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class YoloVisionNode:
    def __init__(self):
        rospy.init_node('yolo_vision_node', anonymous=True)
        rospy.loginfo("--- YOLO Vision Node (Decoupled) ---")
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using compute device: {self.device}")
        self.proc_width = rospy.get_param('~proc_width', 640)
        self.proc_height = rospy.get_param('~proc_height', 480)
        try:
            # 기존 모델 로드
            supply_model_path = rospy.get_param('~supply_model_path', './tracking1.pt')
            self.supply_model = YOLO(supply_model_path).to(self.device)
            marker_model_path = rospy.get_param('~marker_model_path', './vision_enemy.pt')
            self.marker_model = YOLO(marker_model_path).to(self.device)
            path_model_path = rospy.get_param('~yolo_model_path', './weights.pt')
            self.path_model = YOLO(path_model_path).to(self.device)
            self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']

            new_model_path = rospy.get_param('~new_model_path', './traffic_light.pt')
            self.new_detection_model = YOLO(new_model_path).to(self.device)
            self.new_model_class_names = ['red', 'green']

        except Exception as e:
            rospy.logerr(f"Failed to load YOLO models: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return
        self.yolo_confidence = rospy.get_param('~yolo_confidence', 0.7)
        self.drivable_class_index = rospy.get_param('~drivable_class_index', 0)
        self.scaled_camera_intrinsics = None
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=1)
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz/compressed', CompressedImage, queue_size=1)
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz/compressed', CompressedImage, queue_size=1)
        self.mask_pub = rospy.Publisher('/path_planning/yolo/mask', Image, queue_size=1)
        self.depth_for_path_pub = rospy.Publisher('/path_planning/yolo/depth', Image, queue_size=1)
        self.info_for_path_pub = rospy.Publisher('/path_planning/yolo/info', CameraInfo, queue_size=1)
        realsense_img_topic = '/camera/color/image_raw/compressed'
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        info_topic = "/camera/color/camera_info"
        realsense_img_sub = message_filters.Subscriber(realsense_img_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([realsense_img_sub, depth_sub, info_sub], queue_size=5, slop=0.5)
        self.ts.registerCallback(self.realsense_callback)
        usb_cam_topic = '/usb_cam/image_raw/compressed'
        self.usb_cam_sub = rospy.Subscriber(usb_cam_topic, CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        rospy.loginfo("✅ YOLO Vision Node initialized successfully.")

    def realsense_callback(self, compressed_image_msg, depth_msg, info_msg):
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            cv_color = cv2.resize(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), (self.proc_width, self.proc_height))
            cv_depth = cv2.resize(self.bridge.imgmsg_to_cv2(depth_msg, '16UC1'), (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            if self.scaled_camera_intrinsics is None:
                self.scale_camera_info(info_msg)
            self.run_supply_tracking(cv_color, cv_depth)
            drivable_mask = self.create_yolo_drivable_mask(cv_color)
            if drivable_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(drivable_mask, "mono8")
                mask_msg.header = depth_msg.header
                self.mask_pub.publish(mask_msg)
                depth_img_msg = self.bridge.cv2_to_imgmsg(cv_depth, "16UC1")
                depth_img_msg.header = depth_msg.header
                self.depth_for_path_pub.publish(depth_img_msg)
                self.info_for_path_pub.publish(info_msg)
                cv_color[drivable_mask > 0] = cv2.addWeighted(cv_color[drivable_mask > 0], 0.5, np.full_like(cv_color[drivable_mask > 0], (0, 255, 0)), 0.5, 0)
            self.publish_compressed_viz(self.realsense_viz_pub, cv_color)
        except Exception as e:
            rospy.logerr(f"Error in Realsense callback: {e}", exc_info=True)

    def scale_camera_info(self, info_msg):
        scale_x = self.proc_width / info_msg.width
        scale_y = self.proc_height / info_msg.height
        self.scaled_camera_intrinsics = {
            'fx': info_msg.K[0] * scale_x, 'fy': info_msg.K[4] * scale_y,
            'ppx': info_msg.K[2] * scale_x, 'ppy': info_msg.K[5] * scale_y
        }
        rospy.loginfo(f"Cam intrinsics scaled for vision node: {self.scaled_camera_intrinsics}")

    def create_yolo_drivable_mask(self, color_image):
        results = self.path_model(color_image, conf=self.yolo_confidence, verbose=False)
        result = results[0]
        if result.masks is None: return None
        final_mask = np.zeros((self.proc_height, self.proc_width), dtype=np.uint8)
        drivable_indices = np.where(result.boxes.cls.cpu().numpy() == self.drivable_class_index)[0]
        if len(drivable_indices) == 0: return None
        for idx in drivable_indices:
            mask_data = result.masks.data.cpu().numpy()[idx]
            resized_mask = cv2.resize(mask_data, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            final_mask = np.maximum(final_mask, (resized_mask * 255).astype(np.uint8))
        return final_mask

    def run_supply_tracking(self, color_image, depth_image):
        if self.scaled_camera_intrinsics is None: return
        results = self.supply_model(color_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                    depth_in_meters = depth_image[cy, cx] / 1000.0
                    if depth_in_meters > 0:
                        fx, fy, ppx, ppy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy'], self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
                        x = (cx - ppx) * depth_in_meters / fx
                        y = (cy - ppy) * depth_in_meters / fy
                        point_msg = Point(x=depth_in_meters, y=-x, z=-y)
                        self.distance_pub.publish(point_msg)
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 1. 기존 marker_model(vision_enemy.pt) 추론 및 결과 그리기
            results_marker = self.marker_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_marker_detections(cv_image, results_marker)
            
            # 2. 새로운 new_detection_model 추론 및 결과 그리기
            results_new = self.new_detection_model(cv_image, conf=0.5, iou=0.45, verbose=False)
            annotated_image = self.draw_new_detections(annotated_image, results_new)

            # 3. 최종 결과 이미지 발행
            self.publish_compressed_viz(self.usb_cam_viz_pub, annotated_image)
        except Exception as e:
            rospy.logerr(f"Error in USB Cam callback: {e}")

    def publish_compressed_viz(self, publisher, cv_image):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)

    def draw_marker_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                # 초록색(0, 255, 0)으로 바운딩 박스 표시
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    # <<< [추가] 새로운 모델의 탐지 결과를 시각화하는 함수 >>>
    def draw_new_detections(self, image, results):
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.new_model_class_names[cls_id] if cls_id < len(self.new_model_class_names) else "Unknown"
                # 파란색(255, 0, 0)으로 바운딩 박스를 그려 기존 탐지와 구분합니다.
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

if __name__ == '__main__':
    try:
        node = YoloVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass