#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
# [차선 주행 전용 노드 - 정적 전방주시]
# 설명: Realsense로 차선/보급품을, USB 캠으로 마커를 인식하는 최종 버전입니다.
#       조향각은 Float64, 모든 시각화는 RQT로 확인합니다.
# 생성자: Hinton (사용자의 요청에 따라 재구성)
# 버전: 8.0 (2025-07-25)
# ===================================================================================

import rospy
import argparse
import cv2
import torch
import numpy as np
import message_filters
from math import atan2, sqrt, degrees

from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float64, Bool
from geometry_msgs.msg import Point

# ────────── 유틸리티 함수 ────────── #

def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None

def final_filter(bev_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    f1 = cv2.morphologyEx(bev_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(f1, connectivity=8)
    if num_labels <= 1: return np.zeros_like(bev_mask)
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 300]
    comps.sort(key=lambda x: x[1], reverse=True)
    cleaned = np.zeros_like(bev_mask)
    for i in range(min(len(comps), 2)):
        cleaned[labels == comps[i][0]] = 255
    return cleaned

def overlay_polyline(image, coeff, color, thickness=2):
    if coeff is None: return
    h, w = image.shape[:2]
    draw_points = []
    for y in range(h - 1, h // 2, -2):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            draw_points.append((int(x), int(y)))
    if len(draw_points) > 1:
        cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)

# ────────── 메인 로직 클래스 ────────── #

class UnifiedNode:
    def __init__(self, opt):
        self.opt = opt
        self.bridge = CvBridge()
        self.device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() and torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"[Unified Node] Using device: {self.device}")

        # --- 모델 로딩 ---
        self.lane_model = YOLO(opt.lane_weights).to(self.device)
        self.supply_model = YOLO(opt.supply_weights).to(self.device)
        self.marker_model = YOLO(opt.marker_weights).to(self.device)
        rospy.loginfo("[Unified Node] All 3 models loaded.")

        # --- 파라미터 및 변수 초기화 ---
        self.marker_class_names = ['A', 'E', 'Enemy', 'Heart', 'K', 'M', 'O', 'R', 'ROKA', 'Y']
        self.bev_params = np.load(opt.param_file)
        self.bev_h, self.bev_w = int(self.bev_params['warp_h']), int(self.bev_params['warp_w'])
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA, self.MAX_LANE_AGE = 0.6, 7
        
        # 수정된 파라미터
        self.L = 0.58  # 휠베이스 (m)
        self.LOOKAHEAD_DISTANCE = 1.0 # 정적 전방주시거리 (m)

        self.proc_width, self.proc_height = 640, 480
        self.scaled_camera_intrinsics = None

        # --- ROS Publisher ---
        self.pub_steering = rospy.Publisher('steering_angle', Float64, queue_size=1)
        self.pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=1)
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz/compressed', CompressedImage, queue_size=1)
        self.bev_viz_pub = rospy.Publisher('/unified_vision/bev/viz/compressed', CompressedImage, queue_size=1)
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz/compressed', CompressedImage, queue_size=1)

        # --- ROS Subscriber ---
        # 1. Realsense 구독 (차선 주행, 보급품 추적용)
        realsense_color_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        realsense_depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        realsense_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.realsense_ts = message_filters.ApproximateTimeSynchronizer([realsense_color_sub, realsense_depth_sub, realsense_info_sub], queue_size=10, slop=0.5)
        self.realsense_ts.registerCallback(self.realsense_callback)
        
        # 2. USB Cam 구독 (마커 인식용)
        self.usb_cam_sub = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24)

        # 3. 스로틀 구독 제거됨

        rospy.loginfo("✅ Unified Node initialized. Realsense for Driving/Supply, USB Cam for Markers.")

    def realsense_callback(self, compressed_color_msg, depth_msg, info_msg):
        """Realsense 카메라로 차선 주행과 보급품 추적을 처리합니다."""
        try:
            # 1. 이미지 준비
            np_arr = np.frombuffer(compressed_color_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

            # 2. 차선 주행 로직 실행
            self.process_lanes(cv_color)

            # 3. 보급품 추적 로직 실행
            resized_color = cv2.resize(cv_color, (self.proc_width, self.proc_height))
            resized_depth = cv2.resize(cv_depth, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            if self.scaled_camera_intrinsics is None: self.scale_camera_info(info_msg)
            self.run_supply_tracking(resized_color, resized_depth)

            # 4. Realsense 시각화(보급품) 발행
            viz_msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
            viz_msg.data = np.array(cv2.imencode('.jpg', resized_color)[1]).tobytes()
            self.realsense_viz_pub.publish(viz_msg)

        except Exception as e:
            rospy.logerr(f"Realsense callback error: {e}", exc_info=True)

    def usb_cam_callback(self, compressed_msg):
        """USB 카메라로 마커 인식을 처리합니다."""
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 마커 인식 실행 및 이미지에 그리기
            self.run_marker_detection(cv_image)

            # USB 캠 시각화 발행
            viz_msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
            viz_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.usb_cam_viz_pub.publish(viz_msg)

        except Exception as e:
            rospy.logerr(f"USB Cam callback error: {e}", exc_info=True)
    
    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def process_lanes(self, cv_image):
        bev_image = self.do_bev_transform(cv_image)
        results = self.lane_model(bev_image, imgsz=self.opt.img_size, conf=self.opt.conf_thres, iou=self.opt.iou_thres, device=self.device, verbose=False)
        combined_mask = np.zeros(results[0].orig_shape, dtype=np.uint8)
        if results[0].masks:
            for conf, mask_tensor in zip(results[0].boxes.conf, results[0].masks.data):
                if conf >= 0.5:
                    mask_np = cv2.resize((mask_tensor.cpu().numpy() * 255).astype(np.uint8), (self.bev_w, self.bev_h))
                    combined_mask = np.maximum(combined_mask, mask_np)
        final_mask = final_filter(combined_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask)
        detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, 2)
                    if coeff is not None:
                        detections.append({'coeff': coeff, 'x_bottom': np.polyval(coeff, self.bev_h - 1)})
            detections.sort(key=lambda c: c['x_bottom'])
        left, right = self.tracked_lanes['left'], self.tracked_lanes['right']
        current_left, current_right = None, None
        if len(detections) == 2: current_left, current_right = detections[0], detections[1]
        elif len(detections) == 1:
            d = detections[0]
            if left['coeff'] is not None and right['coeff'] is not None:
                if abs(d['x_bottom'] - np.polyval(left['coeff'], self.bev_h - 1)) < abs(d['x_bottom'] - np.polyval(right['coeff'], self.bev_h - 1)): current_left = d
                else: current_right = d
            elif d['x_bottom'] < self.bev_w / 2: current_left = d
            else: current_right = d
        for side, current in [('left', current_left), ('right', current_right)]:
            if current:
                if self.tracked_lanes[side]['coeff'] is None: self.tracked_lanes[side]['coeff'] = current['coeff']
                else: self.tracked_lanes[side]['coeff'] = self.SMOOTHING_ALPHA * current['coeff'] + (1 - self.SMOOTHING_ALPHA) * self.tracked_lanes[side]['coeff']
                self.tracked_lanes[side]['age'] = 0
            else: self.tracked_lanes[side]['age'] += 1
            if self.tracked_lanes[side]['age'] > self.MAX_LANE_AGE: self.tracked_lanes[side]['coeff'] = None
        final_left_coeff, final_right_coeff = self.tracked_lanes['left']['coeff'], self.tracked_lanes['right']['coeff']
        self.pub_lane_status.publish(Bool(data=(final_left_coeff is not None or final_right_coeff is not None)))
        steering_angle_deg, goal_point_vehicle = self.calculate_steering(final_left_coeff, final_right_coeff)
        if steering_angle_deg is not None:
            self.pub_steering.publish(Float64(data=steering_angle_deg))
        self.publish_bev_viz(bev_image, final_left_coeff, final_right_coeff, goal_point_vehicle, steering_angle_deg)
        
    def calculate_steering(self, final_left_coeff, final_right_coeff):
        if final_left_coeff is None and final_right_coeff is None:
            self.tracked_center_path['coeff'] = None
            return None, None
        center_points = []
        lane_width_pixels = 1.5 / self.m_per_pixel_x
        for y in range(self.bev_h - 1, self.bev_h // 2, -1):
            x_center = None
            if final_left_coeff is not None and final_right_coeff is not None:
                x_center = (np.polyval(final_left_coeff, y) + np.polyval(final_right_coeff, y)) / 2
            elif final_left_coeff is not None: x_center = np.polyval(final_left_coeff, y) + lane_width_pixels / 2
            else: x_center = np.polyval(final_right_coeff, y) - lane_width_pixels / 2
            center_points.append([x_center, y])
        target_center_coeff = polyfit_lane(np.array(center_points)[:, 1], np.array(center_points)[:, 0], 2)
        if target_center_coeff is not None:
            if self.tracked_center_path['coeff'] is None: self.tracked_center_path['coeff'] = target_center_coeff
            else: self.tracked_center_path['coeff'] = self.SMOOTHING_ALPHA * target_center_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path['coeff']
        final_center_coeff = self.tracked_center_path['coeff']
        if final_center_coeff is None: return None, None
        
        # 정적 전방주시거리 사용
        goal_point_vehicle = None
        for y_bev in range(self.bev_h - 1, -1, -1):
            x_bev = np.polyval(final_center_coeff, y_bev)
            x_veh = (self.bev_h - y_bev) * self.m_per_pixel_y + self.y_offset_m
            y_veh = (self.bev_w / 2 - x_bev) * self.m_per_pixel_x
            if sqrt(x_veh**2 + y_veh**2) >= self.LOOKAHEAD_DISTANCE:
                goal_point_vehicle = (x_veh, y_veh)
                break
        if goal_point_vehicle is None: return None, None

        x_g, y_g = goal_point_vehicle
        steering_angle_rad = atan2(2.0 * self.L * y_g, x_g**2 + y_g**2)
        steering_angle_deg = -degrees(steering_angle_rad)
        return np.clip(steering_angle_deg, -25.0, 25.0), goal_point_vehicle

    def publish_bev_viz(self, bev_image, cL, cR, goal, steer):
        if self.bev_viz_pub.get_num_connections() == 0: return
        viz = bev_image.copy()
        overlay_polyline(viz, cL, (255, 0, 0))
        overlay_polyline(viz, cR, (0, 0, 255))
        overlay_polyline(viz, self.tracked_center_path['coeff'], (0, 255, 0), 3)
        if goal is not None:
            x_g, y_g = goal
            u_g = self.bev_w / 2 - y_g / self.m_per_pixel_x
            v_g = self.bev_h - (x_g - self.y_offset_m) / self.m_per_pixel_y
            if 0 <= u_g < self.bev_w and 0 <= v_g < self.bev_h:
                cv2.circle(viz, (int(u_g), int(v_g)), 10, (0, 255, 255), -1)
        cv2.putText(viz, f"Steer: {steer:.1f}" if steer is not None else "Steer: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
        msg.data = np.array(cv2.imencode('.jpg', viz)[1]).tobytes()
        self.bev_viz_pub.publish(msg)

    def scale_camera_info(self, info_msg):
        scale_x = self.proc_width / info_msg.width
        scale_y = self.proc_height / info_msg.height
        self.scaled_camera_intrinsics = {'fx': info_msg.K[0] * scale_x, 'fy': info_msg.K[4] * scale_y, 'ppx': info_msg.K[2] * scale_x, 'ppy': info_msg.K[5] * scale_y}
        rospy.loginfo_once(f"[Unified Node] Realsense intrinsics scaled.")
        
    def run_supply_tracking(self, color_image, depth_image):
        if self.scaled_camera_intrinsics is None: return
        results = self.supply_model(color_image, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) == 0 and box.conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cy < self.proc_height and 0 <= cx < self.proc_width:
                    depth_m = depth_image[cy, cx] / 1000.0
                    if 0.1 < depth_m < 10.0:
                        intr = self.scaled_camera_intrinsics
                        cam_x = (cx - intr['ppx']) * depth_m / intr['fx']
                        cam_y = (cy - intr['ppy']) * depth_m / intr['fy']
                        point_msg = Point(x=depth_m, y=-cam_x, z=-cam_y) # point_msg 계산
                        self.distance_pub.publish(point_msg)
                        
                        # 텍스트 레이블을 x, y, z 좌표가 모두 포함된 형식으로 변경
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        # 수정한 레이블을 이미지에 표시 (긴 텍스트를 위해 폰트 크기 조절)
                        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def run_marker_detection(self, color_image):
        results = self.marker_model(color_image, conf=0.5, iou=0.45, verbose=False)
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls_id = box.conf[0], int(box.cls[0])
                label = self.marker_class_names[cls_id] if cls_id < len(self.marker_class_names) else "Unknown"
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# ────────── 메인 함수 ────────── #
def main():
    rospy.init_node('unified_node', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane_weights', type=str, default='./YOLOTL.pt', help="Lane detection model")
    parser.add_argument('--supply_weights', type=str, default='./tracking1.pt', help="Supply box detection model")
    parser.add_argument('--marker_weights', type=str, default='./vision_enemy.pt', help="Marker detection model")
    parser.add_argument('--param-file', type=str, default='./bev_params_y_5.npz', help="BEV parameters file")
    parser.add_argument('--device', default='0', help='CUDA device, e.g. 0 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    opt, _ = parser.parse_known_args()
    try:
        node = UnifiedNode(opt)
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("Shutting down unified node.")

if __name__ == '__main__':
    main()