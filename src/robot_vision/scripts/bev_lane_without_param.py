#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================================
# [차선 주행 전용 노드 - BEV 비활성화 디버깅]
# 설명: BEV 변환 없이 원본 이미지에서 차선 인식을 시도하고,
#       마스크와 조향각을 포함한 모든 시각화 결과를 RQT로 확인합니다.
# 생성자: Hinton (사용자의 요청에 따라 재구성)
# 버전: 9.0 (2025-07-25)
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
        # bev_h, bev_w는 do_bev_transform이 주석처리되어 직접 사용되지 않지만,
        # calculate_steering의 좌표 계산에 사용될 수 있어 유지합니다.
        self.bev_h, self.bev_w = int(self.bev_params['warp_h']), int(self.bev_params['warp_w'])
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA, self.MAX_LANE_AGE = 0.6, 7
        
        self.L = 0.58
        self.LOOKAHEAD_DISTANCE = 1.0

        self.proc_width, self.proc_height = 640, 480
        self.scaled_camera_intrinsics = None

        # --- ROS Publisher ---
        self.pub_steering = rospy.Publisher('steering_angle', Float64, queue_size=1)
        self.pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/supply_distance', Point, queue_size=1)
        self.realsense_viz_pub = rospy.Publisher('/unified_vision/realsense/viz/compressed', CompressedImage, queue_size=1)
        self.lane_driving_viz_pub = rospy.Publisher('/unified_vision/lane_driving/viz/compressed', CompressedImage, queue_size=1) # 이름 변경
        self.lane_mask_pub = rospy.Publisher('/unified_vision/lane_mask/compressed', CompressedImage, queue_size=1) # 마스크 시각화용
        self.usb_cam_viz_pub = rospy.Publisher('/unified_vision/usb_cam/viz/compressed', CompressedImage, queue_size=1)

        # --- ROS Subscriber ---
        realsense_color_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        realsense_depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        realsense_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.realsense_ts = message_filters.ApproximateTimeSynchronizer([realsense_color_sub, realsense_depth_sub, realsense_info_sub], queue_size=10, slop=0.5)
        self.realsense_ts.registerCallback(self.realsense_callback)
        self.usb_cam_sub = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.usb_cam_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("✅ Unified Node initialized. BEV transform is DISABLED.")

    def realsense_callback(self, compressed_color_msg, depth_msg, info_msg):
        try:
            np_arr = np.frombuffer(compressed_color_msg.data, np.uint8)
            cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

            self.process_lanes(cv_color)

            resized_color = cv2.resize(cv_color, (self.proc_width, self.proc_height))
            resized_depth = cv2.resize(cv_depth, (self.proc_width, self.proc_height), interpolation=cv2.INTER_NEAREST)
            if self.scaled_camera_intrinsics is None: self.scale_camera_info(info_msg)
            self.run_supply_tracking(resized_color, resized_depth)
            
            viz_msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
            viz_msg.data = np.array(cv2.imencode('.jpg', resized_color)[1]).tobytes()
            self.realsense_viz_pub.publish(viz_msg)

        except Exception as e:
            rospy.logerr(f"Realsense callback error: {e}", exc_info=True)

    def usb_cam_callback(self, compressed_msg):
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.run_marker_detection(cv_image)
            viz_msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
            viz_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.usb_cam_viz_pub.publish(viz_msg)
        except Exception as e:
            rospy.logerr(f"USB Cam callback error: {e}", exc_info=True)
    
    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def process_lanes(self, cv_image):
        # 1. BEV 변환을 주석 처리
        # bev_image = self.do_bev_transform(cv_image)
        
        # 2. 원본 이미지(cv_image)를 모델의 입력으로 사용
        #    (주의: 모델이 BEV용으로 학습되었다면 성능이 저하될 수 있음)
        #    모델의 입력 사이즈에 맞게 리사이즈가 필요할 수 있으나, 우선 원본으로 시도
        h, w, _ = cv_image.shape
        results = self.lane_model(cv_image, imgsz=self.opt.img_size, conf=self.opt.conf_thres, iou=self.opt.iou_thres, device=self.device, verbose=False)
        
        # 3. 결과 마스크 생성
        combined_mask = np.zeros((h, w), dtype=np.uint8) # 원본 이미지 크기
        if results[0].masks:
            for conf, mask_tensor in zip(results[0].boxes.conf, results[0].masks.data):
                if conf >= 0.5:
                    # 마스크를 원본 이미지 크기로 리사이즈
                    mask_np = cv2.resize((mask_tensor.cpu().numpy() * 255).astype(np.uint8), (w, h))
                    combined_mask = np.maximum(combined_mask, mask_np)
        
        # 필터링 로직은 그대로 사용
        final_mask = final_filter(combined_mask)

        # 4. 차선 마스크 시각화 발행 (RQT 디버깅용)
        if self.lane_mask_pub.get_num_connections() > 0:
            mask_viz = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            mask_msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
            mask_msg.data = np.array(cv2.imencode('.jpg', mask_viz)[1]).tobytes()
            self.lane_mask_pub.publish(mask_msg)

        # 5. 이후 로직은 생성된 마스크를 기반으로 동일하게 수행
        #    (단, 좌표계가 다르므로 결과의 정확도는 보장되지 않음)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask)
        detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, 2)
                    if coeff is not None:
                        detections.append({'coeff': coeff, 'x_bottom': np.polyval(coeff, h - 1)})
            detections.sort(key=lambda c: c['x_bottom'])
            
        # ... (이하 차선 추적 및 스무딩 로직은 동일) ...
        # (생략된 코드는 이전 답변과 동일합니다)

        final_left_coeff, final_right_coeff = self.tracked_lanes['left']['coeff'], self.tracked_lanes['right']['coeff']
        steering_angle_deg, goal_point_img = self.calculate_steering_no_bev(final_left_coeff, final_right_coeff, h, w)
        
        if steering_angle_deg is not None:
            self.pub_steering.publish(Float64(data=steering_angle_deg))
            
        self.publish_lane_driving_viz(cv_image, final_left_coeff, final_right_coeff, goal_point_img, steering_angle_deg)
        
    def calculate_steering_no_bev(self, final_left_coeff, final_right_coeff, h, w):
        # BEV가 아닐 때의 조향각 계산 (단순화된 모델)
        # 이 부분은 BEV 좌표계가 아니므로 정확한 물리량을 계산하기 어려움
        # 따라서, 이미지 중심을 기준으로 경로의 이탈 정도를 조향각으로 변환하는 방식을 사용
        if final_left_coeff is None and final_right_coeff is None: return None, None
        
        # 이미지 하단에서 중앙 경로의 x좌표 계산
        y_eval = h - 1 # 이미지 맨 아래
        x_center = None
        if final_left_coeff is not None and final_right_coeff is not None:
            x_center = (np.polyval(final_left_coeff, y_eval) + np.polyval(final_right_coeff, y_eval)) / 2
        elif final_left_coeff is not None: x_center = np.polyval(final_left_coeff, y_eval) + w/4 # 임의의 오프셋
        elif final_right_coeff is not None: x_center = np.polyval(final_right_coeff, y_eval) - w/4 # 임의의 오프셋
        
        if x_center is None: return None, None
        
        # 이미지 중심(w/2)과의 오차 계산
        error = x_center - w / 2
        
        # 오차에 비례하는 조향각 생성 (P제어와 유사)
        steering_angle_deg = -error * 0.1 # 게인 값(0.1)은 튜닝 필요
        
        return np.clip(steering_angle_deg, -25.0, 25.0), (int(x_center), int(y_eval))

    def publish_lane_driving_viz(self, cv_image, cL, cR, goal, steer):
        if self.lane_driving_viz_pub.get_num_connections() == 0: return
        viz = cv_image.copy()
        overlay_polyline(viz, cL, (255, 0, 0)) # Left - Blue
        overlay_polyline(viz, cR, (0, 0, 255)) # Right - Red
        
        # 중앙 경로 시각화는 생략 (또는 단순화된 경로로 대체)
        if goal is not None:
            cv2.circle(viz, goal, 10, (0, 255, 255), -1)

        cv2.putText(viz, f"Steer: {steer:.1f}" if steer is not None else "Steer: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        msg = CompressedImage(header=rospy.Header(stamp=rospy.Time.now()), format="jpeg")
        msg.data = np.array(cv2.imencode('.jpg', viz)[1]).tobytes()
        self.lane_driving_viz_pub.publish(msg)

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
                        point_msg = Point(x=depth_m, y=-cam_x, z=-cam_y)
                        self.distance_pub.publish(point_msg)
                        label = f"Supply Box: x={point_msg.x:.2f}m, y={point_msg.y:.2f}m, z={point_msg.z:.2f}m"
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
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