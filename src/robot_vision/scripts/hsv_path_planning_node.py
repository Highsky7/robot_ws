#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FILE: hsv_path_planning_node.py (Corrected)

import rospy
import cv2
import numpy as np
import tf2_ros
from tf.transformations import quaternion_matrix
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class HsvPathPlanningNode:
    def __init__(self):
        rospy.init_node('hsv_path_planning_node', anonymous=True)
        rospy.loginfo("--- HSV Path Planning Node (Decoupled) ---")
        self.bridge = CvBridge()
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.path_lookahead = rospy.get_param('~path_lookahead', 3.0)
        self.num_path_points = rospy.get_param('~num_path_points', 20)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.5)
        self.scaled_camera_intrinsics = None
        self.smoothed_path_points_3d = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.path_pub = rospy.Publisher('/competition_path_hsv', Path, queue_size=1)
        mask_topic = '/path_planning/hsv/mask'
        depth_topic = '/path_planning/hsv/depth'
        info_topic = '/path_planning/hsv/info'
        mask_sub = message_filters.Subscriber(mask_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([mask_sub, depth_sub, info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.planning_callback)
        rospy.loginfo(f"Subscribing to intermediate topics: {mask_topic}, {depth_topic}, {info_topic}")
        rospy.loginfo("✅ HSV Path Planning Node initialized successfully.")

    def planning_callback(self, mask_msg, depth_msg, info_msg):
        try:
            cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            if self.scaled_camera_intrinsics is None:
                # [수정] 수신된 메시지로 스케일링
                self.scale_camera_info(depth_msg, info_msg)
            contours, _ = cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return
            contour_points_2d = np.concatenate(contours, axis=0).squeeze(axis=1)
            points_3d = self.unproject_contours_to_3d(contour_points_2d, cv_depth)
            if points_3d.shape[0] < 50: return
            self.generate_and_publish_path(points_3d, mask_msg.header)
        except Exception as e:
            rospy.logerr(f"Error in planning callback: {e}", exc_info=True)

    def scale_camera_info(self, depth_msg, info_msg):
        # [수정] 파라미터 서버 대신, 수신한 메시지의 크기로부터 스케일링 비율을 직접 계산
        proc_width = depth_msg.width
        proc_height = depth_msg.height
        orig_width = info_msg.width
        orig_height = info_msg.height

        if orig_width == 0 or orig_height == 0:
            rospy.logwarn("Original camera info width/height is zero, cannot scale intrinsics yet.")
            return

        scale_x = proc_width / orig_width
        scale_y = proc_height / orig_height
        self.scaled_camera_intrinsics = {
            'fx': info_msg.K[0] * scale_x, 'fy': info_msg.K[4] * scale_y,
            'ppx': info_msg.K[2] * scale_x, 'ppy': info_msg.K[5] * scale_y
        }
        rospy.loginfo_once(f"Path planner intrinsics scaled: {self.scaled_camera_intrinsics}")

    def unproject_contours_to_3d(self, contour_points, cv_depth):
        if self.scaled_camera_intrinsics is None: return np.array([])
        u, v = contour_points[:, 0], contour_points[:, 1]
        depths = cv_depth[v, u]
        valid = depths > 0
        u, v, depths = u[valid], v[valid], depths[valid]
        if len(u) == 0: return np.array([])
        fx, fy, cx, cy = self.scaled_camera_intrinsics['fx'], self.scaled_camera_intrinsics['fy'], self.scaled_camera_intrinsics['ppx'], self.scaled_camera_intrinsics['ppy']
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
                self.smoothed_path_points_3d = np.array(raw_path)
            else:
                self.smoothed_path_points_3d = self.smoothing_factor * np.array(raw_path) + (1 - self.smoothing_factor) * self.smoothed_path_points_3d
            path_msg = Path(header=rospy.Header(stamp=header.stamp, frame_id=self.robot_base_frame))
            for p in self.smoothed_path_points_3d:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p[0], p[1], p[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")

    def transform_to_matrix(self, t):
        trans, rot = t.transform.translation, t.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[:3, 3] = [trans.x, trans.y, trans.z]
        return mat

if __name__ == '__main__':
    try:
        node = HsvPathPlanningNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass