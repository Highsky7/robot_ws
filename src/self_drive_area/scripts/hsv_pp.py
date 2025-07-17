#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_matrix
import threading
import math # <<< 추가

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist # <<< Twist 추가
import message_filters
from cv_bridge import CvBridge

class AdvancedHsvPlanner:
    def __init__(self):
        rospy.loginfo("Initializing Advanced HSV Path Planner with Path Follower...")
        self.bridge = CvBridge()
        self.latest_color_msg, self.latest_depth_msg, self.latest_info_msg = None, None, None
        self.data_lock = threading.Lock()
        
        self.cv_window_name = "Drivable Area Visualization (Advanced HSV)"

        # 파라미터
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'camera_link') # 터틀봇의 경우 'base_footprint'가 더 적합할 수 있습니다.
        self.path_lookahead = rospy.get_param('~path_lookahead', 1.5) # 경로 생성 시 내다볼 거리
        self.num_path_points = rospy.get_param('~num_path_points', 15)
        self.point_downsample_rate = rospy.get_param('~downsample', 5)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.4)
        
        # <<< 경로 추종 제어 파라미터 추가
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.7) # 로봇이 따라갈 전방 목표 지점 거리 (m)
        self.linear_velocity = rospy.get_param('~linear_velocity', 0.15) # 로봇의 최대 전진 속도 (m/s)
        self.angular_gain = rospy.get_param('~angular_gain', 1.2) # 회전 제어 게인 (클수록 급격히 회전)
        
        self.LOWER_MINT_HSV = np.array([100, 9, 55])
        self.UPPER_MINT_HSV = np.array([140, 30, 220])
        self.kernel = np.ones((5, 5), np.uint8)
        self.smoothed_path_points = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        color_topic, depth_topic, info_topic = "/camera/color/image_raw/compressed", "/camera/aligned_depth_to_color/image_raw", "/camera/color/camera_info"
        color_sub, depth_sub, info_sub = message_filters.Subscriber(color_topic, CompressedImage), message_filters.Subscriber(depth_topic, Image), message_filters.Subscriber(info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], 2, 0.5)
        self.ts.registerCallback(self.synchronized_callback)

        self.path_pub = rospy.Publisher("/path", Path, queue_size=1)
        self.viz_pub = rospy.Publisher("/drivable_area/viz_2d", Image, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1) # <<< 로봇 구동 명령 Publisher 추가
        
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        rospy.loginfo("Node initialization complete. Advanced HSV Planner is running.")

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
            
            path_msg = None
            try:
                np_arr = np.frombuffer(local_color_msg.data, np.uint8)
                cv_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv_depth = self.bridge.imgmsg_to_cv2(local_depth_msg, "16UC1")
                
                mask_2d = self.create_mask_from_color(cv_color)
                if np.any(mask_2d):
                    points_in_camera = self.get_3d_points(mask_2d, cv_depth, local_info_msg)
                    if points_in_camera is not None and points_in_camera.shape[0] > 100:
                        # <<< 함수 이름을 변경하고, 경로 메시지를 반환받도록 수정
                        path_msg = self.generate_path(points_in_camera, local_info_msg)

                # <<< 경로 생성 유무에 따라 로봇 제어
                if path_msg and path_msg.poses:
                    self.path_pub.publish(path_msg)
                    self.follow_the_path(path_msg)
                else:
                    self.stop_robot()

                path_points_2d = self.generate_path_from_mask_2d(mask_2d)
                viz_image = self.create_visualization_image(cv_color, mask_2d, path_points_2d)
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
                cv2.imshow(self.cv_window_name, viz_image)
                cv2.waitKey(1)

            except Exception as e:
                rospy.logerr(f"Processing loop error: {e}")
                self.stop_robot() # <<< 에러 발생 시 로봇 정지
            
            rate.sleep()

    # <<< [핵심 추가] Pure Pursuit 알고리즘 기반 경로 추종 함수
    def follow_the_path(self, path_msg):
        # 로봇의 기준 프레임(base_footprint)이 (0,0)에 있다고 가정합니다.
        # path_msg의 좌표들은 이미 로봇 기준 프레임으로 변환되어 있습니다.
        
        # 1. Look-ahead distance에 가장 가까운 경로 포인트를 찾습니다.
        goal_point = None
        for pose in reversed(path_msg.poses): # 경로의 끝점부터 탐색하여 가장 먼 지점을 우선으로 함
            point_dist = math.sqrt(pose.pose.position.x**2 + pose.pose.position.y**2)
            if point_dist >= self.lookahead_distance:
                goal_point = pose.pose.position
                break
        
        # 적절한 목표 지점을 찾지 못하면 마지막 점을 목표로 하거나 정지합니다.
        if goal_point is None and path_msg.poses:
            goal_point = path_msg.poses[-1].pose.position
        elif goal_point is None:
            self.stop_robot()
            return
            
        # 2. 목표 지점까지의 각도(theta)를 계산합니다.
        # atan2(y, x)를 사용하여 정확한 사분면의 각도를 구합니다.
        angle_to_goal = math.atan2(goal_point.y, goal_point.x)
        
        # 3. 각속도(angular.z)를 계산합니다.
        # 목표 지점과의 각도 차이에 비례하여 회전 속도를 결정합니다.
        angular_z = self.angular_gain * angle_to_goal
        
        # 4. 선속도(linear.x)를 결정합니다.
        # 큰 각도로 회전 시 속도를 줄여 안정성을 높입니다.
        # abs(angle_to_goal)이 클수록 감속량이 커집니다.
        turn_damping = 1.0 - min(1.0, abs(angle_to_goal) / (math.pi / 2)) # 90도 이상 회전 시 0
        linear_x = self.linear_velocity * turn_damping

        # 5. Twist 메시지를 생성하고 발행합니다.
        twist_cmd = Twist()
        twist_cmd.linear.x = linear_x
        twist_cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_cmd)

    # <<< [핵심 추가] 로봇 정지 함수
    def stop_robot(self):
        twist_cmd = Twist()
        twist_cmd.linear.x = 0.0
        twist_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_cmd)
        
    # <<< 함수명 변경 및 return 값 추가
    def generate_path(self, points_3d_cam, info_msg):
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_base_frame, info_msg.header.frame_id, info_msg.header.stamp, rospy.Duration(0.1))
            trans_matrix = self.transform_to_matrix(transform)
            points_homogeneous = np.hstack((points_3d_cam, np.ones((points_3d_cam.shape[0], 1))))
            points_3d_robot = (trans_matrix @ points_homogeneous.T).T[:, :3]

            valid_x_indices = (points_3d_robot[:, 0] > 0.1) & (points_3d_robot[:, 0] < self.path_lookahead)
            if np.sum(valid_x_indices) < 20: return None

            x_coords = points_3d_robot[valid_x_indices, 0]
            y_coords = points_3d_robot[valid_x_indices, 1]
            
            path_poly_coeffs = np.polyfit(x_coords, y_coords, 2)
            path_poly = np.poly1d(path_poly_coeffs)

            path_x = np.linspace(0.2, self.path_lookahead, self.num_path_points)
            path_y = path_poly(path_x)

            raw_path_points = []
            for px, py in zip(path_x, path_y):
                search_radius = 0.15
                distances = np.linalg.norm(points_3d_robot[:, :2] - np.array([px, py]), axis=1)
                nearby_points = points_3d_robot[distances < search_radius]
                if nearby_points.shape[0] > 5:
                    z = np.median(nearby_points[:, 2])
                    raw_path_points.append(np.array([px, py, z]))

            if len(raw_path_points) < self.num_path_points / 2: return None

            alpha = self.smoothing_factor
            if self.smoothed_path_points is None or len(self.smoothed_path_points) != len(raw_path_points):
                self.smoothed_path_points = raw_path_points
            else:
                for i in range(len(raw_path_points)):
                    self.smoothed_path_points[i] = alpha * raw_path_points[i] + (1 - alpha) * self.smoothed_path_points[i]
            
            path_msg = Path(header=rospy.Header(stamp=info_msg.header.stamp, frame_id=self.robot_base_frame))
            for p in self.smoothed_path_points:
                pose = PoseStamped(header=path_msg.header)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
            return path_msg # <<< 경로 메시지 반환

        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Path generation failed: {e}")
            return None # <<< 실패 시 None 반환

    def create_mask_from_color(self, cv_color):
        hsv = cv2.cvtColor(cv_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_MINT_HSV, self.UPPER_MINT_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask

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
        if mask_2d is not None and np.any(mask_2d):
            overlay_color = np.array([224, 255, 144], dtype=np.uint8)
            viz_image[mask_2d > 0] = (0.5 * viz_image[mask_2d > 0].astype(np.float32) + 0.5 * overlay_color.astype(np.float32)).astype(np.uint8)
        if path_points_2d:
            for x, y in path_points_2d:
                cv2.circle(viz_image, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        return viz_image

    def generate_path_from_mask_2d(self, mask):
        path_points = []
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
    rospy.init_node('advanced_hsv_planner', anonymous=True)
    try:
        planner = AdvancedHsvPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # <<< 종료 시 로봇을 정지시키는 로직 추가
        # planner가 정의되었는지 확인 후 정지 명령 전달
        if 'planner' in locals() and isinstance(planner, AdvancedHsvPlanner):
             planner.stop_robot()
             rospy.loginfo("Planner shutting down. Sending stop command to robot.")
        cv2.destroyAllWindows()
        rospy.loginfo("OpenCV windows closed.")

if __name__ == '__main__':
    main()