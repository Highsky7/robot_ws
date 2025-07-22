#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class RealtimeObjectDetector:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('realtime_object_detector', anonymous=True)

        # 클래스 이름 정의 (학습 시 사용한 순서와 정확히 일치해야 합니다)
        self.class_names = ['A', 'E', 'Heart', 'K', 'M', 'O', 'R', 'Y']

        # YOLO 모델 로드 (학습된 가중치 파일 경로)
        # best.pt 파일을 이 스크립트와 같은 폴더에 위치시키거나 절대 경로를 지정하세요.
        self.model_path = './vision_marker.pt'
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return

        # CompressedImage 토픽 구독 설정
        self.image_sub = rospy.Subscriber(
            "usb_cam/image_raw/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24 # 버퍼 사이즈를 충분히 크게 설정하여 이미지 드랍 방지
        )

        rospy.loginfo("Realtime Object Detector Node has been started.")

    def image_callback(self, msg):
        """
        CompressedImage 토픽을 수신할 때마다 호출되는 콜백 함수
        """
        try:
            # 1. ROS CompressedImage 메시지를 OpenCV 이미지로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Failed to decode image: {e}")
            return

        # 2. YOLO 모델을 사용하여 객체 탐지 수행
        # conf: confidence threshold (탐지 신뢰도 임계값)
        # iou: IoU(Intersection over Union) threshold for NMS
        results = self.model(cv_image, conf=0.5, iou=0.45)

        # 3. 탐지 결과를 원본 이미지에 시각화
        annotated_image = self.draw_detections(cv_image, results)

        # 4. 결과 이미지를 화면에 표시
        cv2.imshow("Realtime Object Detection", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User pressed 'q' to exit.")
            cv2.destroyAllWindows()


    def draw_detections(self, image, results):
            """
            탐지된 객체들의 바운딩 박스와 클래스 이름을 이미지에 그리는 함수
            """
            # results[0]에 탐지 결과가 담겨 있습니다.
            for result in results:
                boxes = result.boxes.cpu().numpy() # 바운딩 박스 정보 (xyxy, conf, cls)
                for box in boxes:
                    # 바운딩 박스 좌표 추출 (정수형으로 변환)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 신뢰도와 클래스 ID 추출
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])

                    # 클래스 ID를 실제 클래스 이름으로 변환
                    if class_id < len(self.class_names):
                        label = self.class_names[class_id]
                    else:
                        label = "Unknown"

                    # 바운딩 박스 그리기 (색상: (0, 255, 0) - 초록색, 두께: 2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 클래스 이름과 신뢰도 텍스트 표시
                    text = f"{label}: {confidence:.2f}"

                    # --- 텍스트 크기 및 두께 수정 ---
                    font_scale = 2.0  # 폰트 크기를 0.5에서 1.0으로 키움
                    font_thickness = 2 # 폰트 두께를 1에서 2로 늘림

                    # 텍스트 크기에 맞는 배경 사각형 계산 및 그리기
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                    
                    # 수정된 크기와 두께로 텍스트 쓰기 (색상: (0, 0, 0) - 검은색)
                    cv2.putText(image, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            return image

    def run(self):
        """
        ROS 노드를 계속 실행 상태로 유지
        """
        rospy.spin()
        # ROS 종료 시 OpenCV 창 닫기
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        detector = RealtimeObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()