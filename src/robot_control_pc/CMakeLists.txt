cmake_minimum_required(VERSION 3.0.2)
project(robot_control_pc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(twist_to_diff src/twist_to_diff.cpp)
target_link_libraries(twist_to_diff ${catkin_LIBRARIES})
install(TARGETS twist_to_diff
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# --- 아래 내용 추가 ---
# Python 스크립트를 설치 대상으로 추가
catkin_install_python(PROGRAMS src/network_sender.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
