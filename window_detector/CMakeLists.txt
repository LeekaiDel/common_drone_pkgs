cmake_minimum_required(VERSION 2.8.3)
project(window_detector)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  rospy
  roscpp
  sensor_msgs
  std_msgs
  stereo_msgs
  drone_msgs
)

find_package(OpenCV REQUIRED)

if(cv_bridge_VERSION VERSION_GREATER "1.12.0")
  add_compile_options(-std=c++11)
endif()


## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/WindowDetector.cfg
)

## catkin specific configuration
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES window_detector
#  CATKIN_DEPENDS cv_bridge dynamic_reconfigure geometry_msgs rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS bin/window_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Cpp executable nodes 1
#add_executable(window_detector_real
#               src/window_detector_real.cpp)
#target_link_libraries(window_detector_real
#                      ${catkin_LIBRARIES}
#                      ${OpenCV_LIBRARIES})

# Cpp executable nodes 2
#add_executable(window_detector_real_2
#               src/window_detector_real_2.cpp)
#target_link_libraries(window_detector_real_2
#                      ${catkin_LIBRARIES}
#                      ${OpenCV_LIBRARIES})

# Cpp executable nodes 3
add_executable(window_detector_real_3
               src/window_detector_real_3.cpp
               src/window_detector_debug.h)
target_link_libraries(window_detector_real_3
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES})

