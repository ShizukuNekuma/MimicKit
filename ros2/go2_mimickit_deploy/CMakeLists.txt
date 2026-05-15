cmake_minimum_required(VERSION 3.5)
project(go2_mimickit_deploy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(unitree_go REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/onnxruntime/include
    $ENV{ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/onnxruntime/lib
    $ENV{ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime was not found. Set ONNXRUNTIME_ROOT or install libonnxruntime.")
endif()

add_executable(go2_policy_node src/go2_policy_node.cpp)
target_include_directories(go2_policy_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(go2_policy_node ${ONNXRUNTIME_LIBRARY})
ament_target_dependencies(go2_policy_node rclcpp unitree_go)

install(TARGETS go2_policy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
