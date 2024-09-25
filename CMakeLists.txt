cmake_minimum_required(VERSION 3.28)
project(t3)

set(CMAKE_CXX_STANDARD 17)

add_executable(t3

        src/main.cpp

        src/libs/color_recognition.cpp

        src/libs/neural_recogniton.cpp
        include/ov_yolov8.h
        src/libs/ov_yolov8.cpp
        include/pnp_distance.h
        src/libs/pnp_distance.cpp
        include/ekf_Kalman.h
        src/libs/ekf_Kalman.cpp
        src/libs/serial.cpp
        include/serial.h
        include/packet.hpp
        include/packet_test.h
        src/main.cpp
        src/main.cpp
)
#头文件还有测试文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(./test)


        find_package(OpenCV REQUIRED)
        include_directories(${OpenCV_INCLUDE_DIRS})
        target_link_libraries(t3 ${OpenCV_LIBS})
        ## 打印 OpenCV 信息
        #message(STATUS "OpenCV version: ${OpenCV_VERSION}")
        #message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
        #message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")




        include_directories(/usr/include/openvino)
        target_link_libraries(t3 /usr/lib/libopenvino.so)

        #相机驱动
        include_directories(/home/shur/c++_libs/linuxSDK_V2.1.0.37/include)
        target_link_libraries(t3 /home/shur/c++_libs/linuxSDK_V2.1.0.37/lib/x64/libMVSDK.so)

find_package(yaml-cpp REQUIRED)
# 链接 yaml-cpp 库到可执行目标
target_link_libraries(t3  yaml-cpp)