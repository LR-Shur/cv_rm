cmake_minimum_required(VERSION 3.10)
project(test_color_recognition)
project(test_neural_network_recognition)




add_executable(test_color_recognition test_colar.cpp
        ../include/color_recognition.h
        ../src/libs/color_recognition.cpp
        )
add_executable(test_neural_network_recognition
        test_neural.cpp
        ../include/neural_recognition.h
        ../src/libs/neural_recogniton.cpp
        ../include/ov_yolov8.h
        ../src/libs/ov_yolov8.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_color_recognition ${OpenCV_LIBS})

target_link_libraries(test_neural_network_recognition ${OpenCV_LIBS})

include_directories(/usr/include/openvino)
target_link_libraries(test_neural_network_recognition /usr/lib/libopenvino.so)



#距离估计pnp
add_executable(test_pnp_distance
        ../include/pnp_distance.h
        test_pnp_distance.cpp
        ../src/libs/pnp_distance.cpp
        ../include/color_recognition.h
        ../src/libs/color_recognition.cpp
        ../include/ekf_Kalman.h
        ../src/libs/ekf_Kalman.cpp)
target_link_libraries(test_pnp_distance ${OpenCV_LIBS})

add_executable(test_serial
        test_serial.cpp
        ../include/serial.h
        ../src/libs/serial.cpp
        test_serial.cpp
        test_serial.cpp
)
find_package(yaml-cpp REQUIRED)
# 链接 yaml-cpp 库到可执行目标
target_link_libraries(test_serial  yaml-cpp)