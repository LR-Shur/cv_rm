//
// Created by shur on 24-9-25.
//

#include "../include/serial.h"
#include <thread>

int main() {
    Serial_all serial("/home/shur/CLionProjects/t3/source/config.yaml");
    std::thread receiver(&Serial_all::read_msg, &serial);
    data_3d data;
    data.x = 1.0;
    data.y = 2.0;
    data.z = 3.0;
  serial.send_data(data);
    while (true) {
        serial.send_data(data);
    }
    //serial.send_msg("hello");
    // serial.read_msg();

    // 停止接收线程
    serial.stop();
    receiver.join();
    return 0;
}
