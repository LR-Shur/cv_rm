//
// Created by shur on 24-9-11.
//

#ifndef SERIAL_H
#define SERIAL_H
#include <string>

#include <iostream>
#include <fcntl.h>      // 文件控制定义
#include <errno.h>      // 错误码定义
#include <termios.h>    // 终端IO定义
#include <unistd.h>     // 读写操作
#include <cstring>      // 字符串处理
#include<yaml-cpp/yaml.h>
using namespace std;


struct data_3d {
    float x;
    float y;
    float z;


};


class Serial_all {
public:
    Serial_all(std::string file_path);
    int send_msg(std::string msg1);
    // std::string read_msg();
    void send_data(data_3d data);
    string read_data;
    void stop();
    void read_msg();
    bool keep_reading = true;
private:
    struct termios tty;
    int serial_port;
    std::string device;


};



#endif //SERIAL_H
