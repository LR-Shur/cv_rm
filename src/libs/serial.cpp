//
// Created by shur on 24-9-11.
//

#include "../../include/serial.h"


Serial_all::Serial_all(string file_path) {
    YAML::Node config = YAML::LoadFile(file_path);
    YAML::Node serial_config;

    if (config["serial"]) {
        serial_config = config["serial"];
    } else {
        cout << "未找到 'serial' 配置节点。" << endl;
        return;
    }

    if (serial_config["device_name"]) {
        device = serial_config["device_name"].as<string>();
    } else {
        cout << "未找到 'device_name' 配置项。" << endl;
        return;
    }

    serial_port = open(device.c_str(), O_RDWR | O_NOCTTY);
    if (serial_port == -1) {
        cout << "无法打开串口设备: " << device << ", 错误: " << strerror(errno) << endl;
        return;
    }

    if (tcgetattr(serial_port, &tty) != 0) {
        cout << "无法读取串口设置: " << strerror(errno) << endl;
        close(serial_port);
        return;
    }

    memset(&tty, 0, sizeof tty);

    speed_t speed;
    if (serial_config["baud_rate"]) {
        string speed1 = serial_config["baud_rate"].as<string>();
        switch (stoi(speed1)) {
            case 1000000: speed = B1000000; break;
            case 115200:  speed = B115200; break;
            case 9600:    speed = B9600; break;
            default:
                cout << "不支持的波特率: " << speed1 << endl;
                close(serial_port);
                return;
        }
        cout << "波特率: " << speed1 << endl;
        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);
    } else {
        cout << "未找到 'baud_rate' 配置项。" << endl;
        close(serial_port);
        return;
    }

    // 配置串口参数
    tty.c_cflag &= ~CSIZE; // 清除数据位设置
    tty.c_cflag |= CS8;    // 设置数据位为 8 位
    tty.c_cflag |= (CLOCAL | CREAD); // 启用本地模式和接收模式

    if (serial_config["flow_control"]) {
        if (serial_config["flow_control"].as<string>() == "none") {
            tty.c_cflag &= ~CRTSCTS; // 禁用硬件流控制
            tty.c_iflag &= ~(IXON | IXOFF | IXANY); // 禁用软件流控制
        }
    }

    if (serial_config["parity"]) {
        if (serial_config["parity"].as<string>() == "none") {
            tty.c_cflag &= ~PARENB; // 禁用校验
        }
    }

    if (serial_config["stop_bits"]) {
        if (serial_config["stop_bits"].as<string>() == "1") {
            tty.c_cflag &= ~CSTOPB; // 设置 1 位停止位
        }
    }

    tty.c_lflag = 0; // 非本地模式
    tty.c_iflag = 0; // 原始输入模式
    tty.c_oflag = 0; // 原始输出模式

    // 应用设置
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        cout << "无法应用串口设置: " << strerror(errno) << endl;
        close(serial_port);
        return;
    }
}



int Serial_all::send_msg(string msg1) {

    const char *msg = msg1.c_str();
    int bytes_written = write(serial_port, msg, strlen(msg));
    if (bytes_written == -1) {
        cout << "数据发送失败: " << strerror(errno) << endl;
    } else {
        cout << "发送的数据: " << msg1 << ", 字节数: " << bytes_written << endl;
    }
    return bytes_written;
}


// 读取消息的线程函数
void Serial_all:: read_msg() {
    while (keep_reading) {
        char read_buf[256];
        memset(&read_buf, '\0', sizeof(read_buf));
         // cout<<"正在读取数据"<<endl;
        int num_bytes = read(serial_port, &read_buf, sizeof(read_buf));
        if (num_bytes > 0) {
            std::string message(read_buf);
            read_data = message;
            std::cout << "读取到的数据: " << message << std::endl;
        } else {
            // std::cout << "num_bytes: " << num_bytes << std::endl;
        }
    }
}

// 停止读取
void Serial_all::stop() {
    keep_reading = false;
}


    void Serial_all::send_data(data_3d data) {
        string msg = to_string(data.x) + "," + to_string(data.y) + "," + to_string(data.z) + "\n";
        send_msg(msg);
        //cout<<1<<endl;
    }
