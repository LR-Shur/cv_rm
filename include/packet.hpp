// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rm_serial_driver
{
struct ReceivePacket
{
  uint8_t header = 0x5A;//数据头部消息，代表数据的开始
  uint8_t robot_color : 1;//颜色
  uint8_t task_mode : 2;//模式
  uint8_t reserved : 5;//预留位
  float pitch;//俯仰角
  float yaw;//偏航
  float aim_x;
  float aim_y;
  float aim_z;
  uint16_t checksum = 0;//校验值
} __attribute__((packed));

struct SendPacket
{
  uint8_t header = 0xA5;//数据头部消息，代表数据的开始
  bool tracking;
  float x;
  float y;
  float z;
  float yaw;//偏航
  float vx;
  float vy;
  float vz;
  float v_yaw;
  float r1;
  float r2;
  float z_2;
  uint16_t checksum = 0;
} __attribute__((packed));


/**
 * @brief 这个函数的作用是将一个vector<uint8_t> 类型的数据复制到 ReceivePacket 类型的对象中
 */
inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
  ReceivePacket packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}


/**
 * @brief 这个函数的作用是将一个 SendPacket 类型的对象转换为 vector<uint8_t> 类型的数据
 */
inline std::vector<uint8_t> toVector(const SendPacket & data)
{
  std::vector<uint8_t> packet(sizeof(SendPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(SendPacket), packet.begin());
  return packet;
}

}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
