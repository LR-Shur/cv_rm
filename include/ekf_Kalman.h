//
// Created by shur on 24-8-9.
//

#ifndef EKF_KALMAN_H
#define EKF_KALMAN_H

#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;

using namespace cv;
#endif //EKF_KALMAN_H




class EKFTracker {
public:
    EKFTracker(double dt, const cv::Mat& processNoiseCov, const cv::Mat& measurementNoiseCov, const cv::Mat& errorCovPost);

    // 初始化状态向量和协方差矩阵
    void init(const cv::Mat& initState);

    // 执行预测步骤
    void predict();

    // 执行更新步骤，更新观测值
    void update(const cv::Mat& measurement);

    // 获取当前的状态向量
    cv::Mat getState() const;

private:
    cv::Mat state;                // 状态向量 [tx, ty, tz, vx, vy, vz]
    cv::Mat transitionMatrix;     // 状态转移矩阵 F
    cv::Mat processNoiseCov;      // 过程噪声协方差矩阵 Q
    cv::Mat measurementMatrix;    // 观测矩阵 H
    cv::Mat measurementNoiseCov;  // 观测噪声协方差矩阵 R  只有更新用
    cv::Mat errorCovPost;         // 后验误差协方差矩阵 P  更新预测都用
    Mat previousState;          // 保存上一次的状态向量
    double dt;                    // 时间间隔 Δt
};
