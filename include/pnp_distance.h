//
// Created by shur on 24-8-8.
//

#ifndef PNP_DISTANCE_H
#define PNP_DISTANCE_H

#include<opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
vector<double> pnp_Distance(vector<Point2f> &object_2d_point,Mat cameraMatrix,Mat distCoeffs,Mat& rot1);
void project3DPointsTo2D(const std::vector<cv::Point3f>& points3D,
                         const cv::Mat& cameraMatrix,
                         const cv::Mat& distCoeffs,
                         const cv::Mat& rvec,
                         const cv::Mat& tvec,
                         std::vector<cv::Point2f>& points2D) ;
#endif //PNP_DISTANCE_H
