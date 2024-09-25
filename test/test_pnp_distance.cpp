//
// Created by shur on 24-8-8.
//
#include<pnp_distance.h>
#include "color_recognition.h"
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main() {

    Mat src = imread("/home/shur/CLionProjects/t3/data/9.png");
    vector<Point2f> object2d_point;
    Mat result = colorRecognition(src,object2d_point);
    // 相机内参矩阵
    Mat cameraMatrix = (Mat_<double>(3, 3) << 2.12457367e+03, 0.00000000e+00, 5.99781974e+02, 0, 2.12728011e+03, 4.77982829e+02, 0, 0, 1);

    // 畸变系数
    Mat distCoeffs = (Mat_<double>(5, 1) << -1.64099302e-01, 2.47006825e+00, -1.77395774e-03, -4.78041749e-03, -1.27311747e+01);
    vector<double> object_3d_point;
    double distance;
    object_3d_point =pnp_Distance(object2d_point,cameraMatrix,distCoeffs);
    cout<<"distance = "<<object_3d_point[2]<<endl;
    imshow("result", result);
    waitKey(0);

    return 0;
}
