//
// Created by shur on 24-8-2.
//
#include "color_recognition.h"
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main() {

    Mat src = imread("/home/shur/CLionProjects/t3/data/9.png");
    vector<Point2f> object_2d_point;
    Mat result = colorRecognition(src,object_2d_point);

    imshow("result", result);
    waitKey(0);

    return 0;
}

