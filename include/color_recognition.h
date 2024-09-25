//
// Created by shur on 24-8-2.
//

#ifndef COLOR_RECOGNITION_H
#define COLOR_RECOGNITION_H
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
using namespace cv;
using namespace std;
cv::Mat colorRecognition(cv::Mat frame,vector<Point2f> &object2d_point);

Mat image_processing(Mat frame);




#endif //COLOR_RECOGNITION_H
