//
// Created by shur on 24-8-3.
//

#ifndef NEURAL_RECOGNITION_H
#define NEURAL_RECOGNITION_H
#include <stdio.h>
#include <string>
#include<iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<ov_yolov8.h>
using namespace cv;
using namespace ov;
using namespace std;

int neuralRecognition(Mat frame,String modelPath);
#endif //NEURAL_RECOGNITION_H
