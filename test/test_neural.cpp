//
// Created by shur on 24-8-2.
//

#include "neural_recognition.h"
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main() {

    Mat src = imread("/home/shur/CLionProjects/t3/data/7.png");
    Mat resized_mat;



    int result = neuralRecognition(src, "/home/shur/CLionProjects/t3/best.onnx");
    cout<<result<<endl;
    // imshow("result", result);
    // waitKey(0);

    return 0;
}

