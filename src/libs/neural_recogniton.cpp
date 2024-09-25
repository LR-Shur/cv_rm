#include<neural_recognition.h>


cv::Mat invertIfWhiteDominant(const cv::Mat& grayImg) {
    // 确保图像是灰度图像
    // if (grayImg.type() != CV_8UC1) {
    //     throw std::invalid_argument("图像必须是灰度图像");
    // }

    // 计算白色像素和黑色像素的数量
    int whitePixels = cv::countNonZero(grayImg == 255);
    int blackPixels = cv::countNonZero(grayImg == 0);
    // cout<<whitePixels<<endl;
    // cout<<blackPixels<<endl;
    cv::Mat resultImg;

    // 检查白色像素是否多于黑色像素
    if (whitePixels > blackPixels) {
        // 反转黑白图像
        cv::bitwise_not(grayImg, resultImg);
    } else {
        // 如果白色像素少于或等于黑色像素，返回原图像
        resultImg = grayImg.clone();
    }

    return resultImg;
}


int neuralRecognition(Mat frame,String modelPath_xml) {

    YoloModel yoloModel;
    string device="CPU";
    yoloModel.LoadDetectModel(modelPath_xml,device);
    vector<Object> vecObj;
    Mat dst;
    int result=yoloModel.YoloDetectInfer(frame,0.5,0.5,dst,vecObj);

    return result;































    // Core core;
    // shared_ptr<Model> model = core.read_model(modelPath_xml);
    //
    // auto input = model->input();
    // auto output = model->output();
    // // cout<<1;
    // string device="CPU";
    // //编译模型并加载到设备
    // auto compiled_model = core.compile_model(model, device);
    // //创建推理请求
    // auto infer_request = compiled_model.create_infer_request();
    // // 将图像转换为单通道灰度图像
    // cv::Mat gray_image;
    // cv::cvtColor(frame, gray_image, cv::COLOR_BGR2GRAY);
    // gray_image.convertTo(gray_image, CV_32F, 1.0 / 255.0); // 归一化
    //
    //
    // // 调整图像大小
    // if (gray_image.rows != 28 || gray_image.cols != 28) {
    //     resize(gray_image, gray_image, cv::Size(28, 28));
    // }
    //
    // // // 将灰度图像转换为二值图像
    // // cv::Mat binary_image;
    // // double threshold_value = 127; // 阈值
    // // cv::threshold(gray_image, binary_image, threshold_value, 255, cv::THRESH_BINARY);
    // // imshow("1",binary_image);
    // // waitKey(0);
    //
    // // //一定要黑底的图
    // // Mat gray_image2 = Mat::zeros(28, 28, CV_32F);
    // // gray_image2 = invertIfWhiteDominant(gray_image);
    // // imshow("1",gray_image2);
    // // waitKey(0);
    //
    // // 创建 Blob 对象以存储输入数据
    // auto input_tensor = infer_request.get_input_tensor();
    // auto input_data = input_tensor.data<float>();
    //
    // // 拷贝图像数据到 Blob 中
    // std::memcpy(input_data, gray_image.data, gray_image.total() * gray_image.elemSize());
    //
    // // 运行推理
    // infer_request.infer();
    //
    // // 获取输出
    // auto output_tensor = infer_request.get_output_tensor();
    // auto output_data = output_tensor.data<float>();
    //
    //
    //
    // // for(int i=0;i<10;i++){
    // //     cout<<output_data[i]<<endl;
    // // }
    //
    //
    // // 获取最大值的索引
    // int max_index = 0;
    // float max_value = output_data[0];
    // for (int i = 1; i < 10; i++) {
    //     if (output_data[i] > max_value) {
    //         max_value = output_data[i];
    //         max_index = i;
    //     }
    // }
    // // cout<<max_index<<endl;
    // if(output_data[max_index]>=-20) {
    //     return max_index;
    // }else {
    //     return -1;
    // }



}