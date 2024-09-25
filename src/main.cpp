#include "CameraApi.h" //相机SDK的API头文件

#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include <stdio.h>
#include "color_recognition.h"
#include"neural_recognition.h"
#include"pnp_distance.h"
#include"ekf_Kalman.h"
#include <iostream>
#include<packet_test.h>
#include<serial.h>
#include <thread>
#include <atomic>
#include <chrono>
using namespace cv;
using namespace std;
unsigned char           * g_pRgbBuffer;     //处理后数据缓存区

int main()
{

    int                     iCameraCounts = 1;
    int                     iStatus=-1;
    tSdkCameraDevInfo       tCameraEnumList;
    int                     hCamera;
    tSdkCameraCapbility     tCapability;      //设备描述信息
    tSdkFrameHead           sFrameInfo;
    BYTE*			        pbyBuffer;
    int                     iDisplayFrames = 10000;
    Mat *iplImage = NULL;
    int                     channel=3;
	//设置的值
	double* eptime;
	BOOL* wb;
	int* gain_min;
	int* gain_max;
	int* frame_speed= 0;


    CameraSdkInit(1);

	// 相机内参矩阵
	Mat cameraMatrix = (Mat_<double>(3, 3) << 2.12457367e+03, 0.00000000e+00, 5.99781974e+02, 0, 2.12728011e+03, 4.77982829e+02, 0, 0, 1);

	// 畸变系数
	Mat distCoeffs = (Mat_<double>(5, 1) << -1.64099302e-01, 2.47006825e+00, -1.77395774e-03, -4.78041749e-03, -1.27311747e+01);
	//串口初始化
	Serial_all serial_port= Serial_all("../source/config.yaml");
    //枚举设备，并建立设备列表
    iStatus = CameraEnumerateDevice(&tCameraEnumList,&iCameraCounts);
	printf("state = %d\n", iStatus);

	printf("count = %d\n", iCameraCounts);
    //没有连接设备
    if(iCameraCounts==0){
        return -1;
    }

    //相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
    iStatus = CameraInit(&tCameraEnumList,-1,-1,&hCamera);

    //初始化失败
	printf("state = %d\n", iStatus);
    if(iStatus!=CAMERA_STATUS_SUCCESS){
    	cout<<"init error"<<endl;
        return -1;
    }

    //获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
    CameraGetCapability(hCamera,&tCapability);

    //
    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);
    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    /*让SDK进入工作模式，开始接收来自相机发送的图像
    数据。如果当前相机是触发模式，则需要接收到
    触发帧以后才会更新图像。    */
    CameraPlay(hCamera);

    /*其他的相机参数设置
    例如 CameraSetExposureTime   CameraGetExposureTime  设置/读取曝光时间
         CameraSetImageResolution  CameraGetImageResolution 设置/读取分辨率
         CameraSetGamma、CameraSetConrast、CameraSetGain等设置图像伽马、对比度、RGB数字增益等等。
         本例程只是为了演示如何将SDK中获取的图像，转成OpenCV的图像格式,以便调用OpenCV的图像处理函数进行后续开发
    */
	//相机设置
	// CameraSetExposureTime(hCamera,0.4 * 1000);
	// CameraSetAeState(hCamera,TRUE);
	// CameraSetWbMode(hCamera,FALSE);
	 CameraSetAeAnalogGainRange(hCamera,64,130);
	CameraSetFrameSpeed(hCamera,0.3);
	// //确认设置
	// CameraGetExposureTime(hCamera,eptime);
	// cout<<"曝光时间为："<<*eptime<<endl;
	// CameraGetWbMode(hCamera,wb);
	// cout<<"白平衡为："<<*wb<<endl;
	// CameraGetAeAnalogGainRange(hCamera,gain_min,gain_max);
	// cout<<"增益范围为："<<*gain_min<<"-"<<*gain_max<<endl;
	// CameraGetFrameSpeed(hCamera,frame_speed);
	// cout<<"帧速率为："<<*frame_speed<<endl;

    if(tCapability.sIspCapacity.bMonoSensor){
        channel=1;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel=3;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
    }

	// 初始化 EKF 参数
	// double dt = *frame_speed;
	double dt = 0.3;
	cv::Mat processNoiseCov = cv::Mat::eye(9, 9, CV_64F) * (0.8);
	cv::Mat measurementNoiseCov = cv::Mat::eye(9, 9, CV_64F) * (0.09);
	cv::Mat errorCovPost = cv::Mat::eye(9, 9, CV_64F)*0.1;

	// 创建 EKFTracker 需要的数组啥的
	EKFTracker ekf(dt, processNoiseCov, measurementNoiseCov, errorCovPost);
	int jishu=0;
	vector<double> last_object_3d_vector;
	vector<double> last_object_3d_v ;
    //循环显示1000帧图像

	// 启动接收线程
	std::thread receiver(&Serial_all::read_msg, &serial_port);
    //while(iDisplayFrames--)
    while(true)
    {
		// cout<<"kaitou计数是"<<jishu<<endl;
        if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
		{
		    CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);

		    cv::Mat matImage(
					Size(sFrameInfo.iWidth,sFrameInfo.iHeight),
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
					);
			//白平衡模式
        	CameraSetOnceWB(hCamera);


        	// // 计算新相机矩阵（可以选择进行优化）
        	// Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, image.size(), 1, image.size(), 0);
        	// 矫正图像
        	Mat undistortedImage;
        	undistort(matImage, undistortedImage, cameraMatrix, distCoeffs);


        	//处理图像
			int jieguo;
        	Mat img1;
        	vector<Point2f> object_2d_point;
        	jieguo =neuralRecognition(undistortedImage,"/home/shur/CLionProjects/t3/best.onnx");

        	if (jieguo!=-1) {
				// cout<<jieguo<<endl;
				jishu=jishu+1;
        		img1 =colorRecognition(undistortedImage,object_2d_point);

        	}
			else{
				cout<<"no"<<endl;
				img1 = undistortedImage;
				object_2d_point.clear();
				jishu=0;
			}
        	vector<double> object_3d_vector;

        	double distance;
        	Mat rvec;
        	//如果识别到装甲板了，再进行后续步骤
        	if(object_2d_point.size()!=0) {
        		object_3d_vector =pnp_Distance(object_2d_point,cameraMatrix,distCoeffs, rvec);
        		cout<<"装甲板距离是"<<object_3d_vector[2]<<endl;

				if(jishu==1) {
					// cout<<"现在技术是1"<<endl;
					last_object_3d_vector = object_3d_vector;
					last_object_3d_v = {0,0,0};
					EKFTracker ekf(dt, processNoiseCov, measurementNoiseCov, errorCovPost);
				}
        		if(jishu==2) {//第二帧用来初始化
        			// cout<<"现在技术是2"<<endl;
					double x = object_3d_vector[0] - last_object_3d_vector[0];
        			double y = object_3d_vector[1] - last_object_3d_vector[1];
        			double z = object_3d_vector[2] - last_object_3d_vector[2];
        			double v_x = x / dt;
        			double v_y = y / dt;
        			double v_z = z / dt;
        			double a_x = (v_x - last_object_3d_v[0]) / dt;
        			double a_y = (v_y - last_object_3d_v[1]) / dt;
        			double a_z = (v_z - last_object_3d_v[2]) / dt;
        			last_object_3d_vector = object_3d_vector;
        			last_object_3d_v = {v_x,v_y,v_z};
        			Mat initState = (cv::Mat_<double>(9, 1)
        				<< object_3d_vector[0], object_3d_vector[1], object_3d_vector[2]
        				, v_x, v_y, v_z, a_x, a_y, a_z);
        			ekf.init(initState);
        		}
        		if(jishu>2) {//后面就一直预测就好了
        			// cout<<"现在技术是3"<<endl;
        			double x = object_3d_vector[0] - last_object_3d_vector[0];
        			double y = object_3d_vector[1] - last_object_3d_vector[1];
        			double z = object_3d_vector[2] - last_object_3d_vector[2];
        			double v_x = x / dt;
        			double v_y = y / dt;
        			double v_z = z / dt;
        			double a_x = (v_x - last_object_3d_v[0]) / dt;
        			double a_y = (v_y - last_object_3d_v[1]) / dt;
        			double a_z = (v_z - last_object_3d_v[2]) / dt;
        			last_object_3d_vector = object_3d_vector;
        			last_object_3d_v = {v_x,v_y,v_z};
        			Mat measurement = (cv::Mat_<double>(9, 1)
						<< object_3d_vector[0], object_3d_vector[1], object_3d_vector[2]
						, v_x, v_y, v_z, a_x, a_y, a_z);
        			ekf.predict();
        			ekf.update(measurement);

        			// 获取更新后的状态
        			cv::Mat state = ekf.getState();
        			double tx = state.at<double>(0);
        			double ty = state.at<double>(1);
        			double tz = state.at<double>(2);
        			vector<cv::Point3f> points3D;
        			points3D.push_back(cv::Point3f(0, 0, 0));
        			vector<cv::Point2f> points2D;
        			project3DPointsTo2D(points3D, cameraMatrix, distCoeffs, rvec, cv::Mat(state, cv::Rect(0, 0, 1, 3)), points2D);
        			// cout<<"x:"<<points2D[0].x<<"y:"<<points2D[0].y<<endl;
        			//绘制预测的点
        			circle(img1, points2D[0], 5, cv::Scalar(0, 0, 255), 5);
					//发送数据
        			data_3d data;
        			data.x=x;
        			data.y=y;
        			data.z=z;
        			serial_port.send_data(data);
        			// serial_port.send_msg(to_string(x));

        			// string a = serial_port.read_msg();
        			// cout<<a<<endl;
     //    			if (serial_port.read_data!="") {
					// 	cout<<serial_port.read_data<<endl;
					// }
        		}

        	}else {
        		object_2d_point.clear();
        		jishu=0;
        	}




			//显示图像
			imshow("Opencv Demo", img1);
        	Mat grays=image_processing(undistortedImage);
        	// imshow("gray",grays);
            waitKey(5);





            //在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
			//否则再次调用CameraGetImageBuffer时，程序将被挂起一直阻塞，直到其他线程中调用CameraReleaseImageBuffer来释放了buffer
			CameraReleaseImageBuffer(hCamera,pbyBuffer);

		}
    }

    CameraUnInit(hCamera);
    //注意，现反初始化后再free
    free(g_pRgbBuffer);


    return 0;
}

