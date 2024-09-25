#include "pnp_distance.h"
// 绘制旋转矩形的函数
void drawRotatedRect(cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness)
{
    cv::Point2f Vertex[4]; // 定义四个点的数组，用于存储旋转矩形的四个顶点
    rect.points(Vertex); // 获取旋转矩形的四个顶点
    for(int i = 0 ; i < 4 ; i++)
    {
        // 绘制旋转矩形的边，连接相邻的顶点
        cv::line(img , Vertex[i] , Vertex[(i + 1) % 4] , color , thickness);
    }
}

//将卡尔曼滤波的3d点变为2d的
void project3DPointsTo2D(const std::vector<cv::Point3f>& points3D,
                         const cv::Mat& cameraMatrix,
                         const cv::Mat& distCoeffs,
                         const cv::Mat& rvec,
                         const cv::Mat& tvec,
                         std::vector<cv::Point2f>& points2D) {
    // 将旋转向量转换为旋转矩阵
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // 投影点
    cv::projectPoints(points3D, R, tvec, cameraMatrix, distCoeffs, points2D);
}

// 获取装甲板的二维点
void getTarget2dPoints( vector<Point2f> &object_2d_point,std::vector<Point2f> &object2d_point)
{
    vector<Point2f> vertices; // 存储旋转矩形四个顶点
    vertices =object_2d_point;
    cv::Point2f lu, ld, ru, rd; // 定义四个点，表示矩形的左上、左下、右上、右下
    // cout<<vertices.size()<<"这里是顶点的大小"<<endl;
    if(vertices.size()==0) {
        return ;
    }
    std::sort(vertices.begin(), vertices.end(), [](const cv::Point2f & p1, const cv::Point2f & p2) { return p1.x < p2.x; }); // 按 x 坐标排序顶点
    if (vertices[0].y < vertices[1].y) {
        lu = vertices[0]; // 左上角
        ld = vertices[1]; // 左下角
    } else {
        lu = vertices[1]; // 左上角
        ld = vertices[0]; // 左下角
    }
    if (vertices[2].y < vertices[3].y) {
        ru = vertices[2]; // 右上角
        rd = vertices[3]; // 右下角
    } else {
        ru = vertices[3]; // 右上角
        rd = vertices[2]; // 右下角
    }
    object2d_point.clear(); // 清空二维点向量
    object2d_point.push_back(lu); // 添加左上角
    object2d_point.push_back(ru); // 添加右上角
    object2d_point.push_back(rd); // 添加右下角
    object2d_point.push_back(ld); // 添加左下角
}

// HSV 颜色空间阈值设置（用于图像二值化）
int iLowH = 156;
int iHighH = 180;
int iLowS = 43;
int iHighS = 255;
int iLowV = 46;
int iHighV = 255;



vector<double> pnp_Distance(vector<Point2f> &object_2d_point,Mat cameraMatrix,Mat distCoeffs,Mat& rot1){


            vector<Point2f> object2d_point; // 存储二维图像点
            getTarget2dPoints(object_2d_point, object2d_point); // 获取二维点
            if(object2d_point.size() != 4) // 如果二维点不是四个，返回
            {
                vector<double> no_one;
                return no_one;
            }
            std::vector<cv::Point3f> point3d; // 存储三维世界点

            float half_x = 4.0f / 2.0f; // 装甲板一半宽度2.6
            float half_y = 3.4f / 2.0f; // 装甲板一半高度3.4

            // 定义装甲板的四个角点在世界坐标系中的位置
            point3d.push_back(Point3f(-half_x, half_y, 0)); // 左上角
            point3d.push_back(Point3f(half_x, half_y, 0));  // 右上角
            point3d.push_back(Point3f(half_x, -half_y, 0)); // 右下角
            point3d.push_back(Point3f(-half_x, -half_y, 0)); // 左下角

            cv::Mat rot; // 旋转矩阵

            cv::Mat trans; // 平移矩阵


            // 通过已知的三维点和二维点计算相机姿态
            cv::solvePnP(point3d, object2d_point, cameraMatrix, distCoeffs, rot, trans);
            rot.copyTo(rot1);

            // 提取平移向量的 x, y, z 分量
            double tx = trans.at<double>(0, 0);
            double ty = trans.at<double>(1, 0);
            double tz = trans.at<double>(2, 0);
            // 计算目标的距离
            // double dis = sqrt(tx * tx + ty * ty + tz * tz);
            // cout << "dis: " << dis << endl; // 输出距离
            vector<double> rvec;
            rvec.push_back(tx);
            rvec.push_back(ty);
            rvec.push_back(tz);

    return rvec;
}




