#include "color_recognition.h"


// 计算两点之间的距离
double distance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}


//给一个矩形点的数组，通过看边长找到我要的点，封装，很神奇吧
vector<Point2f> getPoints(Point2f vertices[4]) {
    // // 找到左下角的点作为参考点
    // Point2f refPoint = vertices[0];
    // for (int i = 1; i < 4; ++i) {
    //     if (vertices[i].y > refPoint.y || (vertices[i].y == refPoint.y && vertices[i].x < refPoint.x)) {
    //         refPoint = vertices[i];
    //     }
    // }

    // 初始化前两个最大 y 值的点
    Point2f maxYPoint1 = vertices[0];
    Point2f maxYPoint2 = vertices[1];

    // 确保 maxYPoint1 是最大的 y 值点
    if (maxYPoint2.y > maxYPoint1.y || (maxYPoint2.y == maxYPoint1.y && maxYPoint2.x > maxYPoint1.x)) {
        swap(maxYPoint1, maxYPoint2);
    }

    // 遍历剩下的点，更新最大的两个 y 值点
    for (int i = 2; i < 4; ++i) {
        if (vertices[i].y > maxYPoint1.y || (vertices[i].y == maxYPoint1.y && vertices[i].x > maxYPoint1.x)) {
            maxYPoint2 = maxYPoint1;
            maxYPoint1 = vertices[i];
        } else if (vertices[i].y > maxYPoint2.y || (vertices[i].y == maxYPoint2.y && vertices[i].x > maxYPoint2.x)) {
            maxYPoint2 = vertices[i];
        }
    }

    // 在这两个点中选择 x 值最大的点作为 refPoint
    Point2f refPoint = (maxYPoint1.x >= maxYPoint2.x) ? maxYPoint1 : maxYPoint2;


    // 计算其他点到参考点的距离
    vector<pair<float, Point2f>> distances;
    for (int i = 0; i < 4; ++i) {
        distances.push_back(make_pair(distance(refPoint, vertices[i]), vertices[i]));
    }

    // 根据距离排序
    sort(distances.begin(), distances.end(), [](pair<float, Point2f> a, pair<float, Point2f> b) {
        return a.first < b.first;
    });

    // 返回排序后的点
    vector<Point2f> result;
    for (int i = 0; i < 4; ++i) {
        result.push_back(distances[i].second);
    }

    return result;
}


Mat image_processing(Mat frame) {

    Mat hsv;
    cvtColor( frame, hsv, COLOR_BGR2HSV );

    // Scalar lower_white(0, 0, 200);
    // cv::Scalar upper_white(180, 30, 255);
    Scalar lower_white(0, 0, 200);
    cv::Scalar upper_white(20, 30, 255);

    cv::Mat mask;
    cv::inRange(hsv, lower_white, upper_white, mask);

    Mat result1;
    cv::bitwise_and(frame, frame, result1, mask);

    // 转换为灰度图像
    Mat grayResult;
    cvtColor(result1, grayResult, COLOR_BGR2GRAY);

    // imshow("result0", grayResult);
    // waitKey(0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // 执行开运算
    cv::Mat image1;
    cv::morphologyEx(grayResult, image1, cv::MORPH_OPEN, kernel);

    // 执行闭运算
    cv::Mat image2;
    cv::morphologyEx(image1, image2, cv::MORPH_CLOSE, kernel);
    return image2;




}

cv::Mat colorRecognition(cv::Mat src,vector<Point2f> &object2d_point) {
    Mat image2;
    image2 = image_processing(src);
    // imshow("result1", image2);
    // waitKey(0);


    Mat image3;
    vector<vector<Point>> contours;
    findContours(image2, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 按轮廓面积进行排序
    std::sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
        return contourArea(a) > contourArea(b);
    });

    vector<RotatedRect> validRects;

    // 筛选有效矩形
    for (const auto& contour : contours) {
        if (contour.size() >= 5) { // 确保轮廓足够复杂
            RotatedRect rect = minAreaRect(contour);
            float width = rect.size.width;
            float height = rect.size.height;

            // 确保长宽比至少为5:1
            if ((width / height >= 4.0) || (height / width >= 4.0)) {
                 // cout<<width / height <<"这里这里这里是长宽比"<<endl;

                validRects.push_back(rect);
                // cout<<validRects.size()<<"数量"<<endl;
            }else {
                // cout<<width / height <<"不够的"<<endl;
            }
        }
    }

    if (validRects.size() > 1) {
        // 获取最大的两个轮廓
        vector<Point> max_contour1 = contours[0];
        vector<Point> max_contour2 = contours[1];


        // 获取最大的两个轮廓
         max_contour1 = contours[0];
        max_contour2 = contours[1];
        // cout<<max_contour1.size()<<"这里这里这里是轮廓的大小"<<endl;
        // cout<<max_contour2.size()<<"这里这里这里是轮廓的大小"<<endl;
        RotatedRect rect1 = minAreaRect(max_contour1);
        RotatedRect rect2 = minAreaRect(max_contour2);



        Point2f vertices1[4];
        Point2f vertices2[4];
        rect1.points(vertices1);
        rect2.points(vertices2);
        vector<Point2f> result1 = getPoints(vertices1);
        vector<Point2f> result2 = getPoints(vertices2);
        //四个关键点
        Point2f key_points[4]={result1[0],result1[2],result2[0],result2[2]};
        object2d_point.push_back(result1[0]);
        object2d_point.push_back(result1[2]);
        object2d_point.push_back(result2[0]);
        object2d_point.push_back(result2[2]);


        // 找到最远的一对点
        float max_dist = 0;
        int point1 = 0, point2 = 1;

        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                float dist = distance(key_points[i], key_points[j]);
                if (dist > max_dist) {
                    max_dist = dist;
                    point1 = i;
                    point2 = j;
                }
            }
        }

        // 另一个对角线的两个点
        vector<int> remaining_points;
        for (int i = 0; i < 4; ++i) {
            if (i != point1 && i != point2) {
                remaining_points.push_back(i);
            }
        }

        Point2f p0 = key_points[point1];
        Point2f p1 = key_points[point2];
        Point2f p2 = key_points[remaining_points[0]];
        Point2f p3 = key_points[remaining_points[1]];



        //中心点
        Point2f p4 = cv::Point((p0.x+p1.x+p2.x+p3.x)/4, (p0.y+p1.y+p2.y+p3.y)/4);

        circle(src, p4, 5, cv::Scalar(255, 0, 0), 5);



        //画点
        int thickness= 1.5;
        Scalar red1 =(0,255,255);
        int fontScale = 1;
        putText(src, "0", p0, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,255), thickness);
        putText(src, "1", p1, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,255), thickness);
        putText(src, "2", p2, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,255), thickness);
        putText(src, "3", p3, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,255), thickness);
        //画线
        line(src, p0, p1, Scalar(255, 0, 0), 2); // 从 p0 到 p2 的对角线
        line(src, p2, p3, Scalar(255, 0, 0), 2); // 从 p1 到 p3 的对角线


        //画出最大的两个轮廓

        Rect rect_1 =rect1.boundingRect();
        Rect rect_2 =rect2.boundingRect();

        cv::rectangle(src, rect_1, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(src, rect_2, cv::Scalar(0, 255, 0), 2);
        Scalar color(0, 255, 0); // 绿色
        for (int i = 0; i < 4; i++) {
            line(src, vertices1[i], vertices1[(i+1)%4], color, 2); // 连接点
            circle(src, vertices1[i], 5, color, -1); // 绘制点
        }
        for (int i = 0; i < 4; i++) {
            line(src, vertices2[i], vertices2[(i+1)%4], color, 2); // 连接点
            circle(src, vertices2[i], 5, color, -1); // 绘制点
        }

        //



        return src;
    }
    else {
        return src;
    }
}