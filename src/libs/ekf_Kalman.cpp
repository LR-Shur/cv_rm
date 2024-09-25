
#include<ekf_Kalman.h>

EKFTracker::EKFTracker(double dt, const cv::Mat& processNoiseCov, const cv::Mat& measurementNoiseCov, const cv::Mat& errorCovPost)
: dt(dt), processNoiseCov(processNoiseCov), measurementNoiseCov(measurementNoiseCov), errorCovPost(errorCovPost) {
    // 初始化状态转移矩阵 F
    transitionMatrix = (cv::Mat_<double>(9, 9) <<
   1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0,  // tx的状态转移
   0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0,  // ty的状态转移
   0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt,  // tz的状态转移
   0, 0, 0, 1, 0, 0, dt, 0, 0,         // vx的状态转移
   0, 0, 0, 0, 1, 0, 0, dt, 0,         // vy的状态转移
   0, 0, 0, 0, 0, 1, 0, 0, dt,         // vz的状态转移
   0, 0, 0, 0, 0, 0, 1, 0, 0,         // ax的状态转移（假设加速度不变）
   0, 0, 0, 0, 0, 0, 0, 1, 0,         // ay的状态转移（假设加速度不变）
   0, 0, 0, 0, 0, 0, 0, 0, 1          // az的状态转移（假设加速度不变）
);
    // 初始化观测矩阵 H, 只观测位置 tx, ty, tz
    measurementMatrix =(cv::Mat_<double>(9, 9) <<
   1, 0, 0, 0, 0, 0, 0, 0, 0,   // 观测到 tx
   0, 1, 0, 0, 0, 0, 0, 0, 0,   // 观测到 ty
   0, 0, 1, 0, 0, 0, 0, 0, 0,  // 观测到 tz
   0, 0, 0, 1, 0, 0, 0, 0, 0,   // 观测到 vx
   0, 0, 0, 0, 1, 0, 0, 0, 0,   // 观测到 vy
   0, 0, 0, 0, 0, 1, 0, 0, 0 , // 观测到 vz
   0, 0, 0, 0, 0, 0, 1, 0, 0 ,
   0, 0, 0, 0, 0, 0, 0, 1, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 1



);
}

void EKFTracker::init(const cv::Mat& initState) {
    state = initState.clone();  // 初始化状态向量
    previousState = initState.clone(); // 保存初始状态
}

void EKFTracker::predict() {
    // 预测状态：x_k = F * x_{k-1}
    // cout<<transitionMatrix.size()<<endl;
    // cout<<state.size()<<endl;
    state = transitionMatrix * state;
    // 协方差矩阵预测 P'_t = F * P_{t-1} * F^T + Q
    errorCovPost = transitionMatrix * errorCovPost * transitionMatrix.t() + processNoiseCov;
}

void EKFTracker::update(const cv::Mat& measurement) {
    // 计算卡尔曼增益 K_t
    // std::cout << "processNoiseCov: " << processNoiseCov.size() << std::endl;
    // std::cout << "measurementNoiseCov: " << measurementNoiseCov.size()  << std::endl;
    // std::cout << "errorCovPost: " << errorCovPost.size()  << std::endl;
    // std::cout << "measurementMatrix: " << measurementMatrix.size()  << std::endl;
    // std::cout << "state: " << state.size()  << std::endl;
    // std::cout << "measurement: " << measurement.size()  << std::endl;


    cv::Mat temp = measurementMatrix * errorCovPost * measurementMatrix.t() + measurementNoiseCov;
    cv::Mat kalmanGain = errorCovPost * measurementMatrix.t() * temp.inv();

    // 更新状态 x_t = x'_t + K_t * (z_t - H * x'_t)
    cv::Mat innovation = measurement - measurementMatrix * state; // z_t - H * x'_t
    state = state + kalmanGain * innovation;

    // 更新协方差矩阵 P_t = (I - K_t * H) * P_t
    cv::Mat identity = cv::Mat::eye(errorCovPost.size(), errorCovPost.type());
    //  std::cout << "measurementMatrix: " << measurementMatrix.size()  << std::endl;
    // std::cout << "errorCovPost: " << errorCovPost.size()  << std::endl;
    // std::cout << "kalmanGain: " << kalmanGain.size()  << std::endl;
    // std::cout << "identity: " << identity.size()  << std::endl;

    errorCovPost = (identity - kalmanGain * measurementMatrix) * errorCovPost;
}

cv::Mat EKFTracker::getState() const {
    return state.clone();  // 返回当前状态向量
}
