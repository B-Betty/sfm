#ifndef BA_hpp
#define BA_hpp

#include <stdio.h>
#include "feature_extractor.h"
#include "feature_manager.h"
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include "initial_sfm.h"
#include "solve_5pts.h"

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, Vector3d>& _points, double _t):points{_points},t{_t},is_key_frame{true}
    {
    };
    map<int, Vector3d> points;
    double t;
    Matrix3d R;
    Vector3d T;
    bool is_key_frame;
};



class VINS
{
  public:
    
    VINS();
    
    FeatureManager f_manager;
    MotionEstimator m_estimator;
    int frame_count;
    
    Matrix3d ric;
    Vector3d tic;
    bool Rs_init;
    Vector3d Ps[10 * (WINDOW_SIZE + 1)];
    Vector3d Vs[10 * (WINDOW_SIZE + 1)];
    Matrix3d Rs[10 * (WINDOW_SIZE + 1)];
    cv::Mat image_buf[10* (WINDOW_SIZE + 1)];
    
    vector<Vector3f> point_cloud;
    
    int feature_num;
    bool propagate_init_ok;
    
    vector<double> dt_buf[10 * (WINDOW_SIZE + 1)];
    double Headers[10 * (WINDOW_SIZE + 1)];
    Vector3d g;
    
    vector<Vector3d> init_point_cloud;
    vector<Vector3d> margin_map;
    vector<Vector3d> key_poses;
    vector<Vector3d> init_poses;
    vector<Vector3d> init_velocity;
    vector<Matrix3d> init_rotation;
    vector<Eigen::Quaterniond> init_quaternion;
    double initial_timestamp;
    vector<Vector3d> gt_init_poses;
    Vector3d init_P;
    Vector3d init_V;
    Matrix3d init_R;
    
    
    //for initialization
    map<double, ImageFrame> all_image_frame;
    Matrix3d back_R0;
    Vector3d back_P0;
    //for falure detection
    double spatial_dis;
    bool failure_hand;
    bool is_failure;
    Vector3d ypr_imu;
    Vector3d Ps_his[WINDOW_SIZE];
    Matrix3d Rs_his[WINDOW_SIZE];
    double Headers_his[WINDOW_SIZE];
    bool need_recover;
    
    int parallax_num_view;
    int fail_times;
    double final_cost;
    double visual_cost;
    int visual_factor_num;
    
    void solve_ceres(int buf_num);
    void solveCalibration();
    void old2new();
    void new2old();
    void clearState();
    void setIMUModel();
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void processImage(map<int, Vector3d> &image_msg, double header, cv::Mat image, cv::Mat &debug_image);
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void changeState();
    bool solveInitial(cv::Mat &debug_image);
    bool relativePose(int camera_id, Matrix3d &relative_R, Vector3d &relative_T, int &l, cv::Mat &debug_image);
    bool visualInitialAlign();
    void failureDetection();
    void failureRecover();
    void update_loop_correction();
};
#endif /* VINS_hpp */