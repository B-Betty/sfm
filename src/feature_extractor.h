#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "DBoW/FBrief.h"
#include "DVision/DVision.h"
#include "tic_toc.h"
#include "solve_5pts.h"
#include "initial_sfm.h"

using namespace Eigen;
using namespace cv;
using namespace DBoW2;
using namespace std;
using namespace DVision;

#define U0 1520.69
#define V0 1006.81
#define FX 2759.48
#define FY 2764.16
#define COL 3072
#define ROW 2048
//#define USE_ORB true

template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

private:
  DVision::BRIEF m_brief;
};


class FeatureTracker
{
  public:
    FeatureTracker();

	void addPoints();

	void rejectWithF();

	void matchByDes(const std::vector<cv::KeyPoint> &cur_keys, std::vector<BRIEF::bitset> &new_cur_des);

	void readImage(const cv::Mat &_img, cv::Mat &result);

	bool updateID(unsigned int i);

	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

	bool inAera(cv::Point2f pt, cv::Point2f center, float area_size);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	MotionEstimator m_estimator;
    cv::Mat pre_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> pre_pts, cur_pts;

    cv::Mat pre_des_orb, cur_des_orb, n_des_orb;
#ifdef USE_ORB
    vector<cv::Mat> pre_des, cur_des, n_des;
#else
    vector<BRIEF::bitset> pre_des, cur_des, n_des;
#endif
    vector<int> ids;
    vector<int> track_cnt;

    static int n_id, img_cnt;
    map<int, Vector3d> image_msg;
};
