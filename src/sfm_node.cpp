#include <iostream>
#include <fstream>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "dirent.h"
#include <unistd.h>
#include <string>
#include <typeinfo>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Eigen>
#include <queue>
#include "feature_extractor.h"

//for rand()
#include <stdlib.h>
#include <time.h>
#include <numeric>
#include <time.h>

using namespace cv;
using namespace Eigen;
using namespace std;

ros::Publisher pub_img;
ros::Publisher pub_imu;

FeatureTracker feature_tracker;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sfm");
    ros::NodeHandle n("~");

    //publish iphone sensor data
    pub_img = n.advertise<sensor_msgs::Image>("/iphone/cam", 1000);
    pub_imu = n.advertise<sensor_msgs::Imu>("/iphone/imu", 1000);

    string data_dir;
    n.getParam("data_dir", data_dir);
    cout << "data dir: " << data_dir << endl;
    //get images from folder
    std::string inputDirectory = data_dir + "%01d.png";
    VideoCapture sequence(inputDirectory);
    if (!sequence.isOpened())
    {
      cerr << "Failed to open Image Sequence!\n" << endl;
      return 1;
    }
    //image timestamp file
    std::fstream imageTimeFile(data_dir + "image_time", std::ios_base::in);
    //imu data file
    std::fstream imuDataFile(data_dir + "imu_data", std::ios_base::in);
    Mat rawImage;
    namedWindow("track", CV_WINDOW_NORMAL);
    
    while(true)
    {
        sequence >> rawImage;
        if(rawImage.empty())
        {
            cout << "End of Sequence" << endl;
            break;
        }
        else
        {
            cv::Mat grayImage;
            cv::cvtColor(rawImage, grayImage, CV_BGRA2GRAY);
            cv::Mat resultImage;
            feature_tracker.readImage(grayImage, resultImage);
            imshow("track",resultImage);
            waitKey(-1);
            //sleep(0.3);
        }
    }
    return 0;
    //ros::spin();
}
