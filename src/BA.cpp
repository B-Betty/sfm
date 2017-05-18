#include "BA.h"


VINS::VINS()
:f_manager{}
{
    printf("init VINS begins\n");
    clearState();
}


void VINS::clearState()
{
    frame_count = 0;
    all_image_frame.clear();
    
    printf("clear state\n");
    for (int i = 0; i < 10 * (WINDOW_SIZE + 1); i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        dt_buf[i].clear();
    }
    
    f_manager.clearState();
    
}

void VINS::processImage(map<int, Vector3d> &image_msg, double header, cv::Mat image, cv::Mat &debug_image)
{
    printf("adding feature points %lu\n", image_msg.size());
    f_manager.addFeatureCheckParallax(frame_count, image_msg);
    printf("number of feature: %d %d\n", feature_num = f_manager.getFeatureCount(), frame_count);
    
    Headers[frame_count] = header;
    image_buf[frame_count] = image;

    ImageFrame imageframe(image_msg, header);
    all_image_frame.insert(make_pair(header, imageframe));
        
    if (frame_count == WINDOW_SIZE)
    {
        bool result = false;
        result = solveInitial(debug_image);
    }
    else
        frame_count++; 
}

bool VINS::solveInitial(cv::Mat &debug_image)
{
    printf("solve initial------------------------------------------\n");
    printf("PS %lf %lf %lf\n", Ps[0].x(),Ps[0].y(), Ps[0].z());
    // global sfm
    Quaterniond *Q = new Quaterniond[frame_count + 1];
    Vector3d *T = new Vector3d[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    {
        for (auto &it_per_id : f_manager.feature)
        {
            int imu_j = it_per_id.start_frame - 1;
            
            SFMFeature tmp_feature;
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                Vector3d pts_j = it_per_frame.point;
                tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
            }
            sfm_f.push_back(tmp_feature);
        }
        Matrix3d relative_R;
        Vector3d relative_T;
        int l = 1;
        if (!relativePose(0, relative_R, relative_T, l, debug_image))
        {
            printf("init solve 5pts between first frame and last frame failed\n");
            return false;
        }
        GlobalSFM sfm;
        if(!sfm.construct(frame_count + 1, Q, T, l,
                          relative_R, relative_T,
                          sfm_f, sfm_tracked_points))
        {
            printf("global SFM failed!");
            return false;
        }
    }
    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            cout << "key frame " << i << endl;
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * ric.transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);
        
        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            //cout << "feature id " << feature_id;
            //cout << " pts image_frame " << (i_p.second.head<2>() * 460 ).transpose() << endl;
            it = sfm_tracked_points.find(feature_id);
            if(it != sfm_tracked_points.end())
            {
                Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                
                Vector2d img_pts = id_pts.second.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                     0, 1, 0,
                     0, 0, 1);

        if(pts_3_vector.size() < 6 )
        {
            printf("init Not enough points for solve pnp !\n");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            printf("init solve pnp fail!\n");
            return false;
        }
        cv::Rodrigues(rvec, r);
        //cout << "r " << endl << r << endl;
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //cout << "R_pnp " << endl << R_pnp << endl;
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * ric.transpose();
        frame_it->second.T = T_pnp;
    }
    delete[] Q;
    delete[] T;
    
    printf("init PS after pnp %lf %lf %lf\n", Ps[0].x(),Ps[0].y(), Ps[0].z());   
    
}

bool VINS::relativePose(int camera_id, Matrix3d &relative_R, Vector3d &relative_T, int &l, cv::Mat &debug_image)
{
    printf("relative %d\n", l);
    for (int i = 8; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        printf("corrs between %d and %d are %d\n", i, WINDOW_SIZE, corres.size());
        if (corres.size() > 20)
        {
             printf("draw image\n");
             cv::vconcat(image_buf[i], image_buf[WINDOW_SIZE], debug_image);
             for (int i = 0; i< (int)corres.size(); i++)
             {   
                cv::Point2f pre_pt,cur_pt;
                pre_pt.x = FX * corres[i].first.x()/corres[i].first.z() + U0;
                pre_pt.y = FY * corres[i].first.y()/corres[i].first.z() + V0;

                cur_pt.x = FX * corres[i].second.x()/corres[i].second.z() + U0;
                cur_pt.y = FY * corres[i].second.y()/corres[i].second.z() + V0;
                cur_pt.y += ROW;
                cv::line(debug_image, pre_pt, cur_pt, cvScalar(0), 7, 8, 0);
            }
            ostringstream convert;
                convert << "/home/peiliang/catkin_ws/src/sfm/data/debug.jpg";
                cv::imwrite( convert.str().c_str(), debug_image);
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
                
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            parallax_num_view = average_parallax * 520;
            if(average_parallax * 520 < 30)
            {
                return false;
            }
            if(m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure\n", average_parallax * 520, l);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}
