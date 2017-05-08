#include "feature_extractor.h"

int FeatureTracker::n_id = 0;
template <typename T>
void reduceVector(vector<T> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

template <typename T>
void reduceVectortoNew(vector<T> &v, vector<uchar> status, vector<T> &v_out)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
        else
        	v_out.push_back(v[i]);
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::addPoints()
{
	printf("add %d\n", n_pts.size());
   	for(int i = 0; i < n_pts.size(); i++)
    {
        cur_pts.push_back(n_pts[i]);
        ids.push_back(-1);
        track_cnt.push_back(1);
        cur_des.push_back(n_des[i]);
    }
    printf("add end\n");
}

bool FeatureTracker::inAera(cv::Point2f pt, cv::Point2f center, float area_size)
{
    if(center.x < 0 || center.x > COL || center.y < 0 || center.y > ROW)
        return false;
    if(pt.x > center.x - area_size && pt.x < center.x + area_size &&
       pt.y > center.y - area_size && pt.y < center.y + area_size)
        return true;
    else
        return false;
}

void FeatureTracker::rejectWithF()
{
    if (pre_pts.size() >= 8)
    {
        vector<uchar> status;
        printf("reject\n");
        cv::findFundamentalMat(cur_pts, pre_pts, cv::FM_RANSAC, 10.0, 0.98, status);
        printf("reject finish\n");
        reduceVector(pre_pts, status);
        reduceVector(pre_des, status);
        reduceVectortoNew(cur_pts, status, n_pts);
        reduceVectortoNew(cur_des, status, n_des);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        printf("loop_match ransac old %d, cur %d\n", pre_pts.size(), cur_pts.size());
    }
}
/**
** search matches by guide descriptor match
**
**/

void FeatureTracker::matchByDes(const std::vector<cv::KeyPoint> &cur_keys, std::vector<BRIEF::bitset> &new_cur_des)
{
    printf("loop_match before old %d %d, cur %d %d\n", pre_des.size(), pre_pts.size(), cur_des.size(), cur_keys.size());
    cur_pts.clear();
    n_pts.clear();
	n_des.clear();
    vector<Point2f> pre_pts_new;
    vector<BRIEF::bitset> pre_des_new;
    vector<BRIEF::bitset> cur_des_new;
    std::vector<int> dis_cur_old;
    std::vector<uchar> status;

    vector<int> ids_new, track_cnt_new;
    for(int i = 0; i < cur_des.size(); i++)
    {
        int bestDist = 256;
        int bestIndex = -1;
        for(int j = 0; j < pre_des.size(); j++)
        {
        	if(!inAera(pre_pts[j], cur_keys[i].pt, 100))
        		continue;
        	//cout << "cur: " << cur_keys[j].pt << endl;
            int dis = HammingDis(pre_des[j], cur_des[i]);
            if(dis < bestDist)
            {
                bestDist = dis;
                bestIndex = j;
            }
        }
        if(bestDist < 256)
        {
            pre_pts_new.push_back(pre_pts[bestIndex]);
            pre_des_new.push_back(pre_des[bestIndex]);
            cur_pts.push_back(cur_keys[i].pt);
            cur_des_new.push_back(cur_des[i]);
            ids_new.push_back(ids[i]);
            track_cnt_new.push_back(track_cnt[i]);
        }
        else
        {
        	n_pts.push_back(cur_keys[i].pt);
        	n_des.push_back(cur_des[i]);
        }
    }
    pre_pts = pre_pts_new;
    pre_des = pre_des_new;
    cur_des = cur_des_new;
    track_cnt = track_cnt_new;
    ids = ids_new;
    printf("loop_match after old %d %d, cur %d %d %d %d\n", pre_des.size(), pre_pts.size(), cur_des.size(), cur_pts.size(),track_cnt.size(), ids.size());
}

void FeatureTracker::readImage(const cv::Mat &_img, cv::Mat &result)
{
    cv::Mat img = _img;
    cout << img.size()<< endl;
    result = img;
    if (cur_img.empty())
    {
        pre_img = cur_img = img;
    }
    else
    {
        cur_img = img;
    }

    cur_pts.clear();
    cur_des.clear();

	vector<cv::KeyPoint> cur_keypoints;
 
    std::string PATTERN_FILE = "/home/peiliang/catkin_ws/src/sfm/src/brief_pattern.yml";
    const char *BRIEF_PATTERN_FILE = PATTERN_FILE.c_str();
    BriefExtractor extractor(BRIEF_PATTERN_FILE);
    extractor(cur_img, cur_keypoints, cur_des);

    for(int i = 0; i < cur_keypoints.size(); i++)
    	cv::circle(result, cur_keypoints[i].pt, 20, cv::Scalar(0), 5);
  
    if (pre_pts.size() > 0)
    {
    	printf("match\n");
    	std::vector<BRIEF::bitset> new_cur_des;
    	matchByDes(cur_keypoints, new_cur_des);
    	assert(cur_pts.size() == pre_pts.size());   	
    }
    else
    {
    	for(int i = 0; i < cur_keypoints.size(); i++)
    		cur_pts.push_back(cur_keypoints[i].pt);
    	n_pts = cur_pts;
    	n_des = cur_des;
    	cur_pts.clear();
    	cur_des.clear();
    	printf("first time extract %d %d\n", n_pts.size(), n_des.size());
    }
    cv::Mat match_img;
    cv::vconcat(pre_img, cur_img, match_img);
    cv::Mat loop_match_img1,loop_match_img2;

    //rejectWithF();
    for (int i = 0; i< (int)pre_pts.size(); i++)
    {	
        cv::Point2f cur_pt = cur_pts[i];
        cur_pt.y += ROW;
        cv::line(match_img, pre_pts[i], cur_pt, cvScalar(0), 7, 8, 0);
    }
    result = match_img;

    for (auto &n : track_cnt)
        n++;

    addPoints();
    printf("new point size %d\n", n_pts.size());

    pre_img = cur_img;
    pre_pts = cur_pts;
    pre_des = cur_des;

    image_msg.clear();
    int num_new = 0;
        
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        completed |= updateID(i);
        if (!completed)
            break;
    }
    cout << "update----------------------" << endl;
    for(int i = 0; i<ids.size(); i++)
    {
        double x = (cur_pts[i].x - U0)/FX;
        double y = (cur_pts[i].y - V0)/FY;
        double z = 1.0;
        image_msg[(ids[i])] = (Vector3d(x, y, z));  
        cout << ids[i] << ": " << x<< ", "<< y<< endl;
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}
int FeatureTracker::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}