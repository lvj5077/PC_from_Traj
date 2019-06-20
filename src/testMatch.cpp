#include "slamBase.h"
#include "loadImage.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


#include <pcl/common/transforms.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


// Eigen !
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#include<iostream>

#include <fstream>
#include <string>

#include <stack>
#include <ctime>


using namespace std;
using namespace cv;
using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SIFT;
using namespace Eigen;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 100;

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}

void match(string type, Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    cout << "have desc1 " << desc1.rows<<endl;
    cout << "found desc2 " << desc2.rows<<endl;
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector< std::vector<DMatch> > knn_matches;
    // matcher->knnMatch( desc1, desc2, knn_matches, 2 );
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.7f;
    // std::vector<DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         matches.push_back(knn_matches[i][0]);
    //     }
    // }


    // if (type == "bf") {
    //     BFMatcher desc_matcher(cv::NORM_L2, true);
    //     desc_matcher.match(desc1, desc2, matches, Mat());
    // }
    // if (type == "knn") {
    //     BFMatcher desc_matcher(cv::NORM_L2, true);
    //     vector< vector<DMatch> > vmatches;
    //     desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
    //     for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
    //         if (!vmatches[i].size()) {
    //             continue;
    //         }
    //         matches.push_back(vmatches[i][0]);
    //     }
    // }
    // std::sort(matches.begin(), matches.end());
    // while (matches.front().distance * kDistanceCoef < matches.back().distance) {
    //     matches.pop_back();
    // }
    // while (matches.size() > kMaxMatchingSize) {
    //     matches.pop_back();
    // }


    int k = 2; 
    double sum_dis = 0;     
    double dis_ratio = 0.5; 

    cv::flann::Index* mpFlannIndex = new cv::flann::Index(desc2, cv::flann::KDTreeIndexParams()); 

    int num_features = desc1.rows; 
    cv::Mat indices(num_features, k, CV_32S); 
    cv::Mat dists(num_features, k, CV_32F); 
    cv::Mat relevantDescriptors = desc1.clone(); 

    mpFlannIndex->knnSearch(relevantDescriptors, indices, dists, k, flann::SearchParams(16) ); 

    int* indices_ptr = indices.ptr<int>(0); 
    float* dists_ptr = dists.ptr<float>(0); 
    cv::DMatch m;
    set<int> train_ids; 
    for(int i=0; i<indices.rows; i++){
        float dis_factor = dists_ptr[i*2] / dists_ptr[i*2+1]; 
        if(dis_factor < dis_ratio ){
            int train_id = indices_ptr[i*2]; 
            if(train_ids.count(train_id) > 0) { // already add this feature 
                // TODO: select the best matched pair 
                continue; 
            }
            // add this match pair  
            m.trainIdx = train_id; 
            m.queryIdx = i; 
            m.distance = dis_factor;
            matches.push_back(m);
            train_ids.insert(train_id); 
        }
    }

}


int main(int argc, char** argv)
{
    // 声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1 = cv::imread( "/Users/lingqiujin/Data/06_14_startPoint/color/1260.png");
    cv::Mat depth1 = cv::imread( "/Users/lingqiujin/Data/06_14_startPoint/depth/1260.png", -1);

    cv::Mat rgb2 = cv::imread( "/Users/lingqiujin/Data/testStart/02/color/230.png");
    cv::Mat depth2 = cv::imread( "/Users/lingqiujin/Data/testStart/02/depth/230.png", -1);

    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
    Ptr<Feature2D> sift = SIFT::create();

    cv::Mat gray1,gray2;
    cvtColor( rgb1, gray1, CV_BGR2GRAY );
    cvtColor( rgb2, gray2, CV_BGR2GRAY );
    vector< cv::KeyPoint > kp1, kp2; //关键点
    sift->detect( gray1,kp1 );
    sift->detect( gray2,kp2 );

    // // cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    
    // // 可视化， 显示关键点
    // cv::Mat imgShow;
    // cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    // cv::imshow( "keypoints", imgShow );
    // cv::imwrite( "./data/keypoints.png", imgShow );
    // cv::waitKey(0); //暂停等待一个按键
   
    // 计算描述子
    cv::Mat desp1, desp2;
    sift->compute( gray1, kp1, desp1 );
    sift->compute( gray2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches; 
    cv::BFMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    // cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    // cv::imshow( "matches", imgMatches );
    // cv::imwrite( "./data/matches.png", imgMatches );
    // cv::waitKey( 0 );

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    match("bf", desp1, desp2, goodMatches);

    // // 显示 good matches
    // cout<<"good matches="<<goodMatches.size()<<endl;
    // cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    // cv::imshow( "good matches", imgMatches );
    // cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数
    
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 319.5;
    C.cy = 239.5;
    C.fx = 531.577087;
    C.fy = 531.577148;

    C.scale = 1000.0;


    PointCloud::Ptr cloud1 = image2PointCloud( rgb1.setTo(cv::Scalar(0,255,0)), depth1, C );

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
    }


    C.fx = 530.562866;
    C.fy = 530.562866;
    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 5.0, 0.99, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( gray1, kp1, gray2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::waitKey( 0 );


    Mat mat_T = cv::Mat::eye(4,4,CV_64F);
    Mat mat_r;
    Rodrigues(rvec, mat_r);
    mat_r.copyTo(mat_T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(mat_T(cv::Rect(3, 0, 1, 3)));

    Eigen::Isometry3d T_eigen = cvTtoEigenT(mat_T);


    PointCloud::Ptr cloud2 = image2PointCloud( rgb2.setTo(cv::Scalar(0,0,255)), depth2, C );



    // pcl::transformPointCloud( *cloud2, *cloud2, T_eigen.inverse().matrix() );
    pcl::transformPointCloud( *cloud2, *cloud2, T_eigen.inverse().matrix() );
    *cloud1 += *cloud2;
    pcl::io::savePCDFile("/Users/lingqiujin/work/PC_from_Traj/match.pcd", *cloud1);



    return 0;
}
