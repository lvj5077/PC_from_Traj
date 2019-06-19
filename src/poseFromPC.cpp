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
    // cout << "have " << desc1.rows<<endl;
    // cout << "found " << desc2.rows<<endl;
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


    if (type == "bf") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }


    // int k = 2; 
    // double sum_dis = 0;     
    // double dis_ratio = 0.5; 

    // cv::flann::Index* mpFlannIndex = new cv::flann::Index(desc2, cv::flann::KDTreeIndexParams()); 

    // int num_features = desc1.rows; 
    // cv::Mat indices(num_features, k, CV_32S); 
    // cv::Mat dists(num_features, k, CV_32F); 
    // cv::Mat relevantDescriptors = desc1.clone(); 

    // mpFlannIndex->knnSearch(relevantDescriptors, indices, dists, k, flann::SearchParams(16) ); 

    // int* indices_ptr = indices.ptr<int>(0); 
    // float* dists_ptr = dists.ptr<float>(0); 
    // cv::DMatch m;
    // set<int> train_ids; 
    // for(int i=0; i<indices.rows; i++){
    //     float dis_factor = dists_ptr[i*2] / dists_ptr[i*2+1]; 
    //     if(dis_factor < dis_ratio ){
    //         int train_id = indices_ptr[i*2]; 
    //         if(train_ids.count(train_id) > 0) { // already add this feature 
    //             // TODO: select the best matched pair 
    //             continue; 
    //         }
    //         // add this match pair  
    //         m.trainIdx = train_id; 
    //         m.queryIdx = i; 
    //         m.distance = dis_factor;
    //         matches.push_back(m);
    //         train_ids.insert(train_id); 
    //     }
    // }

}


int main(int argc, char** argv)
{
    cout<<endl<<"Program Started!"<<endl;
    cout <<"~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    string ParameterPath = "/Users/lingqiujin/work/PC_from_Traj/parameters.txt";
    if(argc >=2){
    	ParameterPath = argv[1];
    }

    ParameterReader pd(ParameterPath);

    string image_Path       = pd.getData( "image_Path" );
    string trajectory_Path  = pd.getData( "trajectory_Path" );
    string timestamp_Path   = pd.getData( "timestamp_Path" );
    string output_Path      = pd.getData( "output_Path" );

    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx   =atof(pd.getData( "camera.cx" ).c_str());
    camera.cy   =atof(pd.getData( "camera.cy" ).c_str());
    camera.fx   =atof(pd.getData( "camera.fx" ).c_str());
    camera.fy   =atof(pd.getData( "camera.fy" ).c_str());
    camera.scale=atof(pd.getData( "camera.scale" ).c_str());

    double gridsize = atof( pd.getData("voxel_grid").c_str() );

    int space  = atoi( pd.getData( "space" ).c_str() );

    // Matrix4f Tlast;// = Eigen::Isometry3d::Identity();
    Matrix4f Tnow;
    Tnow << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    // Matrix4f Tm;

    Matrix4f Tfirst = Tnow;

    string timestmp, bufStr;
    double buffer, x,y,z,qx,qy,qz,qw;

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;



    ifstream fAssociation(timestamp_Path.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
          stringstream ss;
          ss << s;
          double t;
          string sRGB, sD;
          ss >> t;
          ss >> sRGB;
          vstrImageFilenamesRGB.push_back(sRGB);
          ss >> t;
          ss >> sD;
          vstrImageFilenamesD.push_back(sD);
        }
    }

    FRAME frame1, frame2;
    frame1.rgb = cv::imread( image_Path+vstrImageFilenamesRGB[1] );
    frame1.depth = cv::imread( image_Path+vstrImageFilenamesD[1], -1);

    // PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    Mat desc; 
            
    cv::Mat image;
    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
    Ptr<Feature2D> sift = SIFT::create();

    Mat brief_descriptors;
    vector<cv::KeyPoint> keypoints;
    const int fast_th = 20; // corner detector response threshold

    cvtColor( frame1.rgb.clone(), image, CV_BGR2GRAY );

    bool useFast = true;
    if(useFast){
        tic();
        cv::FAST(image, keypoints, fast_th, true);
        brief->compute(image, keypoints, brief_descriptors); 
        // sift->detect ( image,keypoints );
        // sift->compute ( image, keypoints, brief_descriptors );
        // sift->detectAndCompute(image, Mat(), keypoints, brief_descriptors);
        toc();
    }
    else
    {
        tic();
        vector<cv::Point2f> tmp_pts;
        cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
        for(int i = 0; i < (int)tmp_pts.size(); i++)
        {
            cv::KeyPoint key;
            key.pt = tmp_pts[i];
            keypoints.push_back(key);
        }
        brief->compute(image, keypoints, brief_descriptors); 
        toc();
    } 



    cv::Mat image1, image2;
    cv::Mat depth1, depth2;
    vector<cv::KeyPoint> keypoints1,keypoints2;
    Mat descriptors1,descriptors2;

    vector<cv::KeyPoint> keypointsAll;
    Mat descriptorsAll;
    vector<cv::Point3f> pts_objAll;

    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    image2 = image;
    depth2 = frame1.depth.clone();
    keypoints2 = keypoints;
    descriptors2 = brief_descriptors;


    ifstream input( trajectory_Path.c_str() );
    
    PointCloud::Ptr output (new PointCloud());
    // output = cloud1;

    int count = 0;
    int res = 500;

    int lastCount = 1;

    Mat currT = cv::Mat::eye(4,4,CV_64F);
    // pcl::visualization::CloudViewer viewer( "viewer" );
    while(!input.eof() )
    {
        res++;
        count ++;
        
        input>>buffer>>x>>y>>z>>qx>>qy>>qz>>qw;
        
        if (res<space){
            continue;
        }
        else
        {
            res = 0;    
            cout << count <<endl;

        
            // Tlast = Tnow;
            Tnow(0,3) = x; 
            Tnow(1,3) = y; 
            Tnow(2,3) = z;

            Tnow(0,0) = 1 - 2*(qy*qy +qz*qz); 
            Tnow(0,1) = 2*(qx*qy-qz*qw); 
            Tnow(0,2) = 2*(qx*qz+qy*qw);  

            Tnow(1,0) = 2*(qx*qy+qz*qw); 
            Tnow(1,1) = 1 - 2*(qx*qx +qz*qz); 
            Tnow(1,2) = 2*(qz*qy-qx*qw); 

            Tnow(2,0) = 2*(qx*qz-qy*qw);
            Tnow(2,1) = 2*(qz*qy+qx*qw); 
            Tnow(2,2) = 1 - 2*(qy*qy +qx*qx);  

            //Tm = Tlast.colPivHouseholderQr().solve(Tnow);

            // Tm = Tfirst.colPivHouseholderQr().solve(Tnow);
            // cout << "Tnow"<<endl<<Tnow<<endl;
            cout << Tnow.matrix() <<endl<<endl;


            string ssRGB = image_Path+vstrImageFilenamesRGB[count];
            cout << ssRGB <<endl;
            frame2.rgb = cv::imread( ssRGB );
            
            string ssDepth = image_Path+vstrImageFilenamesD[count];
            // cout << ssDepth <<endl;
            frame2.depth = cv::imread( ssDepth,-1);

            cvtColor( frame2.rgb.clone(), image, CV_BGR2GRAY );
            
            if(useFast){
                tic();
                cv::FAST(image, keypoints, fast_th, true);
                brief->compute(image, keypoints, brief_descriptors); 

                // sift->detect ( image,keypoints );
                // sift->compute ( image, keypoints, brief_descriptors );

                // sift->detectAndCompute(image, Mat(), keypoints, brief_descriptors);
                toc();
            }
            else
            {
                tic();
                vector<cv::Point2f> tmp_pts;
                cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
                for(int i = 0; i < (int)tmp_pts.size(); i++)
                {
                    cv::KeyPoint key;
                    key.pt = tmp_pts[i];
                    keypoints.push_back(key);
                }
                brief->compute(image, keypoints, brief_descriptors); 
                toc();
            }
            
            // cout << keypoints.size()<<endl;



            // vector<cv::KeyPoint> keypointsAll;
            // Mat descriptorsAll;
            // vector<cv::Point3f> pts_objAll;
            // feature融合 对匹配上的点做3d稀疏点云拓展
            for (size_t i=0; i<keypoints.size(); i++)
            {
                cv::Point2f p = keypoints[i].pt;

                ushort d = frame2.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
                if (d == 0)
                    continue;

                // 将(u,v,d)转成(x,y,z)
                cv::Point3f pt ( p.x, p.y, d );
                cv::Point3f pd1 = point2dTo3d( pt,  camera);

                cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);

                MatrixXf matA(4, 1);
                MatrixXf matB(4, 1);
                matA << pd1.x, pd1.y, pd1.z, 1;
                matB = Tnow.inverse()*matA ;
                cv::Point3f projPd(matB(0,0), matB(1,0),matB(2,0));

                keypointsAll.push_back(keypoints[i]);
                pts_objAll.push_back( projPd );
            }
            // sift->compute( image, keypointsAll, descriptorsAll );
            brief->compute(image, keypointsAll, descriptorsAll); 



            image1 = image2.clone();
            image2 = image.clone();
            depth1 = depth2.clone();
            depth2 = frame2.depth.clone();
            keypoints1 = keypoints2;
            keypoints2 = keypoints;
            descriptors1 = descriptors2.clone();
            descriptors2 = brief_descriptors.clone();

            pts_img.clear();
            pts_obj.clear();

            vector<DMatch> goodMatches;
            match("bf", descriptors1, descriptors2, goodMatches);
            // cout<<"goodMatches: "<<goodMatches.size()<<endl;
            for (size_t i=0; i<goodMatches.size(); i++)
            {
                // query 是第一个, train 是第二个
                cv::Point2f p = keypoints1[goodMatches[i].queryIdx].pt;
                // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
                ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
                if (d == 0)
                    continue;
                pts_img.push_back( cv::Point2f( keypoints2[goodMatches[i].trainIdx].pt ) );

                // 将(u,v,d)转成(x,y,z)
                cv::Point3f pt ( p.x, p.y, d );
                cv::Point3f pd = point2dTo3d( pt,  camera);
                pts_obj.push_back( pd );
            }

            double camera_matrix_data[3][3] = {
                {camera.fx, 0, camera.cx},
                {0, camera.fy, camera.cy},
                {0, 0, 1}
            };

            // 构建相机矩阵
            cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
            cv::Mat rvec, tvec, inliers;
            // 求解pnp
            // cout<<"pts_obj: "<<pts_obj.size()<<endl;
            // cout<<"pts_img: "<<pts_img.size()<<endl;
            cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );

            // cout<<"inliers: "<<inliers.rows<<endl;
            // cout<<"R="<<rvec<<endl;
            // cout<<"t="<<tvec<<endl;
            cv::Mat mat_T = cv::Mat::eye(4,4,CV_64F);
            Mat mat_r;
            Rodrigues(rvec, mat_r);
            mat_r.copyTo(mat_T(cv::Rect(0, 0, 3, 3)));
            tvec.copyTo(mat_T(cv::Rect(3, 0, 1, 3)));
            currT = currT*mat_T.inv();
            cout<<"T="<<endl<<currT<<endl;

            // 画出inliers匹配 
            cv::Mat imgMatches;
            vector< cv::DMatch > matchesShow;
            for (size_t i=0; i<inliers.rows; i++)
            {
                matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
            }

            // cv::drawMatches( image1, keypoints1, image2, keypoints2, matchesShow, imgMatches );
            // cv::imshow( "inlier matches", imgMatches );
            // cv::waitKey( 0 );





            pts_img.clear();
            pts_obj.clear();
            vector<DMatch> gloabalMatches;
            match("bf", descriptorsAll, descriptors2, gloabalMatches);
            cout<<"gloabalMatches: "<<gloabalMatches.size()<<endl;


            vector<cv::KeyPoint> keypoints2Show;
            for (size_t i=0; i<gloabalMatches.size(); i++)
            {
                pts_img.push_back( cv::Point2f( keypoints2[gloabalMatches[i].trainIdx].pt ) );

                cv::Point3f pd = pts_objAll[gloabalMatches[i].queryIdx];
                pts_obj.push_back( pd );
                keypoints2Show.push_back( keypoints2[gloabalMatches[i].trainIdx] );
            }
            // cout<<"pts_obj: "<<pts_obj.size()<<endl;
            // cout<<"pts_img: "<<pts_img.size()<<endl;
            cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );

            cout<<"inliers: "<<inliers.rows<<endl;


            if (count>1){
                for (size_t i=0; i<inliers.rows; i++)
                {
                    int idx = inliers.ptr<int>(i)[0];
                    keypointsAll[idx] = keypointsAll.back();
                    keypointsAll.pop_back();
                    
                    pts_objAll[idx] = pts_objAll.back();
                    pts_objAll.pop_back();
                }
            }





            mat_T = cv::Mat::eye(4,4,CV_64F);
            Rodrigues(rvec, mat_r);
            mat_r.copyTo(mat_T(cv::Rect(0, 0, 3, 3)));
            tvec.copyTo(mat_T(cv::Rect(3, 0, 1, 3)));
            cout<<"T_globalMatch="<<endl<<mat_T<<endl;

            cv::Mat imgShow;
            cv::drawKeypoints( image2, keypoints2Show, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
            cv::imshow( "keypoints", imgShow );
            cv::waitKey(0); 

            // 点云融合 可视化 没啥用
            PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );
            pcl::transformPointCloud( *cloud2, *cloud2, Tnow.matrix() );
            *output += *cloud2;
            static pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize( gridsize, gridsize, gridsize );
            voxel.setInputCloud( output );
            voxel.filter( *output );
            cloud2->points.clear();

            cout << "saved " << keypointsAll.size()<<endl;
            cout<<"====================================================================="<<endl;






        }
    }

    
    
    pcl::io::savePCDFile(output_Path.c_str(), *output);
    
    
    // while( !viewer.wasStopped() )
    // {
        
    // }
    return 0;
}
