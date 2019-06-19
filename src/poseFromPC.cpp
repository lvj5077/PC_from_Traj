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
using namespace Eigen;


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

    Matrix4f Tlast;// = Eigen::Isometry3d::Identity();
    Matrix4f Tnow;
    Tnow << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    Matrix4f Tm;

    Matrix4f Tfirst = Tnow;

    string timestmp, bufStr;
    double buffer, x,y,z,qx,qy,qz,qw;

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;


    // cout << "i am here1"<<endl;


// bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
//                             const std::vector<BRIEF::bitset> &descriptors_old,
//                             const std::vector<cv::KeyPoint> &keypoints_old,
//                             const std::vector<cv::KeyPoint> &keypoints_old_norm,
//                             cv::Point2f &best_match,
//                             cv::Point2f &best_match_norm)
// {
//     cv::Point2f best_pt;
//     int bestDist = 128;
//     int bestIndex = -1;
//     for(int i = 0; i < (int)descriptors_old.size(); i++)
//     {

//         int dis = HammingDis(window_descriptor, descriptors_old[i]);
//         if(dis < bestDist)
//         {
//             bestDist = dis;
//             bestIndex = i;
//         }
//     }
//     //printf("best dist %d", bestDist);
//     if (bestIndex != -1 && bestDist < 80)
//     {
//       best_match = keypoints_old[bestIndex].pt;
//       best_match_norm = keypoints_old_norm[bestIndex].pt;
//       return true;
//     }
//     else
//       return false;
// }

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

    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    

    cv::Mat image = frame1.rgb.clone();
    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
    Mat brief_descriptors;
    vector<cv::KeyPoint> keypoints;

    // cout << "i am here"<<endl;
	const int fast_th = 20; // corner detector response threshold
	if(1){
        tic();
		cv::FAST(image, keypoints, fast_th, true);
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
        toc();
	}
    brief->compute(image, keypoints, brief_descriptors);   


    ifstream input( trajectory_Path.c_str() );
    
    PointCloud::Ptr output (new PointCloud());
    output = cloud1;

    int count = 1;
    int res = 1;


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

            

            Tlast = Tnow;
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

            Tm = Tfirst.colPivHouseholderQr().solve(Tnow);
            cout << Tm.matrix() <<endl<<endl;



            string ssRGB = image_Path+vstrImageFilenamesRGB[count];
            // cout << ssRGB <<endl;
            frame2.rgb = cv::imread( ssRGB );
            
            string ssDepth = image_Path+vstrImageFilenamesD[count];
            // cout << ssDepth <<endl;
            frame2.depth = cv::imread( ssDepth,-1);

            Mat src_gray;
            image = frame2.rgb.clone();
            cvtColor( image, src_gray, CV_BGR2GRAY );

            tic();
            cout << "Fast corner: "<<endl;
            cv::FAST(src_gray, keypoints, fast_th, true);
            toc();
            cout << keypoints.size()<<endl;

            cout << "~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            cout << "goodFeaturesToTrack: "<<endl;
            tic();
            vector<cv::Point2f> tmp_pts;
            cv::goodFeaturesToTrack(src_gray, tmp_pts, 500, 0.01, 10);
            for(int i = 0; i < (int)tmp_pts.size(); i++)
            {
                cv::KeyPoint key;
                key.pt = tmp_pts[i];
                keypoints.push_back(key);
            }
            toc();
            cout << keypoints.size()<<endl;


            PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );

            pcl::transformPointCloud( *cloud2, *cloud2, Tm.matrix() );

            static pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize( gridsize, gridsize, gridsize );
            voxel.setInputCloud( cloud2 );
            voxel.filter( *cloud2 );

            *output += *cloud2;
            cloud2->points.clear();

        }
        // viewer.showCloud( output );
    }

    
    
    pcl::io::savePCDFile(output_Path.c_str(), *output);
    
    
    // while( !viewer.wasStopped() )
    // {
        
    // }
    return 0;
}
