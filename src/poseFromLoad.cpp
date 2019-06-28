#include "loadImage.h"
#include "slamBase.h"

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

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>
// using namespace ceres;
// using ceres::AutoDiffCostFunction;
// using ceres::CostFunction;
// using ceres::Problem;
// using ceres::Solve;
// using ceres::Solver;

using namespace std;
using namespace cv;
using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SIFT;
using namespace Eigen;

const double kDistanceCoef = 5.0;
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

void match(string type, Mat& desc2, Mat& desc1, vector<DMatch>& matches) {
    matches.clear();
    cout << "have saved " << desc2.rows<<endl;
    cout << "query " << desc1.rows<<endl;

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
}

//===================================================================================
struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d& observation)
        : observation(observation)
    {
    }

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }
};


void bundle_adjustment(
    Mat& intrinsic,
    Mat& extrinsics,
    vector<cv::Point2d> pts_img,
    vector<cv::Point3d> pts_obj
)
{
    ceres::Problem problem;

    // load extrinsics (rotations and motions)
    problem.AddParameterBlock(extrinsics.ptr<double>(), 6);



    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.

    for (int  point_idx= 0; point_idx < pts_img.size(); point_idx++)
    {
        Point2d observed = pts_img[point_idx];
        // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
    // fully trust my rgbd map
        problem.SetParameterBlockConstant( &(pts_obj[point_idx].x) );

        problem.AddResidualBlock(
            cost_function,
            loss_function,
            intrinsic.ptr<double>(),            // Intrinsic
            extrinsics.ptr<double>(),           // View Rotation and Translation
            &(pts_obj[point_idx].x)                        // Point in 3D space
        );
    }

    // Solve BA
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }
}

//===================================================================================




int main(int argc, char** argv)
{
    bool useFast = false;
    const int fast_th = 20;

    cout<<endl<<"Program Started!"<<endl;
    cout <<"~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    string ParameterPath = "/Users/jin/work/PC_from_Traj/parameters.txt";
    // if(argc >=2){
    // 	// ParameterPath = argv[1];
    //     useFast= true;
    // }

    useFast= true;
    
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

    Mat descriptorsAll;
    ifstream input( trajectory_Path.c_str() );
    FileStorage inDesp;
    inDesp.open("/Users/jin/work/PC_from_Traj/desp.yml", FileStorage::READ); 
    inDesp["descriptorsAll"]>> descriptorsAll; 
    inDesp.release();

    ifstream inputXYZ("/Users/jin/work/PC_from_Traj/pcXYZ.txt");

    vector<cv::Point3d> pts_objAll;
    pts_objAll.clear();
    while(!inputXYZ.eof())
    {
        cv::Point3d pd;
        inputXYZ>> pd.x >> pd.y >> pd.z;
        pts_objAll.push_back(pd);
    }
    pts_objAll.pop_back();

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("/Users/jin/work/PC_from_Traj/denseMap.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
/* ============================================================================== */

    // 第一个帧的三维点
    vector<cv::Point3d> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2d > pts_img;

    Mat rgb,depth,gray;
    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
    Mat brief_descriptors;
    vector<cv::KeyPoint> keypoints;

    cv::Mat rvec, tvec, inliers;

    // rgb = cv::imread( "/Users/jin/Data/06_14_startPoint/color/1311.png");
    // depth = cv::imread( "/Users/jin/Data/06_14_startPoint/depth/1311.png", -1);


    // Mat rgb = cv::imread( "/Users/jin/Data/testStart/02/color/230.png");
    // Mat depth = cv::imread( "/Users/jin/Data/testStart/02/depth/230.png", -1);


    rgb = cv::imread( "/Users/jin/Data/testStart/02/color/115.png");
    depth = cv::imread( "/Users/jin/Data/testStart/02/depth/115.png", -1);
    camera.fx = 530.562866;
    camera.fy = 530.562927;

    cvtColor( rgb, gray, CV_BGR2GRAY );

    keypoints.clear();  
    tic(); 

    cv::FAST(gray, keypoints, fast_th, true);
    brief->compute(gray, keypoints, brief_descriptors); 
    cout <<"fast time: ";

    pts_img.clear();
    pts_obj.clear();

    vector<DMatch> goodMatches;
    goodMatches.clear();
    match("knn", descriptorsAll, brief_descriptors, goodMatches);
    cout<<"goodMatches: "<<goodMatches.size()<<endl;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        pts_img.push_back( cv::Point2d( keypoints[goodMatches[i].queryIdx].pt ) );

        cv::Point3d pd = pts_objAll[goodMatches[i].trainIdx];
        pts_obj.push_back( pd );
    }
    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    
    // 求解pnp
    cout<<"pts_obj: "<<pts_obj.size()<<endl;
    cout<<"pts_img: "<<pts_img.size()<<endl;


    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 5.0, 0.95, inliers );

    toc();

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;


    Mat mat_T = cv::Mat::eye(4,4,CV_64F);
    Mat mat_r;
    Rodrigues(rvec, mat_r);
    mat_r.copyTo(mat_T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(mat_T(cv::Rect(3, 0, 1, 3)));
    mat_T = mat_T.inv();
    cout<<"T_query="<<endl<<mat_T<<endl;



    Eigen::Isometry3d T_eigen = cvTtoEigenT(mat_T);

    PointCloud::Ptr cloud2 = image2PointCloud( rgb.setTo(cv::Scalar(0,0,255)), depth, camera );
    // pcl::transformPointCloud( *cloud2, *cloud2, T_eigen.inverse().matrix() );
    pcl::transformPointCloud( *cloud2, *cloud2, T_eigen.matrix() );
    *cloud = *cloud2+ *cloud;
    pcl::io::savePCDFile("/Users/jin/work/PC_from_Traj/checkLoad.pcd", *cloud);

    Mat intrinsic(Matx41d(camera.fx, camera.fy, camera.cx, camera.cy));
    Mat extrinsics;

    cout <<"intrinsic now = " << endl << intrinsic << endl;
    cout <<"pnp T = " << endl << mat_T << endl;
    bundle_adjustment(intrinsic, extrinsics, pts_img, pts_obj);
    cout <<"intrinsic after = " << endl << intrinsic << endl;
    cout <<"extrinsics = " << endl << extrinsics << endl;

    return 0;
}