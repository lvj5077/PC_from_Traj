#include "slamBase.h"
#include <pcl/filters/voxel_grid.h>

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0 || d<110 || d>4000)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}


cv::Point3f point2dTo3d( cv::Point2f& point, double& d, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p;
    p.z = float(d) ;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;

    return p;
}

pcl::PointCloud<pcl::PointXYZ> cvPtsToPCL(vector<Point3f> &p_XYZs)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.points.resize (p_XYZs.size());
    for (size_t i=0; i<p_XYZs.size(); i++) {
        cloud.points[i].x = p_XYZs[i].x;
        cloud.points[i].y = p_XYZs[i].y;
        cloud.points[i].z = p_XYZs[i].z;
    }
    cloud.height = 1;
    cloud.width = cloud.points.size();
    return cloud;
}

// vector<Point3f> imagToCVpt( Mat depth, CAMERA_INTRINSIC_PARAMETERS& camera ){
//     vector<Point3f> pts_cv;

//     for(int i=0;i<camera.width;i++){
//         for(int j=0;j<camera.height;j++){

//             if ((i<176&&i>60) && (j>0&&j<80)){
//                 continue;
//             }

//             cv::Point3f p;
//             double d = depth.at<double>(j,i,2);
//             p.z = float( d) ;
//             // p.x = ( i - camera.cx) * p.z / camera.fx;
//             // p.y = ( j - camera.cy) * p.z / camera.fy;

//             p.x = float( depth.at<double>(j,i,0) ) ;
//             p.y = float( depth.at<double>(j,i,1) ) ;

//             pts_cv.push_back(p);
//         }
//     }

//     return pts_cv;
// }
