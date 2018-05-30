#include "slamBase.h"
#include "loadImage.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
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

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv)
{
    cout<<endl<<"Program Started!"<<endl;
    cout <<"~~~~~~~~~~~~~~~~~~"<<endl<<endl;
    string ParameterPath = "/home/jin/Lingqiu_Jin/PC_from_Traj/parameters.txt";
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

    string timestmp;
    double x,y,z,qx,qy,qz,qw;

    string ImageFilenamesRGB;
    string ImageFilenamesD;

    pcl::visualization::CloudViewer viewer( "viewer" );

    FRAME frame;

    PointCloud::Ptr output (new PointCloud());

    ifstream fAssociation(trajectory_Path.c_str());

    int count = 0;
    int res = 1;

    int firstRun = 0;

    cout << "space = " << space <<endl;

    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            res++;
            count ++;

            stringstream ss;
            ss << s;
            ss >> timestmp>>x>>y>>z>>qx>>qy>>qz>>qw;


            if (res<space && count>5){
                cout << "I skipped !!"<<endl;
                continue;
            }
            else{
            res = 0;  
            cout << count <<endl;
            ImageFilenamesRGB = image_Path+"color/"+timestmp+".png";
            ImageFilenamesD = image_Path+"depth/"+timestmp+".png";

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

            Tm = Tfirst.colPivHouseholderQr().solve(Tnow);
            cout << Tm.matrix() <<endl<<endl;

            frame.rgb = cv::imread( ImageFilenamesRGB );
            frame.depth = cv::imread( ImageFilenamesD, -1);

            PointCloud::Ptr cloud = image2PointCloud( frame.rgb, frame.depth, camera );

            pcl::transformPointCloud( *cloud, *cloud, Tm.matrix() );

            static pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize( gridsize, gridsize, gridsize );
            voxel.setInputCloud( cloud );
            voxel.filter( *cloud );

            *output += *cloud;
            cloud->points.clear();

            }

            viewer.showCloud( output );

        }
    }

    pcl::io::savePCDFile(output_Path.c_str(), *output);
    
    while( !viewer.wasStopped() )
    {
        
    }
    return 0;
}
