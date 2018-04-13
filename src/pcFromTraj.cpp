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
    frame1.rgb = cv::imread( vstrImageFilenamesRGB[0] );
    frame1.depth = cv::imread( vstrImageFilenamesD[0], -1);

    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    

    ifstream input( trajectory_Path.c_str() );
    
    PointCloud::Ptr output (new PointCloud());
    output = cloud1;

    int count = 0;
    int res = 1;


    pcl::visualization::CloudViewer viewer( "viewer" );
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

        PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );

        pcl::transformPointCloud( *cloud2, *cloud2, Tm.matrix() );

        static pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize( gridsize, gridsize, gridsize );
        voxel.setInputCloud( cloud2 );
        voxel.filter( *cloud2 );

        *output += *cloud2;
        cloud2->points.clear();

        }
        viewer.showCloud( output );
    }

    
    
    pcl::io::savePCDFile(output_Path.c_str(), *output);
    
    while( !viewer.wasStopped() )
    {
        
    }
    return 0;
}
