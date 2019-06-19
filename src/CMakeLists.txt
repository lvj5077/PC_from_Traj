INCLUDE_DIRECTORIES ("/usr/local/include/eigen3/") #/usr/local/Cellar/eigen/3.3.7

SET(PCL_DIR "/usr/local/Cellar/pcl/1.9.1_4/share/pcl-1.9")
FIND_PACKAGE( PCL 1.9 REQUIRED COMPONENTS common io visualization filters )
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )
link_directories( /usr/local/Cellar/flann/1.9.1_7/lib )

SET(OpenCV_DIR "/usr/local/Cellar/opencv@3/3.4.5_2/share/OpenCV")
find_package( OpenCV 3.4 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIB_DIR})



ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( PC_from_Traj pcFromTraj.cpp)
TARGET_LINK_LIBRARIES( PC_from_Traj
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( poseFromPC poseFromPC.cpp)
TARGET_LINK_LIBRARIES( poseFromPC
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )