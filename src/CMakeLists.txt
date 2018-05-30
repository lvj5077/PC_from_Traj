include_directories(/usr/include/eigen3)

SET("PCL_DIR" "/home/jin/Third_Party_Packages/pcl-pcl-1.8.1/build")
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
SET("OpenCV_DIR" "/home/jin/Third_Party_Packages/opencv-3.4.1/build")
find_package(OpenCV REQUIRED)

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# INCLUDE_DIRECTORIES( ${PROJECT_SOURSE_DIR}/include )


ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( PC_from_Traj pcFromTraj.cpp)
TARGET_LINK_LIBRARIES( PC_from_Traj
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( PC_from_Traj_name pcFromTraj_name.cpp) # not work yet
TARGET_LINK_LIBRARIES( PC_from_Traj_name
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( MypcFromTraj MypcFromTraj.cpp) # not work yet
TARGET_LINK_LIBRARIES( MypcFromTraj
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )