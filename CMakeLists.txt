cmake_minimum_required(VERSION 2.8.3)
project(sfm)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    cv_bridge 
    )


find_package(OpenCV REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)


catkin_package(
    )

add_executable(sfm
    src/sfm_node.cpp
    src/feature_extractor.cpp
    src/DUtils/Random.cpp
    src/DUtils/Timestamp.cpp
    src/DVision/BRIEF.cpp
    src/DBoW/FBrief.cpp
    src/solve_5pts.cpp
    src/initial_sfm.cpp
    src/feature_manager.cpp
    src/BA.cpp)

target_link_libraries(sfm ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
