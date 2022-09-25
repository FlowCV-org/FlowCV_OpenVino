cmake_minimum_required(VERSION 3.21)
project(FlowCV_OpenVino_Plugins)

set(CMAKE_CXX_STANDARD 17)

find_package( FlowCV REQUIRED )

add_subdirectory(Plugins/Pose_Estimation_2D)
add_subdirectory(Plugins/Head_Face_Emotion)
