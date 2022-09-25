project(Pose_Estimation_2D)

include(${CMAKE_SOURCE_DIR}/CMake/OpenVino_Config.cmake)

if (WIN32)
    LIST(APPEND SYS_LIB shlwapi)
endif ()

add_library(
        ${PROJECT_NAME} SHARED
        pose_estimation_2d.cpp
        ${FlowCV_SRC}
        ${IMGUI_SRC}
        ${DSPatch_SRC}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${OpenVino_LIBS}
        ${SYS_LIB}
)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        SUFFIX ".fp"
        )
