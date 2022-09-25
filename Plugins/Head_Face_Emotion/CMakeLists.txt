project(Head_Face_Emotion)

include(${CMAKE_SOURCE_DIR}/CMake/OpenVino_Config.cmake)

include_directories("./include")
include_directories("../Common")

# Head Tracker Source Files
file(GLOB HEAD_TRACKER_SRC "./src/*.cpp")

if (WIN32)
    LIST(APPEND SYS_LIB shlwapi)
endif ()

add_library(
        ${PROJECT_NAME} SHARED
        head_face_emotion.cpp
        ${FlowCV_SRC}
        ${IMGUI_SRC}
        ${DSPatch_SRC}
        ${HEAD_TRACKER_SRC}
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
