# OpenVino CMake Config

# Configure Compile Flags
if (WIN32)
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX -DWITH_EXTENSIONS")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_SSE -DHAVE_AVX2 -DHAVE_AVX512F")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") #treating warnings as errors
    endif ()

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror") #treating warnings as errors
    if (APPLE)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()
endif()

# Get/Set OpenVino Env Paths
if(DEFINED ENV{INTEL_OPENVINO_DIR})
    set(INTEL_OPENVINO_DIR "$ENV{INTEL_OPENVINO_DIR}")
else()
    find_path(INTEL_OPENVINO_DIR "INTEL_OPENVINO_DIR")
endif()
message(STATUS "OpenVino Dir: ${INTEL_OPENVINO_DIR}")

if(WIN32)
    if(DEFINED ENV{OPENVINO_LIB_PATHS})
        set(OPENVINO_LIB_PATHS "$ENV{OPENVINO_LIB_PATHS}")
    else()
        find_path(OPENVINO_LIB_PATHS "OPENVINO_LIB_PATHS")
    endif()
else()
    if(DEFINED ENV{LD_LIBRARY_PATH})
        set(OPENVINO_LIB_PATHS "$ENV{LD_LIBRARY_PATH}")
        string(REPLACE ":" ";" OPENVINO_LIB_PATHS ${OPENVINO_LIB_PATHS})
    else()
        find_path(OPENVINO_LIB_PATHS "OPENVINO_LIB_PATHS")
    endif()
endif()
message(STATUS "OpenVino Lib Paths: ${OPENVINO_LIB_PATHS}")

if(DEFINED ENV{InferenceEngine_DIR})
    set(InferenceEngine_DIR "$ENV{InferenceEngine_DIR}")
else()
    set(InferenceEngine_DIR ${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/share)
endif()
message(STATUS "OpenVino Inference Engine Dir: ${InferenceEngine_DIR}")

if(DEFINED ENV{ngraph_DIR})
    set(ngraph_DIR "$ENV{ngraph_DIR}")
else()
    set(ngraph_DIR ${INTEL_OPENVINO_DIR}/deployment_tools/ngraph/cmake)
endif()
message(STATUS "NGraph Dir: ${ngraph_DIR}")

if (DEFINED ENV{TBB_DIR})
    set(TBB_DIR "$ENV{TBB_DIR}")
else()
    set(TBB_DIR ${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/cmake)
endif()
message(STATUS "TBB Dir: ${TBB_DIR}")

find_path(OpenVinoBuild_DIR "OpenVinoBuild_DIR")
if ("${OpenVinoBuild_DIR}" STREQUAL "OpenVinoBuild_DIR-NOTFOUND")
    message(FATAL_ERROR "No OpenVino Build Dir Found")
endif()

if(WIN32)
    link_directories(${OpenVinoBuild_DIR}/intel64/Release)
else()
    link_directories(${OpenVinoBuild_DIR}/intel64/Release/lib)
endif()
message(STATUS "OpenVino Build Dir: ${OpenVinoBuild_DIR}")

FIND_PACKAGE( OpenCV REQUIRED )

# Include OpenVino cmake configs
include(${TBB_DIR}/TBBConfig.cmake)
include(${InferenceEngine_DIR}/InferenceEngineConfig.cmake)
include(${ngraph_DIR}/ngraphConfig.cmake)

# Additional Include Paths
INCLUDE_DIRECTORIES("${InferenceEngine_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${InferenceEngine_DIR}/../demos/common/cpp/models/include")
INCLUDE_DIRECTORIES("${InferenceEngine_DIR}/../demos/common/cpp/pipelines/include")
INCLUDE_DIRECTORIES("${InferenceEngine_DIR}/../demos/common/cpp/utils/include")
INCLUDE_DIRECTORIES("${InferenceEngine_DIR}/../demos/common/cpp/monitors/include")
INCLUDE_DIRECTORIES("${InferenceEngine_DIR}/../demos/common/format_reader")
INCLUDE_DIRECTORIES("${ngraph_DIR}/../include")
INCLUDE_DIRECTORIES("${OpenVinoBuild_DIR}/thirdparty/gflags/include")

# Library Paths
link_directories(${OPENVINO_LIB_PATHS})

if(WIN32)
    list(APPEND OpenVino_LIBS
            monitors.lib
            models.lib
            pipelines.lib
            utils.lib
            common.lib
            gflags_nothreads_static.lib
            ${NGRAPH_LIBRARIES}
            ${InferenceEngine_LIBRARIES}
            )
else()
    list(APPEND OpenVino_LIBS
            libmonitors.a
            libmodels.a
            libpipelines.a
            libutils.a
            libmulti_channel_common.a
            libgflags_nothreads.a
            ${NGRAPH_LIBRARIES}
            ${InferenceEngine_LIBRARIES}
            )
endif()
