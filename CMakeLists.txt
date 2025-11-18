cmake_minimum_required(VERSION 3.18)
project(TFFInfer CXX)

set(CMAKE_CURRENT_SOURCE_DIR .)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS ON)
SET(CMAKE_CUDA_ARCHITECTURES "86")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)



include(FetchContent)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(nlohmann_json)
if (NOT nlohmann_json_FOUND)
    #json
    FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.0
    )
    FetchContent_MakeAvailable(nlohmann_json)
    get_target_property(json_include_dirs nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Include directories for nlohmann_json: ${json_include_dirs}")
endif ()

find_package(glog)
if (NOT glog_FOUND)
    #glog
    FetchContent_Declare(
            glog
            GIT_REPOSITORY https://github.com/google/glog.git
            GIT_TAG v0.7.1
    )
    FetchContent_MakeAvailable(glog)
    get_target_property(GLOG_INCLUDE_DIRS glog::glog INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Glog include directories: ${GLOG_INCLUDE_DIRS}")
endif ()
#fmt
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        12.0.0  # 推荐稳定版本
)
FetchContent_MakeAvailable(fmt)

find_package(Taskflow)
if (NOT Taskflow_FOUND)
    #taskflow
    FetchContent_Declare(
            taskflow
            GIT_REPOSITORY https://github.com/taskflow/taskflow
            GIT_TAG v3.10.0
    )
    FetchContent_MakeAvailable(taskflow)
    get_target_property(taskflow_include_dirs Taskflow INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Include directories for taskflow: ${taskflow_include_dirs}")
    include_directories(${taskflow_include_dirs})
endif ()


set(${PROJECT_NAME}_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/)
add_subdirectory(test)
add_subdirectory(src/taskgraph)
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCE_LIST
        CONFIGURE_DEPENDS
        FOLLOW_SYMLINKS
        "src/**/*.cpp"
        "src/**/*.cu"
        "src/**/*.h"
        "include/**/*.h"
)

#file(GLOB ${PROJECT_NAME}_SOURCE_LIST
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/factory/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/factory/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/cuda/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/cuda/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/cpu/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/cpu/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/device/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/mem/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/mem/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/quant/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/quant/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/global/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/global/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/runtime/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/runtime/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/runtime/*.cu
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/quant/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/quant/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/quant/*.cu
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/*.cu
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/base/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/base/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/llama/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/model/llama/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/graph/op/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/graph/op/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/include/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/cuda/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/cuda/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/cuda/*.cu
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/factory/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/factory/*.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

message("-- TFFInfer sources: " ${${PROJECT_NAME}_SOURCE_LIST})
add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCE_LIST})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
    message("-- cmake_build_type:" ${CMAKE_BUILD_TYPE})
endif ()
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/log)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/model)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/factory)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
target_include_directories(${PROJECT_NAME} PUBLIC ${json_include_dirs})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart nlohmann_json::nlohmann_json glog::glog fmt::fmt)



