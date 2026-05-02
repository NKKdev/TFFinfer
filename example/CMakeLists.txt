cmake_minimum_required(VERSION 3.18)
project(tffInfer_example CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "120")
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
message("CMAKE_CURRENT_SOURCE_DIR: ",${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB ${PROJECT_NAME}_SOURCE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
message("-- tffInfer_example sources: " ${${PROJECT_NAME}_SOURCE_LIST})
add_subdirectory(load_model)







