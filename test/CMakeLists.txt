cmake_minimum_required(VERSION 3.18)
project(tffInfer_test CXX)
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
#cutlass
#FetchContent_Declare(
#        cutlass
#        GIT_REPOSITORY https://gitee.com/git_mirror/cutlass.git
#        GIT_TAG v3.7.0
#)
#FetchContent_MakeAvailable(cutlass)
#get_target_property(cutlass_INCLUDE_DIRS CUTLASS INTERFACE_INCLUDE_DIRECTORIES)
#message(STATUS "cutlass include directories: ${cutlass_INCLUDE_DIRS}")

file(GLOB ${PROJECT_NAME}_SOURCE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
message("-- tffInfer_test sources: " ${${PROJECT_NAME}_SOURCE_LIST})


add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCE_LIST}
        rms_norm_kernel.cu
        gemm_nt.cu
        quant_8_0.cu
        quant_q_8_0_gemm.cu
        quant_q_8_0.cuh
        gemm_half_scalar.cu
        gemm_vec.cu
        gemm_vec_vec.cu
        flash_attention.cu
        flash_attention_opt.cu
        flash_attention_multi_head.cu
        flash_attetntion_rope.cu
        quant_q_8_0_gemm_reshape_half.cu)
#target_compile_options(${PROJECT_NAME}
#        PRIVATE
#        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
#        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
#        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
#        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>
#)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
    message("-- cmake_build_type:" ${CMAKE_BUILD_TYPE})
endif ()
message(" -- TFFInfer_SOURCE_ROOT " ${CMAKE_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/log)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/3rdparty)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/core)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/model)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/factory)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/kernel)
target_include_directories(${PROJECT_NAME} PUBLIC ${json_include_dirs})
target_include_directories(${PROJECT_NAME} PUBLIC ${cutlass_INCLUDE_DIRS})
message(" -- CMAKE_LIBRARY_OUTPUT_DIRECTORY :" ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart nlohmann_json::nlohmann_json glog::glog TFFInfer)





