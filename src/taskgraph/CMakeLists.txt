cmake_minimum_required(VERSION 3.18)
project(taskflowframe CXX)

set(CMAKE_CURRENT_SOURCE_DIR .)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_language(CUDA)

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
file(GLOB ${PROJECT_NAME}_SOURCE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cu)
message(" -- ${PROJECT_NAME} sources : " ${${PROJECT_NAME}_SOURCE_LIST})
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCE_LIST})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart Taskflow)