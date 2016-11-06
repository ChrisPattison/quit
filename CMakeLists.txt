cmake_minimum_required(VERSION 3.0)

project (MPPAMC)
set (MPPAMC_VERSION_MAJOR 1)
set (MPPAMC_VERSION_MINOR 0)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

subdirs(src)
