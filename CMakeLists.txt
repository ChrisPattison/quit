cmake_minimum_required(VERSION 3.7)

project (QUIT CXX)
set (QUIT_VERSION_MAJOR 1)
set (QUIT_VERSION_MINOR 0)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(SHARED_FAST_FLAGS "-Wall -Wno-misleading-indentation -Wno-int-in-bool-context -march=native")

# Enable interprocedural optimization if supported
set(IPO_SUPPORTED NO)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9")
    cmake_policy(VERSION 3.9)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION IPO_SUPPORTED)
endif()
message("* Using interprocedural optimizations : ${IPO_SUPPORTED}")

# Enable OpenMP if supported
set(OPENMP_FOUND NO)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(SHARED_FAST_FLAGS "${SHARED_FAST_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message("* Using OpenMP : ${OPENMP_FOUND}")

# -------- Fast
if(NOT CMAKE_CXX_FLAGS_FAST)
    set(CMAKE_CXX_FLAGS_FAST "${SHARED_FAST_FLAGS} -O3 -DNDEBUG" CACHE STRING
        "Flags used by the C++ compiler during fast release builds."
        FORCE)
endif()

if(NOT CMAKE_C_FLAGS_FAST)
    SET( CMAKE_C_FLAGS_FAST "${SHARED_FAST_FLAGS} -O3 -DNDEBUG" CACHE STRING
        "Flags used by the C compiler during fast release builds."
        FORCE )
endif()

if(NOT CMAKE_EXE_LINKER_FLAGS_FAST)
    SET( CMAKE_EXE_LINKER_FLAGS_FAST
        "" CACHE STRING
        "Flags used for linking binaries during fast release builds."
        FORCE )
endif()

if(NOT CMAKE_SHARED_LINKER_FLAGS_FAST)
    SET( CMAKE_SHARED_LINKER_FLAGS_FAST
        "" CACHE STRING
        "Flags used by the shared libraries linker during fast release builds."
        FORCE )
endif()
# -------- Fast Debug

if(NOT CMAKE_CXX_FLAGS_FASTDEBUG)
    SET( CMAKE_CXX_FLAGS_FASTDEBUG "${SHARED_FAST_FLAGS} -g -O2" CACHE STRING 
        "Flags used by the C++ compiler during fast debugging builds." 
        FORCE )
endif()

if(NOT CMAKE_C_FLAGS_FASTDEBUG)
    SET( CMAKE_C_FLAGS_FASTDEBUG "${SHARED_FAST_FLAGS} -g -O2" CACHE STRING
        "Flags used by the C compiler during fast debugging builds."
        FORCE )
endif()

if(NOT CMAKE_EXE_LINKER_FLAGS_FASTDEBUG)
    SET( CMAKE_EXE_LINKER_FLAGS_FASTDEBUG
        "" CACHE STRING
        "Flags used for linking binaries during fast debugging builds."
        FORCE )
endif()

if(NOT CMAKE_SHARED_LINKER_FLAGS_FASTDEBUG)
    SET( CMAKE_SHARED_LINKER_FLAGS_FASTDEBUG
        "" CACHE STRING
        "Flags used by the shared libraries linker during fast debugging builds."
        FORCE )
endif()

MARK_AS_ADVANCED (
    CMAKE_CXX_FLAGS_FAST
    CMAKE_C_FLAGS_FAST
    CMAKE_EXE_LINKER_FLAGS_FAST
    CMAKE_SHARED_LINKER_FLAGS_FAST

    CMAKE_CXX_FLAGS_FASTDEBUG
    CMAKE_C_FLAGS_FASTDEBUG
    CMAKE_EXE_LINKER_FLAGS_FASTDEBUG
    CMAKE_SHARED_LINKER_FLAGS_FASTDEBUG )

# -------- Build Type

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Fast
        CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel FastDebug Fast."
       FORCE)
endif()

message("* Current build type is : ${CMAKE_BUILD_TYPE}")

subdirs(src doc test)

message("* Build docs with 'make doc'")