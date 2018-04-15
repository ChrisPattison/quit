include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(TIMESTAMP BUILD_TIME UTC)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.hpp"
)

add_definitions(-DEIGEN_DONT_PARALLELIZE)

find_package(TBB REQUIRED)
set(LIBS ${LIBS} ${TBB_IMPORTED_TARGETS})
include_directories(${TBB_INCLUDE_DIRS})

include_directories(BEFORE ${CMAKE_BINARY_DIR}/parallelstl/include)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(dSFMT)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dSFMT)

add_subdirectory(XSadd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/XSadd)

set(SRCS
    graph.cpp 
    random_number_generator.cpp 
    parse.cpp 
    output.cpp
    string_util.cpp 
    compare.cpp
    log_lookup.cpp
    spin_vector_monte_carlo.cpp
    parallel_tempering_base.cpp
    parallel_tempering.cpp
)

add_executable(
    quit_solve
    main.cpp 
    ${SRCS}
)

install(TARGETS quit_solve
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

set_target_properties(quit_solve PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

find_package(Boost REQUIRED COMPONENTS program_options)
set(LIBS ${LIBS} ${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(LIBS ${LIBS} dSFMT XSadd)

target_link_libraries(quit_solve ${LIBS})


add_library(quit_lib STATIC ${SRCS})
set_target_properties(quit_lib PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_include_directories(quit_lib INTERFACE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(quit_lib ${LIBS})
