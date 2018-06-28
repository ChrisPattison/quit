include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(TIMESTAMP BUILD_TIME UTC)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DDSFMT_MEXP=19937)
add_subdirectory(dSFMT)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dSFMT)

add_subdirectory(XSadd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/XSadd)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.hpp"
)

if(NOT QUIT_LOG_LOOKUP_SIZE)
    set(QUIT_LOG_LOOKUP_SIZE 1024
        CACHE STRING "Set number of entries in log lookup table."
        FORCE)
endif()
message("* Using ${QUIT_LOG_LOOKUP_SIZE} Log lookup table entries")

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/constant.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/constant.hpp"
)

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

set(LIBS ${LIBS} dSFMT XSadd)

target_link_libraries(quit_solve ${LIBS})


add_library(quit_lib STATIC ${SRCS})
set_target_properties(quit_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(quit_lib ${LIBS})
