find_package(GTest)

if(GTEST_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})

    add_executable(
        tests
        random_number_generator_test.cpp
        spin_vector_monte_carlo_test.cpp
        main.cpp
    )

    set_target_properties(tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/XSadd)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/dSFMT)

    set(LIBS quit_lib ${GTEST_LIBRARIES})
    target_link_libraries(tests ${LIBS})

endif()
