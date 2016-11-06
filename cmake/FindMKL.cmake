# Hacked up from https://github.com/Eyescale/CMake/blob/master/FindMKL.cmake

set(MKL_ARCH_DIR "intel64")

find_path(MKL_ROOT_DIR
    include/mkl_cblas.h
    PATHS
    $ENV{MKLROOT}
    /opt/intel/mkl/*/
    /opt/intel/cmkl/*/
)

find_path(MKL_INCLUDE_DIR
  mkl_cblas.h
  PATHS
    ${MKL_ROOT_DIR}/include
    ${INCLUDE_INSTALL_DIR}
)

find_library(MKL_CORE_LIBRARY
  mkl_core
  PATHS
    ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR}
    ${MKL_ROOT_DIR}/lib/
)

find_library(MKL_ILP_LIBRARY
    mkl_intel_ilp64
    PATHS
    ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR}
    ${MKL_ROOT_DIR}/lib/
)

find_library(MKL_SEQUENTIAL_LIBRARY
    mkl_sequential
    PATHS
    ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR}
    ${MKL_ROOT_DIR}/lib/
)

set(MKL_ILP_SEQUENTIAL_LIBRARIES
    ${MKL_ILP_LIBRARY} ${MKL_SEQUENTIAL_LIBRARY} ${MKL_CORE_LIBRARY}
)

set(MKL_LIBRARIES ${MKL_ILP_SEQUENTIAL_LIBRARIES})
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
if(MKL_ROOT_DIR)
    set(MKL_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mkl, MKL_LIBRARIES, MKL_INCLUDE_DIR)

mark_as_advanced(MKL_LIBRARIES, MKL_INCLUDE_DIR)