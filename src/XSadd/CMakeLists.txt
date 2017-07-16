add_library(XSadd STATIC xsadd.cpp)

set_target_properties(XSadd PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED NO
    CXX_EXTENSIONS NO
    LINKER_LANGUAGE CXX
)