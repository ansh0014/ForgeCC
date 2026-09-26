add_library(forgecc_rt STATIC
    src/cpu_runtime.cpp
    src/gpu_runtime.cpp
)

target_include_directories(forgecc_rt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

