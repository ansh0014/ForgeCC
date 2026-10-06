add_library(forgecc_rt STATIC
    src/cpu_runtime.cpp
    src/gpu_runtime.cpp
    src/universal_api.cpp
)

target_include_directories(forgecc_rt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

add_library(forgecc_rt_shared SHARED
    src/cpu_runtime.cpp
    src/gpu_runtime.cpp
    src/universal_api.cpp
)

set_target_properties(forgecc_rt_shared PROPERTIES
    OUTPUT_NAME "forgecc_rt"
)

target_include_directories(forgecc_rt_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

add_custom_command(TARGET forgecc_rt_shared POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:forgecc_rt_shared>
    ${CMAKE_SOURCE_DIR}/python/forgecc/
)

