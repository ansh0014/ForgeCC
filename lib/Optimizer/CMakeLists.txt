add_library(ForgeCCOptimizer Passes.cpp)
target_include_directories(ForgeCCOptimizer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ForgeCCOptimizer PRIVATE ForgeCCIR)
