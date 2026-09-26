add_library(ForgeCCCodeGen CodeGen.cpp)
target_include_directories(ForgeCCCodeGen PUBLIC ${CMAKE_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})
target_link_libraries(ForgeCCCodeGen PRIVATE LLVM)
