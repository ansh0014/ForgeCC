add_library(ForgeCCSema Sema.cpp)
target_include_directories(ForgeCCSema PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ForgeCCSema PRIVATE ForgeCCAST)

