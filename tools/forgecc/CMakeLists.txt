add_executable(forgecc
    main.cpp
    Driver.cpp
)

target_include_directories(forgecc PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(forgecc PRIVATE LLVM version ole32)