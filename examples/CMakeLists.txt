add_executable(ai_model_inference
    ai_model_inference.cpp
)

target_include_directories(ai_model_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runtime/include
)

target_link_libraries(ai_model_inference PRIVATE
    forgecc_rt
)
