add_executable(dense_baseline
  dense_baseline.cpp
)

target_link_libraries(dense_baseline PRIVATE us4_runtime_core)

target_compile_definitions(dense_baseline PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(dense_baseline
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)

add_executable(matrix_runner
  matrix_runner.cpp
)

target_link_libraries(matrix_runner PRIVATE us4_runtime_core)

target_compile_definitions(matrix_runner PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(matrix_runner
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)
