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

add_executable(real_forward_throughput
  real_forward_throughput.cpp
)

target_link_libraries(real_forward_throughput PRIVATE us4_runtime_core)

target_compile_definitions(real_forward_throughput PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(real_forward_throughput
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)

add_executable(kernel_cpu_benchmarks
  kernel_cpu_benchmarks.cpp
  ../core/hardware_probe.cpp
  ../core/runtime_mode.cpp
  ../core/tensor.cpp
  ../cpu/scalar_matmul.cpp
  ../cpu/scalar_attention.cpp
  ../neon/kernel_profile.cpp
  ../neon/neon_matmul.cpp
  ../neon/neon_attention.cpp
)

target_compile_definitions(kernel_cpu_benchmarks PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(kernel_cpu_benchmarks
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)

add_subdirectory(correctness)
