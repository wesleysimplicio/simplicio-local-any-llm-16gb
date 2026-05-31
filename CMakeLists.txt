cmake_minimum_required(VERSION 3.27)

project(us4 VERSION 0.2.7 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(US4_BUILD_TESTS "Build runtime smoke tests" ON)
option(US4_BUILD_BENCHMARKS "Build runtime benchmark placeholders" ON)
option(US4_FETCH_GTEST "Fetch a pinned GoogleTest when no system package is found" ON)
option(US4_ENABLE_COVERAGE "Instrument the runtime and tests for llvm-cov source coverage" OFF)

if(US4_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "US4_ENABLE_COVERAGE requires Clang/AppleClang (source-based coverage via -fprofile-instr-generate).")
  endif()
  add_compile_options(-fprofile-instr-generate -fcoverage-mapping -O0 -g)
  add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/us4/version.h
  @ONLY
)

add_subdirectory(runtime)
add_subdirectory(apps)

if(US4_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
