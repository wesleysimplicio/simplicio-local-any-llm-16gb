find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  find_package(GTest QUIET)
endif()

# The runtime contract suite is the bulk of our coverage. When no system GTest
# is available (default GitHub macos/ubuntu runners, fresh dev boxes) fall back
# to a pinned FetchContent build so the contract tests always compile and run
# instead of being silently skipped.
if(NOT GTest_FOUND AND US4_FETCH_GTEST)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  if(TARGET gtest AND NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
  set(GTest_FOUND TRUE)
endif()

if(NOT GTest_FOUND)
  message(WARNING "GTest not found and US4_FETCH_GTEST is OFF. Building smoke tests only; contract tests will be skipped.")
endif()

add_subdirectory(unit)

add_test(
  NAME adaptive_cache_speculation_tiny
  COMMAND ${CMAKE_SOURCE_DIR}/tests/tiny/run_adaptive_cache_speculation_tiny.sh
)
set_tests_properties(adaptive_cache_speculation_tiny PROPERTIES
  LABELS "runtime;moe;speculative;tiny"
)

find_package(Python3 COMPONENTS Interpreter QUIET)
find_program(US4_ENGINE_TEST_MAKE make)

if(US4_ENGINE_TEST_MAKE)
  add_test(
    NAME engine_colibri_make_test
    COMMAND ${US4_ENGINE_TEST_MAKE} test
  )
  set_tests_properties(engine_colibri_make_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/c
    LABELS "engine;unit;quantization;tokenizer"
  )
else()
  message(WARNING "make not found: engine_colibri_make_test cannot be registered")
endif()

if(Python3_Interpreter_FOUND)
  add_test(
    NAME engine_colibri_forward_oracles
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/fixtures/engine/run_engine_forward_oracles.py
  )
  set_tests_properties(engine_colibri_forward_oracles PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "engine;oracle;forward"
  )

  add_test(
    NAME engine_colibri_tokenizer_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/fixtures/engine/run_engine_tokenizer_contract.py
  )
  set_tests_properties(engine_colibri_tokenizer_contract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "engine;tokenizer"
  )

  add_test(
    NAME engine_colibri_runner_skip_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/engine/test_runner_skip_contract.py
  )
  set_tests_properties(engine_colibri_runner_skip_contract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "engine;contract;skip"
  )
endif()
