find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  find_package(GTest QUIET)
endif()

if(NOT GTest_FOUND)
  message(WARNING "GTest not found. Building smoke tests only; contract tests will be skipped.")
endif()

add_subdirectory(unit)
