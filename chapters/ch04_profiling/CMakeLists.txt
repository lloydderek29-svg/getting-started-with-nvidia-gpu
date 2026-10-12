set(CH04_TOOLS baseline_bench pipeline_demo hist_bench)

foreach(tool IN LISTS CH04_TOOLS)
  add_executable(${tool} src/${tool}.cu)
  target_include_directories(${tool}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${tool}
    PRIVATE gpubook_common ${CMAKE_DL_LIBS})
  target_compile_options(${tool} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endforeach()

if(NOT TARGET GTest::gtest_main)
  include(FetchContent)
  set(GTEST_URL
    https://github.com/google/googletest/archive/refs/tags)
  FetchContent_Declare(googletest
    URL ${GTEST_URL}/v1.15.2.zip)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

add_executable(ch04_tests tests/test_histogram.cu)
target_include_directories(ch04_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ch04_tests
  PRIVATE gpubook_common GTest::gtest_main)
gtest_discover_tests(ch04_tests)
