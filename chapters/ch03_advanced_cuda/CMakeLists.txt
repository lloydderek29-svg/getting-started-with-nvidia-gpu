set(DEMOS
  blocksize_sweep
  matrix_2d
  occupancy_demo
  divergence_demo
)

foreach(demo IN LISTS DEMOS)
  add_executable(${demo} src/${demo}.cu)
  target_link_libraries(${demo} PRIVATE gpubook_common)
endforeach()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_library(ch03_kernels STATIC src/kernels.cu)
target_include_directories(ch03_kernels
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ch03_kernels
  PUBLIC gpubook_common)

add_executable(ch03_tests tests/test_kernels.cu)
target_link_libraries(ch03_tests
  PRIVATE ch03_kernels GTest::gtest_main)
gtest_discover_tests(ch03_tests)
