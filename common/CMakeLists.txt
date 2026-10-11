add_library(gpubook_common INTERFACE)
target_include_directories(gpubook_common
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpubook_common
  INTERFACE CUDA::cudart)
# chapters/ch01_architecture/CMakeLists.txt
add_executable(device_properties
  src/device_properties.cu)
target_link_libraries(device_properties
  PRIVATE gpubook_common)
