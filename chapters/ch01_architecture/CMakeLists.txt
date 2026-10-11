foreach(tool IN ITEMS device_properties
                      bandwidth_test
                      transfer_test
                      runtime_report)
  add_executable(${tool} src/${tool}.cu)
  target_link_libraries(${tool} PRIVATE gpubook_common)
endforeach()
