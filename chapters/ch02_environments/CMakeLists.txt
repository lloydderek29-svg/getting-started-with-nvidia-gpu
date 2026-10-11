add_executable(check_runtime src/check_runtime.cu)
target_link_libraries(check_runtime PRIVATE gpubook_common)

add_test(NAME check_runtime COMMAND check_runtime)
