add_executable(testgetEmbeds test-getEmbeds.cpp)

set_property(TARGET testgetEmbeds PROPERTY CXX_STANDARD 17)

target_link_libraries(testgetEmbeds PRIVATE protorch)
message(WARN ${LLVM_DEFINITIONS})

target_compile_definitions(testgetEmbeds PRIVATE ${LLVM_DEFINITIONS})


add_test(NAME test-getEmbeds COMMAND testgetEmbeds)
