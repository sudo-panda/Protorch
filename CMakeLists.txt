cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ProTorch)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(protorch SHARED src/protorch.cpp)
target_include_directories(protorch PUBLIC include)

include(cmake/SetupLLVM.cmake)

target_compile_definitions(protorch PRIVATE ${LLVM_DEFINITIONS})
target_include_directories(protorch
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

llvm_config(protorch USE_SHARED)

find_package(Torch REQUIRED)
target_link_libraries(protorch PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(protorch PUBLIC "-frtti")

set_property(TARGET protorch PROPERTY CXX_STANDARD 17)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProTorchConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ProTorchConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ProTorch"
)

# # Install the config file
# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/ProTorchConfig.cmake"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ProTorch"
# )

# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/libprotorch.so"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
# )
