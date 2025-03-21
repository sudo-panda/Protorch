cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(ProTorch)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

set(PROTORCH_DIR "${PROJECT_SOURCE_DIR}/")
configure_file(include/config.h.in include/config.h @ONLY)

add_library(protorch SHARED src/protorch.cpp)
target_include_directories(protorch PUBLIC include "${CMAKE_CURRENT_BINARY_DIR}/include")

include(cmake/SetupLLVM.cmake)

target_compile_definitions(protorch PRIVATE ${LLVM_DEFINITIONS})
target_include_directories(protorch
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

llvm_config(protorch USE_SHARED)

find_package(Torch REQUIRED)
target_link_libraries(protorch PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_compile_options(protorch PUBLIC "-frtti")

set_property(TARGET protorch PROPERTY CXX_STANDARD 17)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProTorchConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ProTorchConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ProTorch"
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
# # Install the config file
# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/ProTorchConfig.cmake"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ProTorch"
# )

# install(
#     FILES "${CMAKE_CURRENT_BINARY_DIR}/libprotorch.so"
#     DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
# )
