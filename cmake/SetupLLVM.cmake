# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM REQUIRED CONFIG)

if(NOT LLVM_ENABLE_RTTI)
  message(FATAL_ERROR "ProTorch needs RTTI enabled LLVM build")
endif()
