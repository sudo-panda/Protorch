# conda create --name LLNL python=3.12 llvm=18 llvmdev=18 clang=18 clangxx=18 cuda=12.6 "cmake>=3.23" cudnn pytorch=2.6.0

CONDA_ENV_DIR=$1

if [ -z "$CONDA_ENV_DIR" ]; then
    echo "Usage: setup-conda.sh <Conda ENV dir>"
    return 0
fi 

mkdir build
cd build

cmake .. \
 -DLLVM_INSTALL_DIR=$CONDA_ENV_DIR/include/llvm \
 -DLLVM_VERSION=18 \
 -DCMAKE_CUDA_ARCHITECTURES=70 \
 -DCMAKE_C_COMPILER="$CONDA_ENV_DIR/bin/clang" \
 -DCMAKE_CXX_COMPILER="$CONDA_ENV_DIR/bin/clang++" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
 -DCMAKE_INSTALL_PREFIX="install" \
 -DTorch_DIR=$CONDA_ENV_DIR/share/cmake/Torch \
 -DPython_EXECUTABLE=$CONDA_ENV_DIR/bin/python \
 -Dnvtx3_dir=$CONDA_ENV_DIR/nsight-compute-2024.3.2/host/target-linux-x64/nvtx/include/nvtx3
