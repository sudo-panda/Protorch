ml load cmake/3.23.1
ml load cuda/12.2.2

LLVM_INSTALL_DIR=$1

if [ -z "$LLVM_INSTALL_DIR" ]; then
    echo "Usage: setup-lassen.sh <LLVM installation dir>"
    return 0
fi

export PATH="$LLVM_INSTALL_DIR/bin":$PATH

mkdir build-lassen
pushd build-lassen

cmake .. \
-DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
-DLLVM_VERSION=18 \
-DENABLE_CUDA=on \
-DCMAKE_CUDA_ARCHITECTURES=70 \
-DCMAKE_C_COMPILER=/usr/tce/packages/gcc/gcc-11.2.1/bin/gcc \
-DCMAKE_CXX_COMPILER=/usr/tce/packages/gcc/gcc-11.2.1/bin/g++ \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on \
-DCMAKE_INSTALL_PREFIX="install-lassen" \
-DTorch_DIR=/usr/WS2/LExperts/spack_env/spack/opt/spack/linux-rhel7-power9le/gcc-11.2.1/py-torch-2.5.1-ehahjndagztrk4dlzkfe3q4gqeglkrus/lib/python3.11/site-packages/torch/share/cmake/Torch \
-DCUDNN_LIBRARY_PATH=/usr/WS2/LExperts/spack_env/spack/opt/spack/linux-rhel7-power9le/gcc-11.2.1/cudnn-8.9.7.29-12-i3lq7ofrjshbhexur47ml3ag5qylrhgq/lib/libcudnn.so \
-DCUDNN_INCLUDE_PATH=/usr/WS2/LExperts/spack_env/spack/opt/spack/linux-rhel7-power9le/gcc-11.2.1/cudnn-8.9.7.29-12-i3lq7ofrjshbhexur47ml3ag5qylrhgq/include

popd
