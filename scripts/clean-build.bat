rmdir /s /q build
cmake -G "MinGW Makefiles" -B build -S . -DCMAKE_PREFIX_PATH="C:/msys64/ucrt64" -DLLVM_DIR="C:/msys64/ucrt64/lib/cmake/llvm"
cmake --build build

