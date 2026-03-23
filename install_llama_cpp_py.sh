#!/bin/bash

# CUDACXX=/usr/local/cuda-12/bin/nvcc cmake -B build -S . -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native

pip install --upgrade pip setuptools wheel

rm -rf llama-cpp-python/
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python && \
    CUDACXX=/usr/local/cuda-12/bin/nvcc cmake -B build -S . -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 && \
    cmake --build build && \
    pip install .

pip install scikit-build-core cmake ninja

CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade