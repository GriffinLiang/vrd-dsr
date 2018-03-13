#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd roi_pooling/src/cuda

echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py
