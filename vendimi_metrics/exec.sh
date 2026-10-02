#!/bin/bash

# Activate virtualenv
ENV_PATH=/path/to/env
source $ENV_PATH/bin/activate

# GPUGen vendimi computation with CFG file
# python compute_vendimi_gpu.py \
#     --kernel-file ../src/cuda/batch/kernels/vatom_simple_batch.cuh \
#     --gpugen-config ../config/classes.cfg \
#     --cfg-select F0-20_K1-200_Q3-1000 \
#     --kernel-res 512 \
#     --min-rand-kernel-res 0 \
#     --nclasses 1000 \
#     --samples-per-class 128 \
#     --model-res 224 \
#     --prep crop \
#     &> progress.txt

# GPUGen vendimi computation with CSV file
# python compute_vendimi_gpu.py \
#     --kernel-file ../src/cuda/batch/kernels/vatom_simple_batch.cuh \
#     --gpugen-config ../config/classes.csv \
#     --kernel-res 512 \
#     --min-rand-kernel-res 0 \
#     --nclasses 1000 \
#     --samples-per-class 128 \
#     --model-res 224 \
#     --prep crop \
#      &> progress.txt

# File vendimi computation
# FILE_SRC=path/to/dataset
# python compute_vendimi_file.py $FILE_SRC \
#     --total-files 128000 \
#     --seed 42 \
#     --batch-size 512 \
#     --num-workers 8 \
#     --model-res 224 \
#     --prep resize &> progress.txt

