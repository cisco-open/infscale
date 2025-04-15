#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Run the profile with offloading
python per_layer_profiling.py \
    --model_type resnet \
    --batch_sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192