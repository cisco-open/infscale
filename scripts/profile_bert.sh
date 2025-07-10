#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Run the profile with offloading
python per_layer_profiling.py \
    --model_type bert \
    --batch_sizes 1 2 4 8 16 32 64 128 256 512 1024 \
    --seq_lengths 128 256 512