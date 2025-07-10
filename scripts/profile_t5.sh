#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Run the profile with offloading
python per_layer_profiling.py \
    --model_type t5 \
    --batch_sizes 1 2 4 8 16 \
    --seq_lengths 1000 2000