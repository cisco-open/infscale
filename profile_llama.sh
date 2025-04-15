#!/bin/bash

# Profiling, we assume the prefill & decode profiling share the same amount of static & kv cache memory. But the dynamic memory are running time are profiled separately.

# # Prefill profiling with different batch sizes
# python per_layer_profiling.py \
#     --model_type llama \
#     --batch_sizes 1 2 4 8 \
#     --seq_lengths 1000 2000

# Decode profiling with different batch sizes
python per_layer_profiling.py \
    --model_type llama \
    --profile_decode \
    --batch_sizes 1 2 4 8 \
    --seq_lengths 1000 2000