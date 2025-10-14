# !/bin/bash

python per_layer_profiling.py --model_type llama \
                             --seq_lengths 64 256 1024 2048 4096 \
                             --batch_sizes 1 2 4 8 16 32 64 128 \
                             --output_suffix "t4"