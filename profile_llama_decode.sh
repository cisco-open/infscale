# !/bin/bash

export CUDA_VISIBLE_DEVICES=1

python per_layer_profiling.py --model_type llama_70b \
                             --profile_decode \
                             --seq_lengths 64 256 1024 2048 4096 \
                             --batch_sizes 1 2 4 8 16 \
                             --trimmed_model_dir "trimmed_model_70b" \
                             --output_suffix "t4"