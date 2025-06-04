#!/bin/bash

python generate_config.py \
    --model_type llama \
    --input_jsons \
    solution/llama_batch2_gpus3.json \
    solution/llama_batch2_gpus6.json \
    --dispatcher_device cuda \
    --gpus_per_machine 4