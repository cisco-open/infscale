#!/bin/bash

python generate_config.py \
    --input_jsons \
    solution/resnet_batch32_gpus4.json \
    solution/resnet_batch32_gpus4.json \
    solution/resnet_batch32_gpus3.json \
    --dispatcher_device cpu \
    --gpus_per_machine 4