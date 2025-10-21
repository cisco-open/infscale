#!/bin/bash


python generate_heter_config.py \
  --input_jsons exp_config/2xT4_1xL4/ppipe.json \
  --model_type llama \
  --gpus_per_machine 4 \
  --ip_yaml ip_example.yaml

python generate_heter_config.py \
  --input_jsons exp_config/2xT4_1xL4/our.json \
  --model_type llama \
  --gpus_per_machine 4 \
  --ip_yaml ip_example.yaml

python generate_heter_config.py \
  --input_jsons exp_config/2xT4_1xL4/oobleck.json \
  --model_type llama \
  --gpus_per_machine 4 \
  --ip_yaml ip_example.yaml