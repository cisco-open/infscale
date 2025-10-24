#!/usr/bin/env bash

python perf_bandwidth.py \
  --role client \
  --master_addr 127.0.0.1 \
  --sizes_mb 1,16,32,64,128,512,1024 \
  --device 1 \
  --iters 50


