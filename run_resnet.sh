#!/bin/bash

# python -m infscale start job examples/resnet152/linear.yaml
python -m infscale start job examples/resnet_profiling/batch_size_128_1stage_2gpu_cuda.yaml

