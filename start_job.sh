#!/bin/bash

# ResNet152
# python -m infscale start job examples/resnet152/static/single_no_rep.yaml
# python -m infscale start job examples/resnet152/static/single_no_rep_2.yaml

# Llama3
# python -m infscale start job examples/llama3/static/no_rep.yaml
# python -m infscale start job examples/llama3/static/no_rep_2.yaml
python -m infscale start job examples/llama3/static/2_stage.yaml