#!/bin/bash

# Run deployment placement optimization for ResNet model
# Arguments:
#   --dir: Directory containing solution files (solutions_index.yaml)
#   -k: Fault tolerance level (number of node failures the deployment can tolerate)
#   --dispatcher: Reserve 1 GPU on a node for dispatcher service
#   -o: Output JSON file path for the placement results
#   --gpu_per_node: Number of GPUs per node
#   --num_nodes: Total number of nodes
python deployment_placement.py --dir solution/resnet -k 2 --dispatcher -o resnet_placement.json --gpu_per_node 4 --num_nodes 10