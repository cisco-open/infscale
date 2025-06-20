#!/usr/bin/env python3
"""
Generate InfScale YAML configuration directly from placement.json.

This script is completely self-contained and does not rely on generate_config.py's
heuristic packing. It trusts only the placement.json produced by the DP packer
and the template files to build the exact configuration.

Usage:
    python generate_config_from_packing_solution.py \
           --placement placement.json \
           --model llama \
           --out infscale_config/packed.yaml
"""

import argparse
import json
import yaml
import os
from pathlib import Path
from collections import defaultdict


class FlowList(list):
    """Custom list class for YAML flow style representation."""
    pass


def represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, represent_flow_list)


def load_template_info(template_path):
    """Load template JSON and extract stage information."""
    with open(template_path) as f:
        data = json.load(f)
    
    stages = data.get("stages", data.get("pipeline_stages", []))
    batch_size = data.get("batch_size", 1)
    
    return {
        "stages": stages,
        "batch_size": batch_size
    }


def allocate_workers_for_deployment(deploy_id, deploy_info, template_info, node_gpu_tracker, stage_id_offset, gpus_per_node):
    """
    Allocate workers for a single deployment based on its node_segments.
    
    Returns:
        workers: list of worker dicts
        worker_to_machine: dict mapping worker_id to node_id
        worker_to_gpu: dict mapping worker_id to local_gpu_id
    """
    template_size = deploy_info["template_size"]
    node_segments = deploy_info["node_segments"]
    stages = template_info["stages"]
    
    # Create a flat list of GPUs available from node_segments
    available_gpus = []  # List of (node_id, local_gpu_id) tuples
    for segment in node_segments:
        node_id = segment["node_id"]
        gpus_in_segment = segment["gpus"]
        
        # Get the starting local GPU ID for this node
        start_local_gpu = node_gpu_tracker[node_id]
        
        # Add GPUs from this segment
        assert gpus_in_segment + start_local_gpu <= gpus_per_node, f"GPUs in segment {gpus_in_segment} is greater than the number of available GPUs {gpus_per_node}"
        for i in range(gpus_in_segment):
            local_gpu = start_local_gpu + i
            available_gpus.append((node_id, local_gpu))
        
        # Update the tracker
        node_gpu_tracker[node_id] += gpus_in_segment
    
    # Now allocate workers based on stages and their gpu_allocation
    workers = []
    worker_to_machine = {}
    worker_to_gpu = {}
    gpu_idx = 0  # Index into available_gpus
    
    for stage in stages:
        stage_id = stage["stage_id"] + stage_id_offset
        layer_start, layer_end = stage["layer_range"]
        num_replicas = stage["num_replicas"]
        gpu_allocation = stage["gpu_allocation"]
        
        # Calculate total GPUs needed for this stage
        total_gpus_for_stage = sum(gpu_allocation.values())
        
        # Allocate replicas for this stage
        replica_idx = 0
        for machine_offset_str, gpu_count in gpu_allocation.items():
            for _ in range(gpu_count):
                if replica_idx < num_replicas and gpu_idx < len(available_gpus):
                    worker_id = f"{stage_id}-{replica_idx}"
                    node_id, local_gpu = available_gpus[gpu_idx]
                    
                    workers.append({
                        "id": worker_id,
                        "device": f"cuda:{local_gpu}",
                        "stage": {"start": layer_start, "end": layer_end}
                    })
                    
                    worker_to_machine[worker_id] = node_id
                    worker_to_gpu[worker_id] = local_gpu
                    
                    replica_idx += 1
                    gpu_idx += 1
    
    return workers, worker_to_machine, worker_to_gpu, node_gpu_tracker


def build_flow_graph(all_workers, worker_to_machine, server_machine, model_type, dispatcher_device, deployment_workers):
    """Build the flow graph for InfScale communication."""
    # Generate cluster IPs
    max_node_id = max(worker_to_machine.values()) if worker_to_machine else 0
    max_node_id = max(max_node_id, server_machine)
    cluster_ips = [f"10.20.1.{70 + i}" for i in range(max_node_id + 1)] # TODO: Current hardcoded IP range; should be read from the cluster_ips.txt file etc
    
    flow_graph = {}
    world_id = 0
    server_backend = "nccl" if dispatcher_device == "cuda" else "gloo"
    
    # Process each deployment separately
    for deploy_id, deploy_workers in deployment_workers.items():
        # Group workers by stage_id within this deployment
        stages = {}
        for worker in deploy_workers:
            stage_id = int(worker["id"].split("-")[0])
            if stage_id not in stages:
                stages[stage_id] = []
            stages[stage_id].append(worker)
        
        stage_ids = sorted(stages.keys())
        
        # Build connections for this deployment's pipeline
        for i, stage_id in enumerate(stage_ids):
            stage_workers = stages[stage_id]
            
            for worker in stage_workers:
                worker_id = worker["id"]
                worker_node = worker_to_machine[worker_id]
                worker_addr = cluster_ips[worker_node]
                
                connections = []
                
                # Connection to previous stage or server
                if i == 0:  # First stage connects to server
                    connections.append({
                        "name": f"w{world_id}",
                        "peers": FlowList(["s-0"]),
                        "addr": worker_addr,
                        "backend": server_backend
                    })
                    world_id += 1
                else:  # Connect to all workers in previous stage
                    prev_stage_id = stage_ids[i-1]
                    prev_workers = stages[prev_stage_id]
                    for prev_worker in prev_workers:
                        connections.append({
                            "name": f"w{world_id}",
                            "peers": FlowList([prev_worker["id"]]),
                            "addr": worker_addr,
                            "backend": "nccl"
                        })
                        world_id += 1
                
                # For llama, add feedback connections from last stage to first stage
                if model_type == "llama" and i == 0 and len(stage_ids) > 1:
                    last_stage_workers = stages[stage_ids[-1]]
                    for last_worker in last_stage_workers:
                        connections.append({
                            "name": f"w{world_id}",
                            "peers": FlowList([last_worker["id"]]),
                            "addr": worker_addr,
                            "backend": "nccl"
                        })
                        world_id += 1
                
                flow_graph[worker_id] = connections
    
    # Add server connections - connect to last stage of each deployment
    server_connections = []
    for deploy_workers in deployment_workers.values():
        # Find last stage workers in this deployment
        max_stage_id = max(int(w["id"].split("-")[0]) for w in deploy_workers)
        last_stage_workers = [w for w in deploy_workers if int(w["id"].split("-")[0]) == max_stage_id]
        
        for worker in last_stage_workers:
            server_connections.append({
                "name": f"w{world_id}",
                "peers": FlowList([worker["id"]]),
                "addr": cluster_ips[server_machine],
                "backend": server_backend
            })
            world_id += 1
    
    flow_graph["s-0"] = server_connections
    
    return flow_graph


def main():
    parser = argparse.ArgumentParser(description="Generate InfScale config from placement.json")
    parser.add_argument("--placement", required=True, help="Path to placement.json")
    parser.add_argument("--model", choices=["resnet", "llama"], default="llama", help="Model type")
    parser.add_argument("--out", required=True, help="Output YAML file path")
    parser.add_argument("--dispatcher_device", choices=["cpu", "cuda"], default="cuda", help="Dispatcher device")
    parser.add_argument("--max_inflight", type=int, default=8, help="Max inflight batches")
    args = parser.parse_args()
    
    # Load placement data
    with open(args.placement) as f:
        placement = json.load(f)
    
    gpus_per_node = placement["meta"]["gpus_per_node"]

    # Validate dispatcher exists
    if "dispatcher" not in placement["deployments"]:
        raise RuntimeError("No dispatcher found in placement.json!")
    
    # Extract dispatcher info
    dispatcher_info = placement["deployments"]["dispatcher"]
    server_machine = dispatcher_info["node_segments"][0]["node_id"]
    
    # Get all model deployments (excluding dispatcher)
    model_deployments = {
        deploy_id: info for deploy_id, info in placement["deployments"].items()
        if deploy_id != "dispatcher"
    }
    
    if not model_deployments:
        raise RuntimeError("No model deployments found!")
    
    # Track GPU allocation per node
    node_gpu_tracker = {}
    for key in placement["nodes"]:
        node_gpu_tracker[int(key)] = 0
    
    # Process each deployment
    all_workers = []
    worker_to_machine = {}
    worker_to_gpu = {}
    deployment_workers = {}  # Track workers per deployment
    batch_size = 1
    stage_id_offset = 0
    
    # Process deployments in the exact order from JSON file
    for deploy_id, deploy_info in model_deployments.items():
        template_size = deploy_info["template_size"]
        template_path = placement["template_solutions"][str(template_size)]
        
        # Load template info
        template_info = load_template_info(template_path)
        batch_size = template_info["batch_size"]
        
        # Allocate workers for this deployment
        workers, w_to_m, w_to_g, node_gpu_tracker = allocate_workers_for_deployment(
            deploy_id, deploy_info, template_info, node_gpu_tracker, stage_id_offset, gpus_per_node
        )
        
        deployment_workers[deploy_id] = workers
        all_workers.extend(workers)
        worker_to_machine.update(w_to_m)
        worker_to_gpu.update(w_to_g)
        
        # Update stage_id_offset for next deployment
        max_stage_id = max(int(w["id"].split("-")[0]) for w in workers)
        stage_id_offset = max_stage_id + 1
    
    # Add dispatcher worker
    dispatcher_gpu = node_gpu_tracker[server_machine] if args.dispatcher_device == "cuda" else None
    dispatcher_device = f"cuda:{dispatcher_gpu}" if args.dispatcher_device == "cuda" else "cpu"
    
    dispatcher_worker = {
        "id": "s-0",
        "device": dispatcher_device,
        "is_server": True,
        "stage": {"start": -1, "end": -1}
    }
    
    all_workers.insert(0, dispatcher_worker)  # Server first
    
    # Build flow graph
    flow_graph = build_flow_graph(all_workers, worker_to_machine, server_machine, args.model, args.dispatcher_device, deployment_workers)
    
    # Model configurations
    model_configs = {
        "resnet": {
            "name": "resnet152_packed",
            "model": "microsoft/resnet-152",
            "dataset": {"path": "cifar10", "name": "", "split": "test"},
            "fwd_policy": "rr",
            "num_layers": 51
        },
        "llama": {
            "name": "llama3_packed",
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "dataset": {"path": "fka/awesome-chatgpt-prompts", "name": "", "split": "train"},
            "fwd_policy": "rr",
            "num_layers": 34
        }
    }
    assert args.model in model_configs, f"Model {args.model} not found in model_configs"
    
    config = model_configs[args.model]
    
    # Build final configuration
    final_config = {
        "name": config["name"],
        "model": config["model"],
        "nfaults": 1,
        "micro_batch_size": batch_size,
        "fwd_policy": config["fwd_policy"],
        "job_id": "job1",
        "max_inflight": args.max_inflight,
        "flow_graph": flow_graph,
        "dataset": config["dataset"],
        "workers": all_workers
    }
    
    # Write YAML
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"InfScale configuration written to {output_path}")
    print(f"Total workers: {len(all_workers)}")
    print(f"Server on node: {server_machine}")
    print(f"Model deployments: {len(model_deployments)}")


if __name__ == "__main__":
    main() 