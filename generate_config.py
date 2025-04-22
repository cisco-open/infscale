import json
import yaml
import argparse
from pathlib import Path
from yaml.representer import SafeRepresenter
from collections import defaultdict

class FlowList(list):
    pass

def represent_flow_list(dumper, data):
    # Represent this sequence in flow style, e.g. [s-0]
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

# Register the representer for FlowList
yaml.add_representer(FlowList, represent_flow_list)

def generate_unified_allocation(pipeline_stages, gpus_per_machine, cluster_ips, machine_offset=0, stage_id_offset=0):
    """
    Create a unified allocation mapping of workers to machines and GPUs.
    
    Returns:
    - worker_to_machine: Dict mapping worker ID to machine ID
    - worker_to_gpu: Dict mapping worker ID to local GPU ID
    - server_machine: Machine ID for the server/dispatcher
    """
    # Validate that no machine has more workers than gpus_per_machine
    machine_worker_count = defaultdict(int)
    for stage in pipeline_stages:
        for machine_id_str, count in stage["gpu_allocation"].items():
            machine_id = int(machine_id_str) + machine_offset
            assert machine_id < len(cluster_ips), f"Machine ID {machine_id} is out of range for the number of machines ({len(cluster_ips)})"
            machine_worker_count[machine_id] += count
            
            if machine_worker_count[machine_id] > gpus_per_machine:
                raise ValueError(f"Machine {machine_id} has {machine_worker_count[machine_id]} workers allocated, but only {gpus_per_machine} GPUs per machine are configured.")
    
    # Find the last machine that has a worker (for server placement)
    last_worker_machine = 0
    for machine_id in machine_worker_count.keys():
        last_worker_machine = max(last_worker_machine, machine_id)
    server_machine = last_worker_machine
    
    # Create worker ID to machine ID mapping
    worker_to_machine = {}
    worker_to_gpu = {}  # Maps worker ID to local GPU ID on the machine
    
    # Track already allocated GPUs per machine to assign local GPU IDs
    allocated_gpus = defaultdict(set)  # machine_id -> set of used local GPU ids
    
    # Directly assign each worker to a machine and GPU
    for stage in pipeline_stages:
        orig_stage_id = stage["stage_id"]
        stage_id = orig_stage_id + stage_id_offset
        
        # Build a list of (machine_id, count) pairs for worker assignment
        machine_allocs = []
        for machine_id_str, count in stage["gpu_allocation"].items():
            machine_id = int(machine_id_str) + machine_offset
            machine_allocs.append((machine_id, count))
        
        # Assign workers to machines according to the allocation
        worker_idx = 0
        for machine_id, count in machine_allocs:
            for _ in range(count):
                if worker_idx < stage["num_replicas"]:
                    wid = f"{stage_id}-{worker_idx}"
                    worker_to_machine[wid] = machine_id
                    
                    # Find next available local GPU on this machine
                    for local_gpu in range(gpus_per_machine):
                        if local_gpu not in allocated_gpus[machine_id]:
                            allocated_gpus[machine_id].add(local_gpu)
                            worker_to_gpu[wid] = local_gpu
                            break
                    else:
                        # If we get here, we couldn't find an available GPU
                        raise ValueError(f"No available GPUs on machine {machine_id} for worker {wid}")
                    
                    worker_idx += 1
    
    return worker_to_machine, worker_to_gpu, server_machine

def create_flow_graph(pipeline_stages, cluster_ips, gpus_per_machine, worker_to_machine, server_machine, model_type="resnet", dispatcher_device="cpu", stage_id_offset=0, world_id_offset=0):
    """Create flow graph configuration with distributed address mapping based on planning JSON stages."""
    flow_graph = {}
    current_world_id = world_id_offset
    server_backend = "gloo" if dispatcher_device == "cpu" else "nccl"
    
    # Add server connections 
    server_connections = []
    
    # Get the last stage for connections
    last_stage = pipeline_stages[-1]
    last_stage_id = last_stage["stage_id"] + stage_id_offset
    for r in range(last_stage["num_replicas"]):
        peer_id = f"{last_stage_id}-{r}"
        server_connections.append({
            "name": None,
            "peers": FlowList([peer_id]),
            "addr": None,  # Server's own IP
            "backend": server_backend
        })
    
    assert "s-0" not in flow_graph, "Server should not be in the flow graph"

    # Add worker connections
    for stage in pipeline_stages:
        orig_stage_id = stage["stage_id"]
        stage_id = orig_stage_id + stage_id_offset
        
        # Find the previous stage
        prev = None
        prev_stage_id = None
        if orig_stage_id > 0:
            for s in pipeline_stages:
                if s["stage_id"] == orig_stage_id - 1:
                    prev = s
                    prev_stage_id = prev["stage_id"] + stage_id_offset
                    break
        
        for r in range(stage["num_replicas"]):
            wid = f"{stage_id}-{r}"
            # Get worker's machine from the unified allocation
            worker_machine = worker_to_machine[wid]
                
            if orig_stage_id == 0:
                peers = ["s-0"]; backend = server_backend
            else:
                peers = [f"{prev_stage_id}-{i}" for i in range(prev["num_replicas"])]
                backend = "nccl"

            connections = []
            for peer in peers:
                connections.append({
                    "name": f"w{current_world_id}",
                    "peers": FlowList([peer]),
                    "addr": cluster_ips[worker_machine],  # Worker's own IP
                    "backend": backend if peer != "s-0" else server_backend
                })
                current_world_id += 1
            flow_graph[wid] = connections

    return flow_graph, server_connections

def create_workers(pipeline_stages, model_layers, worker_to_machine, worker_to_gpu, server_machine, dispatcher_device="cpu", gpus_per_machine=4, stage_id_offset=0):
    """Create workers configuration with proper GPU assignments."""
    used_gpus_on_server_machine = set()
    for wid, gpu_id in worker_to_gpu.items():
        if worker_to_machine[wid] == server_machine:
            used_gpus_on_server_machine.add(gpu_id)

    workers = []
    
    # Assign stage workers
    for stage in pipeline_stages:
        orig_stage_id = stage["stage_id"]
        stage_id = orig_stage_id + stage_id_offset
        layer_start, layer_end = stage["layer_range"]
        
        for r in range(stage["num_replicas"]):
            wid = f"{stage_id}-{r}"
            local_gpu = worker_to_gpu[wid]
                
            workers.append({
                "id": wid,
                "device": f"cuda:{local_gpu}",
                "stage": {"start": layer_start, "end": layer_end}
            })

    return workers

def process_pipeline_config(json_data, cluster_ips, model_type, dispatcher_device, gpus_per_machine, 
                          machine_offset=0, stage_id_offset=0, world_id_offset=0):
    """Process a single pipeline configuration."""
    stages = json_data.get("stages", json_data.get("pipeline_stages", []))
    
    # Generate unified allocation
    worker_to_machine, worker_to_gpu, server_machine = generate_unified_allocation(
        stages, gpus_per_machine, cluster_ips, machine_offset, stage_id_offset
    )

    print(f"worker_to_machine: {worker_to_machine}")
    print(f"worker_to_gpu: {worker_to_gpu}")
    print(f"server_machine: {server_machine}")
    
    # Get model layers based on model type
    if model_type == "resnet":
        model_layers = 51
    elif model_type == "llama":
        model_layers = 34
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Create flow graph with world ID offset
    flow_graph, server_connections = create_flow_graph(
        stages, cluster_ips, gpus_per_machine,
        worker_to_machine, server_machine,
        model_type=model_type, 
        dispatcher_device=dispatcher_device,
        stage_id_offset=stage_id_offset,
        world_id_offset=world_id_offset
    )
    
    # Create workers
    workers = create_workers(
        stages, 
        model_layers,
        worker_to_machine, worker_to_gpu, server_machine,
        dispatcher_device=dispatcher_device,
        gpus_per_machine=gpus_per_machine,
        stage_id_offset=stage_id_offset
    )
    
    # Count total world IDs used
    total_world_ids = 0
    for connections in flow_graph.values():
        total_world_ids += len(connections)
    
    return {
        "flow_graph": flow_graph,
        "workers": workers,
        "server_machine": server_machine,
        "worker_to_machine": worker_to_machine,
        "worker_to_gpu": worker_to_gpu,
        "total_world_ids": total_world_ids,
        "server_connections": server_connections
    }

def process_multiple_json_configs(json_files, model_type, max_inflight, dispatcher_device, gpus_per_machine):
    """Process multiple JSON configuration files and generate a unified config."""
    # Define cluster IPs - may need to expand this for many files
    cluster_ips = ["10.20.1.70", "10.20.1.71", "10.20.1.72", "10.20.1.73", "10.20.1.74", 
                  "10.20.1.75", "10.20.1.76", "10.20.1.77", "10.20.1.78", "10.20.1.79"]
    
    model_config = {
        "resnet": {
            "name": "resnet152_example",
            "model": "microsoft/resnet-152",
            "dataset": {
                "path": "cifar10",
                "name": "",
                "split": "test"
            },
            "fwd_policy": "rr",
            "num_layers": 51
        },
        "llama": {
            "name": "llama3_linear_example",
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "dataset": {
                "path": "fka/awesome-chatgpt-prompts",
                "name": "",
                "split": "train"
            },
            "fwd_policy": "rr",
            "num_layers": 34
        }
    }
    
    # Keep track of machine and stage ID offsets
    machine_offset = 0
    stage_id_offset = 0
    world_id_offset = 0
    
    # Combined flow graph and workers
    combined_flow_graph = {}
    combined_flow_graph["s-0"] = []
    combined_workers = []
    combined_worker_to_gpu = {}
    combined_worker_to_machine = {}
    
    # Last server machine to place the dispatcher
    final_server_machine = 0

    # All last stage replicas connections to the server
    final_server_connections = []
    
    for json_file in json_files:
        # Read JSON file
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # Process this pipeline config
        result = process_pipeline_config(
            json_data, 
            cluster_ips, 
            model_type, 
            dispatcher_device, 
            gpus_per_machine,
            machine_offset,
            stage_id_offset,
            world_id_offset
        )

        print(f"flow_graph: {result['flow_graph']}")
        print(f"workers: {result['workers']}")
        
        # Update the combined flow graph
        for worker_id, connections in result["flow_graph"].items():
            if worker_id in combined_flow_graph:
                combined_flow_graph[worker_id].extend(connections)
            else:
                combined_flow_graph[worker_id] = connections
        
        # Add workers to combined list
        combined_workers.extend(result["workers"])
        combined_worker_to_gpu.update(result["worker_to_gpu"])
        combined_worker_to_machine.update(result["worker_to_machine"])
        
        # Update final server machine
        final_server_machine = result["server_machine"]

        # Update final server connections
        final_server_connections.extend(result["server_connections"])
        
        # Update offsets for next JSON file
        max_machine_id = 0
        for worker in result["workers"]:
            if worker["id"] != "s-0":
                # Extract machine ID from the worker information
                worker_id = worker["id"]
                for pipeline_worker_id, machine_id in result.get("worker_to_machine", {}).items():
                    if pipeline_worker_id == worker_id:
                        max_machine_id = max(max_machine_id, machine_id)
                        break
        
        machine_offset = max_machine_id + 1
        
        # Find max stage ID in this config
        max_stage_id = 0
        for worker in result["workers"]:
            if worker["id"] != "s-0":
                stage_id = int(worker["id"].split("-")[0])
                max_stage_id = max(max_stage_id, stage_id)
        
        stage_id_offset = max_stage_id + 1
        
        # Update world ID offset for next pipeline
        world_id_offset += result["total_world_ids"]
    
    # Now add the server/dispatcher once for all pipelines
    server_worker = {
        "id": "s-0",
        "device": "cpu" if dispatcher_device == "cpu" else f"cuda",  # Simple allocation for dispatcher
        "is_server": True,
        "stage": {"start": -1, "end": -1}
    }

    # Check the number of GPU allocated on the final server machine
    final_server_machine_gpu_count = 0
    final_server_machine_gpu_used = set()
    for worker in combined_worker_to_machine:
        if combined_worker_to_machine[worker] == final_server_machine:
            final_server_machine_gpu_used.add(combined_worker_to_gpu[worker])
            final_server_machine_gpu_count += 1
    if dispatcher_device == "cuda":
        assert final_server_machine_gpu_count <= gpus_per_machine-1, f"Final server machine with {gpus_per_machine} GPUs has {final_server_machine_gpu_count} GPUs allocated for workers, but we need to reserve one for the dispatcher"

    # Find the first available GPU on the final server machine
    if dispatcher_device == "cuda":
        for gpu_id in range(gpus_per_machine):
            if gpu_id not in final_server_machine_gpu_used:
                server_worker["device"] = f"cuda:{gpu_id}"
                break
    
    # Add server at the beginning of the workers list
    combined_workers = [server_worker] + combined_workers

    # Add server to the flow graph
    server_starting_world_id = world_id_offset
    if dispatcher_device == "cuda":
        server_backend = "nccl"
    else:
        server_backend = "gloo"
    for connection in final_server_connections:
        # We need to update the address and backend of the server connections
        connection["addr"] = cluster_ips[final_server_machine]
        connection["name"] = f"w{server_starting_world_id}"
        server_starting_world_id += 1
    combined_flow_graph["s-0"] = final_server_connections
    
    # Create the final config
    config = {
        "name": model_config[model_type]["name"],
        "model": model_config[model_type]["model"],
        "nfaults": 1,
        "micro_batch_size": json_data["batch_size"],  # Use batch size from the last JSON
        "fwd_policy": model_config[model_type]["fwd_policy"],
        "job_id": "job1",
        "max_inflight": max_inflight,
        "flow_graph": combined_flow_graph,
        "dataset": model_config[model_type]["dataset"],
        "workers": combined_workers
    }
    
    return config

def json_to_yaml_config(json_files, output_path, model_type="resnet", max_inflight=None, dispatcher_device="cpu", gpus_per_machine=4):
    """Convert JSON pipeline configurations to a single YAML format."""
    # Process all JSON files
    config = process_multiple_json_configs(
        json_files, 
        model_type, 
        max_inflight, 
        dispatcher_device, 
        gpus_per_machine
    )
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(description='Convert JSON pipeline config to YAML')
    parser.add_argument('--input_jsons', type=str, nargs='+', required=True,
                       help='Input JSON file paths (one or more)')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'llama'], default='resnet',
                       help='Model type to generate config for (default: resnet)')
    parser.add_argument('--max_inflight', type=int, default=8,
                       help='Maximum number of inflight batches (default: number of pipeline stages)')
    parser.add_argument('--dispatcher_device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device for the dispatcher (s-0) node (default: cpu)')
    parser.add_argument('--gpus_per_machine', type=int, default=4,
                       help='Number of GPUs per machine (default: 4)')
    args = parser.parse_args()
    
    # Infer output path from first input file
    input_path = Path(args.input_jsons[0])
    if len(args.input_jsons) > 1:
        if args.dispatcher_device == "cuda":
            output_path = Path('infscale_config') / f"multi_{len(args.input_jsons)}_pipelines_cuda_dispatcher.yaml"
        else:
            output_path = Path('infscale_config') / f"multi_{len(args.input_jsons)}_pipelines_cpu_dispatcher.yaml"
    else:
        if args.dispatcher_device == "cuda":
            output_path = Path('infscale_config') / input_path.name.replace('.json', '_cuda_dispatcher.yaml')
        else:
            output_path = Path('infscale_config') / input_path.name.replace('.json', '_cpu_dispatcher.yaml')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert and save YAML
    json_to_yaml_config(args.input_jsons, output_path, 
                       model_type=args.model_type, 
                       max_inflight=args.max_inflight,
                       dispatcher_device=args.dispatcher_device,
                       gpus_per_machine=args.gpus_per_machine)
    print(f"Generated YAML configuration at: {output_path}")

if __name__ == "__main__":
    main()
