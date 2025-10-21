import torch
import torch.distributed as dist
import time
import numpy as np
import argparse

def calculate_statistics(times, data_size_gb):
    avg_time = np.mean(times)
    avg_speed = np.mean([data_size_gb / t for t in times])
    return avg_time, avg_speed

def format_size(size_gb):
    """Format size as GB or MB depending on magnitude"""
    if size_gb >= 1.0:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_gb * 1024:.2f} MB"

def run(rank, size, backend, tensor_size_mb):
    """ Test communication between processes using either NCCL (GPU) or GLOO (CPU). """
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'
    else:
        device = 'cpu'
    
    # Initialize the process group
    dist_url = "tcp://127.0.0.1:29500"
    dist.init_process_group(backend, rank=rank, world_size=size, init_method=dist_url)
    print(f"Rank {rank}: Initialized process group on {device} using {backend}")

    # Convert MB to number of elements (assuming float32)
    num_elements = (tensor_size_mb * (1024 ** 2)) // 4
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
    
    # Calculate size of data being transferred (in bytes)
    data_size_bytes = tensor.element_size() * tensor.nelement()
    data_size_gb = data_size_bytes / (1024 ** 3)  # Convert to GB

    times = []
    speeds = []

    for i in range(50):
        # Create timing events
        if backend == "nccl":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        else:
            start_time = time.time()
        
        if rank == 0:
            if backend == "nccl":
                start_event.record()
            else:
                start_time = time.time()
            dist.send(tensor, dst=1)
            if backend == "nccl":
                torch.cuda.current_stream().synchronize()
                end_event.record()
            else:
                elapsed_time = time.time() - start_time
        else:
            if backend == "nccl":
                start_event.record()
            else:
                start_time = time.time()
            dist.recv(tensor, src=0)
            if backend == "nccl":
                torch.cuda.current_stream().synchronize()
                end_event.record()
            else:
                elapsed_time = time.time() - start_time
        
        # Get timing
        if backend == "nccl":
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        speed_gbs = data_size_gb / elapsed_time
        times.append(elapsed_time)
        speeds.append(speed_gbs)

    times = times[1:-1] # Remove one maximum and one minimum value
    avg_time, avg_speed = calculate_statistics(times, data_size_gb)
    formatted_size = format_size(data_size_gb)

    if rank == 0:
        print(f"Rank 0 (GPU 0): Sending tensor of size {formatted_size} to rank 1 (GPU 1)")
        print(f"\nTimes: {times}")
        print(f"Average Send time: {avg_time:.6f} seconds, Average Speed: {avg_speed:.2f} GB/s")
    else:
        print(f"Rank 1 (GPU 1): Receiving tensor of size {formatted_size} from rank 0 (GPU 0)")
        print(f"\nTimes: {times}")
        print(f"Average Receive time: {avg_time:.6f} seconds, Average Speed: {avg_speed:.2f} GB/s")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure communication bandwidth between processes')
    parser.add_argument('--size_mb', type=int, default=1024, help='Tensor size in MB (default: 1024 MB = 1 GB)')
    args = parser.parse_args()
    
    size = 2
    # Test NCCL (GPU)
    print(f"\nTesting NCCL backend (GPU-to-GPU) with tensor size: {args.size_mb} MB:")
    torch.multiprocessing.spawn(run, args=(size, "nccl", args.size_mb), nprocs=size, join=True)
    
    print("--------------------------------")

    # Test GLOO (CPU)
    print(f"\nTesting GLOO backend (CPU-to-CPU) with tensor size: {args.size_mb} MB:")
    torch.multiprocessing.spawn(run, args=(size, "gloo", args.size_mb), nprocs=size, join=True)
