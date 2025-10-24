import os
import torch
import torch.distributed as dist
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

def run(rank, size, backend, tensor_size_mb, device_idx=0, iters=50):
    """NCCL point-to-point bandwidth test supporting multiple sizes in one session."""
    if backend != "nccl":
        raise ValueError("Only NCCL backend is supported")

    torch.cuda.set_device(device_idx)
    device = f'cuda:{device_idx}'

    # Initialize the process group using env:// for multi-node runs
    dist.init_process_group(backend, rank=rank, world_size=size, init_method="env://")
    print(f"Rank {rank}: Initialized process group on {device} using {backend}")

    sizes_mb = tensor_size_mb if isinstance(tensor_size_mb, (list, tuple)) else [tensor_size_mb]
    results = []  # Only populated on client (rank 0)

    for size_mb in sizes_mb:
        # Convert MB to number of elements (assuming float32)
        num_elements = (size_mb * (1024 ** 2)) // 4
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

        # Calculate size of data being transferred (in bytes)
        data_size_bytes = tensor.element_size() * tensor.nelement()
        data_size_gb = data_size_bytes / (1024 ** 3)

        times = []
        speeds = []

        for _ in range(iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            if rank == 0:
                start_event.record()
                dist.send(tensor, dst=1)
                torch.cuda.current_stream().synchronize()
                end_event.record()
            else:
                start_event.record()
                dist.recv(tensor, src=0)
                torch.cuda.current_stream().synchronize()
                end_event.record()

            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0

            speed_gbs = data_size_gb / elapsed_time
            times.append(elapsed_time)
            speeds.append(speed_gbs)

        times = times[1:-1]
        avg_time, avg_speed = calculate_statistics(times, data_size_gb)
        formatted_size = format_size(data_size_gb)

        if rank == 0:
            print(f"Rank 0 (client): Sending tensor of size {formatted_size} to rank 1 (server)")
            print(f"\nTimes: {times}")
            print(f"Average Send time: {avg_time:.6f} seconds, Average Speed: {avg_speed:.2f} GB/s")
            std_time = float(np.std(times))
            results.append((formatted_size, float(avg_time), std_time, float(avg_speed)))
        else:
            print(f"Rank 1 (server): Receiving tensor of size {formatted_size} from rank 0 (client)")
            print(f"\nTimes: {times}")
            print(f"Average Receive time: {avg_time:.6f} seconds, Average Speed: {avg_speed:.2f} GB/s")

    if rank == 0 and results:
        # ASCII summary table
        header = (
            "+----------------+---------------------------+-----------------------+\n"
            "| Size           | Avg Time (s) (std)       | Avg Bandwidth (GB/s) |\n"
            "+----------------+---------------------------+-----------------------+"
        )
        print("\n" + header)
        for size_label, avg_t, std_t, avg_bw in results:
            row = f"| {size_label:<14} | {avg_t:>10.6f} ({std_t:>10.6f}) | {avg_bw:>21.2f} |"
            print(row)
        print("+----------------+---------------------------+-----------------------+\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCCL bandwidth test (iperf-style client/server)')
    parser.add_argument('--role', choices=['server', 'client'], default='client', help='Process role')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address (server node IP)')
    parser.add_argument('--master_port', type=int, default=29500, help='Master port')
    parser.add_argument('--sizes_mb', type=str, default='1024', help='Comma-separated tensor sizes in MB')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use on this node')
    parser.add_argument('--iters', type=int, default=50, help='Iterations per size')
    args = parser.parse_args()

    sizes_mb = [int(s) for s in args.sizes_mb.split(',') if s]

    # Map iperf semantics: client sends (rank 0), server receives (rank 1)
    rank = 0 if args.role == 'client' else 1
    world_size = 2

    # env:// rendezvous for multi-node
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    print(f"\nNCCL test: role={args.role}, sizes_mb={sizes_mb}, device={args.device}, master={args.master_addr}:{args.master_port}")
    run(rank, world_size, "nccl", sizes_mb, device_idx=args.device, iters=args.iters)
