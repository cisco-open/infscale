#!/usr/bin/env python3
"""
This script parses a GPU log file with lines in the following format:

2025-02-03 21:49:59,959 | GPU0: Util=0%, Mem=0.5% | GPU1: Util=0%, Mem=0.5% | GPU2: Util=0%, Mem=0.5% | GPU3: Util=0%, Mem=0.5%

It extracts the timestamp and GPU utilization values from each line, then plots the utilization over time.
The log file is provided as a command-line argument.
"""

import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_gpu_log(file_path):
    """
    Parse a GPU log file where each line is formatted as:
    
    2025-02-03 21:49:59,959 | GPU0: Util=0%, Mem=0.5% | GPU1: Util=0%, Mem=0.5% | GPU2: Util=0%, Mem=0.5% | GPU3: Util=0%, Mem=0.5%
    
    Returns:
      - time_seconds: list of timestamps (in seconds) relative to the first timestamp.
      - gpu_utils: list of lists where gpu_utils[i] holds the utilization values for GPU i.
    """
    # Assume 4 GPUs by default
    gpu_count = 4
    gpu_utils = [[] for _ in range(gpu_count)]
    time_stamps = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" | ")
            # Parse timestamp from the first part:
            try:
                dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S,%f")
            except Exception as e:
                print(f"Skipping line (timestamp parse error): {line}")
                continue
            time_stamps.append(dt)
            
            # Parse GPU info from the rest of the parts.
            # Each token is expected to be in the form: "GPUX: Util=Y%, Mem=Z%"
            for token in parts[1:]:
                m = re.match(r"GPU(\d+):\s+Util=(\d+)%", token)
                if m:
                    gpu_idx = int(m.group(1))
                    util = float(m.group(2))
                    while gpu_idx >= len(gpu_utils):
                        gpu_utils.append([])
                    gpu_utils[gpu_idx].append(util)
    
    if not time_stamps:
        return [], gpu_utils
    
    # Convert timestamp to seconds relative to the first timestamp.
    base_time = time_stamps[0]
    time_seconds = [(ts - base_time).total_seconds() for ts in time_stamps]
    return time_seconds, gpu_utils

def plot_gpu_utilization(times, gpu_utils, log_file_name, start_time_sec=0, time_limit=None, gpus_to_plot=[0,1,2,3]):
    """
    Plot GPU utilization based on the parsed log data.

    Parameters:
      times (list): List of time points (in seconds relative to the start).
      gpu_utils (list of lists): GPU utilization data for each GPU.
      log_file_name (str): Original log file name (used for naming the output directory).
      start_time_sec (float): Start time in seconds for plotting.
      time_limit (float): End time in seconds for plotting. If None, the end of the log is used.
      gpus_to_plot (list): List of GPU indices to be plotted.
    """
    if len(times) == 0:
        print("No data to plot.")
        return

    if time_limit is None:
        time_limit = times[-1]

    # Determine the indices corresponding to the specified time window.
    start_idx = next((i for i, t in enumerate(times) if t >= start_time_sec), 0)
    end_idx = next((i for i, t in enumerate(times) if t > time_limit), len(times))

    # Create output directory for plots
    log_base_name = os.path.splitext(os.path.basename(log_file_name))[0]
    output_dir = f'output/{log_base_name}_{start_time_sec}s_to_{time_limit}s'
    os.makedirs(output_dir, exist_ok=True)

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    labels = [f'GPU {i}' for i in range(len(gpu_utils))]
    
    # Combined plot of all selected GPUs:
    plt.figure(figsize=(15, 8))
    combined_values = []
    for i in gpus_to_plot:
        if i < len(gpu_utils) and len(gpu_utils[i]) >= end_idx:
            values = gpu_utils[i][start_idx:end_idx]
            combined_values.extend(values)
            plt.plot(times[start_idx:end_idx], values,
                     label=labels[i],
                     color=colors[i % len(colors)],
                     alpha=0.7)
        else:
            print(f"Warning: Incomplete data for GPU {i}.")
    
    if combined_values:
        total_avg = np.mean(combined_values)
        total_max = np.max(combined_values)
        total_min = np.min(combined_values)
    else:
        total_avg = total_max = total_min = 0.0

    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Utilization (%)')
    plt.title(f'GPU Utilization\nTotal Avg: {total_avg:.2f}%, Max: {total_max:.2f}%, Min: {total_min:.2f}%')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time_sec, time_limit)
    plt.ylim(-5, 105)
    combined_plot_file = os.path.join(output_dir, 'gpu_utilization_combined.png')
    plt.savefig(combined_plot_file)
    plt.close()
    print(f"Combined plot saved: {combined_plot_file}")

    # Individual plots per GPU:
    for i in gpus_to_plot:
        if i < len(gpu_utils) and len(gpu_utils[i]) >= end_idx:
            values = gpu_utils[i][start_idx:end_idx]
            avg_val = np.mean(values)
            max_val = np.max(values)
            min_val = np.min(values)
            plt.figure(figsize=(15, 6))
            plt.plot(times[start_idx:end_idx], values,
                     color=colors[i % len(colors)],
                     alpha=0.7,
                     linewidth=2)
            plt.xlabel('Time (seconds)')
            plt.ylabel('GPU Utilization (%)')
            plt.title(f'GPU {i} Utilization\nAvg: {avg_val:.2f}%, Max: {max_val:.2f}%, Min: {min_val:.2f}%')
            plt.grid(True, alpha=0.3)
            plt.xlim(start_time_sec, time_limit)
            plt.ylim(-5, 105)
            individual_plot_file = os.path.join(output_dir, f'gpu{i}_utilization.png')
            plt.savefig(individual_plot_file)
            plt.close()
            print(f"GPU {i} plot saved: {individual_plot_file}")
        else:
            print(f"Skipping GPU {i} plot due to insufficient data.")

    # Print statistics for each GPU plotted
    for i in gpus_to_plot:
        if i < len(gpu_utils) and len(gpu_utils[i]) >= end_idx:
            values = gpu_utils[i][start_idx:end_idx]
            print(f"\nGPU {i} Statistics:")
            print(f"Average utilization: {np.mean(values):.2f}%")
            print(f"Maximum utilization: {np.max(values):.2f}%")
            print(f"Minimum utilization: {np.min(values):.2f}%")
        else:
            print(f"No statistics available for GPU {i}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse GPU log file and plot GPU utilization")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the GPU log file")
    parser.add_argument("--start_time", type=float, default=0, help="Start time (in seconds) for plotting")
    parser.add_argument("--time_limit", type=float, default=None, help="End time (in seconds) for plotting")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated list of GPU indices to plot (e.g. '0,3')")
    args = parser.parse_args()

    # Parse the list of GPUs to plot from the command-line argument.
    gpus_to_plot = [int(x.strip()) for x in args.gpus.split(",")]

    times, gpu_utils = parse_gpu_log(args.log_file)
    if not times:
        print("No valid data found in the log file.")
    else:
        plot_gpu_utilization(times, gpu_utils,
                               log_file_name=args.log_file,
                               start_time_sec=args.start_time,
                               time_limit=args.time_limit,
                               gpus_to_plot=gpus_to_plot) 