import re
import sys
import statistics
import os
from pathlib import Path
import argparse

def parse_log(file_path: str, output_path: str):
    """
    Parses a log file and computes statistics excluding outliers:
      - For [SEND] lines (grouped by world), it computes average, min, and max send times in ms.
      - For [COMPUTE] lines (grouped by stage), it computes average, min, and max compute times in ms.
      - Additionally, for both SEND and COMPUTE lines, statistics are computed separately
        based on the 'count' field.
      - For [LATENCY] lines, the numerical value is assumed to be in seconds (even if misprinted with "ms"),
        so it is converted to ms before computing statistics.
      - Outliers are excluded using the IQR method.
      - [TOTAL LATENCY] is printed in seconds.
    """
    # Dictionaries to hold measurements.
    send_stats = {}    # key: world (e.g., 'w0'), value: list of times in ms.
    compute_stats = {} # key: stage (e.g., '0-0'), value: list of times in ms.
    # New dictionaries for statistics by count.
    send_stats_count = {}    # key: (world, count), value: list of times in ms.
    compute_stats_count = {} # key: (stage, count), value: list of times in ms.
    latency_list = []  # list of latency times (converted to ms).
    total_latency = None  # total latency in seconds.
    
    # New: Add warmup statistics
    warmup_stats = []  # list of warmup times in ms

    # Regular expressions to extract values.
    pattern_send = re.compile(
        r"\[SEND\].*?seqno\s+(-?\d+).*?world\s+(\w+).*?count:\s*(\d+).*?time:\s*([\d\.]+)\s*ms"
    )
    pattern_compute = re.compile(
        r"\[COMPUTE\].*?seqno\s+(-?\d+).*?stage\s+(\S+).*?count:\s*(\d+).*?time:\s*([\d\.]+)\s*ms"
    )
    # For latency, regardless of the unit, we'll treat the number as seconds.
    pattern_latency = re.compile(r"\[LATENCY\].*?seqno\s+(-?\d+).*?time:\s*([\d\.]+)\s*(s|ms)")
    pattern_total = re.compile(r"\[TOTAL LATENCY\].*?total time:\s*([\d\.]+)\s*s")
    pattern_warmup = re.compile(r"\[WARMUP\].*?seqno\s+(\d+).*?time:\s*([\d\.]+)\s*s")

    with open(file_path, "r") as f:
        for line in f:
            match = pattern_warmup.search(line)
            if match:
                seqno = int(match.group(1))
                time_s = float(match.group(2))
                warmup_stats.append(time_s * 1000)  # Convert to ms
                continue

            # Process [SEND] lines.
            match = pattern_send.search(line)
            if match:
                seqno = int(match.group(1))
                if seqno == -1:  # Skip warmup entries
                    continue
                world = match.group(2)
                count_value = int(match.group(3))
                time_ms = float(match.group(4))
                # Overall statistics by world.
                send_stats.setdefault(world, []).append(time_ms)
                # Statistics grouped by (world, count).
                key = (world, count_value)
                send_stats_count.setdefault(key, []).append(time_ms)
                continue

            # Process [COMPUTE] lines.
            match = pattern_compute.search(line)
            if match:
                seqno = int(match.group(1))
                if seqno == -1:  # Skip warmup entries
                    continue
                stage = match.group(2)
                count_value = int(match.group(3))
                time_ms = float(match.group(4))  # Already in ms.
                # Overall statistics by stage.
                compute_stats.setdefault(stage, []).append(time_ms)
                # Statistics grouped by (stage, count).
                key = (stage, count_value)
                compute_stats_count.setdefault(key, []).append(time_ms)
                continue

            # Process [LATENCY] lines.
            match = pattern_latency.search(line)
            if match:
                seqno = int(match.group(1))
                if seqno == -1:  # Skip warmup entries
                    continue
                time_val = float(match.group(2))
                # Even if the unit is misprinted as "ms", treat the value as seconds.
                time_ms = time_val * 1000  # convert seconds to ms.
                latency_list.append(time_ms)
                continue

            # Process [TOTAL LATENCY] line.
            match = pattern_total.search(line)
            if match:
                total_latency = float(match.group(1))  # In seconds.
                continue

    def filter_outliers(values):
        # """
        # Excludes outliers from a list of numbers using the IQR method.
        # Outliers are values outside the range:
        #      [Q1 - 1.5*IQR, Q3 + 1.5*IQR],
        # where IQR = Q3 - Q1.
        # If the list has fewer than 4 values, no filtering is applied.
        # """
        # if len(values) < 4:
        #     return values
        # sorted_vals = sorted(values)
        # n = len(sorted_vals)
        # # Split data into lower and upper halves.
        # if n % 2 == 0:
        #     lower_half = sorted_vals[:n//2]
        #     upper_half = sorted_vals[n//2:]
        # else:
        #     lower_half = sorted_vals[:n//2]
        #     upper_half = sorted_vals[n//2+1:]
        # q1 = statistics.median(lower_half)
        # q3 = statistics.median(upper_half)
        # iqr = q3 - q1
        # lower_bound = q1 - 10 * iqr
        # upper_bound = q3 + 10 * iqr
        # return [v for v in values if lower_bound <= v <= upper_bound]

        return values   # Due to the warmup, we don't filter outliers

    def compute_summary(values):
        """
        Computes the average, minimum, and maximum of a list of numbers.
        """
        avg = sum(values) / len(values)
        return avg, min(values), max(values), statistics.median(values)

    # Prepare output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_f:
        # Print WARMUP statistics
        print("WARMUP statistics (ms):", file=out_f)
        if warmup_stats:
            avg, min_t, max_t, median_t = compute_summary(warmup_stats)
            print(f"  ({len(warmup_stats)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)
        else:
            print("  No WARMUP entries found.", file=out_f)
        
        print("", file=out_f)  # Empty line for separation

        # Print SEND statistics (excluding outliers).
        print("SEND statistics (ms) [excluding outliers]:", file=out_f)
        for world in sorted(send_stats.keys(), key=lambda x: int(x[1:])):  # Sort by number after 'w'
            times = send_stats[world]
            filtered_times = filter_outliers(times)
            if not filtered_times:
                filtered_times = times
            avg, min_t, max_t, median_t = compute_summary(filtered_times)
            print(f"  World {world} ({len(filtered_times)}/{len(times)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)

        # Print SEND statistics grouped by count.
        print("\nSEND statistics by count (ms) [excluding outliers]:", file=out_f)
        for (world, count) in sorted(send_stats_count.keys(), key=lambda x: (int(x[0][1:]), x[1])):
            times = send_stats_count[(world, count)]
            filtered_times = filter_outliers(times)
            if not filtered_times:
                filtered_times = times
            avg, min_t, max_t, median_t = compute_summary(filtered_times)
            print(f"  World {world}, Count {count} ({len(filtered_times)}/{len(times)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)

        # Print COMPUTE statistics (excluding outliers).
        print("\nCOMPUTE statistics (ms) [excluding outliers]:", file=out_f)
        for stage in sorted(compute_stats.keys(), key=lambda x: tuple(map(int, x.split('-')))):  # Sort by numeric components
            times = compute_stats[stage]
            filtered_times = filter_outliers(times)
            if not filtered_times:
                filtered_times = times
            avg, min_t, max_t, median_t = compute_summary(filtered_times)
            print(f"  Stage {stage} ({len(filtered_times)}/{len(times)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)

        # Print COMPUTE statistics grouped by count.
        print("\nCOMPUTE statistics by count (ms) [excluding outliers]:", file=out_f)
        for (stage, count) in sorted(compute_stats_count.keys(), key=lambda x: (tuple(map(int, x[0].split('-'))), x[1])):
            times = compute_stats_count[(stage, count)]
            filtered_times = filter_outliers(times)
            if not filtered_times:
                filtered_times = times
            avg, min_t, max_t, median_t = compute_summary(filtered_times)
            print(f"  Stage {stage}, Count {count} ({len(filtered_times)}/{len(times)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)

        # Print statistics grouped by main stage
        main_stage_stats = {}
        for stage, times in compute_stats.items():
            main_stage = stage.split('-')[0]
            filtered_times = filter_outliers(times)
            if not filtered_times:
                filtered_times = times
            main_stage_stats.setdefault(main_stage, []).extend(filtered_times)

        print("\nCOMPUTE statistics by stage (ms) [excluding outliers]:", file=out_f)
        for main_stage, times in sorted(main_stage_stats.items()):
            avg, min_t, max_t, median_t = compute_summary(times)
            print(f"  Stage {main_stage} ({len(times)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)

        # Print LATENCY statistics (excluding outliers).
        print("\nLATENCY statistics (ms) [excluding outliers]:", file=out_f)
        if latency_list:
            filtered_latencies = filter_outliers(latency_list)
            if not filtered_latencies:
                filtered_latencies = latency_list
            avg, min_t, max_t, median_t = compute_summary(filtered_latencies)
            print(f"  ({len(filtered_latencies)}/{len(latency_list)} entries): "
                  f"avg = {avg:.3f} ms, min = {min_t:.3f} ms, max = {max_t:.3f} ms, median = {median_t:.3f} ms",
                  file=out_f)
        else:
            print("  No LATENCY entries found.", file=out_f)

        # Print TOTAL LATENCY (in seconds).
        print("\nTOTAL LATENCY:", file=out_f)
        if total_latency is not None:
            print(f"  {total_latency:.3f} s", file=out_f)
        else:
            print("  No TOTAL LATENCY entry found.", file=out_f)

def process_folder(folder_path: str):
    """
    Process all log files in the given folder and create parsed output files
    in a 'parsed' subfolder.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)

    # Create parsed subfolder
    parsed_folder = folder_path / 'parsed'
    
    # Process each .log file in the folder
    for file_path in folder_path.glob('*.log'):
        if file_path.name.startswith('parsed_'):
            continue  # Skip already parsed files
        
        output_path = parsed_folder / f'parsed_{file_path.name}'
        print(f"Processing {file_path.name}...")
        parse_log(str(file_path), str(output_path))
        print(f"Output written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Parse log files and compute statistics.')
    parser.add_argument('--log_path', required=True,
                       help='Path to the folder containing log files')
    args = parser.parse_args()
    
    process_folder(args.log_path)

if __name__ == "__main__":
    main() 