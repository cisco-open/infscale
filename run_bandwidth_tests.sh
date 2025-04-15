#!/bin/bash

# Default data sizes in MB to test if none provided
DEFAULT_SIZES=(1 10 50 100 200)

echo "PyTorch Bandwidth Test Runner"
echo "=============================="

# Use provided sizes or default ones
if [ $# -eq 0 ]; then
    echo "Using default data sizes: ${DEFAULT_SIZES[@]} MB"
    SIZES=("${DEFAULT_SIZES[@]}")
else
    echo "Using provided data sizes: $@ MB"
    SIZES=("$@")
fi

# Create results directory
RESULTS_DIR="bandwidth_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Saving results to: $RESULTS_DIR"

# Run tests for each data size
for size in "${SIZES[@]}"; do
    echo ""
    echo "Running test with data size: $size MB"
    echo "-----------------------------------"
    
    # Run the test and capture output
    OUTPUT_FILE="$RESULTS_DIR/bandwidth_${size}MB.log"
    python perf_bandwidth.py --size_mb "$size" | tee "$OUTPUT_FILE"
    
    echo "Test complete for $size MB, results saved to $OUTPUT_FILE"
    echo ""
done

echo "All tests completed. Results saved to $RESULTS_DIR/"

# Extract and display summary
echo ""
echo "Bandwidth Summary"
echo "================="
echo "Size (MB) | NCCL Speed (GB/s) | GLOO Speed (GB/s)"
echo "---------|--------------------|------------------"

for size in "${SIZES[@]}"; do
    LOG_FILE="$RESULTS_DIR/bandwidth_${size}MB.log"
    
    # Extract NCCL speed - first occurrence of "Average Speed:" after "Testing NCCL"
    NCCL_SPEED=$(grep -A 20 "Testing NCCL" "$LOG_FILE" | grep "Average Speed:" | head -n 1 | awk '{print $(NF-1)" "$NF}')
    
    # Extract GLOO speed - first occurrence of "Average Speed:" after "Testing GLOO"
    GLOO_SPEED=$(grep -A 20 "Testing GLOO" "$LOG_FILE" | grep "Average Speed:" | head -n 1 | awk '{print $(NF-1)" "$NF}')
    
    # If extraction failed, use placeholder
    [ -z "$NCCL_SPEED" ] && NCCL_SPEED="N/A"
    [ -z "$GLOO_SPEED" ] && GLOO_SPEED="N/A"
    
    printf "%-9s | %-18s | %-18s\n" "$size" "$NCCL_SPEED" "$GLOO_SPEED"
done 