#!/bin/bash

# Set the environment variables
export INFSCALE_LOG_LEVEL=PROFILE
export CUDA_VISIBLE_DEVICES=2,3

# Function to handle termination
cleanup() {
    echo "Terminating all processes..."
    kill -TERM "$CONTROLLER_PID"
    wait "$CONTROLLER_PID"
    exit 0
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

python -m infscale start controller &
CONTROLLER_PID=$!

python -m infscale start agent id0

wait $CONTROLLER_PID
