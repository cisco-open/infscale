#!/bin/bash

# Function to cleanup on exit
cleanup() {
    echo "Stopping processes..."
    kill $CONTROLLER_PID $AGENT_PID 2>/dev/null
    wait $CONTROLLER_PID $AGENT_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Allow overriding max tokens for Llama3 generation
export LLAMA3_MAX_NEW_TOKENS=1

# Start controller and agent
python -m infscale start controller --policy static &
CONTROLLER_PID=$!

# Wait a moment for controller to start
sleep 2

# Start agent
python -m infscale start agent id123 &
AGENT_PID=$!

# Wait for both processes
wait $CONTROLLER_PID $AGENT_PID

