#!/bin/bash
set -e

# Start NeMo-Gym resources servers for RLVR1 training
# Run this inside the Slime Docker container before launching training

cd /workspace/nemo-gym

SKIP_SERVERS="${SKIP_SERVERS:-}"  # comma-separated list of servers to skip

echo "============================================"
echo "Starting NeMo-Gym Resources Servers"
echo "============================================"

# Initialize Ray if not already running (needed for code_gen)
if ! ray status &>/dev/null 2>&1; then
    echo "Note: Ray not running. code_gen server will initialize Ray on startup."
fi

# Parse skip list
IFS=',' read -ra SKIP_ARR <<< "$SKIP_SERVERS"

should_skip() {
    local name=$1
    for skip in "${SKIP_ARR[@]}"; do
        if [ "$skip" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

start_server() {
    local name=$1
    local port=$2
    if should_skip "$name"; then
        echo "  Skipping $name"
        return
    fi
    echo "  Starting $name on port $port..."
    python -m slime_integration.server_launcher --server-type "$name" --port "$port" &
}

start_server single_step_tool_use 10001
start_server instruction_following 10002
start_server code_gen 10003
start_server math 10004
start_server mcqa 10005
start_server structured_outputs 10006
start_server calendar 10007
start_server reasoning_gym 10008
start_server math_formal_lean 10009
start_server workplace_assistant 10010

echo ""
echo "Waiting for servers to be ready..."
sleep 5

# Health check loop
MAX_WAIT=120
ELAPSED=0
ALL_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    ALL_READY=true
    for port in 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010; do
        if ! curl -s "http://localhost:$port/docs" > /dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    if $ALL_READY; then
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if $ALL_READY; then
    echo ""
    echo "============================================"
    echo "All servers ready!"
    echo "============================================"
else
    echo ""
    echo "WARNING: Some servers not ready after ${MAX_WAIT}s"
    echo "Checking individual servers..."
    for port in 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010; do
        if curl -s "http://localhost:$port/docs" > /dev/null 2>&1; then
            echo "  Port $port: OK"
        else
            echo "  Port $port: NOT READY"
        fi
    done
fi
