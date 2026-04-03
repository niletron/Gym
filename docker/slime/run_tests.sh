#!/bin/bash
set -e

echo "==========================================="
echo "Slime + NeMo-Gym Integration Test Runner"
echo "==========================================="

# Check GPU availability
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPUs detected"
echo ""

# Start SGLang server in background
echo "Starting SGLang server with ${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}..."
python -m sglang.launch_server \
    --model-path "${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --host 0.0.0.0 \
    --port 30000 \
    --dp 1 \
    --tp 1 \
    --mem-fraction-static 0.8 \
    &
SGLANG_PID=$!
echo "SGLang PID: $SGLANG_PID"

# Start NeMo-Gym reward adapter in background
echo "Starting NeMo-Gym reward adapter..."
python -c "
import uvicorn
import sys
sys.path.insert(0, '/workspace/nemo-gym')
from resources_servers.slime_reward_adapter.app import SlimeRewardAdapter, SlimeRewardAdapterConfig
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient

config = SlimeRewardAdapterConfig(
    host='0.0.0.0',
    port=8100,
    entrypoint='app.py',
    name='slime_reward_adapter',
    reward_type='exact_match',
)
adapter = SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))
app = adapter.setup_webserver()
uvicorn.run(app, host='0.0.0.0', port=8100)
" &
REWARD_PID=$!
echo "Reward adapter PID: $REWARD_PID"

# Run integration tests
echo ""
echo "Running integration tests..."
python /workspace/nemo-gym/docker/slime/test_integration.py
TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ]; then
    echo ""
    echo "Running Slime compatibility tests..."
    python /workspace/nemo-gym/docker/slime/test_slime_compat.py
    TEST_EXIT=$?
fi

if [ $TEST_EXIT -eq 0 ]; then
    echo ""
    echo "Running all-environment compatibility tests..."
    python /workspace/nemo-gym/docker/slime/test_all_envs.py
    TEST_EXIT=$?
fi

# Cleanup
echo ""
echo "Cleaning up..."
kill $SGLANG_PID 2>/dev/null || true
kill $REWARD_PID 2>/dev/null || true

exit $TEST_EXIT
