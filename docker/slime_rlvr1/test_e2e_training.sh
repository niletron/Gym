#!/bin/bash
# End-to-end training test: servers + Slime + Qwen 0.5B
set -ex

cd /workspace/nemo-gym

# 1. Start resources servers (skip lean and code_gen for quick test)
echo "=== Starting NeMo-Gym resources servers ==="
for server in mcqa math structured_outputs calendar reasoning_gym; do
    echo "  Starting $server..."
    python -m slime_integration.server_launcher --server-type $server &
done

sleep 10

# Health check
for port in 10004 10005 10006 10007 10008; do
    if curl -s "http://localhost:$port/docs" > /dev/null 2>&1; then
        echo "  Port $port: OK"
    else
        echo "  Port $port: NOT READY - waiting..."
        sleep 10
        if curl -s "http://localhost:$port/docs" > /dev/null 2>&1; then
            echo "  Port $port: OK (retry)"
        else
            echo "  Port $port: STILL NOT READY"
        fi
    fi
done

# 2. Download model if needed
MODEL_PATH="/root/models/Qwen2.5-0.5B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "=== Downloading Qwen2.5-0.5B-Instruct ==="
    mkdir -p /root/models
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_PATH"
fi

# 3. Create a tiny test subset (5 samples from envs we started)
echo "=== Creating test data ==="
python3 -c "
import json
# Filter test data to only envs we have running
running_envs = {'mcqa', 'math', 'structured_outputs', 'calendar', 'reasoning_gym'}
kept = 0
with open('/data/rlvr1_slime_test50.jsonl') as fin, open('/tmp/train_test.jsonl', 'w') as fout:
    for line in fin:
        d = json.loads(line)
        if d['metadata']['env_type'] in running_envs:
            fout.write(line)
            kept += 1
print(f'Kept {kept} samples for training test')
"

# 4. Start Ray
echo "=== Starting Ray ==="
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

# 5. Run training (2 rollouts, minimal batch)
echo "=== Running training (2 rollouts) ==="
export NEMOGYM_ROUTING_TABLE="/workspace/nemo-gym/routing_table.json"

# Find Slime's train.py
SLIME_DIR=""
for d in /workspace/slime /root/slime; do
    if [ -f "$d/train.py" ]; then
        SLIME_DIR="$d"
        break
    fi
done
if [ -z "$SLIME_DIR" ]; then
    echo "ERROR: Cannot find Slime train.py"
    exit 1
fi
echo "Using Slime at: $SLIME_DIR"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{
        \"working_dir\": \"$SLIME_DIR\",
        \"env_vars\": {
            \"PYTHONPATH\": \"/workspace/nemo-gym:$SLIME_DIR\",
            \"NEMOGYM_ROUTING_TABLE\": \"/workspace/nemo-gym/routing_table.json\",
            \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
        }
    }" \
    -- python3 $SLIME_DIR/train.py \
    --hf-checkpoint "$MODEL_PATH" \
    --save /tmp/output_test \
    \
    --prompt-data /tmp/train_test.jsonl \
    --input-key messages \
    --label-key label \
    --metadata-key metadata \
    --apply-chat-template \
    --rollout-shuffle \
    --custom-rm-path slime_integration.routing_rm.routing_batched_rm \
    --num-rollout 2 \
    --rollout-batch-size 4 \
    --n-samples-per-prompt 2 \
    --rollout-max-response-len 256 \
    --rollout-temperature 1.0 \
    --global-batch-size 8 \
    \
    --advantage-estimator grpo \
    --kl-loss-coef 0.00 \
    --kl-coef 0.00 \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    --train-backend fsdp \
    --update-weight-buffer-size 536870912 \
    \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.4

echo ""
echo "=== E2E Training Test Complete ==="
