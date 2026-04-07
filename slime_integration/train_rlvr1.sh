#!/bin/bash
# Train Qwen2.5-0.5B on RLVR1 data using Slime + NeMo-Gym
#
# Prerequisites:
#   1. NeMo-Gym resources servers running (start_all_servers.sh)
#   2. Data converted to Slime format (rlvr1_to_slime.py)
#   3. Model downloaded (Qwen/Qwen2.5-0.5B-Instruct)
#
# Usage:
#   # Quick test (5 rollouts, small batch)
#   MODE=test bash train_rlvr1.sh
#
#   # Full training
#   MODE=full bash train_rlvr1.sh

set -ex

MODE="${MODE:-test}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_PATH="${DATA_PATH:-/data/rlvr1_slime.jsonl}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
SAVE_PATH="${SAVE_PATH:-/root/output/rlvr1_qwen05b}"
ROUTING_TABLE="${ROUTING_TABLE:-/workspace/nemo-gym/routing_table.json}"

export PYTHONPATH="/workspace/nemo-gym:${PYTHONPATH}"
export NEMOGYM_ROUTING_TABLE="${ROUTING_TABLE}"

# Download model if not present
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading Qwen2.5-0.5B-Instruct..."
    mkdir -p "$(dirname $MODEL_PATH)"
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_PATH"
fi

# Start Ray if not running
if ! ray status &>/dev/null 2>&1; then
    ray start --head --node-ip-address 127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats
fi

# Mode-specific settings
if [ "$MODE" = "test" ]; then
    NUM_ROLLOUT=5
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES=2
    GLOBAL_BATCH_SIZE=16
    MAX_RESPONSE_LEN=512
    EVAL_ARGS=""
    WANDB_ARGS="--use-wandb false"
elif [ "$MODE" = "full" ]; then
    NUM_ROLLOUT=200
    ROLLOUT_BATCH_SIZE=32
    N_SAMPLES=4
    GLOBAL_BATCH_SIZE=128
    MAX_RESPONSE_LEN=2048
    EVAL_ARGS=""
    WANDB_ARGS="--use-wandb --wandb-project slime-rlvr1 --wandb-group qwen05b-rlvr1"
else
    echo "Unknown MODE: $MODE"
    exit 1
fi

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{
        \"env_vars\": {
            \"PYTHONPATH\": \"/workspace/nemo-gym\",
            \"NEMOGYM_ROUTING_TABLE\": \"${ROUTING_TABLE}\",
            \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
        }
    }" \
    -- python3 train.py \
    --hf-checkpoint "$MODEL_PATH" \
    --save "$SAVE_PATH" \
    --save-interval 50 \
    \
    --prompt-data "$DATA_PATH" \
    --input-key messages \
    --label-key label \
    --metadata-key metadata \
    --tool-key tools \
    --apply-chat-template \
    --rollout-shuffle \
    --custom-rm-path slime_integration.routing_rm.routing_batched_rm \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES" \
    --rollout-max-response-len "$MAX_RESPONSE_LEN" \
    --rollout-temperature 1.0 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
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
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$NUM_GPUS" \
    --colocate \
    --train-backend fsdp \
    --update-weight-buffer-size 536870912 \
    \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.4 \
    \
    $EVAL_ARGS \
    $WANDB_ARGS
