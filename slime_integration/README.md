# Slime + NeMo-Gym Integration

This directory contains utilities for integrating [Slime](https://github.com/THUDM/slime) RL training with NeMo-Gym environments.

## Architecture

```
Slime Training Loop (Megatron)
    |
    v
SGLang (inference) <--- NeMo-Gym SGLang Model Server
    |
    v
Slime remote_rm ---> NeMo-Gym Reward Adapter ---> NeMo-Gym Resources Server
    |
    v
reward -> training
```

## Components

### 1. SGLang Model Server (`responses_api_models/sglang_model/`)

NeMo-Gym model server that connects to SGLang inference endpoints. SGLang exposes OpenAI-compatible `/v1/chat/completions`, so this server extends the existing VLLMModel with SGLang-specific features.

### 2. Slime Reward Adapter (`resources_servers/slime_reward_adapter/`)

Bridge server between Slime's `remote_rm` protocol and NeMo-Gym's verify endpoint:
- Accepts Slime's `{prompt, response, label}` POST format
- Translates to NeMo-Gym's `BaseVerifyRequest`
- Can proxy to any upstream NeMo-Gym resources server
- Built-in reward functions: `exact_match`, `contains`

### 3. Custom RM Function (`slime_integration/custom_rm.py`)

Drop-in Slime custom RM that calls NeMo-Gym:
```bash
python train.py ... \
    --custom-rm-path slime_integration.custom_rm.nemogym_rm \
    --rm-url http://localhost:8100
```

### 4. Data Converter (`slime_integration/data_converter.py`)

Convert between Slime and NeMo-Gym JSONL formats:
```bash
python -m slime_integration.data_converter input.jsonl output.jsonl --direction slime_to_nemogym
```

## Quick Start with Docker

```bash
# Build the test image
docker build -t nemogym-slime-test -f docker/slime/Dockerfile .

# Run all integration tests with GPUs
docker run --rm --gpus all --ipc=host --shm-size=16g \
    -e WANDB_MODE=disabled \
    nemogym-slime-test \
    -c "bash /workspace/nemo-gym/docker/slime/run_tests.sh"
```

## Using with Slime Training

### Option A: Slime remote_rm (simplest)

1. Start the NeMo-Gym reward adapter:
   ```bash
   # Inside Docker or with NeMo-Gym installed
   python -c "
   import uvicorn
   from resources_servers.slime_reward_adapter.app import SlimeRewardAdapter, SlimeRewardAdapterConfig
   config = SlimeRewardAdapterConfig(host='0.0.0.0', port=8100, entrypoint='app.py', name='adapter', reward_type='exact_match')
   from unittest.mock import MagicMock
   from nemo_gym.server_utils import ServerClient
   adapter = SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))
   uvicorn.run(adapter.setup_webserver(), host='0.0.0.0', port=8100)
   "
   ```

2. Run Slime training with remote_rm:
   ```bash
   python train.py ... \
       --rm-type remote_rm \
       --rm-url http://localhost:8100/slime_reward
   ```

### Option B: Custom RM function (more control)

```bash
python train.py ... \
    --custom-rm-path slime_integration.custom_rm.nemogym_rm \
    --rm-url http://localhost:8100
```

### Option C: Proxy to existing NeMo-Gym resources server

Configure the adapter to proxy to any NeMo-Gym resources server:
```yaml
slime_reward_adapter:
  resources_servers:
    slime_reward_adapter:
      entrypoint: app.py
      reward_type: proxy
      upstream_server_url: http://localhost:10001  # Your NeMo-Gym resources server
```
