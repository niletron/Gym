# Slime Integration for NeMo-Gym

This module enables [Slime](https://github.com/THUDM/slime) (an LLM post-training framework for RL scaling) to use **all 42 NeMo-Gym RLVR environments** as reward sources during training.

## Table of Contents

- [Architecture](#architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [Environment Support](#environment-support)
- [Data Format](#data-format)
- [Configuration Reference](#configuration-reference)
- [Docker Testing](#docker-testing)
- [Advanced Usage](#advanced-usage)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Slime Training Loop                         │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐  │
│  │ Megatron │◄───│  Data    │◄───│  SGLang   │    │ Reward      │  │
│  │ (train)  │    │  Buffer  │    │  (infer)  │    │ (remote RM) │  │
│  └──────────┘    └──────────┘    └───────────┘    └──────┬──────┘  │
│                                                          │         │
└──────────────────────────────────────────────────────────┼─────────┘
                                                           │
                              HTTP POST                    │
                    {prompt, response, label, metadata}    │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   NeMo-Gym Reward Adapter                          │
│                                                                     │
│  Verify mode (simple envs)        Agent mode (tool-calling envs)   │
│  POST /slime_reward ──────►       POST /slime_agent_run ──────►    │
│         │                                  │                        │
│         ▼                                  ▼                        │
│  Resources Server /verify          Agent /run                      │
│  (math, mcqa, code_gen, …)        (tavily, google, ns_tools, …)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Verify mode**: Slime generates text with SGLang → the adapter sends the response + `verifier_metadata` to a NeMo-Gym resources server's `/verify` endpoint → scalar reward returned.

**Agent mode**: The adapter delegates the full generate → tool-call → verify loop to a NeMo-Gym agent's `/run` endpoint. SGLang generation on the Slime side is skipped for these samples.

---

## Components

| Component | Path | Purpose |
|-----------|------|---------|
| **SGLang model server** | `responses_api_models/sglang_model/` | NeMo-Gym model server for SGLang (extends VLLMModel) |
| **Reward adapter** | `resources_servers/slime_reward_adapter/` | Translates Slime ↔ NeMo-Gym protocols |
| **Custom RM functions** | `slime_integration/custom_rm.py` | Drop-in reward functions for Slime's `--custom-rm-path` |
| **Data converter** | `slime_integration/data_converter.py` | Bidirectional JSONL format conversion |
| **Example configs** | `slime_integration/configs/` | Ready-to-use YAML configurations |
| **Docker tests** | `docker/slime/` | Integration test suite (28 tests) |

---

## Quick Start

### 1. Start the Reward Adapter

**Standalone (built-in exact-match reward):**

```bash
python -c "
import uvicorn
from resources_servers.slime_reward_adapter.app import SlimeRewardAdapter, SlimeRewardAdapterConfig
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient

config = SlimeRewardAdapterConfig(
    host='0.0.0.0', port=8100, entrypoint='app.py', name='adapter',
    reward_type='exact_match',
)
adapter = SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))
uvicorn.run(adapter.setup_webserver(), host='0.0.0.0', port=8100)
"
```

**With a NeMo-Gym resources server (proxy mode):**

```bash
# First, start NeMo-Gym servers:
ng_run "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge.yaml, \
                        responses_api_models/sglang_model/configs/sglang_model.yaml]"

# Then point the adapter at the resources server:
# (set upstream_server_url in config or environment)
```

### 2. Run Slime Training

**Option A — Remote RM (simplest):**

```bash
python train.py \
    --prompt-data data/math_train.jsonl \
    --rm-type remote_rm \
    --rm-url http://localhost:8100/slime_reward \
    ...
```

**Option B — Custom RM (recommended, supports metadata):**

```bash
python train.py \
    --prompt-data data/math_train.jsonl \
    --custom-rm-path slime_integration.custom_rm.nemogym_rm \
    --rm-url http://localhost:8100 \
    ...
```

**Option C — Agent RM (tool-calling environments):**

```bash
python train.py \
    --prompt-data data/tavily_train.jsonl \
    --custom-rm-path slime_integration.custom_rm.nemogym_agent_rm \
    --rm-url http://localhost:8100 \
    ...
```

### 3. Convert Data (if needed)

```bash
# Slime JSONL → NeMo-Gym JSONL
python -m slime_integration.data_converter input.jsonl output.jsonl \
    --direction slime_to_nemogym

# NeMo-Gym JSONL → Slime JSONL
python -m slime_integration.data_converter input.jsonl output.jsonl \
    --direction nemogym_to_slime
```

---

## Environment Support

### Verify-only environments (~35)

These environments verify a model's text response against expected output. Slime generates with SGLang, and NeMo-Gym scores the result.

| Environment | `verifier_metadata` fields | Example |
|-------------|---------------------------|---------|
| `math_with_judge` | `question`, `expected_answer` | `{"expected_answer": "42"}` |
| `mcqa` / `gpqa_diamond` | `options`, `expected_answer`, `grading_mode` | `{"options": ["A","B","C"], "expected_answer": "B"}` |
| `code_gen` | `unit_tests`, `difficulty` | `{"unit_tests": [{"input": "f(1)", "expected": "2"}]}` |
| `instruction_following` | `instruction_id_list`, `kwargs` | `{"instruction_id_list": ["length:num_lines"]}` |
| `structured_outputs` | `schema_str`, `schema_type` | `{"schema_str": "{...}", "schema_type": "json_schema"}` |
| `xlam_fc` | `expected_answers` | `{"expected_answers": [{"name": "func", "args": {}}]}` |
| `math_formal_lean` | `header`, `formal_statement` | `{"header": "import Mathlib", ...}` |
| `swerl_gen` | `instance`, `dataset_name` | `{"instance": {...}, "dataset_name": "swe-bench"}` |
| `abstention` | `label`, `expected_answer` | `{"label": "answerable"}` |
| … and 26 more | See each server's `app.py` | |

**Slime data format for verify-only environments:**

```jsonl
{"input": "What is 2+2?", "label": "4", "metadata": {"verifier_metadata": {"expected_answer": "4"}}}
```

### Tool-calling environments (~7)

These environments require multi-step tool interaction. The NeMo-Gym agent handles the full loop.

| Environment | Tools | Description |
|-------------|-------|-------------|
| `tavily_search` | `web_search`, `find_in_page`, `scroll_page` | Web search QA |
| `google_search` | `search`, `browse` | Google search QA |
| `ns_tools` | Dynamic (per-task) | General tool-use |
| `workplace_assistant` | Dynamic (per-env) | Workplace tool-use |
| `openenv` | Dynamic (MCP) | Open-ended environments |
| `math_with_code` | `execute_python`, `end_session` | Math with code execution |
| `calendar` | (verify-only but stateful) | Calendar state verification |

**Slime data format for tool-calling environments:**

```jsonl
{"input": "Search the web for...", "label": "expected answer", "metadata": {"verifier_metadata": {"question": "...", "ground_truth": "..."}}}
```

---

## Data Format

### Slime → NeMo-Gym mapping

| Slime field | NeMo-Gym field | Notes |
|-------------|----------------|-------|
| `input` (str or chat list) | `responses_create_params.input` | Auto-converted to message items |
| `label` | `verifier_metadata.label` + `.expected_answer` | Duplicated for compat |
| `metadata` | `verifier_metadata` | Pass-through |
| `metadata.verifier_metadata` | `verifier_metadata` | Explicit nesting supported |

### Example conversions

**Simple math:**
```json
// Slime
{"input": "What is 2+2?", "label": "4"}

// NeMo-Gym
{"responses_create_params": {"input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "What is 2+2?"}]}], "model": "slime"}, "verifier_metadata": {"label": "4", "expected_answer": "4"}}
```

**MCQA with options:**
```json
// Slime
{"input": "What is the capital of France?", "label": "B", "metadata": {"verifier_metadata": {"options": ["London", "Paris", "Berlin"], "expected_answer": "B", "grading_mode": "exact"}}}

// NeMo-Gym
{"responses_create_params": {"input": [...]}, "verifier_metadata": {"options": ["London", "Paris", "Berlin"], "expected_answer": "B", "grading_mode": "exact", "label": "B"}}
```

**Code generation with unit tests:**
```json
// Slime
{"input": "Write add(a,b)", "label": "def add(a,b): return a+b", "metadata": {"verifier_metadata": {"unit_tests": [{"input": "add(1,2)", "expected": "3"}]}}}
```

---

## Configuration Reference

### Reward adapter (`slime_reward_adapter.yaml`)

```yaml
slime_reward_adapter:
  resources_servers:
    slime_reward_adapter:
      entrypoint: app.py
      domain: slime

      # Built-in reward (no upstream needed): "exact_match" or "contains"
      reward_type: exact_match

      # Proxy to a NeMo-Gym resources server /verify (verify-only envs):
      # upstream_server_url: http://localhost:10001

      # Proxy to a NeMo-Gym agent /run (tool-calling envs):
      # upstream_agent_url: http://localhost:10002

      # Optional system prompt prepended to every request:
      # default_system_prompt: "You are a helpful assistant."
```

### SGLang model server (`sglang_model.yaml`)

```yaml
policy_model:
  responses_api_models:
    sglang_model:
      entrypoint: app.py
      base_url: ${policy_base_url}        # e.g. http://localhost:30000/v1
      api_key: ${policy_api_key}           # e.g. EMPTY
      model: ${policy_model_name}          # e.g. Qwen/Qwen2.5-0.5B-Instruct
      return_token_id_information: false
      uses_reasoning_parser: true

      # SGLang-specific (optional):
      # sglang_router_url: http://localhost:30000
      # sglang_sampling_params:
      #   min_new_tokens: 1
```

### Combined config (`sglang_with_reward_adapter.yaml`)

See `slime_integration/configs/sglang_with_reward_adapter.yaml` for a ready-to-use config that wires up both the SGLang model server and reward adapter.

---

## Docker Testing

### Build and run all 28 tests

```bash
# Build the test image (uses lmsysorg/sglang:latest as base)
docker build -t nemogym-slime-test -f docker/slime/Dockerfile .

# Run with GPU access
docker run --rm --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e WANDB_MODE=disabled \
    nemogym-slime-test \
    -c "bash /workspace/nemo-gym/docker/slime/run_tests.sh"
```

### What the tests cover

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_integration.py` | 12 | SGLang health, chat completions, /generate, reward adapter endpoints, concurrency, logprobs, data converter |
| `test_slime_compat.py` | 6 | Slime remote_rm protocol, custom RM functions, batched RM, generate+RM flow, 20-sample concurrent rollout |
| `test_all_envs.py` | 10 | Math/MCQA/code_gen/instruction_following/structured_outputs patterns, metadata passthrough, agent /run endpoint, mixed batches |

---

## Advanced Usage

### Proxying to a real NeMo-Gym resources server

To use an actual NeMo-Gym environment (e.g. `math_with_judge`) instead of the built-in exact-match:

```bash
# 1. Start the NeMo-Gym resources server
ng_run "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge.yaml]"

# 2. Start the adapter pointing to it
# In your adapter config, set:
#   upstream_server_url: http://localhost:<resources_server_port>
```

### Using the SGLang model server with NeMo-Gym's own rollout collection

```bash
# Start SGLang + adapter + agent
ng_run "+config_paths=[slime_integration/configs/sglang_with_reward_adapter.yaml]"

# Collect rollouts (NeMo-Gym's own pipeline)
ng_collect_rollouts \
    +agent_name=slime_simple_agent \
    +input_jsonl_fpath=resources_servers/slime_reward_adapter/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +num_repeats=3
```

### Custom Slime generate function for agent-mode environments

For tool-calling environments where the NeMo-Gym agent handles generation,
you can write a custom Slime generate function:

```python
# my_generate.py
async def generate_with_nemogym(args, sample, sampling_params, evaluation=False):
    """Replace SGLang generation with NeMo-Gym agent /run."""
    from slime_integration.custom_rm import nemogym_agent_rm
    reward = await nemogym_agent_rm(args, sample)
    sample.reward = reward
    sample.response = ""  # Agent handled generation internally
    return sample
```

```bash
python train.py ... --custom-generate-function-path my_generate.generate_with_nemogym
```
