# Dockerfile for Slime + NeMo-Gym integration testing
# Uses the official SGLang image which includes CUDA, torch, and inference deps
FROM lmsysorg/sglang:latest

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy NeMo-Gym source
COPY . /workspace/nemo-gym/

# Install NeMo-Gym core dependencies
RUN pip install --no-build-isolation \
    fastapi uvicorn pydantic pydantic_core aiohttp orjson \
    omegaconf hydra-core "ray[default]" itsdangerous rich \
    pytest pytest-asyncio tqdm psutil yappi devtools \
    "openai<=2.7.2" gprof2dot pydot httpx[http2]

# Install wandb
RUN pip install "wandb>=0.25"

# Fix blinker conflict from debian, then install mlflow
RUN pip install --force-reinstall --no-deps blinker && \
    pip install mlflow-skinny mlflow || true

# Install Slime's dependencies that may be needed for testing
RUN pip install pylatexenc 2>&1 | tail -3

# Verify critical imports work
RUN PYTHONPATH=/workspace/nemo-gym python -c "import wandb; import nemo_gym; print('nemo_gym imports OK')"

# Make nemo_gym importable
ENV PYTHONPATH="/workspace/nemo-gym:${PYTHONPATH}"

EXPOSE 8100 30000

ENTRYPOINT ["/bin/bash"]
