FROM ${BASE_IMAGE}

# Copy NeMo-Gym into the container
COPY . /workspace/nemo-gym
WORKDIR /workspace/nemo-gym

# Install NeMo-Gym core
RUN pip install -e ".[dev]" 2>&1 | tail -5

# Install per-server dependencies
RUN pip install \
    "math-verify>=0.8.0" \
    "reasoning-gym>=0.1.19" \
    "openapi-schema-validator>=0.6.3" \
    "xmltodict>=1.0.2" \
    "datasets>=2.19.0" \
    "pylatexenc" \
    "rich" \
    2>&1 | tail -5

# instruction_following needs verifiable_instructions
RUN pip install "verifiable_instructions @ git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval" 2>&1 | tail -3 || \
    echo "WARNING: verifiable_instructions install failed - instruction_following server will not work"

# Add server-local deps to PYTHONPATH (several servers use local imports)
ENV PYTHONPATH="/workspace/nemo-gym:\
/workspace/nemo-gym/resources_servers/code_gen:\
/workspace/nemo-gym/resources_servers/calendar:\
/workspace/nemo-gym/resources_servers/single_step_tool_use_with_argument_comparison:\
/workspace/nemo-gym/resources_servers/workplace_assistant:\
/workspace/nemo-gym/resources_servers/math_formal_lean:\
/workspace/nemo-gym/resources_servers/instruction_following:\
${PYTHONPATH}"

# Verify key imports
RUN python -c "import nemo_gym; print('nemo_gym OK')" && \
    python -c "from resources_servers.mcqa.app import MCQAResourcesServer; print('mcqa OK')" && \
    python -c "from resources_servers.code_gen.app import CompCodingResourcesServer; print('code_gen OK')" && \
    python -c "from slime_integration.routing_rm import routing_batched_rm; print('routing_rm OK')"

# Copy routing table
COPY docker/slime_rlvr1/routing_table.json /workspace/nemo-gym/routing_table.json

ENV PYTHONPATH="/workspace/nemo-gym:${PYTHONPATH}"
ENV NEMOGYM_ROUTING_TABLE="/workspace/nemo-gym/routing_table.json"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
