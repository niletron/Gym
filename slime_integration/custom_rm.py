"""Custom Slime reward models that call NeMo-Gym environments.

Supports ALL NeMo-Gym environments through two modes:

1. **nemogym_rm** (simple environments): Slime generates with SGLang, then this
   RM calls the reward adapter's /slime_reward endpoint with the response text
   and verifier_metadata. Works for: math, mcqa, code_gen, instruction_following, etc.

2. **nemogym_agent_rm** (tool-calling environments): This RM calls the reward
   adapter's /slime_agent_run endpoint which delegates to NeMo-Gym's agent /run.
   The agent handles generate -> tool call -> verify. Works for: tavily_search,
   google_search, ns_tools, workplace_assistant, openenv, etc.

Usage with Slime:
    # Simple environments (Slime generates, NeMo-Gym verifies):
    python train.py ... --custom-rm-path slime_integration.custom_rm.nemogym_rm \\
        --rm-url http://localhost:8100

    # Tool-calling environments (NeMo-Gym agent generates + verifies):
    python train.py ... --custom-rm-path slime_integration.custom_rm.nemogym_agent_rm \\
        --rm-url http://localhost:8100
"""

import aiohttp


async def nemogym_rm(args, sample, **kwargs):
    """Custom Slime RM for simple verify-only NeMo-Gym environments.

    Sends {prompt, response, label, metadata} to the reward adapter.
    The metadata field should contain verifier_metadata for the target
    NeMo-Gym environment (e.g., unit_tests, expected_answer, options, etc.).

    Args:
        args: Slime argument namespace (must have args.rm_url)
        sample: Slime Sample object with .prompt, .response, .label, .metadata
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Reward value from NeMo-Gym
    """
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }

    # Pass through metadata (contains verifier_metadata for NeMo-Gym environments)
    if hasattr(sample, "metadata") and sample.metadata:
        payload["metadata"] = sample.metadata

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{args.rm_url}/slime_reward",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    reward = result.get("reward", 0.0)
    if isinstance(reward, dict) and hasattr(args, "reward_key") and args.reward_key:
        reward = reward[args.reward_key]
    return reward


async def nemogym_agent_rm(args, sample, **kwargs):
    """Custom Slime RM for tool-calling NeMo-Gym environments.

    Instead of just verifying the response text, this RM delegates to
    the NeMo-Gym agent's /run endpoint which handles the full
    generate -> tool call -> verify loop.

    The Slime sample's metadata must contain:
    - responses_create_params: The NeMo-Gym prompt format
    - verifier_metadata: Task-specific verification data

    For this mode, Slime's SGLang generation is SKIPPED - the NeMo-Gym
    agent handles generation internally. Use a custom generate function
    that returns the agent's response (see nemogym_generate below).

    Args:
        args: Slime argument namespace (must have args.rm_url)
        sample: Slime Sample object
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Reward value from NeMo-Gym
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    # Build the agent run request
    # If metadata has explicit responses_create_params, use it
    if "responses_create_params" in metadata:
        rcp = metadata["responses_create_params"]
    else:
        # Convert Slime's prompt to NeMo-Gym format
        prompt = sample.prompt
        if isinstance(prompt, str):
            input_messages = [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ]
        elif isinstance(prompt, list):
            input_messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "input_text", "text": content}]
                input_messages.append({"role": role, "type": "message", "content": content})
        else:
            input_messages = [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": str(prompt)}],
                }
            ]
        rcp = {"input": input_messages, "model": "slime"}

    verifier_metadata = metadata.get("verifier_metadata", {})
    if sample.label is not None:
        verifier_metadata["label"] = sample.label
        verifier_metadata["expected_answer"] = sample.label

    payload = {
        "responses_create_params": rcp,
        "verifier_metadata": verifier_metadata,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{args.rm_url}/slime_agent_run",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    reward = result.get("reward", 0.0)
    if isinstance(reward, dict) and hasattr(args, "reward_key") and args.reward_key:
        reward = reward[args.reward_key]
    return reward


async def nemogym_batched_rm(args, samples, **kwargs):
    """Batched version of nemogym_rm. Processes all samples concurrently."""
    import asyncio

    tasks = [nemogym_rm(args, sample, **kwargs) for sample in samples]
    return await asyncio.gather(*tasks)


async def nemogym_agent_batched_rm(args, samples, **kwargs):
    """Batched version of nemogym_agent_rm. Processes all samples concurrently."""
    import asyncio

    tasks = [nemogym_agent_rm(args, sample, **kwargs) for sample in samples]
    return await asyncio.gather(*tasks)
