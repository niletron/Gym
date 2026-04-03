"""Custom Slime reward-model functions that call NeMo-Gym environments.

Two modes
---------
``nemogym_rm``
    *Verify-only* environments (math, mcqa, code_gen, …).
    Slime generates with SGLang → this RM sends the response text to the
    reward adapter's ``/slime_reward`` endpoint → reward returned.

``nemogym_agent_rm``
    *Tool-calling* environments (tavily_search, google_search, ns_tools, …).
    This RM posts to ``/slime_agent_run`` which delegates to the NeMo-Gym
    agent's ``/run`` endpoint for the full generate → tool → verify loop.

Usage with Slime::

    # Verify-only environments
    python train.py ... \\
        --custom-rm-path slime_integration.custom_rm.nemogym_rm \\
        --rm-url http://localhost:8100

    # Tool-calling environments
    python train.py ... \\
        --custom-rm-path slime_integration.custom_rm.nemogym_agent_rm \\
        --rm-url http://localhost:8100
"""

import asyncio

import aiohttp

# ---------------------------------------------------------------------------
# Verify-only environments
# ---------------------------------------------------------------------------


async def nemogym_rm(args, sample, **kwargs):
    """Score a Slime ``Sample`` against a NeMo-Gym environment (verify mode).

    The adapter receives ``{prompt, response, label, metadata}`` and returns a
    scalar reward.  ``sample.metadata`` can carry ``verifier_metadata`` for
    rich NeMo-Gym environments (unit tests, options, schemas, …).
    """
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
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

    return _extract_reward(result, args)


async def nemogym_batched_rm(args, samples, **kwargs):
    """Concurrent version of :func:`nemogym_rm`."""
    return await asyncio.gather(*(nemogym_rm(args, s, **kwargs) for s in samples))


# ---------------------------------------------------------------------------
# Tool-calling environments
# ---------------------------------------------------------------------------


async def nemogym_agent_rm(args, sample, **kwargs):
    """Score via a NeMo-Gym *agent* ``/run`` (tool-calling environments).

    The adapter delegates the full generate → tool-call → verify loop to
    NeMo-Gym, so Slime's own SGLang generation is not used for this sample.

    ``sample.metadata`` should contain either:
    * ``responses_create_params`` (NeMo-Gym native prompt), or
    * a plain prompt (auto-converted).
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rcp = metadata.get("responses_create_params") or _prompt_to_rcp(sample.prompt)
    vm = metadata.get("verifier_metadata", {})

    if sample.label is not None:
        vm.setdefault("label", sample.label)
        vm.setdefault("expected_answer", sample.label)

    payload = {"responses_create_params": rcp, "verifier_metadata": vm}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{args.rm_url}/slime_agent_run",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    return _extract_reward(result, args)


async def nemogym_agent_batched_rm(args, samples, **kwargs):
    """Concurrent version of :func:`nemogym_agent_rm`."""
    return await asyncio.gather(*(nemogym_agent_rm(args, s, **kwargs) for s in samples))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_reward(result: dict, args) -> float:
    reward = result.get("reward", 0.0)
    if isinstance(reward, dict) and getattr(args, "reward_key", None):
        reward = reward[args.reward_key]
    return reward


def _prompt_to_rcp(prompt) -> dict:
    """Convert a Slime prompt (str or chat list) to ``responses_create_params``."""
    if isinstance(prompt, str):
        messages = [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": prompt}]}]
    elif isinstance(prompt, list):
        messages = []
        for m in prompt:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
            messages.append({"role": role, "type": "message", "content": content})
    else:
        messages = [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": str(prompt)}]}]
    return {"input": messages, "model": "slime"}
