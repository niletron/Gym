"""Custom Slime reward model that calls NeMo-Gym's full agent /run endpoint.

Usage with Slime:
    python train.py ... --custom-rm-path slime_integration.custom_rm.nemogym_rm --rm-url http://host:port

This custom RM sends the Slime sample to a NeMo-Gym reward adapter
(or directly to a NeMo-Gym resources server's /slime_reward endpoint)
and returns the reward.
"""

import aiohttp


async def nemogym_rm(args, sample, **kwargs):
    """Custom Slime RM that calls NeMo-Gym reward adapter.

    Args:
        args: Slime argument namespace (must have args.rm_url)
        sample: Slime Sample object with .prompt, .response, .label
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Reward value from NeMo-Gym
    """
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }

    # Include metadata if present
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

    # Handle dict rewards if reward_key is set
    if isinstance(reward, dict) and hasattr(args, "reward_key") and args.reward_key:
        reward = reward[args.reward_key]

    return reward


async def nemogym_batched_rm(args, samples, **kwargs):
    """Batched version of the NeMo-Gym custom RM.

    Processes all samples concurrently for better throughput.

    Args:
        args: Slime argument namespace
        samples: List of Slime Sample objects
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Rewards for each sample
    """
    import asyncio

    tasks = [nemogym_rm(args, sample, **kwargs) for sample in samples]
    return await asyncio.gather(*tasks)
