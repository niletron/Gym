"""Test with actual Slime source code.

Verifies that:
1. Slime's Sample dataclass works with our custom RM
2. Slime's remote_rm function can call our adapter
3. Slime's batched_async_rm works with our adapter
4. Slime's data loading produces samples our RM can score

Only runs if /workspace/slime exists (mounted at runtime).
"""

import asyncio
import os
import sys
import time

import aiohttp

sys.path.insert(0, "/workspace/nemo-gym")
sys.path.insert(0, "/workspace/slime")

REWARD_PORT = 8100


async def wait_for_service(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


async def test_slime_sample_with_nemogym_rm():
    """Test that Slime's real Sample dataclass works with our custom RM."""
    print("\n=== Test: Slime Sample → nemogym_rm ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = Sample(prompt="What is 2+2?", response="4", label="4")
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Slime Sample correct: reward={reward} ✓")

    sample = Sample(prompt="What is 2+2?", response="5", label="4")
    reward = await nemogym_rm(args, sample)
    assert reward == 0.0, f"Expected 0.0, got {reward}"
    print(f"  Slime Sample wrong: reward={reward} ✓")

    print("  PASSED")


async def test_slime_sample_with_metadata():
    """Test Slime Sample with metadata passthrough."""
    print("\n=== Test: Slime Sample with Metadata ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = Sample(
        prompt="What is the capital of France?",
        response="Paris",
        label="Paris",
        metadata={"verifier_metadata": {"options": ["London", "Paris", "Berlin"], "expected_answer": "Paris"}},
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Sample with verifier_metadata: reward={reward} ✓")

    print("  PASSED")


async def test_slime_remote_rm_function():
    """Test Slime's actual remote_rm function against our adapter."""
    print("\n=== Test: Slime remote_rm Function ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime.rollout.rm_hub import remote_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}/slime_reward")

    sample = Sample(prompt="What is 2+2?", response="4", label="4")
    result = await remote_rm(args, sample)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result["reward"] == 1.0, f"Expected reward 1.0, got {result}"
    print(f"  Slime remote_rm: result={result} ✓")

    print("  PASSED")


async def test_slime_async_rm_with_remote():
    """Test Slime's async_rm routing to remote_rm."""
    print("\n=== Test: Slime async_rm (remote_rm type) ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime.rollout.rm_hub import async_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}/slime_reward",
        rm_type="remote_rm",
        custom_rm_path=None,
        reward_key=None,
    )

    sample = Sample(prompt="What is 3+3?", response="6", label="6")
    result = await async_rm(args, sample)
    # remote_rm returns the full dict
    if isinstance(result, dict):
        assert result["reward"] == 1.0
    else:
        assert result == 1.0
    print(f"  async_rm(remote_rm): {result} ✓")

    print("  PASSED")


async def test_slime_batched_async_rm():
    """Test Slime's batched_async_rm with our adapter."""
    print("\n=== Test: Slime batched_async_rm ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime.rollout.rm_hub import batched_async_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}/slime_reward",
        rm_type="remote_rm",
        custom_rm_path=None,
        reward_key=None,
    )

    samples = [
        Sample(prompt="1+1?", response="2", label="2"),
        Sample(prompt="2+2?", response="4", label="4"),
        Sample(prompt="3+3?", response="7", label="6"),
    ]

    results = await batched_async_rm(args, samples)
    print(f"  batched results: {results}")

    # remote_rm returns dicts, extract rewards
    rewards = []
    for r in results:
        if isinstance(r, dict):
            rewards.append(r["reward"])
        else:
            rewards.append(r)

    assert rewards[0] == 1.0
    assert rewards[1] == 1.0
    assert rewards[2] == 0.0
    print(f"  rewards: {rewards} ✓")

    print("  PASSED")


async def test_slime_custom_rm_path():
    """Test using our RM via Slime's --custom-rm-path mechanism."""
    print("\n=== Test: Slime custom_rm_path Loading ===")
    from types import SimpleNamespace
    from slime.utils.types import Sample
    from slime.utils.misc import load_function

    # This is what Slime does internally with --custom-rm-path
    rm_fn = load_function("slime_integration.custom_rm.nemogym_rm")
    assert callable(rm_fn), f"Expected callable, got {type(rm_fn)}"
    print(f"  load_function: loaded {rm_fn.__name__} ✓")

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)
    sample = Sample(prompt="5+5?", response="10", label="10")
    reward = await rm_fn(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Loaded RM reward: {reward} ✓")

    print("  PASSED")


async def main():
    if not os.path.exists("/workspace/slime"):
        print("Slime source not found at /workspace/slime — skipping")
        return

    print("=" * 60)
    print("Slime Source Code Integration Tests")
    print("=" * 60)

    ok = await wait_for_service(f"http://localhost:{REWARD_PORT}/docs")
    if not ok:
        print("FATAL: Reward adapter not ready")
        sys.exit(1)

    tests = [
        test_slime_sample_with_nemogym_rm,
        test_slime_sample_with_metadata,
        test_slime_remote_rm_function,
        test_slime_async_rm_with_remote,
        test_slime_batched_async_rm,
        test_slime_custom_rm_path,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            await fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Slime Source Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if failed:
        sys.exit(1)
    print("\nALL SLIME SOURCE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
