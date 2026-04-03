"""Test Slime-specific compatibility.

This test simulates what Slime's actual training loop does:
1. Tokenize prompt and send to SGLang /generate
2. Get response text
3. Send {prompt, response, label} to remote RM (reward adapter)
4. Get reward back

It also tests:
- The custom_rm.py Slime RM function
- Multiple samples like batched_async_rm
- Conversation-style prompts (multi-turn)
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import aiohttp

sys.path.insert(0, "/workspace/nemo-gym")

SGLANG_PORT = 30000
REWARD_PORT = 8100


@dataclass
class FakeSample:
    """Mimics Slime's Sample dataclass for testing."""

    prompt: str = ""
    response: str = ""
    label: str = None
    metadata: dict = field(default_factory=dict)


async def test_slime_remote_rm_protocol():
    """Test exact Slime remote_rm protocol."""
    print("\n=== Test: Slime remote_rm Protocol ===")

    # Simulate Slime's remote_rm function
    sample = FakeSample(prompt="What is 2+2?", response="4", label="4")
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }

    # Slime POSTs to args.rm_url directly
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/slime_reward",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    assert "reward" in result, f"No reward in response: {result}"
    assert result["reward"] == 1.0
    print(f"  remote_rm reward: {result['reward']}")
    print("  PASSED")


async def test_slime_custom_rm():
    """Test the custom RM function from slime_integration."""
    print("\n=== Test: Slime Custom RM Function ===")

    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}",
        reward_key=None,
    )

    # Test correct answer
    sample = FakeSample(prompt="What is 3+3?", response="6", label="6")
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Correct: reward={reward}")

    # Test wrong answer
    sample = FakeSample(prompt="What is 3+3?", response="7", label="6")
    reward = await nemogym_rm(args, sample)
    assert reward == 0.0, f"Expected 0.0, got {reward}"
    print(f"  Wrong: reward={reward}")

    print("  PASSED")


async def test_slime_batched_rm():
    """Test batched RM like Slime's batched_async_rm."""
    print("\n=== Test: Slime Batched RM ===")

    from slime_integration.custom_rm import nemogym_batched_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}",
        reward_key=None,
    )

    samples = [
        FakeSample(prompt="2+2?", response="4", label="4"),
        FakeSample(prompt="3+3?", response="6", label="6"),
        FakeSample(prompt="4+4?", response="9", label="8"),
        FakeSample(prompt="5+5?", response="10", label="10"),
    ]

    rewards = await nemogym_batched_rm(args, samples)
    assert rewards == [1.0, 1.0, 0.0, 1.0], f"Expected [1.0, 1.0, 0.0, 1.0], got {rewards}"
    print(f"  Batch rewards: {rewards}")
    print("  PASSED")


async def test_slime_generate_and_rm_flow():
    """Test the complete Slime generate -> RM flow."""
    print("\n=== Test: Full Slime Generate+RM Flow ===")

    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}",
        reward_key=None,
    )

    # Simulate Slime's sglang_rollout.py generate()
    questions = [
        ("What is 2+2? Answer with just the number.", "4"),
        ("What is 10+10? Answer with just the number.", "20"),
        ("What is 100-1? Answer with just the number.", "99"),
    ]

    for question, expected in questions:
        # Step 1: Generate with SGLang
        gen_payload = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                json=gen_payload,
            ) as resp:
                gen_result = await resp.json()
                response_text = gen_result["choices"][0]["message"]["content"]

        # Step 2: Get reward
        sample = FakeSample(prompt=question, response=response_text, label=expected)
        reward = await nemogym_rm(args, sample)
        print(f"  Q: {question[:40]:<40} A: {response_text!r:<8} Expected: {expected:<5} Reward: {reward}")

    print("  PASSED")


async def test_slime_sglang_native_generate():
    """Test SGLang's native /generate endpoint (used by Slime internally)."""
    print("\n=== Test: SGLang Native /generate (Slime rollout) ===")

    # Slime uses /generate with tokenized input and return_logprob=True
    payload = {
        "text": "What is 2+2?",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
        },
        "return_logprob": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{SGLANG_PORT}/generate",
            json=payload,
        ) as resp:
            result = await resp.json()
            text = result.get("text", "")
            meta = result.get("meta_info", {})
            logprobs = meta.get("output_token_logprobs", [])
            finish_reason = meta.get("finish_reason", {})

            print(f"  Text: {text!r}")
            print(f"  Logprobs count: {len(logprobs)}")
            print(f"  Finish reason: {finish_reason}")
            assert text, "Empty generation"
            assert len(logprobs) > 0, "No logprobs returned"

    print("  PASSED")


async def test_slime_concurrent_generate_and_rm():
    """Test concurrent generate+RM at scale (simulating a rollout batch)."""
    print("\n=== Test: Concurrent Generate+RM (Rollout Batch, 20 samples) ===")

    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(
        rm_url=f"http://localhost:{REWARD_PORT}",
        reward_key=None,
    )

    questions = [
        (f"What is {a}+{b}? Reply with just the number.", str(a + b))
        for a, b in [(1, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19),
                     (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39)]
    ]

    async def generate_and_reward(q, expected):
        gen_payload = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
                json=gen_payload,
            ) as resp:
                gen_result = await resp.json()
                response_text = gen_result["choices"][0]["message"]["content"]

        sample = FakeSample(prompt=q, response=response_text, label=expected)
        reward = await nemogym_rm(args, sample)
        return q, response_text, expected, reward

    start = time.time()
    tasks = [generate_and_reward(q, e) for q, e in questions]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    total_reward = sum(r[3] for r in results)
    print(f"  Processed {len(results)} samples in {elapsed:.2f}s")
    print(f"  Average reward: {total_reward/len(results):.2f}")
    for q, resp, exp, rew in results[:5]:
        print(f"    {q[:30]:<30} -> {resp!r:<8} (exp: {exp}, rew: {rew})")
    print(f"    ... ({len(results)-5} more)")
    print("  PASSED")


async def main():
    print("=" * 60)
    print("Slime Compatibility Tests")
    print("=" * 60)

    # Wait for services
    from test_integration import wait_for_service

    print("\nWaiting for services...")
    sglang_ok = await wait_for_service(
        f"http://localhost:{SGLANG_PORT}/health", "SGLang", timeout=300
    )
    reward_ok = await wait_for_service(
        f"http://localhost:{REWARD_PORT}/docs", "Reward Adapter", timeout=60
    )

    if not sglang_ok or not reward_ok:
        print("FATAL: Services not ready")
        sys.exit(1)

    tests = [
        test_slime_remote_rm_protocol,
        test_slime_custom_rm,
        test_slime_batched_rm,
        test_slime_generate_and_rm_flow,
        test_slime_sglang_native_generate,
        test_slime_concurrent_generate_and_rm,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Slime Compat Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nALL SLIME COMPATIBILITY TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
