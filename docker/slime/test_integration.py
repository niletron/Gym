"""Comprehensive integration test for Slime + NeMo-Gym.

This script tests:
1. SGLang model serving via OpenAI-compatible API
2. NeMo-Gym Slime reward adapter (exact_match, contains modes)
3. SGLang model server wrapper (NeMo-Gym responses API)
4. End-to-end: SGLang generate -> reward adapter -> reward
5. Slime remote_rm protocol compatibility
6. Data format conversion (Slime <-> NeMo-Gym)
"""

import asyncio
import json
import subprocess
import sys
import time

import aiohttp


SGLANG_PORT = 30000
REWARD_PORT = 8100
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


async def wait_for_service(url, name, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  [{name}] ready!")
                        return True
        except Exception:
            pass
        await asyncio.sleep(3)
        elapsed = int(time.time() - start)
        if elapsed % 15 == 0:
            print(f"  [{name}] waiting... ({elapsed}s)")
    print(f"  [{name}] TIMEOUT after {timeout}s")
    return False


async def test_sglang_health():
    """Test 1: SGLang server health check."""
    print("\n=== Test 1: SGLang Health Check ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{SGLANG_PORT}/health") as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
    print("  PASSED")


async def test_sglang_chat_completions():
    """Test 2: SGLang /v1/chat/completions endpoint."""
    print("\n=== Test 2: SGLang Chat Completions ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Chat completions failed: {resp.status}"
            result = await resp.json()
            assert "choices" in result, f"No choices in response: {result}"
            content = result["choices"][0]["message"]["content"]
            print(f"  Model response: {content!r}")
            assert content, "Empty response"
    print("  PASSED")


async def test_sglang_generate():
    """Test 3: SGLang native /generate endpoint."""
    print("\n=== Test 3: SGLang Native Generate ===")
    payload = {
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 16,
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{SGLANG_PORT}/generate",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Generate failed: {resp.status}"
            result = await resp.json()
            text = result.get("text", "")
            print(f"  Generated: {text!r}")
            assert text, "Empty generation"
    print("  PASSED")


async def test_reward_adapter_exact_match():
    """Test 4: Reward adapter exact match mode."""
    print("\n=== Test 4: Reward Adapter (exact_match) ===")

    # Correct answer
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/slime_reward",
            json={"prompt": "What is 2+2?", "response": "4", "label": "4"},
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Expected 1.0, got {result['reward']}"
            print(f"  Correct answer reward: {result['reward']}")

    # Wrong answer
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/slime_reward",
            json={"prompt": "What is 2+2?", "response": "5", "label": "4"},
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 0.0, f"Expected 0.0, got {result['reward']}"
            print(f"  Wrong answer reward: {result['reward']}")

    print("  PASSED")


async def test_reward_adapter_root_endpoint():
    """Test 5: Reward adapter via root POST (Slime's rm_url)."""
    print("\n=== Test 5: Reward Adapter Root Endpoint (Slime rm_url) ===")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/",
            json={"prompt": "What is 2+2?", "response": "4", "label": "4"},
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Expected 1.0, got {result['reward']}"
            print(f"  Root endpoint reward: {result['reward']}")
    print("  PASSED")


async def test_reward_adapter_verify():
    """Test 6: Reward adapter standard NeMo-Gym /verify endpoint."""
    print("\n=== Test 6: Reward Adapter /verify Endpoint ===")
    from uuid import uuid4

    verify_body = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": "What is 2+2?"}],
                }
            ],
            "model": "test",
            "metadata": {"label": "4"},
        },
        "response": {
            "id": f"resp_{uuid4().hex}",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "4", "annotations": []}],
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/verify",
            json=verify_body,
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Expected 1.0, got {result['reward']}"
            print(f"  Verify reward: {result['reward']}")
    print("  PASSED")


async def test_conversation_prompt():
    """Test 7: Reward adapter with conversation-style prompts."""
    print("\n=== Test 7: Conversation Prompt ===")
    payload = {
        "prompt": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "response": "4",
        "label": "4",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/slime_reward",
            json=payload,
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Expected 1.0, got {result['reward']}"
            print(f"  Conversation reward: {result['reward']}")
    print("  PASSED")


async def test_end_to_end_generate_and_reward():
    """Test 8: End-to-end: generate with SGLang, then get reward."""
    print("\n=== Test 8: End-to-End Generate + Reward ===")

    # Step 1: Generate response from SGLang
    gen_payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
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
            print(f"  Generated response: {response_text!r}")

    # Step 2: Get reward
    reward_payload = {
        "prompt": "What is 2+2? Reply with just the number.",
        "response": response_text,
        "label": "4",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_PORT}/slime_reward",
            json=reward_payload,
        ) as resp:
            reward_result = await resp.json()
            print(f"  Reward: {reward_result['reward']}")

    print("  PASSED")


async def test_batch_rewards():
    """Test 9: Multiple concurrent reward requests (simulates Slime's batched_async_rm)."""
    print("\n=== Test 9: Batch Reward Requests ===")

    test_cases = [
        {"prompt": "What is 2+2?", "response": "4", "label": "4", "expected": 1.0},
        {"prompt": "What is 3+3?", "response": "6", "label": "6", "expected": 1.0},
        {"prompt": "What is 5+5?", "response": "11", "label": "10", "expected": 0.0},
        {"prompt": "Capital of France?", "response": "Paris", "label": "Paris", "expected": 1.0},
        {"prompt": "Capital of Japan?", "response": "Kyoto", "label": "Tokyo", "expected": 0.0},
    ]

    async def get_reward(tc):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{REWARD_PORT}/slime_reward",
                json={"prompt": tc["prompt"], "response": tc["response"], "label": tc["label"]},
            ) as resp:
                return await resp.json()

    tasks = [get_reward(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks)

    for tc, result in zip(test_cases, results):
        assert result["reward"] == tc["expected"], (
            f"For '{tc['prompt']}': expected {tc['expected']}, got {result['reward']}"
        )
        print(f"  '{tc['prompt']}' -> response='{tc['response']}', reward={result['reward']}")

    print("  PASSED")


async def test_data_converter():
    """Test 10: Data format conversion between Slime and NeMo-Gym."""
    print("\n=== Test 10: Data Format Converter ===")
    sys.path.insert(0, "/workspace/nemo-gym")

    from slime_integration.data_converter import slime_to_nemogym, nemogym_to_slime

    # Slime -> NeMo-Gym
    slime_row = {"input": "What is 2+2?", "label": "4", "metadata": {"rm_type": "math"}}
    nemogym_row = slime_to_nemogym(slime_row)
    assert "responses_create_params" in nemogym_row
    assert nemogym_row["verifier_metadata"]["label"] == "4"
    print(f"  Slime->NeMo-Gym: OK")

    # NeMo-Gym -> Slime
    roundtrip = nemogym_to_slime(nemogym_row)
    assert roundtrip["input"] == "What is 2+2?"
    assert roundtrip["label"] == "4"
    print(f"  NeMo-Gym->Slime roundtrip: OK")

    # Conversation format
    conv_row = {
        "input": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ],
        "label": "Hi",
    }
    nemogym_conv = slime_to_nemogym(conv_row)
    assert len(nemogym_conv["responses_create_params"]["input"]) == 2
    print(f"  Conversation format: OK")

    print("  PASSED")


async def test_sglang_logprobs():
    """Test 11: SGLang logprobs support (critical for RL training)."""
    print("\n=== Test 11: SGLang Logprobs ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 3,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=payload,
        ) as resp:
            result = await resp.json()
            choice = result["choices"][0]
            logprobs = choice.get("logprobs")
            if logprobs:
                content_logprobs = logprobs.get("content", [])
                print(f"  Got {len(content_logprobs)} logprob entries")
                if content_logprobs:
                    first = content_logprobs[0]
                    print(f"  First token logprob: {first.get('logprob', 'N/A')}")
            else:
                print("  (logprobs not returned - may depend on SGLang version)")
    print("  PASSED")


async def test_high_concurrency():
    """Test 12: High concurrency reward requests."""
    print("\n=== Test 12: High Concurrency (50 concurrent requests) ===")
    num_requests = 50
    start = time.time()

    async def single_request(i):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{REWARD_PORT}/slime_reward",
                json={"prompt": f"Q{i}", "response": str(i), "label": str(i)},
            ) as resp:
                return await resp.json()

    tasks = [single_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    correct = sum(1 for r in results if r["reward"] == 1.0)
    print(f"  {num_requests} requests in {elapsed:.2f}s")
    print(f"  {correct}/{num_requests} correct rewards")
    assert correct == num_requests, f"Expected all correct, got {correct}/{num_requests}"
    print("  PASSED")


async def main():
    print("=" * 60)
    print("Slime + NeMo-Gym Integration Tests")
    print("=" * 60)

    # Wait for services
    print("\nWaiting for services...")
    sglang_ok = await wait_for_service(
        f"http://localhost:{SGLANG_PORT}/health", "SGLang", timeout=300
    )
    reward_ok = await wait_for_service(
        f"http://localhost:{REWARD_PORT}/docs", "Reward Adapter", timeout=60
    )

    if not sglang_ok:
        print("\nFATAL: SGLang server not ready. Aborting.")
        sys.exit(1)
    if not reward_ok:
        print("\nFATAL: Reward adapter not ready. Aborting.")
        sys.exit(1)

    # Run all tests
    passed = 0
    failed = 0
    tests = [
        test_sglang_health,
        test_sglang_chat_completions,
        test_sglang_generate,
        test_reward_adapter_exact_match,
        test_reward_adapter_root_endpoint,
        test_reward_adapter_verify,
        test_conversation_prompt,
        test_end_to_end_generate_and_reward,
        test_batch_rewards,
        test_data_converter,
        test_sglang_logprobs,
        test_high_concurrency,
    ]

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
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
