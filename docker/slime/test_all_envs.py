"""Test that all NeMo-Gym environment categories work with Slime.

Tests:
1. Simple verify with verifier_metadata (math, mcqa, code_gen patterns)
2. verifier_metadata passthrough (unit_tests, options, expected_answer)
3. Agent /run mode for tool-calling environments
4. Conversation-format prompts
5. Complex metadata structures
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import aiohttp

sys.path.insert(0, "/workspace/nemo-gym")

REWARD_PORT = 8100


async def wait_for_service(url, name, timeout=60):
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


@dataclass
class FakeSample:
    prompt: str = ""
    response: str = ""
    label: str = None
    metadata: dict = field(default_factory=dict)


async def test_math_environment_pattern():
    """Test math_with_judge pattern: verifier_metadata has question + expected_answer."""
    print("\n=== Test: Math Environment Pattern ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = FakeSample(
        prompt="Solve: What is the integral of 2x?",
        response="x^2 + C",
        label="x^2 + C",
        metadata={
            "verifier_metadata": {
                "question": "What is the integral of 2x?",
                "expected_answer": "x^2 + C",
            }
        },
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Math with verifier_metadata: reward={reward}")
    print("  PASSED")


async def test_mcqa_environment_pattern():
    """Test mcqa pattern: verifier_metadata has options + expected_answer."""
    print("\n=== Test: MCQA Environment Pattern ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = FakeSample(
        prompt="What is the capital of France? A) London B) Paris C) Berlin D) Madrid",
        response="Paris",
        label="Paris",
        metadata={
            "verifier_metadata": {
                "options": ["London", "Paris", "Berlin", "Madrid"],
                "expected_answer": "Paris",
                "grading_mode": "exact",
            }
        },
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  MCQA with options: reward={reward}")
    print("  PASSED")


async def test_code_gen_environment_pattern():
    """Test code_gen pattern: verifier_metadata has unit_tests."""
    print("\n=== Test: Code Gen Environment Pattern ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    # For code_gen, the response would be code. With exact_match adapter,
    # we test that metadata flows correctly.
    sample = FakeSample(
        prompt="Write a function that adds two numbers",
        response="def add(a, b): return a + b",
        label="def add(a, b): return a + b",
        metadata={
            "verifier_metadata": {
                "unit_tests": [
                    {"input": "add(1, 2)", "expected": "3"},
                    {"input": "add(-1, 1)", "expected": "0"},
                ],
                "difficulty": "easy",
            }
        },
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Code gen with unit_tests metadata: reward={reward}")
    print("  PASSED")


async def test_instruction_following_pattern():
    """Test instruction_following pattern: metadata has instruction_id_list."""
    print("\n=== Test: Instruction Following Pattern ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = FakeSample(
        prompt="Write a poem about cats in exactly 4 lines",
        response="Cats are great",
        label="Cats are great",
        metadata={
            "verifier_metadata": {
                "instruction_id_list": ["length:num_lines"],
                "kwargs": [{"num_lines": 4}],
                "grading_mode": "exact",
            }
        },
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Instruction following: reward={reward}")
    print("  PASSED")


async def test_structured_outputs_pattern():
    """Test structured_outputs pattern: metadata has schema_str."""
    print("\n=== Test: Structured Outputs Pattern ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = FakeSample(
        prompt='Output a JSON object with name and age fields: {"name": "Alice", "age": 30}',
        response='{"name": "Alice", "age": 30}',
        label='{"name": "Alice", "age": 30}',
        metadata={
            "verifier_metadata": {
                "schema_str": '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
                "schema_type": "json_schema",
            }
        },
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Structured outputs: reward={reward}")
    print("  PASSED")


async def test_metadata_without_verifier_metadata_key():
    """Test that plain metadata (without explicit verifier_metadata key) also works."""
    print("\n=== Test: Plain Metadata (No verifier_metadata Key) ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    # When metadata doesn't have "verifier_metadata" key, the whole metadata
    # dict is used as verifier_metadata
    sample = FakeSample(
        prompt="What is 2+2?",
        response="4",
        label="4",
        metadata={"rm_type": "math", "difficulty": "easy"},
    )
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Plain metadata passthrough: reward={reward}")
    print("  PASSED")


async def test_no_metadata():
    """Test that samples without metadata still work (label-only)."""
    print("\n=== Test: No Metadata (Label Only) ===")
    from slime_integration.custom_rm import nemogym_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    sample = FakeSample(prompt="What is 2+2?", response="4", label="4")
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  Label-only: reward={reward}")
    print("  PASSED")


async def test_agent_run_endpoint_exists():
    """Test that the /slime_agent_run endpoint is available."""
    print("\n=== Test: Agent Run Endpoint Available ===")

    # Without upstream_agent_url configured, it should return error metadata
    payload = {
        "responses_create_params": {
            "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "test"}]}],
            "model": "test",
        },
        "verifier_metadata": {"expected_answer": "test"},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://localhost:{REWARD_PORT}/slime_agent_run", json=payload) as resp:
            result = await resp.json()
            assert result["reward"] == 0.0
            assert "error" in result.get("metadata", {})
            print(f"  Agent run endpoint (no upstream): reward={result['reward']}, error={result['metadata']['error']}")
    print("  PASSED")


async def test_nemogym_data_with_verifier_metadata():
    """Test that NeMo-Gym JSONL data with verifier_metadata converts correctly to Slime."""
    print("\n=== Test: NeMo-Gym JSONL -> Slime Conversion with verifier_metadata ===")
    from slime_integration.data_converter import nemogym_to_slime, slime_to_nemogym

    # NeMo-Gym format with rich verifier_metadata
    nemogym_row = {
        "responses_create_params": {
            "input": [
                {"role": "user", "type": "message", "content": [{"type": "input_text", "text": "Solve: 2+2"}]}
            ]
        },
        "verifier_metadata": {
            "expected_answer": "4",
            "unit_tests": [{"input": "2+2", "expected": "4"}],
            "difficulty": "easy",
        },
    }

    # Convert to Slime
    slime_row = nemogym_to_slime(nemogym_row)
    assert slime_row["label"] == "4"
    assert "unit_tests" in slime_row.get("metadata", {})
    print(f"  NeMo-Gym -> Slime: label={slime_row['label']}, has unit_tests={True}")

    # Convert back to NeMo-Gym
    roundtrip = slime_to_nemogym(slime_row)
    assert roundtrip["verifier_metadata"]["expected_answer"] == "4"
    assert "unit_tests" in roundtrip["verifier_metadata"]
    print(f"  Roundtrip: verifier_metadata preserved")

    print("  PASSED")


async def test_batch_with_mixed_metadata():
    """Test batch of samples with different metadata structures."""
    print("\n=== Test: Batch with Mixed Metadata Structures ===")
    from slime_integration.custom_rm import nemogym_batched_rm

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_PORT}", reward_key=None)

    samples = [
        # Simple label-only
        FakeSample(prompt="2+2?", response="4", label="4"),
        # With verifier_metadata
        FakeSample(
            prompt="Capital?",
            response="Paris",
            label="Paris",
            metadata={"verifier_metadata": {"options": ["London", "Paris"]}},
        ),
        # Plain metadata
        FakeSample(
            prompt="Code?",
            response="print(1)",
            label="print(1)",
            metadata={"rm_type": "code_gen"},
        ),
        # Wrong answer
        FakeSample(prompt="3+3?", response="7", label="6"),
    ]

    rewards = await nemogym_batched_rm(args, samples)
    assert rewards == [1.0, 1.0, 1.0, 0.0], f"Expected [1.0, 1.0, 1.0, 0.0], got {rewards}"
    print(f"  Mixed batch rewards: {rewards}")
    print("  PASSED")


async def main():
    print("=" * 60)
    print("All-Environment Compatibility Tests")
    print("=" * 60)

    reward_ok = await wait_for_service(f"http://localhost:{REWARD_PORT}/docs", "Reward Adapter")
    if not reward_ok:
        print("FATAL: Reward adapter not ready")
        sys.exit(1)

    tests = [
        test_math_environment_pattern,
        test_mcqa_environment_pattern,
        test_code_gen_environment_pattern,
        test_instruction_following_pattern,
        test_structured_outputs_pattern,
        test_metadata_without_verifier_metadata_key,
        test_no_metadata,
        test_agent_run_endpoint_exists,
        test_nemogym_data_with_verifier_metadata,
        test_batch_with_mixed_metadata,
    ]

    passed = failed = 0
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
    print(f"All-Env Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if failed:
        sys.exit(1)
    print("\nALL ENVIRONMENT TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
