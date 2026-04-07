"""Test the routing RM end-to-end with real NeMo-Gym servers."""

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import aiohttp

sys.path.insert(0, "/workspace/nemo-gym")


@dataclass
class FakeSample:
    prompt: str = ""
    response: str = ""
    label: str = None
    metadata: dict = field(default_factory=dict)


async def wait_for_service(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


async def main():
    print("=" * 60)
    print("Routing RM End-to-End Tests")
    print("=" * 60)

    # Start servers
    servers = {"mcqa": 10005, "math": 10004, "structured_outputs": 10006}
    procs = []
    for stype, port in servers.items():
        p = subprocess.Popen(
            [sys.executable, "-m", "slime_integration.server_launcher", "--server-type", stype, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)

    # Wait
    print("Waiting for servers...")
    for stype, port in servers.items():
        ok = await wait_for_service(f"http://localhost:{port}/docs", timeout=60)
        if not ok:
            print(f"  {stype}: TIMEOUT")
            for p in procs:
                p.kill()
            sys.exit(1)
        print(f"  {stype}: ready")

    # Import routing RM
    from slime_integration.routing_rm import routing_rm, routing_batched_rm

    args = SimpleNamespace(rm_url=None, custom_rm_path=None, reward_key=None)

    passed = 0
    failed = 0

    # Test 1: Single MCQA sample
    print("\n=== Test: Single MCQA sample ===")
    sample = FakeSample(
        prompt=[{"role": "user", "content": "Pick A or B"}],
        response="Answer: B",
        metadata={
            "env_type": "mcqa",
            "expected_answer": "B",
            "options": [{"A": "wrong"}, {"B": "right"}],
            "grading_mode": "lenient_answer_colon",
        },
    )
    reward = await routing_rm(args, sample)
    if reward == 1.0:
        print(f"  PASSED (reward={reward})")
        passed += 1
    else:
        print(f"  FAILED (reward={reward}, expected 1.0)")
        failed += 1

    # Test 2: Math sample
    print("\n=== Test: Math sample ===")
    sample = FakeSample(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        response="The answer is \\boxed{4}",
        label="4",
        metadata={"env_type": "math", "question": "What is 2+2?", "expected_answer": "4"},
    )
    reward = await routing_rm(args, sample)
    if reward == 1.0:
        print(f"  PASSED (reward={reward})")
        passed += 1
    else:
        print(f"  FAILED (reward={reward}, expected 1.0)")
        failed += 1

    # Test 3: Wrong math answer
    print("\n=== Test: Wrong math answer ===")
    sample = FakeSample(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        response="The answer is \\boxed{5}",
        label="4",
        metadata={"env_type": "math", "question": "What is 2+2?", "expected_answer": "4"},
    )
    reward = await routing_rm(args, sample)
    if reward == 0.0:
        print(f"  PASSED (reward={reward})")
        passed += 1
    else:
        print(f"  FAILED (reward={reward}, expected 0.0)")
        failed += 1

    # Test 4: Structured outputs
    print("\n=== Test: Structured outputs ===")
    sample = FakeSample(
        prompt=[{"role": "user", "content": "Generate JSON"}],
        response='{"name": "test", "age": 25}',
        metadata={
            "env_type": "structured_outputs",
            "schema_str": '{"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
            "schema_type": "json",
        },
    )
    reward = await routing_rm(args, sample)
    if reward == 1.0:
        print(f"  PASSED (reward={reward})")
        passed += 1
    else:
        print(f"  FAILED (reward={reward}, expected 1.0)")
        failed += 1

    # Test 5: Batched mixed-env
    print("\n=== Test: Batched mixed-env (3 samples) ===")
    samples = [
        FakeSample(
            prompt=[{"role": "user", "content": "Pick"}],
            response="Answer: A",
            metadata={"env_type": "mcqa", "expected_answer": "A", "options": [{"A": "right"}], "grading_mode": "lenient_answer_colon"},
        ),
        FakeSample(
            prompt=[{"role": "user", "content": "2+2?"}],
            response="\\boxed{4}",
            metadata={"env_type": "math", "question": "2+2?", "expected_answer": "4"},
        ),
        FakeSample(
            prompt=[{"role": "user", "content": "unknown"}],
            response="hello",
            metadata={"env_type": "nonexistent"},
        ),
    ]
    rewards = await routing_batched_rm(args, samples)
    if rewards[0] == 1.0 and rewards[1] == 1.0 and rewards[2] == 0.0:
        print(f"  PASSED (rewards={rewards})")
        passed += 1
    else:
        print(f"  FAILED (rewards={rewards}, expected [1.0, 1.0, 0.0])")
        failed += 1

    # Test 6: Unknown env_type (should return 0.0, not crash)
    print("\n=== Test: Unknown env_type ===")
    sample = FakeSample(
        prompt="test",
        response="test",
        metadata={"env_type": "bogus"},
    )
    reward = await routing_rm(args, sample)
    if reward == 0.0:
        print(f"  PASSED (reward={reward})")
        passed += 1
    else:
        print(f"  FAILED (reward={reward}, expected 0.0)")
        failed += 1

    # Cleanup
    for p in procs:
        p.kill()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}")
    if failed:
        sys.exit(1)
    print("ALL ROUTING RM TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
