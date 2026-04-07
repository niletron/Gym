"""Quick test of NeMo-Gym resources servers inside Docker.

Starts a few servers, sends test verify requests, checks rewards.
"""

import asyncio
import json
import subprocess
import sys
import time

import aiohttp

sys.path.insert(0, "/workspace/nemo-gym")


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


def _rcp(text="Q"):
    return {
        "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": text}]}],
        "model": "test",
    }


def _response(text):
    return {
        "id": "r1",
        "created_at": 0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "m1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


TESTS = [
    {
        "name": "mcqa",
        "port": 10005,
        "payload": {
            "responses_create_params": _rcp("Pick the right answer: A) wrong B) right"),
            "response": _response("Answer: B"),
            "expected_answer": "B",
            "options": [{"A": "wrong"}, {"B": "right"}],
            "grading_mode": "lenient_answer_colon",
        },
        "expected_reward": 1.0,
    },
    {
        "name": "math (library-only)",
        "port": 10004,
        "payload": {
            "responses_create_params": _rcp("What is 2+2?"),
            "response": _response("The answer is \\boxed{4}"),
            "question": "What is 2+2?",
            "expected_answer": "4",
        },
        "expected_reward": 1.0,
    },
    {
        "name": "structured_outputs",
        "port": 10006,
        "payload": {
            "responses_create_params": _rcp("Generate JSON"),
            "response": _response('{"name": "test", "age": 25}'),
            "schema_str": '{"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
            "schema_type": "json",
        },
        "expected_reward": 1.0,
    },
    {
        "name": "calendar",
        "port": 10007,
        "payload": {
            "responses_create_params": _rcp("Schedule events"),
            "response": _response("Here is the schedule:\n10:00am - Meeting A (60 min)\n11:00am - Meeting B (30 min)"),
            "exp_cal_state": {
                "0": {"event_id": 0, "duration": 60, "constraint": "at 10am", "min_time": "10:00", "max_time": "16:00"},
                "1": {"event_id": 1, "duration": 30, "constraint": "at 11am", "min_time": "10:00", "max_time": "16:00"},
            },
        },
        "expected_reward": None,  # Just check it doesn't crash
    },
]


async def run_test(test):
    name = test["name"]
    port = test["port"]
    url = f"http://localhost:{port}/verify"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=test["payload"], timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  {name}: FAILED (status {resp.status}): {body[:200]}")
                    return False
                result = await resp.json()
                reward = result.get("reward", "N/A")
                if test["expected_reward"] is not None and reward != test["expected_reward"]:
                    print(f"  {name}: WRONG REWARD {reward} (expected {test['expected_reward']})")
                    return False
                print(f"  {name}: reward={reward} OK")
                return True
    except Exception as e:
        print(f"  {name}: ERROR: {e}")
        return False


async def main():
    print("=" * 60)
    print("NeMo-Gym Resources Server Quick Tests")
    print("=" * 60)

    # Start servers
    servers = {
        "mcqa": 10005,
        "math": 10004,
        "structured_outputs": 10006,
        "calendar": 10007,
    }
    procs = []
    for stype, port in servers.items():
        p = subprocess.Popen(
            [sys.executable, "-m", "slime_integration.server_launcher", "--server-type", stype, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)
        print(f"  Started {stype} (pid={p.pid}) on port {port}")

    # Wait for servers
    print("\nWaiting for servers...")
    for stype, port in servers.items():
        ok = await wait_for_service(f"http://localhost:{port}/docs", timeout=60)
        print(f"  {stype}: {'ready' if ok else 'TIMEOUT'}")
        if not ok:
            for p in procs:
                p.kill()
            sys.exit(1)

    # Run tests
    print("\nRunning tests...")
    passed = 0
    failed = 0
    for test in TESTS:
        ok = await run_test(test)
        if ok:
            passed += 1
        else:
            failed += 1

    # Cleanup
    for p in procs:
        p.kill()

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("ALL SERVER TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
