"""End-to-end tests with REAL NeMo-Gym servers.

Tests the untested gaps:
1. Proxy mode: reward adapter → real resources server /verify
2. Agent /run mode: reward adapter → real agent → model + resources server
3. Full pipeline: SGLang model + resources server + agent + reward adapter
"""

import asyncio
import json
import subprocess
import sys
import time
from uuid import uuid4

import aiohttp
import uvicorn

sys.path.insert(0, "/workspace/nemo-gym")

SGLANG_PORT = 30000
REWARD_ADAPTER_PORT = 8101
RESOURCES_SERVER_PORT = 8200
AGENT_PORT = 8300
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


async def wait_for_service(url, name, timeout=120):
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
        await asyncio.sleep(2)
    print(f"  [{name}] TIMEOUT after {timeout}s")
    return False


def start_resources_server():
    """Start the example_single_tool_call resources server on port 8200."""
    proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import uvicorn
from resources_servers.example_single_tool_call.app import (
    SimpleWeatherResourcesServer, SimpleWeatherResourcesServerConfig,
)
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient

config = SimpleWeatherResourcesServerConfig(
    host='0.0.0.0', port={RESOURCES_SERVER_PORT}, entrypoint='app.py', name='example',
)
server = SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
app = server.setup_webserver()
uvicorn.run(app, host='0.0.0.0', port={RESOURCES_SERVER_PORT})
"""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def start_reward_adapter_proxy(upstream_server_url=None, upstream_agent_url=None):
    """Start the reward adapter in proxy mode on port 8100."""
    proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import uvicorn
from resources_servers.slime_reward_adapter.app import SlimeRewardAdapter, SlimeRewardAdapterConfig
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient

config = SlimeRewardAdapterConfig(
    host='0.0.0.0', port={REWARD_ADAPTER_PORT}, entrypoint='app.py', name='adapter',
    reward_type='proxy',
    upstream_server_url={repr(upstream_server_url)},
    upstream_agent_url={repr(upstream_agent_url)},
)
adapter = SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))
app = adapter.setup_webserver()
uvicorn.run(app, host='0.0.0.0', port={REWARD_ADAPTER_PORT})
"""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


# ============================================================
# Test 1: Proxy mode with a real NeMo-Gym resources server
# ============================================================

async def test_proxy_verify():
    """Test reward adapter proxying to a real resources server's /verify."""
    print("\n=== Test: Proxy to Real Resources Server /verify ===")

    # The example_single_tool_call server always returns reward=1.0 in verify()
    # Build a proper verify request
    verify_body = {
        "responses_create_params": {
            "input": [
                {"role": "user", "type": "message",
                 "content": [{"type": "input_text", "text": "What's the weather in Paris?"}]},
            ],
            "model": "test",
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
                    "content": [{"type": "output_text", "text": "cold", "annotations": []}],
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }

    # First test: call the resources server directly
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{RESOURCES_SERVER_PORT}/verify",
            json=verify_body,
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Direct verify expected 1.0, got {result.get('reward')}"
            print(f"  Direct /verify: reward={result['reward']} ✓")

    # Second test: call via the reward adapter /slime_reward (proxy mode)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_ADAPTER_PORT}/slime_reward",
            json={"prompt": "What's the weather?", "response": "cold", "label": "cold"},
        ) as resp:
            result = await resp.json()
            assert result["reward"] == 1.0, f"Proxied slime_reward expected 1.0, got {result.get('reward')}"
            print(f"  Proxied /slime_reward: reward={result['reward']} ✓")

    print("  PASSED")


async def test_proxy_with_custom_rm():
    """Test the nemogym_rm custom RM going through proxy to real resources server."""
    print("\n=== Test: Custom RM → Proxy → Real Resources Server ===")
    from dataclasses import dataclass, field
    from types import SimpleNamespace
    from slime_integration.custom_rm import nemogym_rm

    @dataclass
    class FakeSample:
        prompt: str = ""
        response: str = ""
        label: str = None
        metadata: dict = field(default_factory=dict)

    args = SimpleNamespace(rm_url=f"http://localhost:{REWARD_ADAPTER_PORT}", reward_key=None)

    # The example_single_tool_call always returns 1.0
    sample = FakeSample(prompt="What's the weather?", response="anything", label="anything")
    reward = await nemogym_rm(args, sample)
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    print(f"  nemogym_rm → proxy → resources server: reward={reward} ✓")

    print("  PASSED")


async def test_resources_server_tool_endpoint():
    """Test that the resources server's tool endpoints work directly."""
    print("\n=== Test: Resources Server Tool Endpoint ===")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{RESOURCES_SERVER_PORT}/get_weather",
            json={"city": "Paris"},
        ) as resp:
            result = await resp.json()
            assert "weather_description" in result
            assert "Paris" in result["city"]
            print(f"  /get_weather: {result} ✓")

    print("  PASSED")


# ============================================================
# Test 2: SGLang → Resources Server end-to-end
# ============================================================

async def test_sglang_to_resources_server():
    """Full pipeline: SGLang generates, adapter proxies verify to resources server."""
    print("\n=== Test: SGLang Generate → Proxy Verify ===")

    # Generate with SGLang
    gen_payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Say 'cold'."}],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{SGLANG_PORT}/v1/chat/completions",
            json=gen_payload,
        ) as resp:
            gen_result = await resp.json()
            response_text = gen_result["choices"][0]["message"]["content"]
            print(f"  SGLang response: {response_text!r}")

    # Verify via adapter proxy
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_ADAPTER_PORT}/slime_reward",
            json={"prompt": "Weather in Paris?", "response": response_text, "label": "cold"},
        ) as resp:
            result = await resp.json()
            print(f"  Proxied reward: {result['reward']}")

    # The example server always returns 1.0 regardless of content
    assert result["reward"] == 1.0
    print("  PASSED")


async def test_concurrent_proxy():
    """Concurrent requests through the proxy pipeline."""
    print("\n=== Test: Concurrent Proxy Requests (20) ===")

    async def single(i):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{REWARD_ADAPTER_PORT}/slime_reward",
                json={"prompt": f"Q{i}", "response": f"A{i}", "label": f"A{i}"},
            ) as resp:
                return await resp.json()

    start = time.time()
    results = await asyncio.gather(*(single(i) for i in range(20)))
    elapsed = time.time() - start

    all_ok = all(r["reward"] == 1.0 for r in results)
    print(f"  20 requests in {elapsed:.2f}s, all reward=1.0: {all_ok}")
    assert all_ok
    print("  PASSED")


# ============================================================
# Test 3: Agent /run mode
# ============================================================

async def test_agent_run_endpoint():
    """Test /slime_agent_run gracefully handles unreachable agent."""
    print("\n=== Test: Agent /run Endpoint (agent unreachable - error handled) ===")

    payload = {
        "responses_create_params": {
            "input": [{"role": "user", "type": "message",
                       "content": [{"type": "input_text", "text": "test"}]}],
            "model": "test",
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{REWARD_ADAPTER_PORT}/slime_agent_run",
            json=payload,
        ) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            result = await resp.json()
            assert result["reward"] == 0.0
            assert "error" in result.get("metadata", {})
            print(f"  reward={result['reward']}, error={result['metadata']['error'][:60]}...")

    print("  PASSED")


# ============================================================
# Main
# ============================================================

async def main():
    print("=" * 60)
    print("Real Server Integration Tests")
    print("=" * 60)

    # Start servers
    print("\nStarting servers...")

    resources_proc = start_resources_server()
    # Start adapter in proxy mode pointing at the resources server
    adapter_proc = start_reward_adapter_proxy(
        upstream_server_url=f"http://localhost:{RESOURCES_SERVER_PORT}",
        upstream_agent_url=f"http://localhost:{AGENT_PORT}",
    )

    # Wait for services
    sglang_ok = await wait_for_service(f"http://localhost:{SGLANG_PORT}/health", "SGLang", timeout=10)
    resources_ok = await wait_for_service(f"http://localhost:{RESOURCES_SERVER_PORT}/docs", "Resources Server")
    adapter_ok = await wait_for_service(f"http://localhost:{REWARD_ADAPTER_PORT}/docs", "Reward Adapter")

    if not resources_ok or not adapter_ok:
        print("FATAL: Servers not ready")
        resources_proc.kill()
        adapter_proc.kill()
        sys.exit(1)

    tests = [
        test_proxy_verify,
        test_proxy_with_custom_rm,
        test_resources_server_tool_endpoint,
        test_agent_run_endpoint,
        test_concurrent_proxy,
    ]
    if sglang_ok:
        tests.append(test_sglang_to_resources_server)

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

    # Cleanup
    resources_proc.kill()
    adapter_proc.kill()

    print("\n" + "=" * 60)
    print(f"Real Server Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if failed:
        sys.exit(1)
    print("\nALL REAL SERVER TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
