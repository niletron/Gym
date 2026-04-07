"""Standalone launcher for NeMo-Gym resources servers.

Starts individual resources servers without the full ng_run orchestration.
Used inside the Slime Docker container alongside training.

Usage::

    python -m slime_integration.server_launcher --server-type mcqa --port 10005
    python -m slime_integration.server_launcher --all   # start all servers
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server registry
# ---------------------------------------------------------------------------

SERVER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "single_step_tool_use": {
        "module": "resources_servers.single_step_tool_use_with_argument_comparison.app",
        "class": "SingleStepToolUseArgumentComparisonResourcesServer",
        "config_class": "SingleStepToolUseArgumentComparisonResourcesServerConfig",
        "port": 10001,
        "extra_config": {},
    },
    "instruction_following": {
        "module": "resources_servers.instruction_following.app",
        "class": "InstructionFollowingResourcesServer",
        "config_class": "InstructionFollowingResourcesServerConfig",
        "port": 10002,
        "extra_config": {},
    },
    "code_gen": {
        "module": "resources_servers.code_gen.app",
        "class": "CompCodingResourcesServer",
        "config_class": "CompCodingResourcesServerConfig",
        "port": 10003,
        "extra_config": {"num_processes": 4, "unit_test_timeout_secs": 10, "debug": False},
    },
    "math": {
        "module": "resources_servers.math_with_judge.app",
        "class": "LibraryJudgeMathResourcesServer",
        "config_class": "LibraryJudgeMathResourcesServerConfig",
        "port": 10004,
        "extra_config": {
            "should_use_judge": False,
            "judge_model_server": {"type": "responses_api_models", "name": "dummy"},
            "judge_responses_create_params": {"input": []},
        },
    },
    "mcqa": {
        "module": "resources_servers.mcqa.app",
        "class": "MCQAResourcesServer",
        "config_class": "MCQAResourcesServerConfig",
        "port": 10005,
        "extra_config": {},
    },
    "structured_outputs": {
        "module": "resources_servers.structured_outputs.app",
        "class": "StructuredOutputsResourcesServer",
        "config_class": "StructuredOutputsResourcesServerConfig",
        "port": 10006,
        "extra_config": {},
    },
    "calendar": {
        "module": "resources_servers.calendar.app",
        "class": "CalendarResourcesServer",
        "config_class": "CalendarResourcesServerConfig",
        "port": 10007,
        "extra_config": {},
    },
    "reasoning_gym": {
        "module": "resources_servers.reasoning_gym.app",
        "class": "ReasoningGymResourcesServer",
        "config_class": "ReasoningGymResourcesServerConfig",
        "port": 10008,
        "extra_config": {},
    },
    "math_formal_lean": {
        "module": "resources_servers.math_formal_lean.app",
        "class": "MathFormalLeanResourcesServer",
        "config_class": "MathFormalLeanResourcesServerConfig",
        "port": 10009,
        "extra_config": {
            "sandbox_host": os.environ.get("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
            "sandbox_port": int(os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000")),
        },
    },
    "workplace_assistant": {
        "module": "resources_servers.workplace_assistant.app",
        "class": "WorkbenchResourcesServer",
        "config_class": "WorkbenchResourcesServerConfig",
        "port": 10010,
        "extra_config": {},
    },
}


def _launch_single(server_type: str, port: Optional[int] = None, **config_overrides):
    """Launch a single resources server."""
    import importlib

    import uvicorn

    from nemo_gym.server_utils import ServerClient

    # Add server-local dirs to sys.path for local imports (utils, common, etc.)
    reg = SERVER_REGISTRY[server_type]
    module_parts = reg["module"].rsplit(".", 1)
    server_dir = os.path.join(os.path.dirname(__file__), "..", *module_parts[0].split("."))
    server_dir = os.path.abspath(server_dir)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    mod = importlib.import_module(reg["module"])
    server_cls = getattr(mod, reg["class"])
    config_cls = getattr(mod, reg["config_class"])

    actual_port = port or reg["port"]
    config_kwargs = {
        "host": "0.0.0.0",
        "port": actual_port,
        "entrypoint": "app.py",
        "name": server_type,
        **reg["extra_config"],
        **config_overrides,
    }

    try:
        config = config_cls(**config_kwargs)
    except Exception:
        # Some config classes have required fields with defaults from YAML
        # Fall back to minimal config
        minimal = {"host": "0.0.0.0", "port": actual_port, "entrypoint": "app.py", "name": server_type}
        minimal.update(config_overrides)
        config = config_cls(**minimal)

    server = server_cls(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()

    logger.info("Starting %s on port %d", server_type, actual_port)
    uvicorn.run(app, host="0.0.0.0", port=actual_port, log_level="warning")


def _launch_in_process(server_type: str, port: Optional[int] = None):
    """Launch a server in a subprocess."""
    p = multiprocessing.Process(
        target=_launch_single,
        args=(server_type,),
        kwargs={"port": port},
        daemon=True,
    )
    p.start()
    return p


def launch_all(skip: Optional[set] = None) -> Dict[str, multiprocessing.Process]:
    """Launch all registered servers as background processes."""
    skip = skip or set()
    processes = {}
    for server_type, reg in SERVER_REGISTRY.items():
        if server_type in skip:
            logger.info("Skipping %s", server_type)
            continue
        p = _launch_in_process(server_type, reg["port"])
        processes[server_type] = p
        logger.info("Launched %s (pid=%d) on port %d", server_type, p.pid, reg["port"])
    return processes


def wait_for_servers(timeout: float = 120.0, skip: Optional[set] = None):
    """Wait for all servers to be healthy."""
    import aiohttp
    import asyncio

    skip = skip or set()

    async def _check(name, port):
        url = f"http://localhost:{port}/docs"
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            return name, True
            except Exception:
                pass
            await asyncio.sleep(2)
        return name, False

    async def _check_all():
        tasks = []
        for name, reg in SERVER_REGISTRY.items():
            if name not in skip:
                tasks.append(_check(name, reg["port"]))
        return await asyncio.gather(*tasks)

    results = asyncio.run(_check_all())
    for name, ok in results:
        status = "ready" if ok else "TIMEOUT"
        logger.info("  %s: %s", name, status)
    return {name: ok for name, ok in results}


def main():
    parser = argparse.ArgumentParser(description="Launch NeMo-Gym resources servers")
    parser.add_argument("--server-type", choices=list(SERVER_REGISTRY.keys()), help="Single server to launch")
    parser.add_argument("--port", type=int, help="Override port")
    parser.add_argument("--all", action="store_true", help="Launch all servers")
    parser.add_argument("--skip", nargs="*", default=[], help="Server types to skip")
    parser.add_argument("--wait", action="store_true", help="Wait for all servers to be healthy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.all:
        skip = set(args.skip)
        processes = launch_all(skip=skip)
        if args.wait:
            wait_for_servers(skip=skip)
        # Keep alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            for p in processes.values():
                p.terminate()
    elif args.server_type:
        _launch_single(args.server_type, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
