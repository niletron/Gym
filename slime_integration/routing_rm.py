"""Routing reward model for mixed-environment RLVR1 training.

Dispatches each sample to the correct NeMo-Gym resources server based on
``sample.metadata["env_type"]``.

Usage with Slime::

    --custom-rm-path slime_integration.routing_rm.routing_batched_rm
    --rm-url /path/to/routing_table.json   # or NEMOGYM_ROUTING_TABLE env var
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP session
# ---------------------------------------------------------------------------

_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(limit=200),
        )
    return _session


# ---------------------------------------------------------------------------
# Default routing table
# ---------------------------------------------------------------------------

DEFAULT_ROUTING: Dict[str, str] = {
    "single_step_tool_use": "http://localhost:10001",
    "instruction_following": "http://localhost:10002",
    "code_gen": "http://localhost:10003",
    "math": "http://localhost:10004",
    "mcqa": "http://localhost:10005",
    "structured_outputs": "http://localhost:10006",
    "calendar": "http://localhost:10007",
    "reasoning_gym": "http://localhost:10008",
    "math_formal_lean": "http://localhost:10009",
    "workplace_assistant": "http://localhost:10010",
}

ENV_TIMEOUTS: Dict[str, float] = {
    "code_gen": 120.0,
    "math_formal_lean": 60.0,
}

_routing_table: Optional[Dict[str, str]] = None


def _load_routing_table(args) -> Dict[str, str]:
    global _routing_table
    if _routing_table is not None:
        return _routing_table

    # Try env var first
    env_val = os.environ.get("NEMOGYM_ROUTING_TABLE")
    if env_val:
        if os.path.isfile(env_val):
            with open(env_val) as f:
                _routing_table = json.load(f)
        else:
            _routing_table = json.loads(env_val)
        return _routing_table

    # Try args.rm_url as file path
    rm_url = getattr(args, "rm_url", None)
    if rm_url and os.path.isfile(rm_url):
        with open(rm_url) as f:
            _routing_table = json.load(f)
        return _routing_table

    _routing_table = DEFAULT_ROUTING
    return _routing_table


# ---------------------------------------------------------------------------
# Response wrapping
# ---------------------------------------------------------------------------


def _wrap_response(text: str) -> Dict[str, Any]:
    """Wrap plain text into NeMoGymResponse dict format."""
    return {
        "id": f"resp_{uuid4().hex[:16]}",
        "created_at": 0.0,
        "model": "slime",
        "object": "response",
        "output": [
            {
                "id": f"msg_{uuid4().hex[:16]}",
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


def _make_rcp(messages: List[Dict], tools: Optional[List] = None) -> Dict[str, Any]:
    """Build responses_create_params from Slime messages."""
    input_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        input_msgs.append({
            "role": role,
            "type": "message",
            "content": [{"type": "input_text", "text": content}],
        })
    rcp: Dict[str, Any] = {"input": input_msgs, "model": "slime"}
    if tools:
        rcp["tools"] = tools
    return rcp


# ---------------------------------------------------------------------------
# Per-environment verify payload builders
# ---------------------------------------------------------------------------


def _build_verify_payload(env_type: str, metadata: Dict, response_text: str, prompt: Any, sample) -> Dict[str, Any]:
    """Build the /verify request body for a specific environment."""
    # Build messages from prompt
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    tools = metadata.get("tools")
    rcp = _make_rcp(messages, tools)
    response = _wrap_response(response_text)

    payload: Dict[str, Any] = {
        "responses_create_params": rcp,
        "response": response,
    }

    if env_type == "single_step_tool_use":
        payload["expected_action"] = metadata.get("expected_action")

    elif env_type == "instruction_following":
        for key in ("instruction_id_list", "kwargs", "grading_mode", "prompt", "id"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "code_gen":
        vm = metadata.get("verifier_metadata", {})
        payload["verifier_metadata"] = vm

    elif env_type == "math":
        payload["question"] = metadata.get("question", "")
        payload["expected_answer"] = metadata.get("expected_answer", "")

    elif env_type == "mcqa":
        for key in ("expected_answer", "options", "grading_mode", "template_metadata", "uuid"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "structured_outputs":
        for key in ("schema_str", "schema_type"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "calendar":
        payload["exp_cal_state"] = metadata.get("exp_cal_state")

    elif env_type == "reasoning_gym":
        payload["question"] = metadata.get("question", "")
        payload["answer"] = metadata.get("answer")
        # reasoning_gym expects its own metadata dict
        payload["metadata"] = metadata.get("reasoning_gym_metadata", {})

    elif env_type == "math_formal_lean":
        for key in ("header", "formal_statement", "informal_prefix", "name"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "workplace_assistant":
        for key in ("ground_truth", "id", "category", "environment_name"):
            if key in metadata:
                payload[key] = metadata[key]

    return payload


# ---------------------------------------------------------------------------
# Core RM functions
# ---------------------------------------------------------------------------


async def _score_single(args, sample, routing: Dict[str, str]) -> float:
    """Score a single sample by routing to the correct resources server."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_type = metadata.get("env_type", "")

    if not env_type:
        logger.warning("Sample has no env_type in metadata, returning 0.0")
        return 0.0

    base_url = routing.get(env_type)
    if not base_url:
        logger.warning("No server configured for env_type=%s, returning 0.0", env_type)
        return 0.0

    try:
        payload = _build_verify_payload(
            env_type=env_type,
            metadata=metadata,
            response_text=sample.response or "",
            prompt=sample.prompt,
            sample=sample,
        )
    except Exception as exc:
        logger.warning("Failed to build payload for env_type=%s: %s", env_type, exc)
        return 0.0

    timeout = ENV_TIMEOUTS.get(env_type, 30.0)
    url = f"{base_url}/verify"

    try:
        session = await _get_session()
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Server %s returned %d: %s", url, resp.status, body[:200])
                return 0.0
            result = await resp.json()
            return float(result.get("reward", 0.0))
    except asyncio.TimeoutError:
        logger.warning("Timeout calling %s for env_type=%s", url, env_type)
        return 0.0
    except Exception as exc:
        logger.warning("Error calling %s: %s", url, exc)
        return 0.0


async def routing_rm(args, sample, **kwargs) -> float:
    """Single-sample routing RM for Slime's ``--custom-rm-path``."""
    routing = _load_routing_table(args)
    return await _score_single(args, sample, routing)


async def routing_batched_rm(args, samples_or_sample, **kwargs):
    """Routing RM for Slime's ``--custom-rm-path``.

    Slime's ``custom_rm_path`` is called from BOTH:
    - ``async_rm(args, sample)`` — single Sample object
    - ``batched_async_rm(args, samples)`` — list of Sample objects

    This function auto-detects which case and handles both.
    """
    routing = _load_routing_table(args)

    # Single sample case (called from async_rm)
    if not isinstance(samples_or_sample, list):
        return await _score_single(args, samples_or_sample, routing)

    # Batched case (called from batched_async_rm)
    samples = samples_or_sample
    start = time.time()
    rewards = await asyncio.gather(*(
        _score_single(args, sample, routing) for sample in samples
    ))
    elapsed = time.time() - start
    logger.info(
        "Scored %d samples in %.1fs (avg reward: %.3f)",
        len(samples), elapsed, sum(rewards) / max(len(rewards), 1),
    )
    return list(rewards)
