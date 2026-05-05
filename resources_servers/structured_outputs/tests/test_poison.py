# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Poison-payload tests for structured_outputs.

structured_outputs runs untrusted JSON schemas through
openapi_schema_validator and untrusted response text through JSON/YAML/XML
parsers. Pathological schemas can cause validator recursion; pathological
YAML/XML can blow up the parser. These tests pin the SIGKILL-pool contract
so a single bad sample cannot wedge the server.

Run::

    cd resources_servers/structured_outputs
    pytest tests/test_poison.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

os.environ.setdefault("STRUCTURED_OUTPUTS_VERIFY_TIMEOUT_S", "5")

from resources_servers.structured_outputs.app import (  # noqa: E402
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
    SchemaType,
    _evaluate_in_worker,
)


def _make_server():
    config = StructuredOutputsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return StructuredOutputsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(text: str, schema_str: str, schema_type: SchemaType = SchemaType.JSON):
    response = NeMoGymResponse(
        id="resp_test", created_at=0.0, model="dummy", object="response",
        output=[{
            "id": "msg_test",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant", "status": "completed", "type": "message",
        }],
        parallel_tool_calls=False, tool_choice="none", tools=[],
    )
    return StructuredOutputsVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "x"}]}],
            "model": "dummy",
        },
        response=response, schema_str=schema_str, schema_type=schema_type,
    )


class TestStructuredOutputsPoison:
    def test_valid_json_scores_1(self):
        server = _make_server()
        schema = json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})
        req = _make_request('{"x": 1}', schema)
        resp = asyncio.run(server.verify(req))
        assert resp.reward == 1.0

    def test_invalid_json_scores_0(self):
        server = _make_server()
        schema = json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})
        req = _make_request('{"x": "not an int"}', schema)
        resp = asyncio.run(server.verify(req))
        assert resp.reward == 0.0

    def test_huge_response_does_not_hang(self):
        server = _make_server()
        schema = json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})
        # 1MB of padding around a valid object — parser should handle or fail
        # fast, not hang.
        req = _make_request("a" * 1_000_000 + '{"x": 1}' + "b" * 1_000_000, schema)
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Huge response must not exceed timeout; took {elapsed:.1f}s"
        assert resp.reward in (0.0, 1.0)

    def test_deeply_nested_schema_does_not_hang(self):
        """Synthesize a deeply-nested schema. openapi_schema_validator's
        recursive strictification + validation can blow the stack or take
        exponential time on some shapes. Pool must cap the cost."""
        # Build a 50-level-deep nested object schema.
        schema: dict = {"type": "object", "properties": {"v": {"type": "integer"}}}
        for _ in range(50):
            schema = {"type": "object", "properties": {"nested": schema}}
        server = _make_server()
        req = _make_request('{"x": 1}', json.dumps(schema))
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Deeply-nested schema should not hang; took {elapsed:.1f}s"
        assert resp.reward == 0.0  # response doesn't match the deep schema


class TestStructuredOutputsHealth:
    def test_health_returns_ok(self):
        server = _make_server()
        asyncio.run(server.verify(_make_request(
            '{"x": 1}', json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})
        )))  # warm pool
        app = server.setup_webserver()
        health = next(r for r in app.routes if getattr(r, "path", None) == "/health")
        start = time.monotonic()
        result = asyncio.run(health.endpoint())
        elapsed = time.monotonic() - start
        assert result["status"] == "ok", f"Expected ok, got {result}"
        assert elapsed < 3.0, f"/health on warm pool should be fast; took {elapsed:.1f}s"


class TestWorkerFunction:
    def test_worker_fn_picklable_and_correct(self):
        # Direct in-process call to prove the worker function runs standalone.
        result = _evaluate_in_worker(
            "json",
            json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}}),
            '{"x": 1}',
            False,
        )
        assert result == 1.0
