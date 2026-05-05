# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Poison-payload tests for reasoning_gym.

reasoning_gym's scorers are third-party code. In the _u5j70izc_resume run
it produced an 8,160-timeout burst over 3 hours on 2026-04-22. The SIGKILL
pool caps worst-case per-call time.

Run::

    cd resources_servers/reasoning_gym
    pytest tests/test_poison.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

os.environ.setdefault("REASONING_GYM_VERIFY_TIMEOUT_S", "5")

# Skip the whole module if reasoning_gym isn't available — the pattern should
# still be provable via the worker function in isolation.
reasoning_gym = pytest.importorskip("reasoning_gym")

from resources_servers.reasoning_gym.app import (  # noqa: E402
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
    _score_in_worker,
)


def _make_server():
    config = ReasoningGymResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


_SENTINEL = object()


def _make_request(text: str, question: str = "2+2?", answer: str = "4", metadata=_SENTINEL):
    response = NeMoGymResponse(
        id="resp_test", created_at=0.0, model="dummy", object="response",
        output=[{
            "id": "msg_test",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant", "status": "completed", "type": "message",
        }],
        parallel_tool_calls=False, tool_choice="none", tools=[],
    )
    md = {"source_dataset": "simple_equations"} if metadata is _SENTINEL else metadata
    return ReasoningGymVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "x"}]}],
            "model": "dummy",
        },
        response=response, question=question, answer=answer,
        metadata=md,
    )


class TestReasoningGymPoison:
    def test_huge_response_does_not_hang(self):
        server = _make_server()
        # 500KB response — parser shouldn't take anywhere near timeout.
        req = _make_request("<answer>" + "x" * 500_000 + "</answer>")
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Huge response should not exceed timeout; took {elapsed:.1f}s"
        assert isinstance(resp.reward, float)

    def test_missing_task_name_raises(self):
        server = _make_server()
        req = _make_request("<answer>4</answer>", metadata={})
        with pytest.raises(ValueError, match="No task name"):
            asyncio.run(server.verify(req))

    def test_unknown_task_name_scores_zero(self):
        server = _make_server()
        req = _make_request("<answer>42</answer>",
                            metadata={"source_dataset": "totally_made_up_task"})
        resp = asyncio.run(server.verify(req))
        assert resp.reward == 0.0  # worker catches the exception


class TestReasoningGymHealth:
    def test_health_returns_ok(self):
        server = _make_server()
        app = server.setup_webserver()
        health = next(r for r in app.routes if getattr(r, "path", None) == "/health")
        start = time.monotonic()
        result = asyncio.run(health.endpoint())
        elapsed = time.monotonic() - start
        assert "status" in result
        # Accept either "ok" or an error status — we only require the endpoint
        # to return something fast, not that the specific canned task is
        # valid in every reasoning_gym version.
        assert elapsed < 10.0, f"/health took too long: {elapsed:.1f}s"


class TestWorkerFunction:
    def test_worker_returns_float(self):
        """Even if the task name is invalid, the worker must return a float."""
        result = _score_in_worker("totally_made_up_task", "0", {"metadata": {}})
        assert isinstance(result, float)
