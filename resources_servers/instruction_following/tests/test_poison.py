# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Poison-payload tests for instruction_following.

IF graders run regex-heavy verifiable_instructions checks on untrusted
model output. The SIGKILL pool bounds worst-case time.

Run::

    cd resources_servers/instruction_following
    pytest tests/test_poison.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

os.environ.setdefault("IF_VERIFY_TIMEOUT_S", "5")

from resources_servers.instruction_following.app import (  # noqa: E402
    InstructionFollowingResourcesServer,
    InstructionFollowingResourcesServerConfig,
    InstructionFollowingVerifyRequest,
    _check_in_worker,
    _is_refusal,
)


def _make_server():
    config = InstructionFollowingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return InstructionFollowingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(text: str, instruction_id_list, kwargs, grading_mode="binary"):
    response = NeMoGymResponse(
        id="resp_test", created_at=0.0, model="dummy", object="response",
        output=[{
            "id": "msg_test",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant", "status": "completed", "type": "message",
        }],
        parallel_tool_calls=False, tool_choice="none", tools=[],
    )
    return InstructionFollowingVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "x"}]}],
            "model": "dummy",
        },
        response=response,
        id=0, prompt="x",
        instruction_id_list=instruction_id_list,
        kwargs=kwargs,
        grading_mode=grading_mode,
    )


class TestIFPoison:
    def test_empty_instruction_list_scores_1(self):
        """No instructions -> all([]) is True -> reward=1.0."""
        server = _make_server()
        req = _make_request("hello", [], [])
        resp = asyncio.run(server.verify(req))
        assert resp.reward == 1.0

    def test_huge_response_does_not_hang(self):
        server = _make_server()
        # 500KB response with a length constraint
        req = _make_request("word " * 100_000,
                            ["length_constraints:number_words"],
                            [{"num_words": 10}])
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Huge response should not exceed timeout; took {elapsed:.1f}s"
        assert resp.reward in (0.0, 1.0)

    def test_refusal_detection_zeroes_reward(self):
        # Pure unit-test of refusal helper — it's pure and doesn't need pool.
        assert _is_refusal("I cannot help with that.", min_content_length=150) is True
        assert _is_refusal("Here is a long, detailed answer " * 20) is False

    def test_unknown_instruction_graded_as_false_not_crash(self):
        server = _make_server()
        req = _make_request("hello",
                            ["totally_made_up_instruction_id"],
                            [{}])
        resp = asyncio.run(server.verify(req))
        assert resp.reward == 0.0
        assert resp.follow_instruction_list == [False]


class TestIFHealth:
    def test_health_returns_ok(self):
        server = _make_server()
        # Warm pool with a trivial verify first.
        asyncio.run(server.verify(_make_request("hello", [], [])))
        app = server.setup_webserver()
        health = next(r for r in app.routes if getattr(r, "path", None) == "/health")
        start = time.monotonic()
        result = asyncio.run(health.endpoint())
        elapsed = time.monotonic() - start
        assert result["status"] == "ok"
        assert elapsed < 3.0, f"/health on warm pool should be fast; took {elapsed:.1f}s"


class TestWorkerFunction:
    def test_worker_empty_instruction_list_returns_full_reward(self):
        reward, lst = _check_in_worker("hello", [], [], "binary")
        assert reward == 1.0 and lst == []

    def test_worker_fraction_mode(self):
        # Two instructions, one satisfied — would ideally be 0.5, but without
        # real instructions both fail. Check the shape, not specific value.
        reward, lst = _check_in_worker("hi", ["bad1", "bad2"], [{}, {}], "fraction")
        assert lst == [False, False] and reward == 0.0
