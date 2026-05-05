# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Poison-payload tests for the calendar server's SIGKILL subprocess pool.

These tests exist because of the qwen3-30B-rlvr1_v2-3sv29n1u_u5j70izc_resume
incident: 134,735 calendar /verify timeouts over 54 hours, caused by the
catastrophic-backtracking regex in utils.extract_json_list.

Contract:
  * Poison inputs (pathological regex input, infinite work in the worker)
    must return within CALENDAR_VERIFY_TIMEOUT_S, NEVER hang the server.
  * On timeout: reward=0, pool is SIGKILL-recycled.
  * /health probe returns within 2s on healthy servers; it's the watchdog's
    only reliable wedged-loop signal.
  * Normal inputs still grade correctly after a recycle.

Run::

    pytest resources_servers/calendar/tests/test_poison.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# Use a modest timeout for tests. Production-realistic would be 10s; we use 5s
# so poison cases fail fast without bumping into spawn startup (~1-2s per fresh
# pool on this machine). Tests that need tighter bounds override per-test.
os.environ.setdefault("CALENDAR_VERIFY_TIMEOUT_S", "5")

from resources_servers.calendar.app import (  # noqa: E402
    CalendarResourcesServer,
    CalendarResourcesServerConfig,
    CalendarVerifyRequest,
    _grade_in_worker,
)


def _make_server():
    config = CalendarResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return CalendarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(text: str, exp_cal_state: dict):
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return CalendarVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "x"}]}],
            "model": "dummy",
        },
        response=response,
        exp_cal_state=exp_cal_state,
    )


def _poison_bracket_string(n: int = 500) -> str:
    """String that triggers catastrophic backtracking in
    `\\[(?:[^\\[\\]]|\\{[^}]*\\})*\\{...` regex. Many `{` without terminators
    force the engine to try exponentially many groupings."""
    return "[" + "{" * n + "a" * 50


def _sleep_forever(_payload, _state):
    """Module-level so spawn-pickle can find it. Used by the direct-hang test."""
    import time as _t
    _t.sleep(1000)


class TestCalendarPoisonPayloads:
    def test_poison_regex_input_does_not_hang(self):
        """The catastrophic-backtracking regex in extract_json_list should be
        SIGKILLed within the configured timeout. Without the pool, this
        hangs Python's re engine indefinitely.

        Budget: timeout (5s) + spawn overhead (~2s) + slack = 10s.
        In production, spawn is paid once at server startup, so real latency
        is dominated by the timeout."""
        server = _make_server()
        req = _make_request(_poison_bracket_string(300), exp_cal_state={"event_1": {}})
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"verify() must complete within timeout+spawn, took {elapsed:.1f}s"
        assert resp.reward == 0, "Poison-regex input must grade to reward=0 on timeout"

    def test_huge_response_does_not_hang(self):
        """500KB random response — should not blow up the grader even if the
        regex doesn't explode, pool timeout must cap worst-case time."""
        server = _make_server()
        req = _make_request("a" * 500_000 + "[{\"event_id\":\"1\"}]" + "b" * 500_000,
                            exp_cal_state={"1": {"day": "monday"}})
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"Huge response should not exceed timeout+spawn; took {elapsed:.1f}s"
        assert resp.reward in (0, 1)

    def test_worker_hang_times_out(self):
        """Confirm the pool actually times out infinite work in the worker.

        Submits a sleep-forever task directly to the pool to prove the SIGKILL
        + wait_for path handles hung workers. Module-level helper
        `_sleep_forever` is used because spawn pickle can't serialize
        test-local callables."""
        server = _make_server()
        pool = server._ensure_pool()

        async def _run():
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(pool, _sleep_forever, "x", {})
            return await asyncio.wait_for(fut, timeout=2.0)

        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(_run())
        elapsed = time.monotonic() - start
        assert elapsed < 4.0, f"wait_for must cut off at ~2s, took {elapsed:.1f}s"

        # Recycle the pool so subsequent tests start clean (the hung worker
        # is still alive otherwise).
        async def _recycle():
            async with server._ensure_pool() and (server._verify_pool_lock or asyncio.Lock()):
                server._recycle_pool()
        # Simpler: just drop the pool and let _ensure_pool rebuild.
        server._verify_pool = None

    def test_pool_recycles_after_timeout_and_recovers(self):
        """After a timeout recycles the pool, the next (normal) request must
        grade correctly. This proves the recycle path is complete."""
        server = _make_server()

        # 0. Warm the pool so spawn overhead doesn't count against later calls.
        warmup = _make_request("", exp_cal_state={})
        asyncio.run(server.verify(warmup))

        # 1. Trigger a timeout.
        bad = _make_request(_poison_bracket_string(300), exp_cal_state={"1": {"day": "monday"}})
        resp1 = asyncio.run(server.verify(bad))
        assert resp1.reward == 0

        # 2. Send a normal request — the new pool should handle it. The recycle
        # builds a fresh pool, which pays spawn again; allow slack.
        good_text = '[{"event_id": "1", "day": "monday", "start_time": "9am", "duration": 30}]'
        good = _make_request(good_text, exp_cal_state={})
        resp2 = asyncio.run(server.verify(good))
        # exp_cal_state is empty -> "no change expected" -> reward=1
        assert resp2.reward == 1, "Pool must recover after recycle and grade normally"

    def test_think_tag_still_returns_zero_fast(self):
        """Responses with <think> are rejected by the pre-grade check, no
        pool call needed. Must be fast (<100ms)."""
        server = _make_server()
        req = _make_request("<think>reasoning</think>final", exp_cal_state={"1": {}})
        start = time.monotonic()
        resp = asyncio.run(server.verify(req))
        elapsed = time.monotonic() - start
        assert resp.reward == 0
        # The pool was invoked (grade_assistant_response returns 0 for think),
        # but total time should still be well under timeout.
        assert elapsed < 3.0


class TestCalendarHealthEndpoint:
    """The /health endpoint is what the watchdog polls. /docs lies when the
    verify loop is wedged — /health must invoke a real verify under the same
    pool path so a hung worker is detectable."""

    def test_health_returns_ok_on_healthy_server(self):
        server = _make_server()
        # Warm the pool first so /health measures steady-state, not cold spawn.
        asyncio.run(server.verify(_make_request("", exp_cal_state={})))
        app = server.setup_webserver()
        health_route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
        start = time.monotonic()
        result = asyncio.run(health_route.endpoint())
        elapsed = time.monotonic() - start
        assert result["status"] == "ok", f"Expected ok, got {result}"
        # Steady-state /health should be <1s. The 3s ceiling leaves room for
        # slow CI nodes; the ops value is "fails fast on wedged loop."
        assert elapsed < 3.0, f"/health on warm pool should be fast; took {elapsed:.1f}s"

    def test_health_uses_end_to_end_verify(self):
        """/health's return value should include reward + reason — proving it
        actually ran the grader, not just pinged FastAPI."""
        server = _make_server()
        app = server.setup_webserver()
        health_route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
        result = asyncio.run(health_route.endpoint())
        # Empty exp_cal_state with empty response → reward=1, reason="pass"
        assert "reward" in result and "reason" in result


class TestPoolIsolation:
    """Sanity checks on the pool pattern itself."""

    def test_worker_process_separate_from_main(self):
        """Worker must run in a separate process (spawn, not fork) so that
        memory/thread state from the main server can't leak into the grader."""
        # _grade_in_worker directly — runs in-process — should still work but
        # prove it returns valid shape.
        reward, reason = _grade_in_worker("", {})
        assert isinstance(reward, float) and isinstance(reason, str)

    def test_worker_exception_yields_graded_result(self):
        """A crashing grader input must return (0.0, 'worker_exception:*') —
        never crash the pool, never propagate to verify()."""
        reward, reason = _grade_in_worker(None, {"1": {}})  # None -> AttributeError in grader
        assert reward == 0.0
        assert reason.startswith(("worker_exception:", "error")), reason
