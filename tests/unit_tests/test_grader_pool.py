# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for nemo_gym.grader_pool — the shared SIGKILL-on-timeout
worker-pool abstraction used by every grader that can hang on untrusted
input (math, calendar, structured_outputs, reasoning_gym, instruction_following).

These tests validate the contract without requiring any resources server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import os
import time

import pytest

from nemo_gym.grader_pool import GraderPool


# ---------------------------------------------------------------------------
# Module-level workers (spawn pickle requirement)
# ---------------------------------------------------------------------------


def _worker_ok(x: int) -> int:
    return x * 2


def _worker_sleep(duration: float) -> int:
    time.sleep(duration)
    return 42


def _worker_raise() -> int:
    raise RuntimeError("boom from worker")


def _worker_sigill() -> None:
    """Crash the worker with SIGILL to test BrokenProcessPool handling."""
    os.kill(os.getpid(), 4)  # SIGILL


# ---------------------------------------------------------------------------
# Minimal stand-in for a "server instance"
# ---------------------------------------------------------------------------


class _FakeServer:
    """Dummy object to hold pool state. GraderPool attaches per-instance
    attributes, so any object will do."""
    pass


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pool_short_timeout(monkeypatch):
    """Pool with a 2s timeout."""
    monkeypatch.setenv("TEST_POOL_TIMEOUT_S", "2")
    monkeypatch.setenv("TEST_POOL_WORKERS", "2")
    return GraderPool(
        name="test_pool",
        timeout_s_env="TEST_POOL_TIMEOUT_S",
        workers_env="TEST_POOL_WORKERS",
    )


@pytest.fixture
def bound_server(pool_short_timeout):
    s = _FakeServer()
    pool_short_timeout.bind(s)
    yield s, pool_short_timeout
    # Shutdown pool to release spawned processes.
    pool = getattr(s, "__pool_test_pool", None)
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestGraderPoolHappyPath:
    def test_returns_worker_result(self, bound_server):
        server, pool = bound_server
        result = asyncio.run(pool.run(server, _worker_ok, 21))
        assert result == 42

    def test_pool_reused_across_calls(self, bound_server):
        server, pool = bound_server
        # First call warm-starts pool; second must be fast (<1s on a warm pool).
        asyncio.run(pool.run(server, _worker_ok, 1))
        start = time.monotonic()
        asyncio.run(pool.run(server, _worker_ok, 2))
        elapsed = time.monotonic() - start
        # Warm-pool call should be well under 1s; allow 2s for slow CI.
        assert elapsed < 2.0, f"Warm pool call took {elapsed:.2f}s (expected <2s)"


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


class TestGraderPoolTimeout:
    def test_timeout_raises_and_recycles(self, bound_server):
        """A worker that exceeds the timeout must be killed and the pool recycled."""
        server, pool = bound_server
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(pool.run(server, _worker_sleep, 10.0))
        elapsed = time.monotonic() - start
        # 2s timeout + spawn overhead (~2s) + slack = 6s ceiling.
        assert elapsed < 6.0, f"Timeout should fire at ~2s, took {elapsed:.1f}s"

    def test_pool_recovers_after_timeout(self, bound_server):
        """After recycle, subsequent calls must succeed on the fresh pool."""
        server, pool = bound_server
        # Warm first.
        asyncio.run(pool.run(server, _worker_ok, 1))
        # Cause a timeout.
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(pool.run(server, _worker_sleep, 10.0))
        # Fresh pool handles subsequent call.
        result = asyncio.run(pool.run(server, _worker_ok, 7))
        assert result == 14

    def test_run_or_zero_returns_zero_on_timeout(self, bound_server):
        server, pool = bound_server
        result, reason = asyncio.run(pool.run_or_zero(server, _worker_sleep, 10.0))
        assert result == 0.0 and reason == "timeout"

    def test_run_or_zero_custom_zero_value(self, bound_server):
        server, pool = bound_server
        result, reason = asyncio.run(
            pool.run_or_zero(server, _worker_sleep, 10.0, zero_value=(0.0, "fallback"))
        )
        assert result == (0.0, "fallback") and reason == "timeout"


# ---------------------------------------------------------------------------
# Worker-death path
# ---------------------------------------------------------------------------


class TestGraderPoolWorkerDeath:
    def test_worker_crash_raises_broken_pool(self, bound_server):
        server, pool = bound_server
        # Warm the pool first.
        asyncio.run(pool.run(server, _worker_ok, 1))
        with pytest.raises((cf.process.BrokenProcessPool, cf.CancelledError)):
            asyncio.run(pool.run(server, _worker_sigill))

    def test_pool_recovers_after_worker_death(self, bound_server):
        server, pool = bound_server
        asyncio.run(pool.run(server, _worker_ok, 1))
        try:
            asyncio.run(pool.run(server, _worker_sigill))
        except (cf.process.BrokenProcessPool, cf.CancelledError):
            pass
        # Fresh pool should handle the next call.
        result = asyncio.run(pool.run(server, _worker_ok, 3))
        assert result == 6

    def test_run_or_zero_handles_worker_death(self, bound_server):
        server, pool = bound_server
        asyncio.run(pool.run(server, _worker_ok, 1))
        result, reason = asyncio.run(pool.run_or_zero(server, _worker_sigill))
        assert result == 0.0
        assert reason in ("worker_died", "cancelled")


# ---------------------------------------------------------------------------
# Worker-raises-regular-exception path
# ---------------------------------------------------------------------------


class TestGraderPoolWorkerException:
    def test_regular_exception_propagates_from_run(self, bound_server):
        """If the worker raises a normal Python exception (not a crash), run()
        lets it propagate so the caller can distinguish bugs from hangs."""
        server, pool = bound_server
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(pool.run(server, _worker_raise))

    def test_run_or_zero_converts_exception_to_reason(self, bound_server):
        server, pool = bound_server
        result, reason = asyncio.run(pool.run_or_zero(server, _worker_raise))
        assert result == 0.0
        assert reason.startswith("grade_error:RuntimeError")


# ---------------------------------------------------------------------------
# Isolation: two servers, same GraderPool class, separate pool state
# ---------------------------------------------------------------------------


class TestGraderPoolIsolation:
    def test_separate_servers_have_separate_pools(self, pool_short_timeout):
        s1 = _FakeServer()
        s2 = _FakeServer()
        pool_short_timeout.bind(s1)
        pool_short_timeout.bind(s2)

        # Warm both.
        asyncio.run(pool_short_timeout.run(s1, _worker_ok, 1))
        asyncio.run(pool_short_timeout.run(s2, _worker_ok, 2))

        p1 = getattr(s1, "__pool_test_pool")
        p2 = getattr(s2, "__pool_test_pool")
        assert p1 is not p2, "Bound servers must each have their own pool"

        # Clean up.
        for p in (p1, p2):
            try:
                p.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
