"""Strict tests for routing_rm: catch the silent-failure modes that let the
qwen3-30B-rlvr1_v2_20k_gen-oeea1i0r run burn 68h of compute on a dead code_gen
grader, AND the 134k-timeout calendar outage in the _u5j70izc_resume run.

Stability contract (v2, 2026-05-04):
  * ``_score_single`` returns ``float('nan')`` on ANY failure — NOT 0.0.
    This makes grader failures distinguishable from genuine wrong answers.
  * ``routing_batched_rm`` mean-fills NaN rewards per env within the batch.
  * Per-env circuit breaker short-circuits after K consecutive failures.
  * Auto-halt if nan_rate stays high for M consecutive batches.

These tests pin that behavior AND assert the specific warning log is emitted,
so regressions (a new error path added without a log) break CI.

Run::

    pytest slime_integration/tests/test_routing_rm_strict.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from slime_integration import routing_rm


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_sample(env_type: str, metadata: Optional[Dict] = None, response: str = "42"):
    """Build a minimal Slime Sample-like object routing_rm expects."""
    md = {"env_type": env_type}
    if metadata:
        md.update(metadata)
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "what is 6*7?"}],
        response=response,
        label=None,
        metadata=md,
        index=0,
    )


class _FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body

    async def text(self):
        if isinstance(self._body, str):
            return self._body
        return json.dumps(self._body)


class _FakeSession:
    """Programmable aiohttp session stub.

    behavior: one of
      - {'status': int, 'body': dict|str}
      - {'raise': Exception}
      - {'sleep_then_status': (float, int, dict)}
    """

    def __init__(self, behavior: Dict):
        self.behavior = behavior
        self.calls: List[Dict] = []
        self.closed = False

    def post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if "raise" in self.behavior:
            raise self.behavior["raise"]
        if "sleep_then_status" in self.behavior:
            sleep, status, body = self.behavior["sleep_then_status"]

            class _Timeout:
                async def __aenter__(_self):
                    await asyncio.sleep(sleep)
                    return _FakeResponse(status, body)

                async def __aexit__(_self, *a):
                    return False

            return _Timeout()
        return _FakeResponse(self.behavior["status"], self.behavior["body"])


@pytest.fixture(autouse=True)
def _reset_routing_state(tmp_path, monkeypatch):
    """Each test gets a fresh routing table cache AND a fresh breaker state.

    Also redirect trajectory writes to a tmp dir so tests don't require
    /shared/dev (which doesn't exist outside the pod)."""
    routing_rm._routing_table = None
    routing_rm._session = None
    routing_rm._reset_circuit_state()
    monkeypatch.setenv("NEMOGYM_TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("NEMOGYM_EXP_NAME", "test_exp")
    yield
    routing_rm._routing_table = None
    routing_rm._session = None
    routing_rm._reset_circuit_state()


@pytest.fixture
def args():
    return SimpleNamespace(rm_url=None)


@pytest.fixture
def routing_table():
    """Minimal routing table used by most tests."""
    return {env: f"http://test.local:{10000 + i}" for i, env in enumerate(routing_rm.DEFAULT_ROUTING.keys())}


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------------------
# Section 1: NaN-sentinel failure contract (v2)
# ---------------------------------------------------------------------------


class TestNaNSentinelFailureModes:
    """Every one of these represents a path that used to return 0.0 silently
    (indistinguishable from a real wrong answer). They MUST now return NaN so
    that the batched path can mean-fill and metrics can detect the failure."""

    def test_http_500_returns_nan_and_logs_warning(self, args, routing_table, caplog):
        sample = _make_sample("code_gen", {"verifier_metadata": {"unit_tests": []}})
        fake = _FakeSession({"status": 500, "body": "Internal Server Error"})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
                reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward), f"HTTP 500 must return NaN, got {reward}"
        assert any("returned 500" in r.message for r in caplog.records), (
            "HTTP 500 must emit a warning log; without it the failure is invisible in production"
        )

    def test_timeout_returns_nan_and_logs_warning(self, args, routing_table, caplog):
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"raise": asyncio.TimeoutError()})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
                reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward), f"Timeout must return NaN, got {reward}"
        assert any("Timeout" in r.message for r in caplog.records)

    def test_connection_reset_returns_nan_after_retries(self, args, routing_table, caplog):
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"raise": ConnectionResetError("reset")})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
                reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward), f"Connection reset must return NaN, got {reward}"
        assert any("disconnected" in r.message.lower() or "error" in r.message.lower() for r in caplog.records)

    def test_missing_env_type_returns_nan_and_logs(self, args, routing_table, caplog):
        sample = SimpleNamespace(prompt="x", response="y", label=None, metadata={}, index=0)
        with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)
        assert any("no env_type" in r.message.lower() for r in caplog.records)

    def test_unknown_env_type_returns_nan_and_logs(self, args, routing_table, caplog):
        sample = _make_sample("totally_made_up_env")
        with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)
        assert any("no server configured" in r.message.lower() for r in caplog.records)

    def test_payload_build_exception_returns_nan_and_logs(self, args, routing_table, caplog):
        sample = _make_sample("math")

        def bad_builder(**_kwargs):
            raise RuntimeError("simulated payload build failure")

        with patch.object(routing_rm, "_build_verify_payload", side_effect=bad_builder):
            with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
                reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)
        assert any("failed to build payload" in r.message.lower() for r in caplog.records)

    def test_200_with_valid_reward_returns_that_reward(self, args, routing_table):
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"reward": 1.0}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert reward == 1.0

    def test_200_with_missing_reward_field_returns_nan(self, args, routing_table, caplog):
        """Server returns 200 but body has no 'reward' key — must be NaN, not 0.0."""
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"some_other_field": 1.0}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            with caplog.at_level(logging.WARNING, logger=routing_rm.logger.name):
                reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)
        assert any("no 'reward' field" in r.message for r in caplog.records)

    def test_200_with_non_numeric_reward_returns_nan(self, args, routing_table):
        """Server returns 200 with reward='oops' — must be NaN."""
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"reward": "oops"}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)

    def test_200_with_nan_reward_returns_nan(self, args, routing_table):
        """If the grader itself returns NaN/Inf, route that through as NaN — never into training."""
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"reward": float("nan")}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)

    def test_200_with_inf_reward_returns_nan(self, args, routing_table):
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"reward": float("inf")}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(reward)


# ---------------------------------------------------------------------------
# Section 2: per-env payload required-field tests (unchanged)
# ---------------------------------------------------------------------------


REQUIRED_FIELDS: Dict[str, set] = {
    "single_step_tool_use": {"responses_create_params", "response", "expected_action"},
    "instruction_following": {"responses_create_params", "response", "instruction_id_list", "kwargs"},
    "code_gen": {"responses_create_params", "response", "verifier_metadata"},
    "math": {"responses_create_params", "response", "question", "expected_answer"},
    "mcqa": {"responses_create_params", "response", "expected_answer", "options"},
    "structured_outputs": {"responses_create_params", "response", "schema_str", "schema_type"},
    "calendar": {"responses_create_params", "response", "exp_cal_state"},
    "reasoning_gym": {"responses_create_params", "response", "question", "answer", "metadata"},
    "math_formal_lean": {"responses_create_params", "response", "header", "formal_statement", "name"},
    "workplace_assistant": {"responses_create_params", "response", "ground_truth", "environment_name"},
}


CANONICAL_METADATA: Dict[str, Dict] = {
    "single_step_tool_use": {"expected_action": {"name": "f", "arguments": {}}},
    "instruction_following": {
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 10}],
        "grading_mode": "strict",
        "prompt": "write a poem",
        "id": "if_0",
    },
    "code_gen": {"verifier_metadata": {"unit_tests": [{"input": "", "output": ""}]}},
    "math": {"question": "6*7?", "expected_answer": "42"},
    "mcqa": {
        "expected_answer": "B",
        "options": ["A", "B", "C", "D"],
        "grading_mode": "exact",
        "template_metadata": {},
        "uuid": "u0",
    },
    "structured_outputs": {
        "schema_str": json.dumps({"type": "object", "properties": {"x": {"type": "number"}}}),
        "schema_type": "json_schema",
    },
    "calendar": {"exp_cal_state": {"events": []}},
    "reasoning_gym": {
        "question": "2+2?",
        "answer": "4",
        "reasoning_gym_metadata": {"source_dataset": "rg_arith"},
    },
    "math_formal_lean": {
        "header": "import Mathlib",
        "formal_statement": "theorem foo : 2+2=4 := by rfl",
        "informal_prefix": "",
        "name": "foo",
    },
    "workplace_assistant": {
        "ground_truth": "answer",
        "id": "wp_0",
        "category": "cat",
        "environment_name": "slack",
    },
}


class TestPerEnvPayloadSchema:
    @pytest.mark.parametrize("env_type", sorted(REQUIRED_FIELDS.keys()))
    def test_payload_has_all_required_fields(self, env_type):
        if env_type not in CANONICAL_METADATA:
            pytest.skip(f"No canonical metadata defined for {env_type}")
        sample = _make_sample(env_type, CANONICAL_METADATA[env_type])
        payload = routing_rm._build_verify_payload(
            env_type=env_type,
            metadata=sample.metadata,
            response_text=sample.response,
            prompt=sample.prompt,
            sample=sample,
        )
        missing = REQUIRED_FIELDS[env_type] - set(payload.keys())
        assert not missing, f"env_type={env_type} payload missing required fields: {missing}"

    def test_response_wrapping_strips_chat_template_tokens(self):
        wrapped = routing_rm._wrap_response("hello <|im_end|>")
        text = wrapped["output"][0]["content"][0]["text"]
        assert "<|" not in text and "hello" in text

    def test_response_wrapping_strips_think_blocks(self):
        wrapped = routing_rm._wrap_response("<think>foo</think>final")
        text = wrapped["output"][0]["content"][0]["text"]
        assert "<think>" not in text and "foo" not in text and "final" in text


# ---------------------------------------------------------------------------
# Section 3: routing table completeness (unchanged) + timeout hygiene
# ---------------------------------------------------------------------------


class TestRoutingTableCompleteness:
    def test_default_routing_has_url_per_required_env(self):
        required = set(REQUIRED_FIELDS.keys())
        assert required.issubset(set(routing_rm.DEFAULT_ROUTING.keys())), (
            f"DEFAULT_ROUTING missing envs: {required - set(routing_rm.DEFAULT_ROUTING.keys())}"
        )

    def test_every_routing_url_is_http(self):
        for env, url in routing_rm.DEFAULT_ROUTING.items():
            assert url.startswith(("http://", "https://")), f"{env} has non-HTTP url: {url}"

    def test_env_timeouts_only_reference_known_envs(self):
        unknown = set(routing_rm.ENV_TIMEOUTS.keys()) - set(routing_rm.DEFAULT_ROUTING.keys())
        assert not unknown, f"ENV_TIMEOUTS has envs not in DEFAULT_ROUTING: {unknown}"

    def test_env_timeouts_are_tight(self):
        """Graders with SIGKILL pools complete in <10s. Per-env timeout should
        be short enough that the circuit breaker trips in minutes, not hours.
        Regressing to 60s defaults (the bug that caused 54h of calendar
        timeouts) must break this test."""
        # Non-code_gen envs should be <= 15s (post-SIGKILL-pool expectation).
        for env in ("math", "mcqa", "structured_outputs", "instruction_following",
                    "reasoning_gym", "calendar"):
            t = routing_rm.ENV_TIMEOUTS.get(env)
            assert t is not None and t <= 15.0, (
                f"env={env} timeout={t}s exceeds 15s; this keeps the router hammering "
                "a slow/wedged grader for too long before the circuit breaker trips"
            )


# ---------------------------------------------------------------------------
# Section 4: Batch-level NaN handling (mean-fill + metrics)
# ---------------------------------------------------------------------------


class TestBatchNaNFilling:
    """The critical change: batched_rm must preserve shape for slime while
    making sure failed samples contribute ~0 to advantage (mean-centered)."""

    def test_fill_nan_rewards_replaces_nan_with_env_mean(self):
        samples = [_make_sample("math", {"question": "q", "expected_answer": "42"}) for _ in range(4)]
        rewards = [1.0, 0.0, float("nan"), 1.0]
        filled, nan_rate, nan_counts, totals = routing_rm._fill_nan_rewards(rewards, samples)
        # Mean of non-NaN math rewards: (1+0+1)/3 = 0.6667
        assert math.isclose(filled[2], 2/3, rel_tol=1e-6), (
            f"NaN sample should be filled with env mean; got {filled[2]}"
        )
        # Non-NaN samples unchanged.
        assert filled[0] == 1.0 and filled[1] == 0.0 and filled[3] == 1.0
        assert nan_rate["math"] == 0.25
        assert nan_counts["math"] == 1
        assert totals["math"] == 4

    def test_fill_nan_rewards_multi_env_isolation(self):
        """NaN in one env must not affect fill value for another env."""
        samples = [
            _make_sample("math", {"question": "q", "expected_answer": "1"}),
            _make_sample("math", {"question": "q", "expected_answer": "1"}),
            _make_sample("code_gen", {"verifier_metadata": {"unit_tests": []}}),
            _make_sample("code_gen", {"verifier_metadata": {"unit_tests": []}}),
        ]
        rewards = [1.0, float("nan"), 0.0, 0.0]
        filled, nan_rate, _, _ = routing_rm._fill_nan_rewards(rewards, samples)
        # Math NaN filled with math mean (1.0), NOT code_gen mean (0.0)
        assert filled[1] == 1.0
        assert filled[0] == 1.0 and filled[2] == 0.0 and filled[3] == 0.0
        assert nan_rate["math"] == 0.5
        assert nan_rate["code_gen"] == 0.0

    def test_fill_nan_rewards_all_nan_for_env_falls_back_to_zero(self):
        """If every sample for an env is NaN (grader fully dead), fill with 0
        rather than crashing — the circuit breaker will stop the hammering."""
        samples = [_make_sample("math", {"question": "q", "expected_answer": "1"}) for _ in range(3)]
        rewards = [float("nan"), float("nan"), float("nan")]
        filled, nan_rate, _, _ = routing_rm._fill_nan_rewards(rewards, samples)
        assert all(r == 0.0 for r in filled)
        assert nan_rate["math"] == 1.0

    def test_batched_rm_returns_same_length_as_input(self, args, routing_table):
        """Slime requires len(rewards) == len(samples). Must hold even when
        many samples time out — NaN is passed through, slime handles masking."""
        samples = [_make_sample("math", {"question": "q", "expected_answer": "42"}) for _ in range(6)]
        fake = _FakeSession({"raise": asyncio.TimeoutError()})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)), \
             patch.object(routing_rm, "_log_trajectories"), \
             patch.dict("os.environ", {"NEMOGYM_NAN_HALT_CONSECUTIVE": "0"}):
            routing_rm.NAN_RATE_HALT_CONSECUTIVE = 0
            try:
                rewards = asyncio.run(routing_rm.routing_batched_rm(args, samples))
            finally:
                routing_rm.NAN_RATE_HALT_CONSECUTIVE = 3
        assert isinstance(rewards, list) and len(rewards) == len(samples)
        # New contract: NaN passes through to slime, which sets remove_sample.
        assert all(_is_nan(r) for r in rewards), (
            "All-fail batch should forward NaN; slime _post_process_rewards "
            "will mark each sample.remove_sample=True"
        )

    def test_batched_rm_mixed_failure_and_success_passes_nan_through(self, args, routing_table):
        """Some samples succeed, some fail. Successful rewards are returned
        verbatim; failures are NaN. No mean-filling at the router layer."""
        samples = [_make_sample("math", {"question": "q", "expected_answer": "42"}) for _ in range(4)]

        call_n = {"n": 0}

        class _MixedSession:
            closed = False

            def post(self, *a, **kw):
                call_n["n"] += 1
                n = call_n["n"]

                class _Ctx:
                    async def __aenter__(_self):
                        if n == 3:  # third call times out
                            raise asyncio.TimeoutError()
                        return _FakeResponse(200, {"reward": 1.0})

                    async def __aexit__(_self, *a):
                        return False

                return _Ctx()

        mixed = _MixedSession()
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=mixed)), \
             patch.object(routing_rm, "_log_trajectories"):
            routing_rm.NAN_RATE_HALT_CONSECUTIVE = 0
            try:
                rewards = asyncio.run(routing_rm.routing_batched_rm(args, samples))
            finally:
                routing_rm.NAN_RATE_HALT_CONSECUTIVE = 3
        assert len(rewards) == 4
        # Three successes at 1.0, one NaN (the third call).
        assert rewards[0] == 1.0 and rewards[1] == 1.0 and rewards[3] == 1.0
        assert _is_nan(rewards[2])


# ---------------------------------------------------------------------------
# Section 5: Circuit breaker behavior
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """After K consecutive failures for an env, the breaker must OPEN and
    short-circuit subsequent calls to NaN — preventing 134k timeouts-per-day
    like the calendar outage."""

    def test_breaker_opens_after_threshold_consecutive_failures(self, args, routing_table, caplog):
        """Drive K+1 failures and confirm the breaker opens."""
        K = routing_rm.CIRCUIT_BREAKER_THRESHOLD
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"raise": asyncio.TimeoutError()})

        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            with caplog.at_level(logging.ERROR, logger=routing_rm.logger.name):
                for _ in range(K):
                    r = asyncio.run(routing_rm._score_single(args, sample, routing_table))
                    assert _is_nan(r)
        assert routing_rm._is_circuit_open("math"), "Breaker must be OPEN after K failures"
        assert any("Circuit breaker OPEN" in r.message for r in caplog.records), (
            "Breaker opening must log an ERROR-level message for ops visibility"
        )

    def test_breaker_short_circuits_without_http_call(self, args, routing_table):
        """Once the breaker is open, subsequent calls must NOT touch the network."""
        # Force the breaker OPEN directly.
        routing_rm._env_circuit_open_until["math"] = routing_rm._now() + 60.0
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})

        fake = _FakeSession({"status": 200, "body": {"reward": 1.0}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            r = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert _is_nan(r), "Breaker-open path must return NaN"
        assert fake.calls == [], f"Breaker-open path must NOT hit HTTP; got calls: {fake.calls}"

    def test_breaker_isolates_envs(self, args, routing_table):
        """math breaker open must not block code_gen calls."""
        routing_rm._env_circuit_open_until["math"] = routing_rm._now() + 60.0
        math_sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        code_sample = _make_sample("code_gen", {"verifier_metadata": {"unit_tests": []}})

        fake = _FakeSession({"status": 200, "body": {"reward": 1.0}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            r_math = asyncio.run(routing_rm._score_single(args, math_sample, routing_table))
            r_code = asyncio.run(routing_rm._score_single(args, code_sample, routing_table))
        assert _is_nan(r_math)
        assert r_code == 1.0

    def test_breaker_resets_on_success(self, args, routing_table):
        """A success should zero the consecutive-failure counter."""
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fail = _FakeSession({"raise": asyncio.TimeoutError()})
        ok = _FakeSession({"status": 200, "body": {"reward": 1.0}})

        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fail)):
            for _ in range(3):
                asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert routing_rm._env_consecutive_failures["math"] == 3

        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=ok)):
            r = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert r == 1.0
        assert routing_rm._env_consecutive_failures.get("math", 0) == 0

    def test_breaker_auto_recovers_after_probe_interval(self, args, routing_table):
        """After probe interval elapses, the circuit reopens for one probe."""
        # Force breaker OPEN with expiry in the past.
        routing_rm._env_circuit_open_until["math"] = routing_rm._now() - 1.0
        # Should be treated as closed now.
        assert not routing_rm._is_circuit_open("math")

        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"status": 200, "body": {"reward": 1.0}})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)):
            r = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert r == 1.0, "After probe interval, a successful probe must land"


# ---------------------------------------------------------------------------
# Section 6: Auto-halt guard
# ---------------------------------------------------------------------------


class TestAutoHaltGuard:
    """If any env has nan_rate > MAX_NAN_RATE_BEFORE_HALT for M consecutive
    batches, routing_batched_rm must raise RuntimeError rather than letting
    training silently corrupt gradient for hours."""

    def test_halt_raised_after_m_consecutive_high_nan_batches(self, args, routing_table):
        # Reduce to 2 for the test to run quickly.
        orig_halt = routing_rm.NAN_RATE_HALT_CONSECUTIVE
        orig_rate = routing_rm.MAX_NAN_RATE_BEFORE_HALT
        routing_rm.NAN_RATE_HALT_CONSECUTIVE = 2
        routing_rm.MAX_NAN_RATE_BEFORE_HALT = 0.5

        try:
            samples = [_make_sample("math", {"question": "q", "expected_answer": "42"}) for _ in range(4)]
            fake = _FakeSession({"raise": asyncio.TimeoutError()})
            with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)), \
                 patch.object(routing_rm, "_log_trajectories"):
                # 1st batch: all NaN -> high nan_rate -> counter = 1, no raise
                asyncio.run(routing_rm.routing_batched_rm(args, samples))
                # 2nd batch: all NaN -> high nan_rate -> counter = 2 >= M=2 -> raise
                with pytest.raises(RuntimeError, match="nan_rate"):
                    asyncio.run(routing_rm.routing_batched_rm(args, samples))
        finally:
            routing_rm.NAN_RATE_HALT_CONSECUTIVE = orig_halt
            routing_rm.MAX_NAN_RATE_BEFORE_HALT = orig_rate

    def test_halt_not_raised_when_recovered(self, args, routing_table):
        """One bad batch then a good batch must reset the counter."""
        orig_halt = routing_rm.NAN_RATE_HALT_CONSECUTIVE
        orig_rate = routing_rm.MAX_NAN_RATE_BEFORE_HALT
        routing_rm.NAN_RATE_HALT_CONSECUTIVE = 2
        routing_rm.MAX_NAN_RATE_BEFORE_HALT = 0.5

        try:
            samples = [_make_sample("math", {"question": "q", "expected_answer": "42"}) for _ in range(4)]

            # Batch 1: all NaN
            fail = _FakeSession({"raise": asyncio.TimeoutError()})
            with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fail)), \
                 patch.object(routing_rm, "_log_trajectories"):
                asyncio.run(routing_rm.routing_batched_rm(args, samples))
            assert routing_rm._env_consecutive_high_nan_batches.get("math", 0) == 1

            # Batch 2: all success -> counter resets
            ok = _FakeSession({"status": 200, "body": {"reward": 1.0}})
            with patch.object(routing_rm, "_get_session", AsyncMock(return_value=ok)), \
                 patch.object(routing_rm, "_log_trajectories"):
                asyncio.run(routing_rm.routing_batched_rm(args, samples))
            assert routing_rm._env_consecutive_high_nan_batches.get("math", 0) == 0
        finally:
            routing_rm.NAN_RATE_HALT_CONSECUTIVE = orig_halt
            routing_rm.MAX_NAN_RATE_BEFORE_HALT = orig_rate


# ---------------------------------------------------------------------------
# Section 7: Retry policy (unchanged behavior)
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_connection_reset_is_retried_before_giving_up(self, args, routing_table):
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})

        call_count = {"n": 0}

        class _FlakySession:
            closed = False

            def post(self, *a, **kw):
                call_count["n"] += 1

                class _Ctx:
                    async def __aenter__(_self):
                        if call_count["n"] < 3:
                            raise ConnectionResetError("reset")
                        return _FakeResponse(200, {"reward": 1.0})

                    async def __aexit__(_self, *a):
                        return False

                return _Ctx()

        flaky = _FlakySession()
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=flaky)):
            reward = asyncio.run(routing_rm._score_single(args, sample, routing_table))
        assert reward == 1.0, "A transient connection reset followed by a success must recover"
        assert call_count["n"] >= 3, "Retries must actually happen"


# ---------------------------------------------------------------------------
# Section 8: Single-sample routing_rm API (NaN -> 0.0 fallback)
# ---------------------------------------------------------------------------


class TestSingleSamplePath:
    def test_single_sample_nan_becomes_zero(self, args, routing_table):
        """routing_rm (not batched) returns 0.0 on grader failure because a
        single sample has no batch peers to mean-fill from. Loud warning
        must fire so the failure is visible."""
        sample = _make_sample("math", {"question": "q", "expected_answer": "42"})
        fake = _FakeSession({"raise": asyncio.TimeoutError()})
        with patch.object(routing_rm, "_get_session", AsyncMock(return_value=fake)), \
             patch.dict("os.environ", {}, clear=False):
            reward = asyncio.run(routing_rm.routing_rm(args, sample))
        assert reward == 0.0, "Single-sample path degrades NaN -> 0.0 (acceptable fallback)"


# ---------------------------------------------------------------------------
# Section 9: Ray version pin (preserved from v1)
# ---------------------------------------------------------------------------


class TestRayVersionSanity:
    """code_gen's auto-init path blew up with 'Version mismatch: cluster Ray
    2.54.0 vs process Ray 2.55.0'. Pin the installed Ray version so CI rejects
    images that drift."""

    EXPECTED_RAY_MAJOR_MINOR = "2.54"

    def test_installed_ray_version_matches_cluster(self):
        ray = pytest.importorskip("ray")
        installed = ".".join(ray.__version__.split(".")[:2])
        assert installed == self.EXPECTED_RAY_MAJOR_MINOR, (
            f"Ray version drift: installed {ray.__version__}, cluster expects "
            f"{self.EXPECTED_RAY_MAJOR_MINOR}.x. This is the oeea1i0r bug — a "
            f"server process with a newer Ray minor will fail every /verify call silently."
        )
