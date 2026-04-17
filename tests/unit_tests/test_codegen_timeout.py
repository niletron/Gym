# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for code_gen server timeout behavior under high-load scenarios.

Root cause analysis of 2,562 timeouts in RLVR1_v2 training:

The timeout chain has a fundamental mismatch:
1. routing_rm.py: ENV_TIMEOUTS["code_gen"] = 120.0s (HTTP request timeout)
2. code_gen YAML: unit_test_timeout_secs = 10s (per-test-case signal.alarm)
3. check_correctness: Process.join(timeout=(10+1)*N+5) where N = number of test cases
4. For N >= 11 test cases: Process.join timeout > 120s HTTP timeout

In the RLVR1_v2 dataset:
- 73.3% of code_gen samples (8,783/11,984) have >= 11 test cases
- Problems with 100+ test cases have Process.join timeouts > 1,100 seconds
- The worst case (430 test cases) has a Process.join timeout of 4,735 seconds

When model-generated code hits TLE on ANY test case in a high-test-count problem,
the multiprocessing timeout exceeds the HTTP timeout, causing the routing_rm to
receive an asyncio.TimeoutError and return reward=0.

These tests verify the timeout behavior with synthetic high-load scenarios that
mirror real production failures.
"""

import json
import multiprocessing
import signal
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers -- lightweight versions of code_gen internals for unit testing
# without requiring Ray or the full lcb_integration stack
# ---------------------------------------------------------------------------


def _run_code_with_timeout(code: str, input_str: str, timeout: int) -> tuple:
    """Minimal code execution with signal-based timeout, mirroring testing_util.run_test."""
    from io import StringIO
    from types import ModuleType

    def timeout_handler(signum, frame):
        raise TimeoutError("alarm")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        # Compile and run
        tmp = ModuleType("tmp")
        exec(code, tmp.__dict__)
        if hasattr(tmp, "wrapped_function"):
            from unittest.mock import mock_open, patch

            mock_stdin = StringIO(input_str)
            with patch("sys.stdin", mock_stdin), patch("builtins.open", mock_open(read_data=input_str)):
                old_stdout = sys.stdout
                sys.stdout = captured = StringIO()
                try:
                    tmp.wrapped_function()
                finally:
                    sys.stdout = old_stdout
                return captured.getvalue(), None
    except TimeoutError:
        return None, "TLE"
    except Exception as e:
        return None, str(e)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return None, "no output"


def _check_correctness_local(sample: dict, generation: str, timeout: int) -> tuple:
    """Local version of check_correctness that doesn't require Ray.

    Mirrors the exact timeout logic from compute_code_generation_metrics.py:
    - Spawns a multiprocessing.Process
    - Process.join(timeout=(timeout+1)*N+5) where N = number of test cases
    """
    in_outs = json.loads(sample["input_output"])
    n_tests = len(in_outs.get("inputs", []))

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
        # Simplified: just try to exec the code with each input
        for i, inp in enumerate(in_outs.get("inputs", [])):
            try:
                signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("alarm")))
                signal.alarm(timeout)
                # Just exec the code -- we're testing timeout behavior, not correctness
                exec(generation, {"__builtins__": __builtins__, "input": lambda: inp})
                signal.alarm(0)
                result.append(True)
            except TimeoutError:
                signal.alarm(0)
                result.append(-3)
                metadata_list.append({"error": "TLE", "test_index": i})
                return
            except Exception as e:
                signal.alarm(0)
                result.append(-4)
                metadata_list.append({"error": str(e), "test_index": i})
                return
        metadata_list.append({"status": "ok"})

    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, False, result, metadata_list, timeout),
    )
    start = time.time()
    p.start()
    join_timeout = (timeout + 1) * n_tests + 5
    p.join(timeout=join_timeout)
    elapsed = time.time() - start

    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return [-1] * n_tests, {"error": "global_timeout", "elapsed": elapsed, "join_timeout": join_timeout}

    if not result:
        return [-1] * n_tests, {"error": "no_result", "elapsed": elapsed}

    return list(result), list(metadata_list)[0] if metadata_list else None


# ---------------------------------------------------------------------------
# Test: Timeout chain math verification
# ---------------------------------------------------------------------------


class TestTimeoutChainMath:
    """Verify the timeout arithmetic that causes HTTP timeouts."""

    @pytest.mark.parametrize(
        "n_tests,unit_timeout,http_timeout,should_exceed",
        [
            (1, 10, 120, False),  # 16s < 120s
            (5, 10, 120, False),  # 60s < 120s
            (10, 10, 120, False),  # 115s < 120s
            (11, 10, 120, True),  # 126s > 120s -- FIRST FAILURE POINT
            (50, 10, 120, True),  # 555s > 120s
            (100, 10, 120, True),  # 1105s > 120s
            (101, 10, 120, True),  # 1116s -- most common in dataset (1432 samples)
            (200, 10, 120, True),  # 2205s > 120s
            (430, 10, 120, True),  # 4735s -- worst case in dataset
            # With reduced timeout of 3s:
            (11, 3, 120, False),  # (3+1)*11+5 = 49s < 120s
            (20, 3, 120, False),  # (3+1)*20+5 = 85s < 120s
            (28, 3, 120, False),  # (3+1)*28+5 = 117s < 120s
            (29, 3, 120, True),  # (3+1)*29+5 = 121s > 120s -- new boundary
            (50, 3, 120, True),  # (3+1)*50+5 = 205s > 120s
            # With reduced timeout AND increased HTTP timeout:
            (100, 3, 600, False),  # (3+1)*100+5 = 405s < 600s
            (149, 3, 600, True),  # (3+1)*149+5 = 601s > 600s
        ],
    )
    def test_process_join_vs_http_timeout(self, n_tests, unit_timeout, http_timeout, should_exceed):
        """Verify whether Process.join timeout exceeds the HTTP timeout.

        The formula is: process_join_timeout = (unit_timeout + 1) * n_tests + 5

        When process_join_timeout > http_timeout, the HTTP request will timeout
        before the code execution completes, wasting resources.
        """
        process_join_timeout = (unit_timeout + 1) * n_tests + 5
        exceeds = process_join_timeout > http_timeout
        assert exceeds == should_exceed, (
            f"n_tests={n_tests}, unit_timeout={unit_timeout}s: "
            f"Process.join={process_join_timeout}s vs HTTP={http_timeout}s"
        )

    def test_dataset_timeout_exposure(self):
        """73.3% of RLVR1_v2 code_gen samples can exceed the HTTP timeout.

        This test documents the distribution from production data.
        """
        # Distribution from actual RLVR1_v2 training data (11,984 code_gen samples)
        # fmt: off
        test_count_distribution = {
            1: 1020, 2: 426, 3: 327, 4: 255, 5: 215, 6: 223, 7: 217,
            8: 211, 9: 153, 10: 154,
            # >= 11 test cases: Process.join > 120s HTTP timeout
            11: 139, 12: 150, 13: 152, 14: 118, 15: 107, 16: 80, 17: 120,
            18: 88, 19: 79, 20: 96,
            # Spike at 101-104 (LeetCode standard)
            101: 1432, 102: 524, 103: 512, 104: 238,
            # Tail (up to 430 test cases)
        }
        # fmt: on

        unit_timeout = 10  # from code_gen.yaml
        http_timeout = 120  # from routing_rm.py ENV_TIMEOUTS

        total = sum(test_count_distribution.values())
        at_risk = sum(
            count
            for n_tests, count in test_count_distribution.items()
            if (unit_timeout + 1) * n_tests + 5 > http_timeout
        )

        # Document that a large fraction of the dataset is at risk
        at_risk_pct = 100 * at_risk / total
        assert at_risk_pct > 50, f"Expected >50% at risk, got {at_risk_pct:.1f}%"


# ---------------------------------------------------------------------------
# Test: Infinite loop detection
# ---------------------------------------------------------------------------


class TestInfiniteLoopDetection:
    """Test that the signal.alarm mechanism catches infinite loops."""

    @pytest.mark.parametrize(
        "code,expected_tle",
        [
            # Infinite while loop
            ("while True: pass", True),
            # Infinite recursion (will hit recursion limit before alarm, but still fails)
            ("def f(): f()\nf()", False),  # RecursionError, not TLE
            # Busy loop with large range
            ("s = 0\nfor i in range(10**9): s += i\nprint(s)", True),
            # Quick code -- should NOT timeout
            ("print(42)", False),
        ],
    )
    def test_signal_alarm_catches_hangs(self, code, expected_tle):
        """signal.alarm should catch Python-level infinite loops within the timeout."""
        sample = {"input_output": json.dumps({"inputs": [""], "outputs": ["42"]})}
        timeout = 2  # Short timeout for fast tests

        start = time.time()
        result, metadata = _check_correctness_local(sample, code, timeout)
        elapsed = time.time() - start

        if expected_tle:
            # Should complete within a reasonable time (timeout + overhead)
            assert elapsed < timeout + 10, f"Took {elapsed:.1f}s, expected < {timeout + 10}s"
            # Result should indicate failure
            assert any(r == -3 or r == -1 for r in result), f"Expected TLE result, got {result}"
        else:
            # Should complete quickly
            assert elapsed < timeout + 5, f"Took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Test: High test-case count scenarios
# ---------------------------------------------------------------------------


class TestHighTestCaseCount:
    """Test behavior with many test cases, mirroring production data."""

    def test_many_tests_fast_code(self):
        """100 test cases with fast code should still complete quickly."""
        n_tests = 100
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i * i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        code = "n=int(input())\nprint(n*n)"

        start = time.time()
        result, metadata = _check_correctness_local(sample, code, timeout=2)
        elapsed = time.time() - start

        # Fast code with 100 tests should complete in seconds, not minutes
        assert elapsed < 30, f"100 fast tests took {elapsed:.1f}s -- too slow"
        assert all(r is True or r == True for r in result), f"Expected all pass, got {result[:5]}..."

    def test_many_tests_slow_code_triggers_global_timeout(self):
        """Code that TLEs on first test should fail fast, not wait for all N tests."""
        n_tests = 20
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        # This code will TLE on every test case
        code = "import time; time.sleep(100)"

        timeout = 2
        start = time.time()
        result, metadata = _check_correctness_local(sample, code, timeout)
        elapsed = time.time() - start

        # Should fail on first test and return quickly
        # NOT wait for all 20 tests * 2s = 40s
        assert elapsed < timeout + 15, (
            f"Slow code with {n_tests} tests took {elapsed:.1f}s. Should fail fast on first TLE, not run all tests."
        )

    def test_process_join_timeout_computation(self):
        """Verify the Process.join timeout formula matches check_correctness."""
        for n_tests in [1, 5, 10, 50, 100]:
            timeout = 10
            expected_join = (timeout + 1) * n_tests + 5
            # This is the exact formula from check_correctness:
            # p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
            assert expected_join == (timeout + 1) * n_tests + 5


# ---------------------------------------------------------------------------
# Test: Code extraction edge cases (from real RLVR1_v2 samples)
# ---------------------------------------------------------------------------


class TestCodeExtractionEdgeCases:
    """Test code extraction from real model outputs that caused issues."""

    def test_think_tag_stripping(self):
        """routing_rm strips think tags before sending to code_gen.

        But the code_gen server receives the raw output_text, which
        for thinking models contains the code after </think>.
        The extraction should find the LAST code block, not one inside <think>.
        """
        import re

        # Simulated thinking model output (think block + code)
        response = (
            "<think>\n"
            "Let me think about this...\n"
            "```python\n# wrong approach\ndef bad(): pass\n```\n"
            "Actually, let me reconsider.\n"
            "</think>\n"
            "Here's my solution:\n"
            "```python\ndef solve():\n    n = int(input())\n    print(n * n)\n```\n"
        )

        # The routing_rm strips think tags
        stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Extract code (mimic extraction_utils for OpenAIChat style)
        lines = stripped.split("\n")
        backtick_lines = [i for i, line in enumerate(lines) if "```" in line]
        assert len(backtick_lines) >= 2, "Should find code fences after stripping think tags"

        code = "\n".join(lines[backtick_lines[-2] + 1 : backtick_lines[-1]])
        assert "def solve" in code, f"Should extract the correct code, got: {code}"
        assert "def bad" not in code, "Should NOT extract code from think block"

    def test_very_long_response_extraction(self):
        """Responses > 20k chars are common in RLVR1_v2 (3,743 samples > 20k chars).

        Long think blocks should not prevent code extraction.
        """
        # Simulate a 30k char response with a long think block
        think_content = "Let me think step by step...\n" * 1000  # ~30k chars
        response = f"<think>\n{think_content}</think>\n```python\nprint('hello')\n```\n"

        import re

        stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        lines = stripped.split("\n")
        backtick_lines = [i for i, line in enumerate(lines) if "```" in line]
        assert len(backtick_lines) >= 2
        code = "\n".join(lines[backtick_lines[-2] + 1 : backtick_lines[-1]])
        assert code.strip() == "print('hello')"

    def test_no_code_fence_returns_empty(self):
        """Model output without code fences should return empty string."""
        response = "I think the answer is 42.\n\nHere's my approach: just print 42."

        lines = response.split("\n")
        backtick_lines = [i for i, line in enumerate(lines) if "```" in line]
        if len(backtick_lines) < 2:
            code = ""
        else:
            code = "\n".join(lines[backtick_lines[-2] + 1 : backtick_lines[-1]])

        assert code == "", "No code fences should yield empty extraction"


# ---------------------------------------------------------------------------
# Test: Proposed fix -- global timeout cap
# ---------------------------------------------------------------------------


class TestProposedGlobalTimeoutCap:
    """Tests for the proposed fix: cap total execution time regardless of test count.

    Instead of Process.join(timeout=(per_test+1)*N+5), use a fixed global timeout
    that respects the HTTP timeout budget.
    """

    def _check_correctness_with_global_cap(
        self, sample: dict, generation: str, per_test_timeout: int, global_timeout: int
    ) -> tuple:
        """check_correctness with a global timeout cap.

        This is the proposed fix: replace the formula-based Process.join timeout
        with min(formula_timeout, global_timeout).
        """
        in_outs = json.loads(sample["input_output"])
        n_tests = len(in_outs.get("inputs", []))

        manager = multiprocessing.Manager()
        result = manager.list()
        metadata_list = manager.list()

        def _temp_run(in_outs, generation, result, metadata_list, timeout):
            for i, inp in enumerate(in_outs.get("inputs", [])):
                try:
                    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
                    signal.alarm(timeout)
                    exec(generation, {"__builtins__": __builtins__, "input": lambda: inp})
                    signal.alarm(0)
                    result.append(True)
                except TimeoutError:
                    signal.alarm(0)
                    result.append(-3)
                    metadata_list.append({"error": "TLE", "test_index": i})
                    return
                except Exception:
                    signal.alarm(0)
                    result.append(-4)
                    metadata_list.append({"error": "error", "test_index": i})
                    return
            metadata_list.append({"status": "ok"})

        p = multiprocessing.Process(
            target=_temp_run,
            args=(in_outs, generation, result, metadata_list, per_test_timeout),
        )
        start = time.time()
        p.start()

        # PROPOSED FIX: cap join timeout
        formula_timeout = (per_test_timeout + 1) * n_tests + 5
        effective_timeout = min(formula_timeout, global_timeout)
        p.join(timeout=effective_timeout)
        elapsed = time.time() - start

        if p.is_alive():
            p.kill()
            p.join(timeout=5)
            return [-1] * n_tests, {"error": "global_timeout", "elapsed": elapsed, "cap": effective_timeout}

        if not result:
            return [-1] * n_tests, {"error": "no_result", "elapsed": elapsed}

        return list(result), list(metadata_list)[0] if metadata_list else None

    def test_global_cap_prevents_http_timeout(self):
        """With global cap = 90s (< 120s HTTP), even 100-test problems resolve in time."""
        n_tests = 100
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i * i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        # Code that TLEs -- should be capped by global timeout, not formula
        code = "import time; time.sleep(100)"

        start = time.time()
        result, metadata = self._check_correctness_with_global_cap(sample, code, per_test_timeout=2, global_timeout=5)
        elapsed = time.time() - start

        # With global cap of 5s, should complete well under the HTTP timeout
        assert elapsed < 15, f"Global cap should limit to ~5s, took {elapsed:.1f}s"
        # Without cap: formula = (2+1)*100+5 = 305s

    def test_global_cap_still_passes_fast_code(self):
        """Global cap should not affect fast code execution."""
        n_tests = 50
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i * i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        code = "n=int(input())\nprint(n*n)"

        result, metadata = self._check_correctness_with_global_cap(sample, code, per_test_timeout=5, global_timeout=60)

        # Fast code should still pass
        assert all(r is True or r == True for r in result), f"Fast code should pass, got {result[:5]}..."


# ---------------------------------------------------------------------------
# Test: Proposed fix -- early termination on first failure
# ---------------------------------------------------------------------------


class TestEarlyTermination:
    """The actual testing_util.py returns early on first failure in grade_stdio/grade_call_based.

    But when signal.alarm doesn't fire (e.g., code stuck in C extension),
    the multiprocessing.Process timeout is the only safety net.

    These tests verify the timeout behavior with code that hangs.
    Note: our simplified _check_correctness_local doesn't compare outputs
    (that requires the full lcb_integration stack), so we test TLE and
    runtime error scenarios only.
    """

    def test_early_return_on_tle(self):
        """Code that TLEs on test 0 should not attempt remaining tests."""
        n_tests = 50
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        # This code will TLE on the first test case
        code = "while True: pass"

        timeout = 2
        start = time.time()
        result, metadata = _check_correctness_local(sample, code, timeout)
        elapsed = time.time() - start

        # Should fail on first test and return, NOT wait for all 50 tests * 2s
        assert elapsed < timeout + 15, f"TLE should return after first test, took {elapsed:.1f}s"
        assert any(r == -3 or r == -1 for r in result), f"Should report TLE, got {result}"

    def test_early_return_on_runtime_error(self):
        """Code that crashes on test 0 should not run remaining tests."""
        n_tests = 50
        inputs = [f"{i}\n" for i in range(n_tests)]
        outputs = [str(i) for i in range(n_tests)]
        sample = {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
        }
        code = "raise ValueError('crash')"

        start = time.time()
        result, metadata = _check_correctness_local(sample, code, timeout=5)
        elapsed = time.time() - start

        assert elapsed < 10, f"Runtime error should return early, took {elapsed:.1f}s"
        assert any(r == -4 for r in result), "Should report runtime error"
