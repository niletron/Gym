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
"""Tests for the multi-box reward-hacking mitigation.

Covers the pure helper `_keep_only_last_boxed` and its integration via
`_verify_in_worker` (same worker-side call path the subprocess pool uses,
but run inline in the pytest process for speed and determinism).
"""
import os

import pytest

from resources_servers.math_with_judge.app import (
    _keep_only_last_boxed,
    _verify_in_worker,
)


# ──────────────────────────────────────────────────────────────────────
# Unit tests on the pure helper
# ──────────────────────────────────────────────────────────────────────


class TestKeepOnlyLastBoxed:
    def test_empty_string(self):
        assert _keep_only_last_boxed("") == ""

    def test_no_boxes(self):
        s = "just some text with no boxes at all"
        assert _keep_only_last_boxed(s) == s

    def test_single_box(self):
        s = r"the answer is \boxed{42}"
        assert _keep_only_last_boxed(s) == s

    def test_single_box_nested_braces(self):
        s = r"the answer is \boxed{\frac{1}{2}}"
        assert _keep_only_last_boxed(s) == s

    def test_two_boxes_simple(self):
        s = r"stuff \boxed{1} more \boxed{2}"
        out = _keep_only_last_boxed(s)
        # Earlier box removed, last one kept, surrounding text preserved.
        assert r"\boxed{1}" not in out
        assert r"\boxed{2}" in out
        assert out.count(r"\boxed{") == 1
        assert "stuff" in out
        assert "more" in out

    def test_nested_braces_keeps_last_intact(self):
        """Regression test: naive regex would truncate \boxed{\frac{1}{2}}
        at the first '}' and leave a dangling '}'. The depth scan must
        preserve the full expression."""
        s = r"\boxed{1} \boxed{\frac{1}{2}}"
        out = _keep_only_last_boxed(s)
        assert r"\boxed{\frac{1}{2}}" in out
        assert r"\boxed{1}" not in out
        # Exactly one \boxed remains.
        assert out.count(r"\boxed{") == 1

    def test_many_boxes_only_last_kept(self):
        s = r"\boxed{a}\boxed{b}\boxed{c}\boxed{d}"
        out = _keep_only_last_boxed(s)
        assert out.count(r"\boxed{") == 1
        assert r"\boxed{d}" in out
        for lost in ("a", "b", "c"):
            assert f"\\boxed{{{lost}}}" not in out

    def test_unclosed_box_dropped_silently(self):
        r"""Unclosed \boxed{ can't be safely stripped (no end), so we
        leave it in place. The well-formed earlier box still gets
        stripped because we only found one complete span — but that
        single span means len(spans) == 1, so we return unchanged.

        Concretely: input "\boxed{1} \boxed{unclosed" has exactly one
        complete box (\boxed{1}) and one unclosed fragment. The helper
        keeps both (returns unchanged) because after the depth scan
        we only have 1 full span. This is the "graceful" choice — we
        never accidentally truncate text that contains unbalanced braces.
        """
        s = r"\boxed{1} \boxed{unclosed"
        out = _keep_only_last_boxed(s)
        # Not crashed. Exactly the input returned (single closed span).
        assert out == s

    def test_multiline_box(self):
        s = "\\boxed{1}\n\n\\boxed{multi\nline\nanswer}"
        out = _keep_only_last_boxed(s)
        assert r"\boxed{1}" not in out
        assert "\\boxed{multi\nline\nanswer}" in out
        assert out.count(r"\boxed{") == 1

    def test_deeply_nested(self):
        s = r"\boxed{1} \boxed{\frac{\sqrt{a+b}}{c^{2}}}"
        out = _keep_only_last_boxed(s)
        assert r"\boxed{\frac{\sqrt{a+b}}{c^{2}}}" in out
        assert r"\boxed{1}" not in out

    def test_surrounding_text_preserved(self):
        s = "Let me think. \\boxed{wrong} Hmm actually \\boxed{42}."
        out = _keep_only_last_boxed(s)
        assert r"\boxed{wrong}" not in out
        assert r"\boxed{42}" in out
        assert "Let me think." in out
        assert "Hmm actually" in out
        # Final period outside the box preserved.
        assert out.endswith(".")

    def test_three_boxes_middle_stripped_too(self):
        s = r"\boxed{1} and \boxed{2} finally \boxed{3}"
        out = _keep_only_last_boxed(s)
        assert out.count(r"\boxed{") == 1
        assert r"\boxed{3}" in out

    def test_escaped_brace_inside_box(self):
        r"""Escaped braces (\{, \}) shouldn't confuse the depth counter."""
        s = r"\boxed{a \{ b \} c} \boxed{final}"
        out = _keep_only_last_boxed(s)
        # First box (with escaped braces) stripped; last kept.
        assert r"\boxed{final}" in out
        assert r"\boxed{a \{ b \} c}" not in out


# ──────────────────────────────────────────────────────────────────────
# Integration tests via _verify_in_worker
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ensure_flag_on(monkeypatch):
    """Make sure the patch is active by default for these tests
    (independent of the caller's environment)."""
    monkeypatch.delenv("MATH_KEEP_ONLY_LAST_BOX", raising=False)
    yield


class TestVerifyInWorkerLastBoxOnly:
    def test_single_box_correct(self):
        reward, _extracted = _verify_in_worker("42", r"The answer is \boxed{42}.")
        assert reward == 1.0

    def test_single_box_wrong(self):
        reward, _extracted = _verify_in_worker("42", r"The answer is \boxed{7}.")
        assert reward == 0.0

    def test_multi_box_correct_last_wrong_earlier(self):
        """Legitimate case — the model worked through wrong guesses,
        settled on the correct final boxed answer. Must still reward."""
        resp = r"Try \boxed{1}. No wait, \boxed{7}. Actually: \boxed{42}."
        reward, _ = _verify_in_worker("42", resp)
        assert reward == 1.0

    def test_multi_box_wrong_last_correct_earlier_EXPLOIT_BLOCKED(self):
        """THE EXPLOIT: correct answer shows up earlier, wrong one last.
        With the old set-merge behavior this would match on the correct
        extraction and reward=1.0. With the fix, only the LAST box is
        considered — so reward MUST be 0.0."""
        resp = r"I think \boxed{42}. Actually, \boxed{7}."
        reward, _ = _verify_in_worker("42", resp)
        assert reward == 0.0

    def test_flooded_boxes_correct_last(self):
        resp = (
            r"\boxed{1} \boxed{2} \boxed{3} \boxed{4} \boxed{5} "
            r"\boxed{6} \boxed{7} \boxed{8} \boxed{9} \boxed{42}"
        )
        reward, _ = _verify_in_worker("42", resp)
        assert reward == 1.0

    def test_flooded_boxes_correct_only_first(self):
        resp = (
            r"\boxed{42} \boxed{2} \boxed{3} \boxed{4} \boxed{5} "
            r"\boxed{6} \boxed{7} \boxed{8} \boxed{9} \boxed{10}"
        )
        reward, _ = _verify_in_worker("42", resp)
        # Last box (\boxed{10}) is evaluated; it's wrong.
        assert reward == 0.0

    def test_nested_box_correct(self):
        resp = r"After simplification, \boxed{\frac{1}{2}}."
        reward, _ = _verify_in_worker(r"\frac{1}{2}", resp)
        assert reward == 1.0

    def test_nested_box_exploit_blocked(self):
        """Multi-box with nested content in last box; previous contains
        the correct answer. Must return 0."""
        resp = r"\boxed{\frac{1}{2}} and then \boxed{\frac{3}{4}}."
        reward, _ = _verify_in_worker(r"\frac{1}{2}", resp)
        assert reward == 0.0


class TestFeatureFlagOptOut:
    """The flag must truly gate the preprocessing, so ops can A/B test or
    roll back without a code change. We verify by inspecting what gets
    handed to math_verify: with flag=0, the full multi-box string passes
    through untouched; with flag=1/unset, only the last box survives."""

    def test_flag_zero_disables_preprocessing(self, monkeypatch):
        """With MATH_KEEP_ONLY_LAST_BOX=0, the helper is not called —
        the second extracted prediction string should show the set-merge
        of all boxes (e.g. '1,2,42'), confirming all boxes reached
        math_verify unmodified."""
        monkeypatch.setenv("MATH_KEEP_ONLY_LAST_BOX", "0")
        resp = r"\boxed{1} \boxed{2} \boxed{42}"
        reward, extracted = _verify_in_worker("99", resp)
        # Gold is 99, none match — reward=0 either way. What matters is
        # that extracted contains evidence of ALL three boxes having been
        # parsed (set-merge by math_verify).
        assert reward == 0.0
        # math_verify returns the string form; with all three boxes
        # present it contains "1", "2", and "42".
        assert extracted is not None
        for digit in ("1", "2", "42"):
            assert digit in extracted, (
                f"Expected {digit!r} in extracted={extracted!r}; flag=0 should "
                f"pass all boxes to math_verify"
            )

    def test_flag_one_enables_preprocessing(self, monkeypatch):
        """With the flag on (explicitly '1'), only the last box is passed
        through. The extracted prediction must reflect only the final
        box's content, not a set-merge of all boxes."""
        monkeypatch.setenv("MATH_KEEP_ONLY_LAST_BOX", "1")
        resp = r"\boxed{1} \boxed{2} \boxed{42}"
        reward, extracted = _verify_in_worker("99", resp)
        assert reward == 0.0
        # Earlier boxes should NOT show up in extracted: the helper
        # stripped them, so math_verify only saw \boxed{42}.
        assert extracted is not None
        assert "1" not in extracted or extracted == "42"
        assert "2" not in extracted or extracted == "42"
        assert "42" in extracted

    def test_flag_unset_defaults_to_on(self, monkeypatch):
        """No env var set → the fix is on (opt-out, not opt-in)."""
        monkeypatch.delenv("MATH_KEEP_ONLY_LAST_BOX", raising=False)
        resp = r"\boxed{1} \boxed{2} \boxed{42}"
        _reward, extracted = _verify_in_worker("42", resp)
        # With the fix on, only \boxed{42} reaches math_verify → extracted == "42".
        assert extracted == "42"


# ──────────────────────────────────────────────────────────────────────
# Direct assertion that preprocessing is idempotent
# ──────────────────────────────────────────────────────────────────────


class TestIdempotence:
    def test_applying_twice_equals_once(self):
        s = r"\boxed{1} \boxed{2} \boxed{\frac{1}{2}}"
        once = _keep_only_last_boxed(s)
        twice = _keep_only_last_boxed(once)
        assert once == twice

    def test_idempotent_on_clean_input(self):
        s = r"Just one: \boxed{x^2 + 1}"
        assert _keep_only_last_boxed(_keep_only_last_boxed(s)) == s
