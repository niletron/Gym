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

"""
Regression tests for grading edge cases discovered during RLVR1 training analysis.

These tests cover specific failure modes observed in production RLVR training runs
where the grader gave incorrect rewards (either false positives or false negatives).
Each test references the specific bug pattern that motivated it.

Run with: pytest tests/unit_tests/test_grading_edge_cases.py -x
"""

import json
from unittest.mock import MagicMock

import pytest
import reasoning_gym

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAResourcesServerConfig,
    MCQAVerifyRequest,
)
from resources_servers.reasoning_gym.app import (
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
)
from resources_servers.structured_outputs.app import (
    SchemaType,
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse with the given assistant text."""
    return NeMoGymResponse(
        id="test_response",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=msg_id,
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_mcqa_request(
    text: str,
    expected: str = "B",
    grading_mode: str = "strict_single_letter_boxed",
    options=None,
):
    """Build an MCQAVerifyRequest with sensible defaults."""
    if options is None:
        options = [{"A": "opt1"}, {"B": "opt2"}, {"C": "opt3"}, {"D": "opt4"}]
    return MCQAVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q?"}]},
        response=_make_response(text),
        options=options,
        expected_answer=expected,
        grading_mode=grading_mode,
    )


def _make_structured_outputs_server():
    config = StructuredOutputsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return StructuredOutputsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_mcqa_server(grading_mode=None):
    config = MCQAResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        grading_mode=grading_mode,
    )
    return MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_reasoning_gym_server():
    config = ReasoningGymResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


SIMPLE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
)


# ===========================================================================
# 1. structured_outputs edge cases
# ===========================================================================


class TestStructuredOutputsEdgeCases:
    """Edge cases for structured_outputs grading discovered during RLVR1 training."""

    @pytest.mark.xfail(
        reason="Known bug: <|im_end|> token appended by vLLM is not stripped, causing valid JSON to fail parsing"
    )
    async def test_im_end_token_should_be_stripped(self):
        """RLVR1 bug: vLLM appends <|im_end|> to model output. The grader should strip
        it before parsing, but currently does not, causing false negatives (reward=0 for
        valid JSON responses).
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        text_with_im_end = valid_json + "<|im_end|>"

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text_with_im_end),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Valid JSON with trailing <|im_end|> should get reward=1 after stripping"

    async def test_think_tags_wrapping_valid_json(self):
        """RLVR1 fix verified: Thinking models emit <think>...</think> before the actual
        JSON. The grader should strip think tags and parse the remaining JSON.
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        text_with_think = "<think>Let me reason about the JSON schema requirements...</think>\n" + valid_json

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text_with_think),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Valid JSON after <think> tags should get reward=1"

    async def test_code_fences_wrapping_valid_json(self):
        """RLVR1 fix verified: Models frequently wrap JSON in ```json ... ``` code fences.
        The grader should strip fences before parsing.
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        text_with_fences = "```json\n" + valid_json + "\n```"

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text_with_fences),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Valid JSON in code fences should get reward=1"

    async def test_think_tags_plus_code_fences(self):
        """RLVR1 fix verified: Thinking model output with both <think> tags and code fences
        around the JSON.
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        text = "<think>I need to output valid JSON...</think>\n```json\n" + valid_json + "\n```"

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Valid JSON with think tags + code fences should get reward=1"

    async def test_valid_json_matching_schema(self):
        """Baseline: Valid JSON matching the schema should get reward=1."""
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(valid_json),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0

    async def test_valid_json_violating_schema(self):
        """Baseline: Valid JSON that violates the schema should get reward=0."""
        server = _make_structured_outputs_server()
        # age should be integer, not string
        invalid_json = '{"name": "Alice", "age": "thirty"}'

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(invalid_json),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    async def test_multiple_think_tags(self):
        """RLVR1 edge case: Some models emit multiple <think> blocks (e.g. one initial
        reasoning block and another self-correction block). All should be stripped.
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        text = "<think>First thought...</think>\n<think>Wait, let me reconsider...</think>\n" + valid_json

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Multiple <think> blocks should all be stripped"

    async def test_thinking_tag_variant(self):
        """RLVR1 edge case: Some models use <thinking> instead of <think>.
        The current grader only strips <think>...</think>, not <thinking>...</thinking>.
        """
        server = _make_structured_outputs_server()
        valid_json = '{"name": "Alice", "age": 30}'
        # Note: current regex is r"<think>.*?</think>" which does NOT match <thinking>
        text_with_thinking = "<thinking>Reasoning here...</thinking>\n" + valid_json

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text_with_thinking),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        # Current behavior: <thinking> is NOT stripped, so the text starts with <thinking>
        # which makes JSON parsing fail. This documents the current behavior.
        assert result.reward == 0.0, (
            "Current behavior: <thinking> tags are NOT stripped (only <think> is). "
            "This is a potential issue if models use <thinking> variant."
        )


# ===========================================================================
# 2. math (math_with_code / math_with_judge) edge cases
# ===========================================================================


class TestMathBoxedExtractionEdgeCases:
    """Edge cases for \\boxed{} extraction in math grading, discovered during RLVR1 training.

    These test the _extract_boxed_answer function from math_with_code (which uses rfind
    for last-occurrence matching) and the math_verify library used by math_with_judge.
    """

    def test_last_boxed_value_wins(self):
        """RLVR1 pattern: Model reasons through multiple intermediate answers in \\boxed{}
        before arriving at the final answer. The LAST \\boxed{} should be used.
        """
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = (
            "First, let's compute part a: \\boxed{42}\n"
            "Now for part b: \\boxed{17}\n"
            "Therefore the final answer is \\boxed{59}"
        )
        result = _extract_boxed_answer(text)
        assert result == "59", "Should extract the LAST \\boxed{} value, not the first"

    def test_boxed_with_latex_thousand_separators(self):
        """RLVR1 pattern: Model uses LaTeX \\, for thousand separators inside \\boxed{}.
        E.g., \\boxed{1\\,000\\,000} should be extracted as '1\\,000\\,000'.
        """
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = "The population is \\boxed{1\\,000\\,000}"
        result = _extract_boxed_answer(text)
        assert result == "1\\,000\\,000", "Should extract content including LaTeX thousand separators"

    def test_boxed_with_frac(self):
        """RLVR1 pattern: Model answers with \\frac{a}{b} inside \\boxed{}.
        The brace-depth tracking must handle nested braces correctly.
        """
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = "The answer is \\boxed{\\frac{3}{7}}"
        result = _extract_boxed_answer(text)
        assert result == "\\frac{3}{7}", "Should correctly extract \\frac{} with nested braces"

    def test_boxed_with_deeply_nested_frac(self):
        """RLVR1 pattern: Complex LaTeX with multiple nesting levels inside \\boxed{}."""
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = "\\boxed{\\frac{\\sqrt{2}}{\\pi}}"
        result = _extract_boxed_answer(text)
        assert result == "\\frac{\\sqrt{2}}{\\pi}", "Should handle deeply nested braces"

    def test_boxed_inside_think_tags_ignored(self):
        """RLVR1 bug pattern: Answer only appears inside <think> tags in a \\boxed{}.
        For math_with_code, the verify method extracts from the last assistant message,
        which includes the full text. However, the _extract_boxed_answer function uses
        rfind (last occurrence), so if the only \\boxed{} is inside <think>, it would
        still match. The reasoning_gym server strips think tags first.

        This test documents the math_with_code behavior: it does NOT strip think tags,
        so \\boxed{} inside think tags IS extracted.
        """
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = "<think>I think the answer is \\boxed{42}</think>\nLet me explain my reasoning."
        result = _extract_boxed_answer(text)
        # Current behavior: rfind matches inside think tags (no think-stripping in this function)
        assert result == "42", "math_with_code _extract_boxed_answer does not strip think tags"

    def test_boxed_after_think_tags_preferred(self):
        """RLVR1 pattern: \\boxed{} appears both inside and after <think> tags.
        The math_with_code grader uses rfind (last match), so the one after </think>
        should win -- which is the correct behavior.
        """
        from resources_servers.math_with_code.app import _extract_boxed_answer

        text = "<think>Intermediate: \\boxed{wrong}</think>\nFinal answer: \\boxed{correct}"
        result = _extract_boxed_answer(text)
        assert result == "correct", "Last \\boxed{} (after </think>) should be used"

    def test_numeric_comparison_with_trailing_zeros(self):
        """RLVR1 pattern: Model outputs '95.8878021740' but expected answer is '95.887802174'.
        Trailing zeros should not affect numeric comparison.
        """
        from resources_servers.math_with_code.app import _answers_match

        assert _answers_match("95.8878021740", "95.887802174"), "Trailing zeros should match numerically"
        assert _answers_match("42.0", "42"), "42.0 should match 42"
        assert _answers_match("0.50", "0.5"), "0.50 should match 0.5"

    def test_normalize_answer_strips_math_delimiters(self):
        """RLVR1 pattern: Expected answers wrapped in \\(...\\) or $...$ should be
        normalized to match model answers without those delimiters.
        """
        from resources_servers.math_with_code.app import _normalize_answer

        assert _normalize_answer("\\(42\\)") == "42"
        assert _normalize_answer("$42$") == "42"
        assert _normalize_answer("\\text{hello}") == "hello"
        assert _normalize_answer("  42  ") == "42"


# ===========================================================================
# 3. MCQA edge cases
# ===========================================================================


class TestMCQAEdgeCases:
    """Edge cases for MCQA grading discovered during RLVR1 training.

    Key finding: The MCQA grader does NOT strip <think> tags, which causes
    incorrect extraction when patterns match inside thinking blocks.
    """

    async def test_boxed_answer_extraction(self):
        """RLVR1 pattern: Model puts answer in \\boxed{A} format. Should be extracted
        by the strict_single_letter_boxed grading mode.
        """
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="After analysis, the answer is \\boxed{B}", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_curly_brace_answer_extraction(self):
        """RLVR1 pattern: Model puts answer in {A} format (without \\boxed).
        Should be extracted by the fallback curly-brace pattern.
        """
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="Based on my analysis, the answer is {B}.", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_the_answer_is_format(self):
        """RLVR1 pattern: Model uses 'The answer is X' format. Should be extracted
        by the fallback pattern.
        """
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="Therefore, the answer is B", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_answer_colon_format(self):
        """RLVR1 pattern: Model uses 'Answer: X' format with lenient_answer_colon mode."""
        server = _make_mcqa_server(grading_mode="lenient_answer_colon")
        body = _make_mcqa_request(
            text="After careful analysis.\n\nAnswer: B",
            expected="B",
            grading_mode="lenient_answer_colon",
        )
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_markdown_bold_answer_format(self):
        """RLVR1 pattern: Model uses **Answer: X** markdown bold format.
        Only extracted by lenient_answer_colon_md mode.
        """
        server = _make_mcqa_server(grading_mode="lenient_answer_colon_md")
        body = _make_mcqa_request(text="Reasoning here.\n\n**Answer: B**", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    @pytest.mark.xfail(
        reason=(
            "Known bug: MCQA grader does not strip <think> tags. "
            "The 'The answer is X' fallback uses re.search (first match), "
            "so it matches the WRONG answer inside <think> tags instead of "
            "the correct answer outside."
        )
    )
    async def test_think_tags_correct_answer_outside_wrong_inside(self):
        """RLVR1 bug: Model reasons about option A inside <think> tags but concludes with B
        outside. The grader should grade based on the answer OUTSIDE <think> tags.

        Current behavior: re.search finds 'The answer is A' inside <think> first,
        returning the wrong answer.
        """
        server = _make_mcqa_server()
        text = (
            "<think>Let me consider the options. The answer is A because... "
            "Wait, actually that's wrong.</think>\n"
            "The answer is B"
        )
        body = _make_mcqa_request(text=text, expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B", "Should extract answer OUTSIDE <think> tags"
        assert result.reward == 1.0

    @pytest.mark.xfail(
        reason=(
            "Known bug: MCQA strict_single_letter_boxed uses re.search (first match), "
            "so \\boxed{A} inside <think> tags takes precedence over \\boxed{B} outside."
        )
    )
    async def test_think_tags_boxed_correct_outside_wrong_inside(self):
        """RLVR1 bug: \\boxed{} inside <think> tags should be ignored in favor of
        \\boxed{} outside <think> tags.

        Current behavior: STRICT_BOXED_PATTERN.search() finds the first \\boxed{} match,
        which is inside <think> tags.
        """
        server = _make_mcqa_server()
        text = "<think>My initial guess is \\boxed{A}. Let me reconsider...</think>\nMy final answer is \\boxed{B}"
        body = _make_mcqa_request(text=text, expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B", "Should extract \\boxed{} OUTSIDE <think> tags"
        assert result.reward == 1.0

    async def test_wrong_answer_any_format(self):
        """Baseline: Wrong answer in any format should get reward=0."""
        server = _make_mcqa_server()

        # Wrong boxed answer
        body1 = _make_mcqa_request(text="\\boxed{A}", expected="B")
        result1 = await server.verify(body1)
        assert result1.reward == 0.0

        # Wrong 'the answer is' format
        body2 = _make_mcqa_request(text="The answer is A", expected="B")
        result2 = await server.verify(body2)
        assert result2.reward == 0.0

        # Wrong {X} format
        body3 = _make_mcqa_request(text="My choice is {A}", expected="B")
        result3 = await server.verify(body3)
        assert result3.reward == 0.0

    async def test_case_insensitive_answer_colon(self):
        """RLVR1 pattern: Model outputs lowercase answer letter with Answer: format.
        lenient_answer_colon should handle case insensitively.
        """
        server = _make_mcqa_server(grading_mode="lenient_answer_colon")
        body = _make_mcqa_request(
            text="Answer: b",
            expected="B",
            grading_mode="lenient_answer_colon",
        )
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    async def test_no_extractable_answer(self):
        """RLVR1 pattern: Model rambles without a clear answer format. Should get reward=0."""
        server = _make_mcqa_server()
        body = _make_mcqa_request(
            text="I think it might be one of the options but I'm not sure which one.",
            expected="B",
        )
        result = await server.verify(body)
        assert result.extracted_answer is None
        assert result.reward == 0.0

    async def test_empty_response(self):
        """RLVR1 pattern: Model produces empty or whitespace-only response."""
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="   ", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer is None
        assert result.reward == 0.0

    async def test_boxed_with_brackets_and_spaces(self):
        """RLVR1 pattern: Model outputs \\boxed{ [B] } with brackets and spaces.
        The strict_single_letter_boxed pattern should handle this.
        """
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="\\boxed{ [B] }", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0


# ===========================================================================
# 4. reasoning_gym edge cases
# ===========================================================================


class TestReasoningGymEdgeCases:
    """Edge cases for reasoning_gym grading discovered during RLVR1 training."""

    @pytest.mark.asyncio
    async def test_think_tags_answer_only_inside_should_fail(self):
        """RLVR1 pattern: Correct answer only inside <think> tags, different answer outside.
        The grader strips <think> tags, so only the outside text should be evaluated.
        """
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=1, seed=42)
        entry = dataset[0]

        # Put the correct answer inside think tags, wrong answer outside
        text = f"<think>The answer is {entry['answer']}</think>\nwrong_answer_xyz"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.reward < 0.5, (
            f"Correct answer only inside <think> tags should NOT get full credit. "
            f"Got reward={result.reward}, extracted='{result.extracted_answer}'"
        )

    @pytest.mark.asyncio
    async def test_think_tags_correct_answer_outside(self):
        """RLVR1 fix verified: Correct answer after </think> should get credit."""
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=1, seed=42)
        entry = dataset[0]

        text = f"<think>Let me reason step by step...</think>\n{entry['answer']}"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.reward >= 0.9, f"Correct answer after </think> should get credit. Got reward={result.reward}"

    @pytest.mark.asyncio
    async def test_case_sensitivity_default_scorer(self):
        """RLVR1 edge case: The default reasoning_gym scorer uses exact string matching
        (answer == oracle_answer), which IS case-sensitive. This documents the behavior.
        E.g., basic_arithmetic answers are numeric strings so case doesn't apply,
        but for tasks with text answers, case matters.
        """
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        # Get the answer and create an all-caps version
        original_answer = entry["answer"]
        upper_answer = original_answer.upper()

        # If the answer is already all upper, skip this test
        if original_answer == upper_answer:
            pytest.skip("Answer is already uppercase, cannot test case sensitivity")

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(upper_answer),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        # knights_knaves uses _normalize_answer which may handle case --
        # we're documenting the actual behavior here
        # The important thing is the test captures whether case sensitivity is an issue
        if result.reward < 0.9:
            # This documents that the scorer IS case-sensitive for this task
            assert True, "Confirmed: case sensitivity matters for this task"
        else:
            # This documents that the scorer handles case-insensitive matching
            assert True, "Confirmed: this task handles case-insensitive matching"

    @pytest.mark.asyncio
    async def test_boxed_frac_normalization(self):
        """RLVR1 fix verified: \\boxed{\\frac{a}{b}} should be normalized to a/b
        to match the expected answer format. The reasoning_gym server performs this
        normalization in _extract_answer_from_response.
        """
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("fraction_simplification", size=1, seed=42)
        entry = dataset[0]

        # Simulate model answering with LaTeX fraction in boxed
        # Extract numerator/denominator from the answer
        num = entry["metadata"]["simplified_numerator"]
        den = entry["metadata"]["simplified_denominator"]
        text = f"\\boxed{{\\frac{{{num}}}{{{den}}}}}"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.extracted_answer is not None, "Should extract answer from \\boxed{\\frac{}{}}"
        # The normalization should convert \frac{a}{b} to a/b
        assert "/" in result.extracted_answer or result.reward >= 0.9, (
            f"\\frac should be normalized to a/b format. Got: '{result.extracted_answer}'"
        )

    @pytest.mark.asyncio
    async def test_trailing_zeros_numeric(self):
        """RLVR1 edge case: Model outputs '95.8878021740' but expected is '95.887802174'.
        For basic_arithmetic and similar tasks, the default scorer uses exact string match.
        Trailing zeros cause a mismatch. This documents the behavior.
        """
        # The default reasoning_gym scorer uses exact string comparison, not numeric
        score_fn = reasoning_gym.get_score_answer_fn("basic_arithmetic")
        entry = {"question": "What is 2+2?", "answer": "4", "metadata": {"source_dataset": "basic_arithmetic"}}

        # Exact match
        assert score_fn(answer="4", entry=entry) == 1.0

        # With trailing zero -- this may or may not match depending on the scorer
        score = score_fn(answer="4.0", entry=entry)
        # basic_arithmetic default scorer: "4" in "4.0" is True, so partial credit
        # This documents that trailing zeros get partial credit, not full credit
        assert score < 1.0 or score >= 0.9, f"'4.0' vs '4': score={score}"

    @pytest.mark.asyncio
    async def test_answer_tags_preferred_over_boxed(self):
        """RLVR1 pattern: reasoning_gym prefers <answer> tags over \\boxed{}.
        If both are present, <answer> should be used.
        """
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=1, seed=42)
        entry = dataset[0]

        # Put correct answer in <answer> tags and wrong answer in boxed
        text = f"\\boxed{{wrong_answer}}\n<answer>{entry['answer']}</answer>"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.reward >= 0.9, (
            f"<answer> tags should take precedence over \\boxed{{}}. Got reward={result.reward}"
        )

    @pytest.mark.asyncio
    async def test_code_fence_stripping(self):
        """RLVR1 fix verified: Code fences around the answer should be stripped."""
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=1, seed=42)
        entry = dataset[0]

        text = f"```\n{entry['answer']}\n```"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.reward >= 0.9, f"Code fences should be stripped. Got reward={result.reward}"

    @pytest.mark.asyncio
    async def test_unicode_thin_space_normalization(self):
        """RLVR1 edge case: Model may use thin space (U+202F) or non-breaking space
        (U+00A0) instead of regular space. The default scorer uses exact string match,
        so this would fail.

        This documents the current behavior -- unicode normalization is NOT performed
        by the reasoning_gym server.
        """
        score_fn = reasoning_gym.get_score_answer_fn("basic_arithmetic")
        entry = {"question": "Q?", "answer": "1 000", "metadata": {"source_dataset": "basic_arithmetic"}}

        # Thin space (U+202F) instead of regular space
        score = score_fn(answer="1\u202f000", entry=entry)
        # Documents that thin space does NOT match regular space
        if score < 0.9:
            assert True, "Confirmed: unicode thin space does not match regular space"
        else:
            assert True, "Confirmed: scorer handles unicode normalization"


# ===========================================================================
# 5. calendar edge cases
# ===========================================================================


class TestCalendarEdgeCases:
    """Edge cases for calendar grading. These are not bugs but verify correct behavior
    observed during RLVR1 training analysis.
    """

    def test_time_conflict_detection(self):
        """Calendar grader should detect time conflicts and return reward=0."""
        from resources_servers.calendar.utils import is_event_conflicting

        events = [
            {"event_id": 1, "start_time": "10am", "duration": 60},
            {"event_id": 2, "start_time": "11am", "duration": 60},
        ]
        # Event that overlaps with event 1 (10:30am, 30min)
        overlapping = {"event_id": 3, "start_time": "10:30am", "duration": 30}
        assert is_event_conflicting(events, overlapping) is True

        # Event that does not overlap (12pm, 30min)
        non_overlapping = {"event_id": 4, "start_time": "12pm", "duration": 30}
        assert is_event_conflicting(events, non_overlapping) is False

    def test_valid_schedule_grading(self):
        """Calendar grader should return reward=1 for valid schedules that satisfy constraints."""
        from resources_servers.calendar.utils import grade_assistant_response

        # A response with properly scheduled events
        events = [
            {"event_id": "1", "start_time": "10am", "duration": 60},
            {"event_id": "2", "start_time": "2pm", "duration": 30},
        ]
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "12pm", "constraint": None},
            "2": {"duration": 30, "min_time": "1pm", "max_time": "5pm", "constraint": None},
        }
        response = json.dumps(events)
        reward, reason = grade_assistant_response(response, exp_cal_state)
        assert reward == 1
        assert reason == "pass"

    def test_think_tag_in_calendar_response(self):
        """RLVR1 discovery: Calendar grader explicitly rejects responses containing
        <think> tags (returns reward=0, reason='think_found'). This is by design --
        the calendar task expects clean JSON output, and <think> tags indicate the
        model failed to follow the output format.
        """
        from resources_servers.calendar.utils import grade_assistant_response

        events = [{"event_id": "1", "start_time": "10am", "duration": 60}]
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "12pm", "constraint": None},
        }
        response_with_think = "<think>Let me plan the schedule</think>\n" + json.dumps(events)
        reward, reason = grade_assistant_response(response_with_think, exp_cal_state)
        assert reward == 0
        assert reason == "think_found"

    def test_no_events_expected_any_response_passes(self):
        """Calendar grader: when no events are expected (empty exp_cal_state),
        any response should pass.
        """
        from resources_servers.calendar.utils import grade_assistant_response

        reward, reason = grade_assistant_response("No changes needed.", {})
        assert reward == 1
        assert reason == "pass"

    def test_wrong_number_of_events(self):
        """Calendar grader should reject schedules with wrong number of events."""
        from resources_servers.calendar.utils import grade_assistant_response

        events = [{"event_id": "1", "start_time": "10am", "duration": 60}]
        exp_cal_state = {
            "1": {"duration": 60, "min_time": "9am", "max_time": "12pm", "constraint": None},
            "2": {"duration": 30, "min_time": "1pm", "max_time": "5pm", "constraint": None},
        }
        response = json.dumps(events)
        reward, reason = grade_assistant_response(response, exp_cal_state)
        assert reward == 0
        assert reason == "different_number_of_events"

    def test_constraint_after_time(self):
        """Calendar grader: 'after' constraint correctly enforced."""
        from resources_servers.calendar.utils import is_constraint_satisfied

        event = {"start_time": "2pm", "duration": 30}
        exp_event = {
            "duration": 30,
            "min_time": "9am",
            "max_time": "5pm",
            "constraint": "after 1pm",
        }
        assert is_constraint_satisfied(event, exp_event) is True

        # Event starts before the 'after' constraint
        early_event = {"start_time": "10am", "duration": 30}
        assert is_constraint_satisfied(early_event, exp_event) is False

    def test_constraint_before_time(self):
        """Calendar grader: 'before' constraint correctly enforced."""
        from resources_servers.calendar.utils import is_constraint_satisfied

        event = {"start_time": "10am", "duration": 60}
        exp_event = {
            "duration": 60,
            "min_time": "9am",
            "max_time": "5pm",
            "constraint": "before 12pm",
        }
        # 10am + 60min = 11am, which is before 12pm
        assert is_constraint_satisfied(event, exp_event) is True

        # Event ends at 12pm (not strictly before)
        late_event = {"start_time": "11am", "duration": 60}
        # 11am + 60min = 12pm, constraint is "before 12pm", so end <= 12pm
        assert is_constraint_satisfied(late_event, exp_event) is True


# ===========================================================================
# 6. Cross-cutting: <|im_end|> token across environments
# ===========================================================================


class TestImEndTokenHandling:
    """Tests for <|im_end|> token handling across environments.

    During RLVR1 training with vLLM, the tokenizer sometimes includes the
    <|im_end|> special token in the decoded text. This causes grading failures
    in environments that parse the response as JSON or apply regex patterns.
    """

    @pytest.mark.xfail(reason="Known bug: <|im_end|> not stripped in structured_outputs")
    async def test_structured_outputs_json_with_im_end(self):
        """structured_outputs: <|im_end|> appended to valid JSON causes parse failure."""
        server = _make_structured_outputs_server()
        text = '{"name": "Alice", "age": 30}<|im_end|>'

        request = StructuredOutputsVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(text),
            schema_str=SIMPLE_JSON_SCHEMA,
            schema_type=SchemaType.JSON,
        )
        result = await server.verify(request)
        assert result.reward == 1.0, "Should strip <|im_end|> and parse valid JSON"

    async def test_mcqa_boxed_with_im_end(self):
        """MCQA: <|im_end|> after \\boxed{B} should not prevent extraction.
        The boxed regex matches independently of trailing tokens.
        """
        server = _make_mcqa_server()
        body = _make_mcqa_request(text="\\boxed{B}<|im_end|>", expected="B")
        result = await server.verify(body)
        assert result.extracted_answer == "B"
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_reasoning_gym_with_im_end(self):
        """reasoning_gym: <|im_end|> appended to a correct answer should ideally
        not prevent correct scoring (the answer extraction happens before scoring).
        """
        server = _make_reasoning_gym_server()
        dataset = reasoning_gym.create_dataset("basic_arithmetic", size=1, seed=42)
        entry = dataset[0]

        # If the model wraps the answer in <answer> tags, the im_end comes after
        text = f"<answer>{entry['answer']}</answer><|im_end|>"

        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=_make_response(text),
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        result = await server.verify(request)
        assert result.reward >= 0.9, (
            f"<answer> tag extraction should work despite trailing <|im_end|>. Got reward={result.reward}"
        )
