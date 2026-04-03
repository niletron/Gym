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
import pytest
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.slime_reward_adapter.app import (
    SlimeRewardAdapter,
    SlimeRewardAdapterConfig,
    SlimeRewardRequest,
)


def _make_adapter(reward_type="exact_match", **kwargs):
    config = SlimeRewardAdapterConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="",
        reward_type=reward_type, **kwargs,
    )
    return SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))


class TestSlimeRewardAdapter:
    def test_sanity(self):
        assert _make_adapter().config.reward_type == "exact_match"

    def test_string_prompt_conversion(self):
        msgs = _make_adapter()._prompt_to_input_messages("What is 2+2?")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "What is 2+2?"

    def test_conversation_prompt_conversion(self):
        prompt = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        msgs = _make_adapter()._prompt_to_input_messages(prompt)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_response_wrapping(self):
        resp = SlimeRewardAdapter._wrap_response_text("The answer is 4.")
        assert resp.output[0]["content"][0]["text"] == "The answer is 4."

    def test_system_prompt_injection(self):
        adapter = _make_adapter(default_system_prompt="Be helpful.")
        msgs = adapter._prompt_to_input_messages("test")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_verifier_metadata_explicit(self):
        adapter = _make_adapter()
        body = SlimeRewardRequest(
            prompt="q", response="a", label="a",
            metadata={"verifier_metadata": {"unit_tests": [{"in": "1", "out": "1"}]}},
        )
        vm = adapter._build_verifier_metadata(body)
        assert "unit_tests" in vm
        assert vm["label"] == "a"

    def test_verifier_metadata_implicit(self):
        adapter = _make_adapter()
        body = SlimeRewardRequest(
            prompt="q", response="a", label="a",
            metadata={"rm_type": "math"},
        )
        vm = adapter._build_verifier_metadata(body)
        assert vm["rm_type"] == "math"
        assert vm["expected_answer"] == "a"

    @pytest.mark.asyncio
    async def test_exact_match_correct(self):
        result = await _make_adapter().slime_reward(
            SlimeRewardRequest(prompt="q", response="4", label="4"),
        )
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_wrong(self):
        result = await _make_adapter().slime_reward(
            SlimeRewardRequest(prompt="q", response="5", label="4"),
        )
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_contains_reward(self):
        result = await _make_adapter(reward_type="contains").slime_reward(
            SlimeRewardRequest(prompt="q", response="The answer is 4.", label="4"),
        )
        assert result.reward == 1.0
