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


class TestSlimeRewardAdapter:
    def _make_adapter(self, reward_type="exact_match", upstream_server_url=None):
        config = SlimeRewardAdapterConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            reward_type=reward_type,
            upstream_server_url=upstream_server_url,
        )
        return SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))

    def test_sanity(self) -> None:
        adapter = self._make_adapter()
        assert adapter.config.reward_type == "exact_match"

    def test_convert_string_prompt(self) -> None:
        adapter = self._make_adapter()
        messages = adapter._convert_prompt_to_input("What is 2+2?")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "What is 2+2?"

    def test_convert_conversation_prompt(self) -> None:
        adapter = self._make_adapter()
        prompt = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        messages = adapter._convert_prompt_to_input(prompt)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_convert_response(self) -> None:
        adapter = self._make_adapter()
        response = adapter._convert_response_to_nemogym("The answer is 4.")
        assert response.output[0]["content"][0]["text"] == "The answer is 4."

    @pytest.mark.asyncio
    async def test_exact_match_reward(self) -> None:
        adapter = self._make_adapter(reward_type="exact_match")
        slime_request = SlimeRewardRequest(prompt="What is 2+2?", response="4", label="4")
        result = await adapter.slime_reward(slime_request)
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_wrong(self) -> None:
        adapter = self._make_adapter(reward_type="exact_match")
        slime_request = SlimeRewardRequest(prompt="What is 2+2?", response="5", label="4")
        result = await adapter.slime_reward(slime_request)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_contains_reward(self) -> None:
        adapter = self._make_adapter(reward_type="contains")
        slime_request = SlimeRewardRequest(prompt="What is 2+2?", response="The answer is 4.", label="4")
        result = await adapter.slime_reward(slime_request)
        assert result.reward == 1.0

    def test_system_prompt_injection(self) -> None:
        config = SlimeRewardAdapterConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            reward_type="exact_match",
            default_system_prompt="You are a helpful assistant.",
        )
        adapter = SlimeRewardAdapter(config=config, server_client=MagicMock(spec=ServerClient))
        messages = adapter._convert_prompt_to_input("test")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["text"] == "You are a helpful assistant."
