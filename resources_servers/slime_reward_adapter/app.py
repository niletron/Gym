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

"""Slime reward adapter bridge server.

This server bridges Slime's remote_rm protocol to NeMo-Gym's verify endpoint.
Slime sends POST requests with {prompt, response, label} and expects a JSON
reward response. This adapter translates those into NeMo-Gym's BaseVerifyRequest
format, calls the underlying resources server's /verify endpoint, and returns
the reward in Slime's expected format.

It also exposes a standard NeMo-Gym resources server interface so it can
function as a normal NeMo-Gym resources server too.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
from fastapi import Body, FastAPI, Request
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker

logger = logging.getLogger(__name__)


class SlimeRewardRequest(BaseModel):
    """Request format from Slime's remote_rm.

    Slime sends {prompt, response, label} to the reward endpoint.
    """

    prompt: Any  # str or list[dict] (conversation format)
    response: str
    label: Optional[str] = None


class SlimeRewardResponse(BaseModel):
    """Response format expected by Slime's remote_rm.

    Returns scalar reward and optional metadata.
    """

    reward: float
    metadata: Optional[Dict[str, Any]] = None


class SlimeRewardAdapterConfig(BaseResourcesServerConfig):
    """Configuration for the Slime reward adapter.

    The adapter can either:
    1. Proxy to an upstream NeMo-Gym resources server (upstream_server_url)
    2. Use a built-in reward function (reward_type)
    """

    # URL of the upstream NeMo-Gym resources server to proxy verify requests to
    upstream_server_url: Optional[str] = None

    # Built-in reward type for standalone operation (e.g., "exact_match", "contains")
    reward_type: str = "proxy"

    # Default system prompt to inject when converting Slime prompts to NeMo-Gym format
    default_system_prompt: Optional[str] = None


class SlimeRewardAdapter(SimpleResourcesServer):
    """Bridge between Slime's remote_rm and NeMo-Gym's verify endpoint.

    Provides two interfaces:
    1. POST /slime_reward - Slime's native {prompt, response, label} format
    2. POST /verify - Standard NeMo-Gym verify endpoint

    When upstream_server_url is configured, /verify requests are proxied
    to the upstream resources server. Otherwise, a simple built-in reward
    function is used.
    """

    config: SlimeRewardAdapterConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Add Slime-compatible reward endpoint
        app.post("/slime_reward")(self.slime_reward)

        # Also mount at root for Slime's rm_url (Slime POSTs to the URL directly)
        app.post("/")(self.slime_reward)

        return app

    def _convert_prompt_to_input(self, prompt: Any) -> List[Dict[str, Any]]:
        """Convert Slime's prompt format to NeMo-Gym's input format."""
        messages = []

        if self.config.default_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "type": "message",
                    "content": [{"type": "input_text", "text": self.config.default_system_prompt}],
                }
            )

        if isinstance(prompt, str):
            messages.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
        elif isinstance(prompt, list):
            # Conversation format: list of {role, content} dicts
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "input_text", "text": content}]
                messages.append(
                    {
                        "role": role,
                        "type": "message",
                        "content": content,
                    }
                )
        else:
            messages.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": [{"type": "input_text", "text": str(prompt)}],
                }
            )

        return messages

    def _convert_response_to_nemogym(self, response_text: str) -> NeMoGymResponse:
        """Convert Slime's response text to NeMo-Gym's NeMoGymResponse format."""
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=0,
            model="slime",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role="assistant",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=response_text,
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                ).model_dump()
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    async def slime_reward(self, body: SlimeRewardRequest) -> SlimeRewardResponse:
        """Handle Slime's remote_rm format and return reward."""
        # Convert Slime format to NeMo-Gym format
        input_messages = self._convert_prompt_to_input(body.prompt)
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            model="slime",
        )
        response = self._convert_response_to_nemogym(body.response)

        verify_request = BaseVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
        )

        # Add label to verifier_metadata if present
        if body.label is not None:
            # Store label in the responses_create_params metadata for the verify endpoint
            if verify_request.responses_create_params.metadata is None:
                verify_request.responses_create_params.metadata = {}
            verify_request.responses_create_params.metadata["label"] = body.label

        if self.config.upstream_server_url:
            return await self._proxy_verify(verify_request, body.label)

        # Use built-in verify
        verify_response = await self.verify(verify_request)
        return SlimeRewardResponse(reward=verify_response.reward)

    async def _proxy_verify(self, verify_request: BaseVerifyRequest, label: Optional[str] = None) -> SlimeRewardResponse:
        """Proxy the verify request to the upstream NeMo-Gym resources server."""
        verify_dict = verify_request.model_dump()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.upstream_server_url}/verify",
                json=verify_dict,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

        reward = result.get("reward", 0.0)
        return SlimeRewardResponse(reward=reward, metadata=result)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Standard NeMo-Gym verify endpoint with built-in reward functions."""
        reward = 0.0

        # Extract the model's response text
        response_text = ""
        if body.response and body.response.output:
            for output_item in body.response.output:
                if isinstance(output_item, dict):
                    if output_item.get("type") == "message":
                        for content_item in output_item.get("content", []):
                            if content_item.get("type") == "output_text":
                                response_text += content_item.get("text", "")
                elif hasattr(output_item, "content"):
                    for content_item in output_item.content:
                        if hasattr(content_item, "text"):
                            response_text += content_item.text

        # Extract label from metadata
        label = None
        if body.responses_create_params.metadata:
            label = body.responses_create_params.metadata.get("label")

        # Apply built-in reward function
        if self.config.reward_type == "exact_match" and label is not None:
            reward = 1.0 if response_text.strip() == label.strip() else 0.0
        elif self.config.reward_type == "contains" and label is not None:
            reward = 1.0 if label.strip() in response_text else 0.0
        elif self.config.reward_type == "proxy":
            # When no upstream URL, default to 0.0
            reward = 0.0
        else:
            reward = 0.0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    SlimeRewardAdapter.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = SlimeRewardAdapter.run_webserver()  # noqa: F401
