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

This server bridges Slime's remote_rm protocol to NeMo-Gym's verify/run endpoints.

Supports ALL NeMo-Gym environments through two modes:

1. **Verify mode** (simple environments): Slime generates text with SGLang, then this
   adapter calls the NeMo-Gym resources server's /verify endpoint with the response
   and verifier_metadata. Works for: math, mcqa, code_gen, instruction_following, etc.

2. **Agent /run mode** (tool-calling environments): This adapter calls the NeMo-Gym
   agent's /run endpoint which handles the full generate -> tool call -> verify loop.
   Works for: tavily_search, google_search, ns_tools, workplace_assistant, openenv, etc.

Slime sends: {prompt, response, label, metadata}
- metadata can contain verifier_metadata for NeMo-Gym environments
- For agent /run mode, metadata should contain the full responses_create_params
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
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
    metadata can contain verifier_metadata for NeMo-Gym environments.
    """

    prompt: Any  # str or list[dict] (conversation format)
    response: str
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SlimeRewardResponse(BaseModel):
    """Response format expected by Slime's remote_rm."""

    reward: float
    metadata: Optional[Dict[str, Any]] = None


class SlimeAgentRunRequest(BaseModel):
    """Request for the /slime_agent_run endpoint.

    Used for tool-calling environments where Slime delegates the full
    generate -> tool call -> verify loop to NeMo-Gym's agent.
    """

    responses_create_params: Dict[str, Any]
    verifier_metadata: Optional[Dict[str, Any]] = None


class SlimeRewardAdapterConfig(BaseResourcesServerConfig):
    """Configuration for the Slime reward adapter.

    Modes:
    1. Built-in: reward_type = "exact_match" or "contains" (standalone, no upstream)
    2. Proxy to resources server: upstream_server_url points to a NeMo-Gym resources server /verify
    3. Agent run: upstream_agent_url points to a NeMo-Gym agent /run endpoint
    """

    # URL of upstream NeMo-Gym resources server for verify-mode proxy
    upstream_server_url: Optional[str] = None

    # URL of upstream NeMo-Gym agent for agent-run mode (tool-calling environments)
    upstream_agent_url: Optional[str] = None

    # Built-in reward type for standalone operation
    reward_type: str = "proxy"

    # Default system prompt to inject when converting Slime prompts
    default_system_prompt: Optional[str] = None


class SlimeRewardAdapter(SimpleResourcesServer):
    """Bridge between Slime and ALL NeMo-Gym environments.

    Endpoints:
    1. POST /slime_reward - Slime's native {prompt, response, label, metadata} format
    2. POST / - Same as /slime_reward (Slime posts to rm_url root)
    3. POST /slime_agent_run - For tool-calling envs, delegates to NeMo-Gym agent /run
    4. POST /verify - Standard NeMo-Gym verify endpoint
    """

    config: SlimeRewardAdapterConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/slime_reward")(self.slime_reward)
        app.post("/")(self.slime_reward)
        app.post("/slime_agent_run")(self.slime_agent_run)

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
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "input_text", "text": content}]
                messages.append({"role": role, "type": "message", "content": content})
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
                        NeMoGymResponseOutputText(type="output_text", text=response_text, annotations=[])
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
        """Handle Slime's remote_rm format and return reward.

        Supports all NeMo-Gym environments by passing verifier_metadata
        from Slime's sample.metadata through to the verify endpoint.
        """
        input_messages = self._convert_prompt_to_input(body.prompt)

        # Build metadata dict: merge label + Slime metadata
        metadata_for_params = {}
        if body.label is not None:
            metadata_for_params["label"] = body.label

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=input_messages,
            model="slime",
            metadata=metadata_for_params if metadata_for_params else None,
        )
        response = self._convert_response_to_nemogym(body.response)

        # Build the verify request dict with verifier_metadata as extra field
        verify_dict = {
            "responses_create_params": responses_create_params.model_dump(),
            "response": response.model_dump(),
        }

        # Pass through verifier_metadata from Slime's sample metadata
        # This is how NeMo-Gym environments receive task-specific data
        # (unit_tests, expected_answer, options, etc.)
        if body.metadata and "verifier_metadata" in body.metadata:
            verify_dict["verifier_metadata"] = body.metadata["verifier_metadata"]
        elif body.metadata:
            # If metadata doesn't have explicit verifier_metadata key,
            # use the entire metadata dict as verifier_metadata
            verify_dict["verifier_metadata"] = body.metadata
            if body.label is not None:
                verify_dict["verifier_metadata"]["label"] = body.label
                verify_dict["verifier_metadata"]["expected_answer"] = body.label

        if self.config.upstream_server_url:
            return await self._proxy_verify(verify_dict)

        # Use built-in verify
        verify_request = BaseVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
        )
        verify_response = await self.verify(verify_request)
        return SlimeRewardResponse(reward=verify_response.reward)

    async def _proxy_verify(self, verify_dict: Dict[str, Any]) -> SlimeRewardResponse:
        """Proxy the verify request to an upstream NeMo-Gym resources server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.upstream_server_url}/verify",
                json=verify_dict,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

        reward = result.get("reward", 0.0)
        return SlimeRewardResponse(reward=reward, metadata=result)

    async def slime_agent_run(self, body: SlimeAgentRunRequest) -> SlimeRewardResponse:
        """Delegate to NeMo-Gym agent's /run endpoint for tool-calling environments.

        This is used for environments that require multi-step tool interaction
        (tavily_search, google_search, ns_tools, workplace_assistant, openenv, etc.).
        The NeMo-Gym agent handles the full generate -> tool call -> verify loop.
        """
        if not self.config.upstream_agent_url:
            return SlimeRewardResponse(
                reward=0.0,
                metadata={"error": "upstream_agent_url not configured for agent run mode"},
            )

        # Build the /run request body
        run_body = {"responses_create_params": body.responses_create_params}
        if body.verifier_metadata:
            run_body["verifier_metadata"] = body.verifier_metadata

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.upstream_agent_url}/run",
                json=run_body,
                timeout=aiohttp.ClientTimeout(total=300),
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
            reward = 0.0
        else:
            reward = 0.0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    SlimeRewardAdapter.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = SlimeRewardAdapter.run_webserver()  # noqa: F401
