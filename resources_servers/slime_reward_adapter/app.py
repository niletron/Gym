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

"""Slime ↔ NeMo-Gym reward adapter.

Bridges `Slime's remote RM protocol <https://github.com/THUDM/slime>`_ to any
NeMo-Gym resources server or agent.

Three operating modes, selected by config:

==============  ===============================  ============================
Mode            Config                           Use when …
==============  ===============================  ============================
Built-in        ``reward_type: exact_match``      Standalone testing / math
Verify proxy    ``upstream_server_url: <url>``    Verify-only environments
Agent proxy     ``upstream_agent_url: <url>``     Tool-calling environments
==============  ===============================  ============================

Endpoints exposed:

* ``POST /slime_reward`` — Slime's ``{prompt, response, label, metadata}``
* ``POST /``            — alias (Slime posts to ``rm_url`` root)
* ``POST /slime_agent_run`` — delegates to a NeMo-Gym agent ``/run``
* ``POST /verify``      — standard NeMo-Gym verify
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

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SlimeRewardRequest(BaseModel):
    """Payload sent by Slime's ``remote_rm`` (or a custom RM function)."""

    prompt: Any  # str | list[dict]
    response: str
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SlimeRewardResponse(BaseModel):
    """Returned to Slime — must contain at least ``reward``."""

    reward: float
    metadata: Optional[Dict[str, Any]] = None


class SlimeAgentRunRequest(BaseModel):
    """Payload for ``/slime_agent_run`` (tool-calling environments)."""

    responses_create_params: Dict[str, Any]
    verifier_metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SlimeRewardAdapterConfig(BaseResourcesServerConfig):
    """
    Attributes:
        upstream_server_url: Proxy ``/slime_reward`` to this resources-server ``/verify``.
        upstream_agent_url: Proxy ``/slime_agent_run`` to this agent ``/run``.
        reward_type: Built-in reward when no upstream is set (``exact_match`` | ``contains``).
        default_system_prompt: Injected as the first system message in every converted prompt.
    """

    upstream_server_url: Optional[str] = None
    upstream_agent_url: Optional[str] = None
    reward_type: str = "proxy"
    default_system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class SlimeRewardAdapter(SimpleResourcesServer):
    config: SlimeRewardAdapterConfig

    # -- FastAPI wiring -------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/slime_reward")(self.slime_reward)
        app.post("/")(self.slime_reward)
        app.post("/slime_agent_run")(self.slime_agent_run)
        return app

    # -- Format conversion helpers -------------------------------------------

    def _prompt_to_input_messages(self, prompt: Any) -> List[Dict[str, Any]]:
        """Convert a Slime prompt (str or chat list) to NeMo-Gym input items."""
        msgs: List[Dict[str, Any]] = []

        if self.config.default_system_prompt:
            msgs.append(self._make_message("system", self.config.default_system_prompt))

        if isinstance(prompt, str):
            msgs.append(self._make_message("user", prompt))
        elif isinstance(prompt, list):
            for m in prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "input_text", "text": content}]
                msgs.append({"role": role, "type": "message", "content": content})
        else:
            msgs.append(self._make_message("user", str(prompt)))

        return msgs

    @staticmethod
    def _make_message(role: str, text: str) -> Dict[str, Any]:
        return {"role": role, "type": "message", "content": [{"type": "input_text", "text": text}]}

    @staticmethod
    def _wrap_response_text(text: str) -> NeMoGymResponse:
        """Wrap a plain-text response into a ``NeMoGymResponse``."""
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=0,
            model="slime",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role="assistant",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    status="completed",
                    type="message",
                ).model_dump()
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    @staticmethod
    def _extract_response_text(body: BaseVerifyRequest) -> str:
        """Pull the assistant's text out of a NeMo-Gym verify request."""
        parts: List[str] = []
        for item in body.response.output or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
            elif hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
        return "".join(parts)

    def _build_verifier_metadata(self, body: SlimeRewardRequest) -> Dict[str, Any]:
        """Derive ``verifier_metadata`` from a Slime request.

        Priority:
        1. ``metadata["verifier_metadata"]`` (explicit)
        2. Entire ``metadata`` dict (implicit)
        3. Empty dict with only ``label`` / ``expected_answer``
        """
        vm: Dict[str, Any] = {}
        if body.metadata:
            if "verifier_metadata" in body.metadata:
                vm = dict(body.metadata["verifier_metadata"])
            else:
                vm = dict(body.metadata)
        if body.label is not None:
            vm.setdefault("label", body.label)
            vm.setdefault("expected_answer", body.label)
        return vm

    # -- Endpoints ------------------------------------------------------------

    async def slime_reward(self, body: SlimeRewardRequest) -> SlimeRewardResponse:
        """Accept Slime's ``{prompt, response, label, metadata}`` and return a reward.

        If ``upstream_server_url`` is set the request is proxied; otherwise the
        built-in reward function (``exact_match`` / ``contains``) is applied.
        """
        input_msgs = self._prompt_to_input_messages(body.prompt)
        vm = self._build_verifier_metadata(body)

        rcp = NeMoGymResponseCreateParamsNonStreaming(
            input=input_msgs,
            model="slime",
            metadata={"label": body.label} if body.label else None,
        )
        response = self._wrap_response_text(body.response)

        if self.config.upstream_server_url:
            verify_dict = {
                "responses_create_params": rcp.model_dump(),
                "response": response.model_dump(),
                "verifier_metadata": vm,
            }
            return await self._proxy_verify(verify_dict)

        verify_req = BaseVerifyRequest(responses_create_params=rcp, response=response)
        result = await self.verify(verify_req)
        return SlimeRewardResponse(reward=result.reward)

    async def slime_agent_run(self, body: SlimeAgentRunRequest) -> SlimeRewardResponse:
        """Delegate to a NeMo-Gym agent's ``/run`` for tool-calling environments."""
        if not self.config.upstream_agent_url:
            return SlimeRewardResponse(reward=0.0, metadata={"error": "upstream_agent_url not configured"})

        run_body: Dict[str, Any] = {"responses_create_params": body.responses_create_params}
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
        return SlimeRewardResponse(reward=result.get("reward", 0.0), metadata=result)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Built-in verify: ``exact_match`` or ``contains`` against the label."""
        text = self._extract_response_text(body)
        label = (body.responses_create_params.metadata or {}).get("label")

        if self.config.reward_type == "exact_match" and label is not None:
            reward = 1.0 if text.strip() == label.strip() else 0.0
        elif self.config.reward_type == "contains" and label is not None:
            reward = 1.0 if label.strip() in text else 0.0
        else:
            reward = 0.0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    async def _proxy_verify(self, verify_dict: Dict[str, Any]) -> SlimeRewardResponse:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.upstream_server_url}/verify",
                json=verify_dict,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
        return SlimeRewardResponse(reward=result.get("reward", 0.0), metadata=result)


if __name__ == "__main__":
    SlimeRewardAdapter.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = SlimeRewardAdapter.run_webserver()  # noqa: F401
