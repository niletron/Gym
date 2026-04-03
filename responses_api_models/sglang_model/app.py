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

"""SGLang model server for NeMo-Gym.

SGLang exposes OpenAI-compatible /v1/chat/completions endpoints, so this server
builds on the existing VLLMModel with SGLang-specific configuration and health
checking via the SGLang /get_model_info and /health_generate endpoints.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request

from nemo_gym.base_responses_api_model import Body
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

logger = logging.getLogger(__name__)


class SGLangModelConfig(VLLMModelConfig):
    """Configuration for SGLang model server.

    Inherits from VLLMModelConfig since SGLang exposes OpenAI-compatible endpoints.
    Adds SGLang-specific configuration for features like the /generate endpoint
    with native logprob support and router health monitoring.
    """

    # SGLang router URL for direct /generate calls (bypasses OpenAI compat layer).
    # If set, used for health checks and can be used for low-level generation.
    sglang_router_url: Optional[str] = None

    # Whether to use SGLang's native /generate endpoint for logprobs
    # instead of the OpenAI-compatible /v1/chat/completions endpoint.
    # The native endpoint returns more detailed logprob information.
    use_native_generate: bool = False

    # SGLang sampling parameters not available in the OpenAI compat API
    sglang_sampling_params: Optional[Dict[str, Any]] = None


class SGLangModel(VLLMModel):
    """SGLang model server that extends VLLMModel.

    SGLang's OpenAI-compatible API is fully compatible with the VLLMModel
    implementation. This subclass adds:
    - SGLang-specific configuration
    - Health checking via SGLang's /health_generate endpoint
    - Support for SGLang's native /generate endpoint (optional)
    """

    config: SGLangModelConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Add SGLang-specific health check endpoint
        app.get("/sglang_health")(self.sglang_health)

        return app

    async def sglang_health(self) -> Dict[str, Any]:
        """Check SGLang engine health via the router or direct engine endpoint."""
        health_info = {"status": "ok", "backend": "sglang"}

        if self.config.sglang_router_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.sglang_router_url}/health_generate",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            health_info["router_status"] = "healthy"
                        else:
                            health_info["router_status"] = f"unhealthy (status {resp.status})"
            except Exception as e:
                health_info["router_status"] = f"unreachable ({e})"

        return health_info

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Handle responses endpoint. Delegates to VLLMModel implementation.

        SGLang's /v1/chat/completions is fully compatible with vLLM's.
        """
        return await super().responses(request, body)

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        """Handle chat completions endpoint. Delegates to VLLMModel implementation.

        SGLang's /v1/chat/completions is fully compatible with vLLM's.
        """
        return await super().chat_completions(request, body)

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SGLang-specific preprocessing on top of VLLMModel's preprocessing."""
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)

        # Merge SGLang-specific sampling params if configured
        if self.config.sglang_sampling_params:
            for key, value in self.config.sglang_sampling_params.items():
                if key not in body_dict:
                    body_dict[key] = value

        return body_dict


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = SGLangModel.run_webserver()  # noqa: F401
