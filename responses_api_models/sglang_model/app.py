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

SGLang exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so this
server inherits from :class:`VLLMModel` and adds SGLang-specific configuration
(router health checks, native sampling params).

Typical config (YAML)::

    policy_model:
      responses_api_models:
        sglang_model:
          entrypoint: app.py
          base_url: http://localhost:30000/v1
          api_key: EMPTY
          model: Qwen/Qwen2.5-0.5B-Instruct
          return_token_id_information: false
          uses_reasoning_parser: true
"""

import logging
from typing import Any, Dict, Optional

import aiohttp
from fastapi import FastAPI, Request

from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

logger = logging.getLogger(__name__)


class SGLangModelConfig(VLLMModelConfig):
    """Extends :class:`VLLMModelConfig` with SGLang-specific fields.

    Attributes:
        sglang_router_url: Base URL of the SGLang router (without ``/v1``).
            Used for health-checking via ``/health_generate``.
        use_native_generate: Reserved for future use — whether to call SGLang's
            ``/generate`` endpoint instead of the OpenAI-compat layer.
        sglang_sampling_params: Extra key/value pairs merged into every chat
            completion request (e.g. ``min_new_tokens``).
    """

    sglang_router_url: Optional[str] = None
    use_native_generate: bool = False
    sglang_sampling_params: Optional[Dict[str, Any]] = None


class SGLangModel(VLLMModel):
    """SGLang model server.

    Since SGLang's OpenAI-compat API is wire-compatible with vLLM, this class
    only adds:

    * ``GET /sglang_health`` — probes the SGLang router's health endpoint.
    * SGLang-specific sampling params merged into every request.
    """

    config: SGLangModelConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/sglang_health")(self.sglang_health)
        return app

    async def sglang_health(self) -> Dict[str, Any]:
        """Probe the SGLang router and return a health summary."""
        info: Dict[str, Any] = {"status": "ok", "backend": "sglang"}
        if self.config.sglang_router_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.sglang_router_url}/health_generate",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        info["router_status"] = "healthy" if resp.status == 200 else f"unhealthy ({resp.status})"
            except Exception as exc:
                info["router_status"] = f"unreachable ({exc})"
        return info

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        if self.config.sglang_sampling_params:
            for key, value in self.config.sglang_sampling_params.items():
                body_dict.setdefault(key, value)
        return body_dict


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = SGLangModel.run_webserver()  # noqa: F401
