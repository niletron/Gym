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
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


def _make_config(**overrides):
    defaults = dict(
        host="0.0.0.0", port=8080, entrypoint="", name="",
        base_url="http://localhost:30000/v1", api_key="EMPTY",
        model="test-model", return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    return SGLangModelConfig(**(defaults | overrides))


class TestSGLangModel:
    def test_sanity(self):
        model = SGLangModel(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        assert model.config.base_url == ["http://localhost:30000/v1"]

    def test_sglang_config_fields(self):
        model = SGLangModel(
            config=_make_config(
                sglang_router_url="http://localhost:30000",
                sglang_sampling_params={"min_new_tokens": 1},
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        assert model.config.sglang_router_url == "http://localhost:30000"
        assert model.config.sglang_sampling_params == {"min_new_tokens": 1}

    def test_multiple_endpoints(self):
        model = SGLangModel(
            config=_make_config(base_url=["http://a:30000/v1", "http://b:30000/v1"]),
            server_client=MagicMock(spec=ServerClient),
        )
        assert len(model.config.base_url) == 2
