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

# ---------------------------------------------------------------------------
# IF graders run verifiable_instructions checks that can call regex on
# untrusted model output (e.g. length/format constraints with user-supplied
# patterns). Catastrophic backtracking here would hang the server. Isolate
# the full check loop in a subprocess pool.
# ---------------------------------------------------------------------------

import re
from typing import Any, ClassVar, List, Literal, Tuple

from fastapi import FastAPI
from verifiable_instructions import instructions_registry

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.grader_pool import GraderPool


REFUSAL_PATTERNS = [
    r"(?i)^[\s]*(?:I\s+)?(?:cannot|can't|will not|won't|am unable to|refuse to)",
    r"(?i)^[\s]*(?:sorry|apologies|I\s+apologize)",
    r"(?i)^[\s]*(?:as an ai|I'm an ai|I am an ai)",
]


def _is_refusal(text: str, min_content_length: int = 150) -> bool:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if len(stripped) < min_content_length:
        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, stripped):
                return True
    return False


def _check_in_worker(
    final_response_text: str,
    instruction_id_list: List[str],
    kwargs_list: List,
    grading_mode: str,
) -> Tuple[float, List[bool]]:
    """Run the full instruction-following check loop in a worker process.

    Returns (reward, is_following_list). Returns (0.0, [False]*N) on any
    catastrophic failure. Individual instruction errors still yield False
    via the inner try-except."""
    try:
        is_following_list: List[bool] = []
        for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
            try:
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)
                if kwargs is None:
                    kwargs = {}
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                instruction.build_description(**filtered_kwargs)
                is_following_list.append(bool(instruction.check_following(final_response_text)))
            except Exception:
                is_following_list.append(False)

        if grading_mode == "binary":
            reward = float(all(is_following_list))
        elif grading_mode == "fraction":
            reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
        else:
            reward = 0.0

        if reward > 0 and _is_refusal(final_response_text):
            reward = 0.0

        return reward, is_following_list
    except BaseException:
        return 0.0, [False] * len(instruction_id_list)


class InstructionFollowingResourcesServerConfig(BaseResourcesServerConfig):
    pass


class InstructionFollowingRunRequest(BaseRunRequest):
    id: int
    instruction_id_list: List
    prompt: str
    kwargs: List
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


class InstructionFollowingVerifyRequest(InstructionFollowingRunRequest, BaseVerifyRequest):
    pass


class InstructionFollowingVerifyResponse(BaseVerifyResponse):
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    kwargs: List
    instruction_id_list: List
    prompt: str
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


class InstructionFollowingResourcesServer(SimpleResourcesServer):
    config: InstructionFollowingResourcesServerConfig

    _POOL: ClassVar[GraderPool] = GraderPool(
        name="instruction_following",
        timeout_s_env="IF_VERIFY_TIMEOUT_S",
        workers_env="IF_POOL_WORKERS",
        default_timeout_s=8.0,  # routing_rm timeout is 10s
        default_workers=4,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_nltk_data()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._POOL.bind(self)

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available at startup.

        nltk.download() always fetches the remote package index even when the
        data is already present. Guard with a local find() first to skip the
        download when the data already exists.
        """
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        except ImportError:
            # ifbench not available, skip
            pass
        except Exception as e:
            print(f"NLTK setup warning: {e}")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.get("/health")
        async def health():
            """End-to-end probe: run a trivial known-good check through the pool."""
            (reward, _), reason = await self._POOL.run_or_zero(
                self,
                _check_in_worker,
                "hello world",
                [],  # no instructions → reward=1.0 (all([])==True)
                [],
                "binary",
            )
            return {"status": "ok" if reason == "ok" else reason, "reward": reward}

        return app

    async def verify(self, body: InstructionFollowingVerifyRequest) -> InstructionFollowingVerifyResponse:
        # Get the final text response from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                # Extract text from the nested content structure
                final_response_text = last_output.content[0].text

        final_response_text = final_response_text.strip()

        reward_mode = getattr(body, "grading_mode", "binary")
        if reward_mode not in ("binary", "fraction"):
            raise ValueError(f"Invalid reward mode: {reward_mode}")

        # Dispatch to pool: the whole check loop, including per-instruction
        # regex work, runs in a worker we can SIGKILL.
        (reward, is_following_list), _reason = await self._POOL.run_or_zero(
            self,
            _check_in_worker,
            final_response_text,
            list(body.instruction_id_list),
            list(body.kwargs),
            reward_mode,
            zero_value=(0.0, [False] * len(body.instruction_id_list)),
        )

        return InstructionFollowingVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            follow_all_instructions=all(is_following_list) if is_following_list else False,
            follow_instruction_list=is_following_list,
        )


if __name__ == "__main__":
    InstructionFollowingResourcesServer.run_webserver()
