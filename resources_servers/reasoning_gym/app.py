# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# reasoning_gym has a 3h timeout burst in the _u5j70izc_resume run (8,160
# timeouts on 2026-04-22). We isolate score_answer_fn calls in a subprocess
# pool so a wedged third-party scorer cannot hang the server.
# ---------------------------------------------------------------------------

import asyncio
import re
from typing import Any, ClassVar, Optional

import reasoning_gym
from fastapi import FastAPI
from reasoning_gym.utils import extract_answer

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.grader_pool import GraderPool


def _score_in_worker(task_name: str, model_answer: str, entry: dict) -> float:
    """Runs in a worker process (spawn). Import inside so child pays cost once."""
    try:
        import reasoning_gym as _rg
        score_fn = _rg.get_score_answer_fn(task_name)
        return float(score_fn(answer=model_answer, entry=entry))
    except BaseException:
        return 0.0


class ReasoningGymResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ReasoningGymVerifyRequest(BaseVerifyRequest):
    question: str
    answer: Optional[str]
    metadata: dict[str, Any]


class ReasoningGymVerifyResponse(BaseVerifyResponse):
    task_name: str
    score: float
    extracted_answer: Optional[str]


class ReasoningGymResourcesServer(SimpleResourcesServer):
    config: ReasoningGymResourcesServerConfig

    _POOL: ClassVar[GraderPool] = GraderPool(
        name="reasoning_gym",
        timeout_s_env="REASONING_GYM_VERIFY_TIMEOUT_S",
        workers_env="REASONING_GYM_POOL_WORKERS",
        default_timeout_s=8.0,  # routing_rm has 10s; leave 2s headroom
        default_workers=4,
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._POOL.bind(self)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.get("/health")
        async def health():
            """End-to-end probe: runs the pool on a trivial known task.

            Uses a low-cost reasoning_gym task ('simple_equations') for the
            round-trip — if this hangs, the pool is wedged."""
            try:
                # Minimal entry — the score function handles bad inputs by
                # returning 0 via the worker's broad except.
                result, reason = await self._POOL.run_or_zero(
                    self, _score_in_worker, "simple_equations", "0", {"metadata": {}}
                )
                return {"status": "ok", "reward": result, "reason": reason}
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

        return app

    async def verify(self, body: ReasoningGymVerifyRequest) -> ReasoningGymVerifyResponse:
        """Uses reasoning gym verifier in a SIGKILL-capable worker pool."""
        model_answer = self._extract_answer_from_response(body.response)

        task_name = body.metadata.get("source_dataset")
        if not task_name:
            raise ValueError(f"No task name found in metadata: {body.metadata}")

        entry = {
            "question": body.question,
            "answer": body.answer,
            "metadata": body.metadata,
        }

        score, _reason = await self._POOL.run_or_zero(
            self, _score_in_worker, task_name, model_answer, entry
        )

        return ReasoningGymVerifyResponse(
            **body.model_dump(),
            reward=score,
            task_name=task_name,
            score=score,
            extracted_answer=model_answer,
        )

    def _extract_answer_from_response(self, response) -> str:
        assistant_responses = []
        for output_item in response.output:
            if output_item.type != "message":
                continue

            if isinstance(output_item.content, str):
                assistant_responses.append(output_item.content)
            else:
                for content_item in output_item.content:
                    if content_item.type != "output_text":
                        continue
                    assistant_responses.append(content_item.text)

        full_text = "".join(assistant_responses)

        # Strip <think>...</think> tags (thinking models emit these before content)
        full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        # Strip markdown code fences (e.g. ```json ... ```)
        full_text = re.sub(r"^```\w*\n?", "", full_text)
        full_text = re.sub(r"\n?```$", "", full_text)

        # Try <answer> tags first (reasoning gym default)
        extracted = extract_answer(full_text, tag_name="answer")
        if extracted is not None:
            return extracted

        # Try \boxed{} if <answer> tags fail
        # this could be a slight instruction following issue, if model is prompted to use <answer> but uses boxed instead
        # found for deepseek-distill-qwen-1.5b it fails to use <answer> tags in favor of boxed, hence this fallback
        # may advise commenting this out for large models who follow instructions to use <answer> well
        boxed_match = re.search(r"\\boxed\{(.+)\}", full_text, flags=re.DOTALL)
        if boxed_match:
            content = boxed_match.group(1).strip()
            # Balance braces: if there are unmatched closing braces, trim to the balanced prefix
            depth = 0
            end_idx = len(content)
            for i, ch in enumerate(content):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        end_idx = i
                        break
                    depth -= 1
            content = content[:end_idx].strip()
            # Normalize LaTeX fractions: \frac{a}{b} -> a/b
            content = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", content)
            return content

        # return full text if <answer> or \boxed{} fail
        return full_text.strip() if full_text.strip() else ""


if __name__ == "__main__":
    ReasoningGymResourcesServer.run_webserver()
