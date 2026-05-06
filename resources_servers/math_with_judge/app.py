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
import asyncio
import concurrent.futures as cf
import contextlib
import logging
import multiprocessing as mp
import os
import signal
from io import StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


# ---------------------------------------------------------------------------
# Subprocess-isolated math_verify worker
#
# math_verify descends into sympy C-extension recursion (factorials, Sum.doit,
# simplify on tower exponents). Its internal SIGALRM timeout cannot preempt
# C-extension frames, so a poison payload wedges the entire async event loop
# indefinitely. We isolate each verify call in a worker process so a timeout
# can be enforced by SIGKILLing the worker. See MATH_OUTAGE_RCA.md.
# ---------------------------------------------------------------------------

_WORKER_VERIFIER = None


def _worker_init():
    """Initializer run once per worker process. Imports math_verify eagerly so
    the per-call overhead is just function dispatch."""
    global _WORKER_VERIFIER
    from math_verify.metric import math_metric as _mm
    from math_verify.parser import (
        ExprExtractionConfig as _EC,
        LatexExtractionConfig as _LC,
    )

    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    _WORKER_VERIFIER = _mm(
        gold_extraction_target=(_LC(),),
        pred_extraction_target=(_EC(), _LC()),
    )


def _strip_math_delimiters_plain(s: str) -> str:
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


def _keep_only_last_boxed(text: str) -> str:
    r"""Strip all \boxed{...} occurrences except the LAST one.

    Blocks a reward-hacking exploit where the model emits many wrong
    boxes around a correct one to game math_verify's set-merge match.
    The model's FINAL answer convention is the LAST \boxed{} (standard
    math-reasoning prompts instruct "put your final answer in \boxed{...}"),
    so we keep only that one and drop earlier ones before handing the
    string to math_verify.

    Handles nested braces correctly by counting depth. Unclosed boxes
    (no matching closing brace) are left in place (treated as plain
    text — math_verify won't extract them anyway).
    """
    if not text:
        return text

    token = "\\boxed{"
    spans: list[tuple[int, int]] = []  # list of (start, end_exclusive) for each full \boxed{...}

    i = 0
    n = len(text)
    while True:
        start = text.find(token, i)
        if start < 0:
            break

        # Scan forward from the char after the opening brace, tracking
        # nesting depth. depth starts at 1 (for \boxed{'s own open brace).
        j = start + len(token)
        depth = 1
        closed_at = -1
        while j < n:
            ch = text[j]
            if ch == "\\" and j + 1 < n:
                # Skip escaped char (covers \{, \}, \\, etc.) so that
                # an escaped brace doesn't change our depth.
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    closed_at = j
                    break
            j += 1

        if closed_at < 0:
            # Unclosed \boxed{ — bail on scanning further from inside it.
            # Advance past the token so we don't loop forever, but do NOT
            # record a span (we can't safely remove an unclosed box).
            i = start + len(token)
            continue

        spans.append((start, closed_at + 1))
        i = closed_at + 1

    if len(spans) < 2:
        return text

    # Remove all spans except the last. Walk from the end backwards so
    # earlier indices stay valid as we splice.
    last = spans[-1]
    result = text
    for s, e in reversed(spans[:-1]):
        result = result[:s] + result[e:]
        # Note: we don't need to adjust `last` because we iterate in
        # reverse and we never touch indices >= e on subsequent passes.
    return result


def _verify_in_worker(expected_answer: str, generated_answer: str):
    """Run math_verify grading in the current (worker) process. Returns
    (reward, extracted_answer) or raises on failure."""
    global _WORKER_VERIFIER
    if _WORKER_VERIFIER is None:
        _worker_init()

    from math_verify import grader as _grader
    from math_verify.errors import TimeoutException as _TO

    try:
        stripped = _strip_math_delimiters_plain(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        # Reward-hacking mitigation: math_verify set-merges all \boxed{}
        # occurrences in the prediction, so a model can flood wrong boxes
        # around a correct one to game the match. Keep only the LAST box
        # (the final-answer convention) before handing to math_verify.
        # Opt-out via MATH_KEEP_ONLY_LAST_BOX=0 for regression testing.
        if os.environ.get("MATH_KEEP_ONLY_LAST_BOX", "1") != "0":
            generated_answer = _keep_only_last_boxed(generated_answer)
        ret_score, extracted_answer = _WORKER_VERIFIER(
            [ground_truth_parsable], [generated_answer]
        )
        reward = float(ret_score)

        ea: Optional[str] = None
        if extracted_answer is not None and len(extracted_answer) == 2:
            extracted_gold, extracted_prediction = extracted_answer
            for pred in extracted_prediction:
                if any(_grader.verify(gold, pred) for gold in extracted_gold):
                    ea = pred
                    break
            else:
                ea = extracted_prediction[0] if extracted_prediction else None
        return reward, ea
    except (Exception, _TO):
        return 0.0, None


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: LibraryJudgeMathResourcesServerConfig

    # Process-pool isolation for math_verify. Tunable via env vars so ops can
    # adjust without code changes.
    _VERIFY_TIMEOUT_S: ClassVar[float] = float(os.environ.get("MATH_VERIFY_TIMEOUT_S", "10"))
    _VERIFY_POOL_WORKERS: ClassVar[int] = int(os.environ.get("MATH_VERIFY_POOL_WORKERS", "4"))

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Kept for the (synchronous) in-process fallback path only; the hot
        # path now dispatches to a worker pool.
        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

        self._verify_pool: Optional[cf.ProcessPoolExecutor] = None
        self._verify_pool_lock: Optional[asyncio.Lock] = None

    def _build_pool(self) -> cf.ProcessPoolExecutor:
        return cf.ProcessPoolExecutor(
            max_workers=self._VERIFY_POOL_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
        )

    def _ensure_pool(self) -> cf.ProcessPoolExecutor:
        if self._verify_pool is None:
            self._verify_pool = self._build_pool()
        if self._verify_pool_lock is None:
            self._verify_pool_lock = asyncio.Lock()
        return self._verify_pool

    def _recycle_pool(self) -> None:
        """SIGKILL all workers of the current pool and replace it. Called when
        a verify call times out, which can only mean a worker is wedged inside
        sympy's C-extension recursion where SIGTERM won't land."""
        old = self._verify_pool
        self._verify_pool = self._build_pool()
        if old is None:
            return
        pids = []
        try:
            pids = list(getattr(old, "_processes", {}).keys())
        except Exception:
            pids = []
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logging.warning("math_verify pool recycle: SIGKILL %s failed: %s", pid, exc)
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        logging.warning("math_verify pool recycled after timeout (killed %d workers)", len(pids))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.get("/health")
        async def health():
            """End-to-end probe: grade a canned "1==1" through the pool.

            Watchdog polls this. /docs responds even when the verify pool is
            wedged — only /health exercises the pool boundary and catches a
            hung sympy worker."""
            try:
                reward, _ = await self._verify_answer_with_library("1", "\\boxed{1}")
                return {"status": "ok", "reward": float(reward)}
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        (
            reward,
            extracted_answer,
            library_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.question, body.expected_answer, combined_response)
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        library_reward, extracted_answer = await self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    async def _verify_answer_with_library(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        # Dispatched to a subprocess pool so we can SIGKILL workers that wedge
        # inside math_verify's sympy C-extension recursion. See
        # MATH_OUTAGE_RCA.md for why SIGALRM-based timeouts fail here.
        pool = self._ensure_pool()
        loop = asyncio.get_running_loop()
        try:
            fut = loop.run_in_executor(pool, _verify_in_worker, expected_answer, generated_answer)
            return await asyncio.wait_for(fut, timeout=self._VERIFY_TIMEOUT_S)
        except asyncio.TimeoutError:
            logging.warning(
                "math_verify timeout (%.1fs) on expected=%r — recycling pool",
                self._VERIFY_TIMEOUT_S,
                expected_answer[:80],
            )
            async with self._verify_pool_lock:
                self._recycle_pool()
            return 0.0, None
        except (cf.process.BrokenProcessPool, cf.CancelledError):
            logging.warning("math_verify worker died — recycling pool")
            async with self._verify_pool_lock:
                self._recycle_pool()
            return 0.0, None
        except (Exception, TimeoutException) as exc:
            logging.warning("math_verify error: %s", exc)
            return 0.0, None

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
