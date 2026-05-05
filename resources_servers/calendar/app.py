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
# Subprocess-isolated calendar verify
#
# `grade_assistant_response` in utils.py calls `extract_json_list` which uses
# a regex with classic catastrophic-backtracking potential:
#
#     pattern = r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]"
#
# Nested `(?:...)*` with overlapping character-classes can exhibit exponential
# match time on untrusted model output containing many brackets/braces. In the
# qwen3-30B-rlvr1_v2-3sv29n1u_u5j70izc_resume run this produced 134,735 client
# timeouts on the calendar server over 54 hours.
#
# We run each grade call in a worker process so that a wedged regex can be
# SIGKILLed after CALENDAR_VERIFY_TIMEOUT_S. Same pattern as math_with_judge.
# ---------------------------------------------------------------------------

import asyncio
import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import signal
from typing import Any, ClassVar, Optional, Tuple

from fastapi import FastAPI
from utils import grade_assistant_response

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


logger = logging.getLogger(__name__)


def _grade_in_worker(assistant_response: str, exp_cal_state: dict) -> Tuple[float, str]:
    """Run grade_assistant_response in the (worker) process. Returns (reward,
    reason). Returns (0.0, 'error') on any exception so the parent gets a
    well-formed value."""
    try:
        # grade_assistant_response returns (reward:int, reason:str). Coerce to
        # (float, str) for consistency with other envs.
        reward, reason = grade_assistant_response(assistant_response, exp_cal_state)
        return float(reward), str(reason)
    except BaseException as exc:  # catch BaseException to survive C-level hangs
        return 0.0, f"worker_exception:{type(exc).__name__}"


class CalendarRunRequest(BaseRunRequest):
    exp_cal_state: dict[str, Any]


class CalendarVerifyRequest(CalendarRunRequest, BaseVerifyRequest):
    pass


class CalendarResourcesServerConfig(BaseResourcesServerConfig):
    pass


class CalendarResourcesServer(SimpleResourcesServer):
    config: CalendarResourcesServerConfig

    # Tunable via env. Keep default under the routing_rm calendar timeout (15s).
    _VERIFY_TIMEOUT_S: ClassVar[float] = float(os.environ.get("CALENDAR_VERIFY_TIMEOUT_S", "10"))
    _VERIFY_POOL_WORKERS: ClassVar[int] = int(os.environ.get("CALENDAR_POOL_WORKERS", "4"))

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._verify_pool: Optional[cf.ProcessPoolExecutor] = None
        self._verify_pool_lock: Optional[asyncio.Lock] = None

    def _build_pool(self) -> cf.ProcessPoolExecutor:
        return cf.ProcessPoolExecutor(
            max_workers=self._VERIFY_POOL_WORKERS,
            mp_context=mp.get_context("spawn"),
        )

    def _ensure_pool(self) -> cf.ProcessPoolExecutor:
        if self._verify_pool is None:
            self._verify_pool = self._build_pool()
        if self._verify_pool_lock is None:
            self._verify_pool_lock = asyncio.Lock()
        return self._verify_pool

    def _recycle_pool(self) -> None:
        """SIGKILL all workers of the current pool and replace it.

        Called when a verify call times out — which for calendar almost
        always means the extract_json_list regex is wedged. SIGTERM won't
        reliably interrupt Python's re engine mid-match, so we SIGKILL.
        """
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
                logger.warning("calendar pool recycle: SIGKILL %s failed: %s", pid, exc)
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        logger.warning("calendar pool recycled after timeout (killed %d workers)", len(pids))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # End-to-end health probe. Sends a canned {input, expected_empty_state}
        # through the real verify pipeline. Must complete in <2s under normal
        # load. The watchdog in server_launcher.py uses this to detect wedged
        # event loops — /docs responds even when the verify pool is hung, so
        # /health is the only reliable signal.
        @app.get("/health")
        async def health():
            try:
                reward, reason = await self._grade_with_pool("[]", {})
                return {"status": "ok", "reward": reward, "reason": reason}
            except asyncio.TimeoutError:
                return {"status": "timeout"}
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

        return app

    async def _grade_with_pool(
        self, assistant_response: str, exp_cal_state: dict
    ) -> Tuple[float, str]:
        """Run the blocking grader inside a pool worker with a hard timeout.

        On timeout: SIGKILL workers, rebuild pool, return (0.0, 'timeout').
        Never raises to the caller — a timeout is a graded outcome, not a bug.
        """
        pool = self._ensure_pool()
        loop = asyncio.get_running_loop()
        try:
            fut = loop.run_in_executor(pool, _grade_in_worker, assistant_response, exp_cal_state)
            return await asyncio.wait_for(fut, timeout=self._VERIFY_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "calendar grade_assistant_response timeout (%.1fs) on response of length %d — recycling pool",
                self._VERIFY_TIMEOUT_S, len(assistant_response),
            )
            async with self._verify_pool_lock:
                self._recycle_pool()
            return 0.0, "timeout"
        except (cf.process.BrokenProcessPool, cf.CancelledError):
            logger.warning("calendar worker died — recycling pool")
            async with self._verify_pool_lock:
                self._recycle_pool()
            return 0.0, "worker_died"
        except Exception as exc:
            logger.warning("calendar grade error: %s", exc)
            return 0.0, f"grade_error:{type(exc).__name__}"

    async def verify(self, body: CalendarVerifyRequest) -> BaseVerifyResponse:
        # Extract the assistant's text response from the last output item.
        #
        # For reasoning models (e.g., with deepseek_r1 reasoning_parser), the output
        # structure is: [ReasoningItem, MessageItem] where:
        #   - ReasoningItem: has .reasoning attribute (thinking/CoT tokens)
        #   - MessageItem: has .content attribute (actual response text)
        #
        # The last item should be a MessageItem with .content, but if the model
        # hit the token limit while still thinking, the last item will be a
        # ReasoningItem without .content. In that case, we return reward=0.
        assistant_response = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                assistant_response = last_output.content[0].text

        if not assistant_response:
            return BaseVerifyResponse(**body.model_dump(), reward=0)

        reward, _reason = await self._grade_with_pool(assistant_response, body.exp_cal_state)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    CalendarResourcesServer.run_webserver()
