# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-isolated grader pool for resources servers.

This module packages the SIGKILL-capable subprocess-pool pattern that was
first deployed in ``resources_servers/math_with_judge/app.py`` (commit
``b3c0edf``) and is needed by any grader that:

* calls C-extension code where SIGALRM cannot preempt (sympy, numpy linalg,
  pandas-via-C),
* runs regex on untrusted model output (catastrophic backtracking),
* parses deeply-nested JSON schemas,
* shells out to binaries,
* or otherwise can block longer than the client's HTTP timeout.

Why processes, not threads: Python's GIL prevents thread-level preemption
and SIGALRM doesn't work across C-extension frames. Only ``os.kill(pid,
SIGKILL)`` reliably stops a wedged worker.

Why spawn, not fork: fork inherits the parent's event loop and file
descriptors, which frequently deadlocks under asyncio. Spawn is clean.

Incidents this prevents (from the RLVR1_v2 analysis):
  * math outage at step ~375: 23,252 timeouts, sympy recursion wedged the
    math server for 4h43m.
  * calendar outage in _u5j70izc_resume: 134,735 timeouts over 54h,
    catastrophic regex backtracking in ``extract_json_list``.

Usage (in a resources server's ``app.py``)::

    from nemo_gym.grader_pool import GraderPool

    def _my_grade_worker(arg1, arg2):
        # Runs in a worker process. Must be module-level (picklable).
        return float_reward, metadata_dict

    class MyServer(SimpleResourcesServer):
        _POOL: ClassVar[GraderPool] = GraderPool(
            name="my_grader",
            timeout_s_env="MY_GRADER_TIMEOUT_S",
            workers_env="MY_GRADER_POOL_WORKERS",
            default_timeout_s=10.0,
        )

        def model_post_init(self, context):
            super().model_post_init(context)
            self._POOL.bind(self)

        async def verify(self, body):
            reward, meta = await self._POOL.run(_my_grade_worker, arg1, arg2)
            return MyVerifyResponse(**body.model_dump(), reward=reward)

Tests should set ``<NAME>_TIMEOUT_S`` and ``<NAME>_POOL_WORKERS`` via env.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import signal
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


class GraderPool:
    """Process pool with SIGKILL-on-timeout recycle.

    Instance state (``_pool``, ``_lock``) is per-server; call ``bind(self)``
    from ``model_post_init`` so each server gets its own pool.
    """

    def __init__(
        self,
        name: str,
        timeout_s_env: str,
        workers_env: str,
        default_timeout_s: float = 10.0,
        default_workers: int = 4,
        initializer: Optional[Callable[[], None]] = None,
    ):
        self.name = name
        self._timeout_s = float(os.environ.get(timeout_s_env, str(default_timeout_s)))
        self._workers = int(os.environ.get(workers_env, str(default_workers)))
        self._initializer = initializer

    def bind(self, server_instance: Any) -> None:
        """Attach fresh pool state to a server instance. Call once per server."""
        # Attributes are stored on the server so each server has an isolated
        # pool even when multiple servers share the same GraderPool instance.
        object.__setattr__(server_instance, f"__pool_{self.name}", None)
        object.__setattr__(server_instance, f"__pool_lock_{self.name}", None)

    def _build_pool(self) -> cf.ProcessPoolExecutor:
        kwargs: dict = {
            "max_workers": self._workers,
            "mp_context": mp.get_context("spawn"),
        }
        if self._initializer is not None:
            kwargs["initializer"] = self._initializer
        return cf.ProcessPoolExecutor(**kwargs)

    def _ensure(self, server: Any) -> cf.ProcessPoolExecutor:
        pool = getattr(server, f"__pool_{self.name}", None)
        if pool is None:
            pool = self._build_pool()
            object.__setattr__(server, f"__pool_{self.name}", pool)
        lock = getattr(server, f"__pool_lock_{self.name}", None)
        if lock is None:
            object.__setattr__(server, f"__pool_lock_{self.name}", asyncio.Lock())
        return pool

    def _recycle(self, server: Any) -> None:
        """SIGKILL all workers and replace the pool. Call under the lock."""
        old = getattr(server, f"__pool_{self.name}", None)
        new = self._build_pool()
        object.__setattr__(server, f"__pool_{self.name}", new)
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
                logger.warning("%s pool recycle: SIGKILL %s failed: %s", self.name, pid, exc)
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        logger.warning("%s pool recycled after timeout (killed %d workers)", self.name, len(pids))

    async def run(
        self, server: Any, fn: Callable, *args, timeout_s: Optional[float] = None, **kwargs
    ) -> Any:
        """Run ``fn(*args, **kwargs)`` in a worker with hard timeout.

        Returns whatever ``fn`` returns. On timeout or worker-death, raises
        ``asyncio.TimeoutError`` or ``cf.process.BrokenProcessPool`` so the
        caller decides what reward to return. The pool is recycled before
        the exception propagates.
        """
        if kwargs:
            # ProcessPoolExecutor.submit doesn't forward kwargs through
            # run_in_executor in a clean way. Wrap instead.
            from functools import partial
            fn = partial(fn, **kwargs)

        t = self._timeout_s if timeout_s is None else timeout_s
        pool = self._ensure(server)
        lock = getattr(server, f"__pool_lock_{self.name}")
        loop = asyncio.get_running_loop()
        try:
            fut = loop.run_in_executor(pool, fn, *args)
            return await asyncio.wait_for(fut, timeout=t)
        except asyncio.TimeoutError:
            logger.warning(
                "%s grader timeout (%.1fs) — recycling pool", self.name, t,
            )
            async with lock:
                self._recycle(server)
            raise
        except (cf.process.BrokenProcessPool, cf.CancelledError):
            logger.warning("%s worker died — recycling pool", self.name)
            async with lock:
                self._recycle(server)
            raise

    async def run_or_zero(
        self,
        server: Any,
        fn: Callable,
        *args,
        timeout_s: Optional[float] = None,
        zero_value: Any = 0.0,
        **kwargs,
    ) -> Tuple[Any, str]:
        """Convenience: run ``fn`` in a worker; on any failure return
        ``(zero_value, reason_str)``. Use this when the caller just wants
        "grade or fail-closed-to-zero" semantics.
        """
        try:
            result = await self.run(server, fn, *args, timeout_s=timeout_s, **kwargs)
            return result, "ok"
        except asyncio.TimeoutError:
            return zero_value, "timeout"
        except cf.process.BrokenProcessPool:
            return zero_value, "worker_died"
        except cf.CancelledError:
            return zero_value, "cancelled"
        except Exception as exc:
            logger.warning("%s grader error: %s", self.name, exc)
            return zero_value, f"grade_error:{type(exc).__name__}"
