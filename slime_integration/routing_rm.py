"""Routing reward model for mixed-environment RLVR1 training.

Dispatches each sample to the correct NeMo-Gym resources server based on
``sample.metadata["env_type"]``.

Stability contract (v3, 2026-05-05):
  * ``_score_single`` returns ``float('nan')`` on ANY failure (timeout,
    HTTP non-200, connection error, payload build error, missing env_type,
    missing routing URL, invalid response body). This makes grader failures
    distinguishable from genuine wrong answers (reward=0.0).
  * ``routing_batched_rm`` passes NaN through to slime UNCHANGED.
    slime's ``_post_process_rewards`` (patched 2026-05-05) honors NaN as
    a sample-level mask: it sets ``sample.remove_sample = True`` — which
    zeros ``loss_mask`` downstream — and replaces NaN with the group-local
    mean so group normalization stays clean. The net effect is that a
    failed sample contributes exactly zero gradient, and neither the
    grader outage nor the NaN pollutes the policy-gradient math.
  * Per-env circuit breaker: after ``NEMOGYM_CIRCUIT_THRESHOLD`` consecutive
    failures for an env, subsequent calls short-circuit to NaN without
    hitting the server. The circuit re-opens after
    ``NEMOGYM_CIRCUIT_PROBE_S`` seconds (one probe call is attempted then).
  * Per-env timeouts tightened to match post-SIGKILL-pool expectations.
  * If any env has nan_rate > ``NEMOGYM_MAX_NAN_RATE`` for
    ``NEMOGYM_NAN_HALT_CONSECUTIVE`` consecutive batches, the RM raises
    ``RuntimeError`` to halt training before a sustained outage silently
    caps training quality. Disable via ``NEMOGYM_NAN_HALT_CONSECUTIVE=0``.

Usage with Slime::

    --custom-rm-path rlvr.recipe.routing_rm.routing_batched_rm
    --rm-url /path/to/routing_table.json   # or NEMOGYM_ROUTING_TABLE env var
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP session
# ---------------------------------------------------------------------------

_session: Optional[aiohttp.ClientSession] = None

# Chunk size for batched RM — process this many samples concurrently per chunk
BATCH_CHUNK_SIZE = int(os.environ.get("NEMOGYM_BATCH_CHUNK_SIZE", "128"))


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(limit=512, force_close=True),
        )
    return _session


# ---------------------------------------------------------------------------
# Default routing table
# ---------------------------------------------------------------------------

DEFAULT_ROUTING: Dict[str, str] = {
    "single_step_tool_use": "http://localhost:10001",
    "instruction_following": "http://localhost:10002",
    "code_gen": "http://localhost:10003",
    "math": "http://localhost:10004",
    "mcqa": "http://localhost:10005",
    "structured_outputs": "http://localhost:10006",
    "calendar": "http://localhost:10007",
    "reasoning_gym": "http://localhost:10008",
    "math_formal_lean": "http://localhost:10009",
    "workplace_assistant": "http://localhost:10010",
}

# Per-env client timeouts (seconds). Tightened post-SIGKILL-pool: graders that
# got subprocess isolation run in <10s, so a 15s client cap is comfortable.
# Stale long defaults keep us hammering a wedged server for minutes before
# the circuit breaker trips.
ENV_TIMEOUTS: Dict[str, float] = {
    "math": 15.0,
    "mcqa": 5.0,
    "structured_outputs": 5.0,
    "instruction_following": 10.0,
    "reasoning_gym": 10.0,
    "calendar": 15.0,
    "code_gen": 95.0,  # matches niletron's global_timeout_secs=90 + 5s buffer
    "math_formal_lean": 60.0,
    "workplace_assistant": 120.0,
    "single_step_tool_use": 30.0,
}

# Default fallback timeout used for envs not listed above. Kept small so new
# envs fail fast rather than burn a 60s wait before surfacing the bug.
DEFAULT_TIMEOUT_S: float = 30.0

_routing_table: Optional[Dict[str, str]] = None


def _load_routing_table(args) -> Dict[str, str]:
    global _routing_table
    if _routing_table is not None:
        return _routing_table

    # Try env var first
    env_val = os.environ.get("NEMOGYM_ROUTING_TABLE")
    if env_val:
        if os.path.isfile(env_val):
            with open(env_val) as f:
                _routing_table = json.load(f)
        else:
            _routing_table = json.loads(env_val)
        return _routing_table

    # Try args.rm_url as file path
    rm_url = getattr(args, "rm_url", None)
    if rm_url and os.path.isfile(rm_url):
        with open(rm_url) as f:
            _routing_table = json.load(f)
        return _routing_table

    _routing_table = DEFAULT_ROUTING
    return _routing_table


# ---------------------------------------------------------------------------
# Circuit breaker state (per-env)
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_THRESHOLD: int = int(os.environ.get("NEMOGYM_CIRCUIT_THRESHOLD", "10"))
CIRCUIT_BREAKER_PROBE_INTERVAL_S: float = float(os.environ.get("NEMOGYM_CIRCUIT_PROBE_S", "30"))
MAX_NAN_RATE_BEFORE_HALT: float = float(os.environ.get("NEMOGYM_MAX_NAN_RATE", "0.5"))
NAN_RATE_HALT_CONSECUTIVE: int = int(os.environ.get("NEMOGYM_NAN_HALT_CONSECUTIVE", "3"))

# Consecutive failures per env; reset on any success.
_env_consecutive_failures: Dict[str, int] = {}
# Wall-clock seconds until which the circuit stays open for each env.
_env_circuit_open_until: Dict[str, float] = {}
# Consecutive batches with nan_rate > MAX_NAN_RATE_BEFORE_HALT per env.
_env_consecutive_high_nan_batches: Dict[str, int] = {}


def _now() -> float:
    """Seam for tests to stub time.time()."""
    return time.time()


def _is_circuit_open(env_type: str) -> bool:
    t = _env_circuit_open_until.get(env_type)
    return bool(t is not None and _now() < t)


def _record_failure(env_type: str) -> None:
    n = _env_consecutive_failures.get(env_type, 0) + 1
    _env_consecutive_failures[env_type] = n
    if n >= CIRCUIT_BREAKER_THRESHOLD and not _is_circuit_open(env_type):
        _env_circuit_open_until[env_type] = _now() + CIRCUIT_BREAKER_PROBE_INTERVAL_S
        logger.error(
            "Circuit breaker OPEN for env=%s after %d consecutive failures; "
            "blocking /verify calls for %.0fs",
            env_type, n, CIRCUIT_BREAKER_PROBE_INTERVAL_S,
        )


def _record_success(env_type: str) -> None:
    if _env_consecutive_failures.get(env_type, 0) > 0:
        logger.info(
            "Circuit breaker RESET for env=%s after success (prev consecutive_failures=%d)",
            env_type, _env_consecutive_failures[env_type],
        )
    _env_consecutive_failures[env_type] = 0
    _env_circuit_open_until.pop(env_type, None)


def _reset_circuit_state() -> None:
    """Testing helper — clear all breaker state."""
    _env_consecutive_failures.clear()
    _env_circuit_open_until.clear()
    _env_consecutive_high_nan_batches.clear()


# ---------------------------------------------------------------------------
# Response wrapping
# ---------------------------------------------------------------------------


def _wrap_response(text: str) -> Dict[str, Any]:
    """Wrap plain text into NeMoGymResponse dict format."""
    # Strip chat template tokens (e.g. <|im_end|>, <|im_start|>) that Slime passes through
    # from the raw model generation — these break downstream JSON parsing in graders.
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    # Strip think blocks so graders see only the final answer, not chain-of-thought.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return {
        "id": f"resp_{uuid4().hex[:16]}",
        "created_at": 0.0,
        "model": "slime",
        "object": "response",
        "output": [
            {
                "id": f"msg_{uuid4().hex[:16]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _make_rcp(messages: List[Dict], tools: Optional[List] = None) -> Dict[str, Any]:
    """Build responses_create_params from Slime messages."""
    input_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        input_msgs.append({
            "role": role,
            "type": "message",
            "content": [{"type": "input_text", "text": content}],
        })
    rcp: Dict[str, Any] = {"input": input_msgs, "model": "slime"}
    if tools:
        rcp["tools"] = tools
    return rcp


# ---------------------------------------------------------------------------
# Per-environment verify payload builders
# ---------------------------------------------------------------------------


def _build_verify_payload(env_type: str, metadata: Dict, response_text: str, prompt: Any, sample) -> Dict[str, Any]:
    """Build the /verify request body for a specific environment."""
    # Build messages from prompt
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    tools = metadata.get("tools")
    rcp = _make_rcp(messages, tools)
    response = _wrap_response(response_text)

    payload: Dict[str, Any] = {
        "responses_create_params": rcp,
        "response": response,
    }

    if env_type == "single_step_tool_use":
        payload["expected_action"] = metadata.get("expected_action")

    elif env_type == "instruction_following":
        for key in ("instruction_id_list", "kwargs", "grading_mode", "prompt", "id"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "code_gen":
        vm = metadata.get("verifier_metadata", {})
        payload["verifier_metadata"] = vm

    elif env_type == "math":
        payload["question"] = metadata.get("question", "")
        payload["expected_answer"] = metadata.get("expected_answer", "")

    elif env_type == "mcqa":
        for key in ("expected_answer", "options", "grading_mode", "template_metadata", "uuid"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "structured_outputs":
        # schema_str/schema_type may be top-level in metadata or nested inside
        # verifier_metadata (depending on data conversion version).
        vm = metadata.get("verifier_metadata", {})
        for key in ("schema_str", "schema_type"):
            if key in metadata:
                payload[key] = metadata[key]
            elif key in vm:
                payload[key] = vm[key]

    elif env_type == "calendar":
        payload["exp_cal_state"] = metadata.get("exp_cal_state")

    elif env_type == "reasoning_gym":
        payload["question"] = metadata.get("question", "")
        payload["answer"] = metadata.get("answer")
        # reasoning_gym expects its own metadata dict
        payload["metadata"] = metadata.get("reasoning_gym_metadata", {})

    elif env_type == "math_formal_lean":
        for key in ("header", "formal_statement", "informal_prefix", "name"):
            if key in metadata:
                payload[key] = metadata[key]

    elif env_type == "workplace_assistant":
        for key in ("ground_truth", "id", "category", "environment_name"):
            if key in metadata:
                payload[key] = metadata[key]

    return payload


# ---------------------------------------------------------------------------
# Core RM functions
# ---------------------------------------------------------------------------


_NAN: float = float("nan")


async def _score_single(args, sample, routing: Dict[str, str]) -> float:
    """Score a single sample by routing to the correct resources server.

    Returns ``float('nan')`` on ANY failure (see module docstring). Callers must
    treat NaN as "grader could not score this sample" and either fill with
    group-mean (see ``routing_batched_rm``) or exclude from advantage
    computation.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_type = metadata.get("env_type") or metadata.get("rm_type") or ""

    if not env_type:
        logger.warning("Sample has no env_type or rm_type in metadata, returning NaN")
        return _NAN

    base_url = routing.get(env_type)
    if not base_url:
        logger.warning("No server configured for env_type=%s, returning NaN", env_type)
        return _NAN

    # Circuit breaker short-circuit: if the breaker is OPEN for this env,
    # skip the network call entirely and return NaN. This keeps the trainer
    # from hammering a known-dead server.
    if _is_circuit_open(env_type):
        return _NAN

    try:
        payload = _build_verify_payload(
            env_type=env_type,
            metadata=metadata,
            response_text=sample.response or "",
            prompt=sample.prompt,
            sample=sample,
        )
    except Exception as exc:
        logger.warning("Failed to build payload for env_type=%s: %s", env_type, exc)
        _record_failure(env_type)
        return _NAN

    timeout = ENV_TIMEOUTS.get(env_type, DEFAULT_TIMEOUT_S)
    url = f"{base_url}/verify"

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            session = await _get_session()
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Server %s returned %d: %s", url, resp.status, body[:200])
                    _record_failure(env_type)
                    return _NAN
                result = await resp.json()
                if "reward" not in result:
                    logger.warning("Server %s returned 200 but body has no 'reward' field", url)
                    _record_failure(env_type)
                    return _NAN
                try:
                    reward = float(result["reward"])
                except (TypeError, ValueError):
                    logger.warning("Server %s returned non-numeric reward=%r", url, result.get("reward"))
                    _record_failure(env_type)
                    return _NAN
                if math.isnan(reward) or math.isinf(reward):
                    logger.warning("Server %s returned nan/inf reward", url)
                    _record_failure(env_type)
                    return _NAN
                _record_success(env_type)
                return reward
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, ConnectionResetError) as exc:
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            logger.warning("Server %s disconnected after %d retries: %s", url, max_retries, exc)
            _record_failure(env_type)
            return _NAN
        except asyncio.TimeoutError:
            logger.warning("Timeout calling %s for env_type=%s", url, env_type)
            _record_failure(env_type)
            return _NAN
        except Exception as exc:
            logger.warning("Error calling %s: %s", url, exc)
            _record_failure(env_type)
            return _NAN
    _record_failure(env_type)
    return _NAN


def _compute_nan_stats(
    rewards: List[float], samples: List[Any]
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Return per-env NaN statistics without modifying rewards.

    With the slime patch (2026-05-05), NaN rewards are masked at the
    advantage-computation stage via ``sample.remove_sample``. Filling them
    here is no longer the router's job; it only needs to observe the rate
    for logging + auto-halt.

    Returns:
        (nan_rate_per_env, nan_count_per_env, total_count_per_env)
    """
    env_nan_count: Dict[str, int] = {}
    env_total: Dict[str, int] = {}
    for sample, r in zip(samples, rewards):
        env = (sample.metadata or {}).get("env_type", "unknown") if isinstance(
            getattr(sample, "metadata", None), dict
        ) else "unknown"
        env_total[env] = env_total.get(env, 0) + 1
        if isinstance(r, float) and math.isnan(r):
            env_nan_count[env] = env_nan_count.get(env, 0) + 1

    nan_rate = {
        env: (env_nan_count.get(env, 0) / env_total[env]) for env in env_total
    }
    return nan_rate, env_nan_count, env_total


# Backwards-compat alias for any external callers / old tests that import
# the previous name. Returns the same tuple shape as the old function but
# fills are NOT performed — the first element is the pass-through rewards.
def _fill_nan_rewards(
    rewards: List[float], samples: List[Any]
) -> Tuple[List[float], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Deprecated: slime now handles NaN masking. Kept for tests that still
    pin the old fill-with-env-mean behavior. Performs the fill in-process
    so those tests keep passing — production callers should use
    ``_compute_nan_stats`` and let slime do the masking."""
    env_valid: Dict[str, List[float]] = {}
    for sample, r in zip(samples, rewards):
        if not (isinstance(r, float) and math.isnan(r)):
            env = (sample.metadata or {}).get("env_type", "unknown") if isinstance(
                getattr(sample, "metadata", None), dict
            ) else "unknown"
            env_valid.setdefault(env, []).append(float(r))
    nan_rate, nan_counts, env_totals = _compute_nan_stats(rewards, samples)
    filled: List[float] = []
    for sample, r in zip(samples, rewards):
        if isinstance(r, float) and math.isnan(r):
            env = (sample.metadata or {}).get("env_type", "unknown") if isinstance(
                getattr(sample, "metadata", None), dict
            ) else "unknown"
            vals = env_valid.get(env, [])
            filled.append(sum(vals) / len(vals) if vals else 0.0)
        else:
            filled.append(float(r))
    return filled, nan_rate, nan_counts, env_totals


def _check_halt_condition(nan_rate: Dict[str, float]) -> None:
    """Update per-env consecutive-high-nan counters and raise if thresholds exceeded.

    Controlled by ``NEMOGYM_NAN_HALT_CONSECUTIVE`` (set to 0 to disable).
    """
    if NAN_RATE_HALT_CONSECUTIVE <= 0:
        return
    for env, rate in nan_rate.items():
        if rate > MAX_NAN_RATE_BEFORE_HALT:
            _env_consecutive_high_nan_batches[env] = _env_consecutive_high_nan_batches.get(env, 0) + 1
            if _env_consecutive_high_nan_batches[env] >= NAN_RATE_HALT_CONSECUTIVE:
                raise RuntimeError(
                    f"routing_rm: env={env} had nan_rate={rate:.1%} for "
                    f"{_env_consecutive_high_nan_batches[env]} consecutive batches "
                    f"(threshold={MAX_NAN_RATE_BEFORE_HALT:.0%}). Halting to "
                    "prevent gradient corruption — the grader is likely wedged. "
                    "Check per-env /health endpoints and the circuit-breaker log."
                )
        else:
            _env_consecutive_high_nan_batches[env] = 0


async def routing_rm(args, sample, **kwargs) -> float:
    """Single-sample routing RM for Slime's ``--custom-rm-path``.

    Returns 0.0 if the grader failed (can't mean-fill from a single sample).
    The batched path ``routing_batched_rm`` is strongly preferred because it
    can mean-fill NaN samples from other successes in the batch.
    """
    routing = _load_routing_table(args)
    reward = await _score_single(args, sample, routing)
    if isinstance(reward, float) and math.isnan(reward):
        logger.warning(
            "routing_rm single-sample path received NaN; returning 0.0 (use batched path to mean-fill)"
        )
        return 0.0
    return reward


async def routing_batched_rm(args, samples_or_sample, **kwargs):
    """Routing RM for Slime's ``--custom-rm-path``.

    Slime's ``custom_rm_path`` is called from BOTH:
    - ``async_rm(args, sample)`` — single Sample object
    - ``batched_async_rm(args, samples)`` — list of Sample objects

    This function auto-detects which case and handles both.
    """
    routing = _load_routing_table(args)

    # Single sample case (called from async_rm)
    if not isinstance(samples_or_sample, list):
        reward = await _score_single(args, samples_or_sample, routing)
        # slime's _post_process_rewards handles NaN natively (treats as mask).
        # Pass through unchanged.
        _log_single_trajectory(samples_or_sample, reward)
        return reward

    # Batched case (called from batched_async_rm).
    #
    # Contract with slime (see slime/ray/rollout.py:_post_process_rewards):
    #   - Samples with reward == NaN are treated as failed-grading and masked
    #     out of the policy loss via sample.remove_sample = True.
    #   - Group-local mean fill happens inside slime, using knowledge of
    #     n_samples_per_prompt that we don't natively have here.
    #   - This function just needs to forward NaN through unchanged.
    samples = samples_or_sample
    start = time.time()
    raw_rewards: List[float] = []
    for i in range(0, len(samples), BATCH_CHUNK_SIZE):
        chunk = samples[i : i + BATCH_CHUNK_SIZE]
        chunk_rewards = await asyncio.gather(*(
            _score_single(args, sample, routing) for sample in chunk
        ))
        raw_rewards.extend(chunk_rewards)
    elapsed = time.time() - start

    # Compute per-env NaN rate + counts for observability / auto-halt, WITHOUT
    # filling the rewards. slime handles the fill.
    nan_rate, nan_counts, env_totals = _compute_nan_stats(raw_rewards, samples)

    # Loud warnings for any env with elevated nan_rate
    for env, rate in sorted(nan_rate.items()):
        if rate > 0.1:
            logger.warning(
                "env=%s nan_rate=%.1f%% (%d/%d); circuit=%s",
                env, rate * 100, nan_counts.get(env, 0), env_totals.get(env, 0),
                "OPEN" if _is_circuit_open(env) else "closed",
            )

    # Auto-halt guard — raises RuntimeError if sustained high nan_rate.
    _check_halt_condition(nan_rate)

    # Per-env reward summary
    env_rewards: Dict[str, List[float]] = {}
    for sample, reward in zip(samples, raw_rewards):
        env = (sample.metadata or {}).get("env_type", "unknown") if isinstance(
            getattr(sample, "metadata", None), dict
        ) else "unknown"
        env_rewards.setdefault(env, []).append(reward)

    env_summary = {}
    for env, rs in sorted(env_rewards.items()):
        valid = [r for r in rs if not (isinstance(r, float) and math.isnan(r))]
        nan_ct = len(rs) - len(valid)
        mean = sum(valid) / len(valid) if valid else float("nan")
        pos = sum(1 for r in valid if r > 0)
        env_summary[env] = f"mean={mean:.3f} pos={pos}/{len(valid)} nan={nan_ct}"

    # Batch-wide mean for logging (excludes NaN).
    valid_all = [r for r in raw_rewards if not (isinstance(r, float) and math.isnan(r))]
    avg_valid = sum(valid_all) / len(valid_all) if valid_all else float("nan")
    logger.info(
        "Scored %d samples in %.1fs (%d chunks, avg_valid=%.3f, nan=%d) | per_env: %s",
        len(samples), elapsed, (len(samples) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE,
        avg_valid, len(raw_rewards) - len(valid_all), env_summary,
    )

    # Log trajectory to per-step JSONL file. NaN is recorded as null +
    # grader_failed flag so post-mortems can distinguish legitimate zero
    # rewards from masked failures.
    traj_dir = _get_traj_dir()
    _log_trajectories(samples, raw_rewards, traj_dir, raw_rewards=raw_rewards)

    return raw_rewards


_traj_counter = 0
_traj_rollout_id = 0
_single_traj_buffer: List = []
_single_traj_flush_size = 512  # Flush every 512 samples
_single_traj_rollout_count = 0  # Samples in current rollout


def _get_traj_dir() -> str:
    """Get trajectory directory from env var, organized by experiment name."""
    exp_name = os.environ.get("NEMOGYM_EXP_NAME", "default")
    base = os.environ.get("NEMOGYM_TRAJECTORY_DIR", "/shared/dev/shuowei/niletron/exp/trajectories")
    traj_dir = os.path.join(base, exp_name)
    os.makedirs(traj_dir, exist_ok=True)
    return traj_dir


def _is_refusal_quick(text: str) -> bool:
    lower = text[:300].lower()
    return any(p in lower for p in ["i cannot", "i can't", "sorry", "i'm unable", "i refuse", "cannot comply"])


def _get_think_length(text: str) -> int:
    blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    return sum(len(b) for b in blocks)


def _log_single_trajectory(sample, reward: float):
    """Buffer single-sample trajectory, flush when buffer is full."""
    global _single_traj_buffer, _single_traj_rollout_count
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    response_text = str(sample.response) if sample.response else ""
    _single_traj_buffer.append({
        "env_type": metadata.get("env_type", "unknown"),
        "reward": reward,
        "prompt_len": len(str(sample.prompt)) if sample.prompt else 0,
        "response_len": len(response_text),
        "prompt": str(sample.prompt) if sample.prompt else "",
        "response": response_text,
        "label": str(sample.label) if sample.label else None,
        "has_think_tag": "<think>" in response_text,
        "is_refusal": _is_refusal_quick(response_text),
        "think_block_length": _get_think_length(response_text),
    })
    _single_traj_rollout_count += 1
    if len(_single_traj_buffer) >= _single_traj_flush_size:
        _flush_single_trajectories()


def _flush_single_trajectories():
    """Flush buffered single trajectories to disk."""
    global _single_traj_buffer, _traj_rollout_id, _single_traj_rollout_count
    if not _single_traj_buffer:
        return
    traj_dir = _get_traj_dir()
    try:
        traj_path = os.path.join(traj_dir, f"rollout_{_traj_rollout_id:05d}.jsonl")
        with open(traj_path, "a") as f:
            for record in _single_traj_buffer:
                record["rollout_step"] = _traj_rollout_id
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Flushed %d trajectories to %s (rollout %d, total %d)",
                     len(_single_traj_buffer), traj_path, _traj_rollout_id, _single_traj_rollout_count)
    except Exception as exc:
        logger.warning("Failed to flush trajectories: %s", exc)
    _single_traj_buffer = []

    # Estimate rollout boundary: typical rollout = batch_size * n_samples (e.g. 192*16=3072)
    rollout_size = int(os.environ.get("NEMOGYM_ROLLOUT_SIZE", "3072"))
    if _single_traj_rollout_count >= rollout_size:
        _traj_rollout_id += 1
        _single_traj_rollout_count = 0


def _log_trajectories(samples, rewards: List[float], traj_dir: str, raw_rewards: Optional[List[float]] = None):
    """Save rollout trajectories to a per-step JSONL file.

    If raw_rewards is provided, each record includes ``raw_reward`` so
    post-mortems can distinguish genuine zeros from NaN-filled samples.
    """
    global _traj_counter
    try:
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, f"rollout_{_traj_counter:05d}.jsonl")
        with open(traj_path, "w") as f:
            for idx, (sample, reward) in enumerate(zip(samples, rewards)):
                metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
                env_type = metadata.get("env_type", "unknown")
                record = {
                    "rollout_step": _traj_counter,
                    "env_type": env_type,
                    "reward": reward,
                    "prompt_len": len(str(sample.prompt)) if sample.prompt else 0,
                    "response_len": len(str(sample.response)) if sample.response else 0,
                    "prompt": str(sample.prompt) if sample.prompt else "",
                    "response": str(sample.response) if sample.response else "",
                    "label": str(sample.label) if sample.label else None,
                }
                if raw_rewards is not None and idx < len(raw_rewards):
                    raw = raw_rewards[idx]
                    # JSON doesn't allow NaN; encode as null + a separate flag
                    if isinstance(raw, float) and math.isnan(raw):
                        record["raw_reward"] = None
                        record["grader_failed"] = True
                    else:
                        record["raw_reward"] = float(raw)
                        record["grader_failed"] = False
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d trajectories to %s", len(rewards), traj_path)
        _traj_counter += 1
    except Exception as exc:
        logger.warning("Failed to log trajectory: %s", exc)
