# Math grader: keep only last `\boxed{}` before math_verify

Closes a reward-hacking exploit in the math resources server where a
model can flood wrong `\boxed{}` entries around a correct one and
still receive reward=1.0.

## The exploit

`math_verify`'s parser set-merges **all** `\boxed{}` occurrences in a
response into a single candidate set (see
`latex2sympy2_extended` → `FiniteSet`). The grader then accepts a
prediction if *any* extracted candidate matches gold. In certain
shapes (sets vs scalars, tuple orders, equivalence under simplify),
that set-merge yields spurious matches — and even when it doesn't,
noisy boxes pollute extraction and occasionally flip a false-negative
parse into a false-positive.

On the currently-running RLVR1_v2 run, the fraction of math reward=1
samples that exhibit multi-box output grew from **12.4% → 39.4%**
over the last few hundred iterations — a classic reward-hacking
signature: the model found a shortcut and is exploiting it harder
every step.

Analysis:
[`/home/shuowei/niletron/rlvr/analyze/qwen3-30B-rlvr1_v2-3sv29n1u_resume_v2_ly38bd/05_math_codegen_deep.md`](file:///home/shuowei/niletron/rlvr/analyze/qwen3-30B-rlvr1_v2-3sv29n1u_resume_v2_ly38bd/05_math_codegen_deep.md)

## The fix

All standard math-reasoning prompts instruct "put your final answer
in `\boxed{...}`". The model's FINAL answer convention is the
**last** `\boxed{}`. So before handing the response string to
`math_verify`, we strip every `\boxed{...}` except the last.

Concretely, we add a pure helper `_keep_only_last_boxed(text)` that:

1. Walks the string, finding every `\boxed{` occurrence.
2. For each, uses a brace-depth counter (handling nested `{}` and
   escaped `\{ \}`) to find the matching closing `}` and record the
   full `[start, end)` span.
3. If `<= 1` spans exist, returns the string unchanged.
4. Otherwise splices out all but the last span, preserving the
   surrounding text.

It's called inside `_verify_in_worker` immediately before the
`_WORKER_VERIFIER([gold], [generated])` call, gated by env var
`MATH_KEEP_ONLY_LAST_BOX` (default `"1"` — on by default, opt-out).

### What it does NOT touch

- The SIGKILL subprocess pool (`_recycle_pool`, `_ensure_pool`,
  timeout handling) — untouched.
- `/health` endpoint — untouched.
- The judge-fallback path (`_verify_answer_with_judge`) — untouched.
- The `math_verify` library itself — we preprocess our input
  string; we do NOT monkey-patch or vendor-fork math_verify.
- Gold (expected answer) — only the generated answer is preprocessed.

## Files changed

| File | Change |
|---|---|
| `resources_servers/math_with_judge/app.py` | +~75 lines: `_keep_only_last_boxed` helper + one flag-gated call site in `_verify_in_worker` |
| `resources_servers/math_with_judge/tests/test_last_box_only.py` | NEW: 26 unit + integration tests |

## Apply locally

```bash
cd /home/shuowei/Gym
git apply --check patches/math_last_box_only/math_last_box_only.patch    # dry-run
git apply        patches/math_last_box_only/math_last_box_only.patch
python -m pytest resources_servers/math_with_judge/tests/test_last_box_only.py -v
```

## Apply on a P11 training pod

The math server on the training cluster lives at
`/shared/dev/shuowei/niletron/Gym/resources_servers/math_with_judge/app.py`
(the active Gym checkout driven by the training recipe — NOT
`/root/slime` which is a different repo entirely).

```bash
bash ~/verify.sh    # refresh kubectl auth

# Patch just the math server pod (pod 3 in RLVR1_v2 topology):
./apply_on_pod.sh shuowei-qwen35-slime-rlvr-<suffix>-startraycluster-0-3

# Or all pods if multiple checkouts:
for i in 0 1 2 3; do
  ./apply_on_pod.sh "shuowei-qwen35-slime-rlvr-<suffix>-startraycluster-0-${i}"
done
```

The apply script is idempotent (checks for the `_keep_only_last_boxed`
marker) and tries strict `git apply` first, falling back to
`patch --fuzz=3` on version drift.

### Restarting the math server after applying

**The apply script does NOT restart the server** — that's too easy
to misfire. After applying, restart it manually:

```bash
# Find the existing launcher for math_with_judge on pod 3 (10.209.87.44):
kubectl exec -n application-nonprod <pod> -- \
    bash -c "ps auxf | grep -E 'server_launcher|math_with_judge' | grep -v grep"

# SIGTERM the math server_launcher process (safe — it's a child of
# start_ray_cluster, not PID 1):
kubectl exec -n application-nonprod <pod> -- kill -TERM <PID>

# Re-launch with the same env vars the training recipe uses.
# (See the training recipe or slime Ray job submit args for the
# exact command; do NOT invent env vars.)
```

## Roll back

```bash
# Local:
cd /home/shuowei/Gym && git apply --reverse patches/math_last_box_only/math_last_box_only.patch

# On pod:
kubectl exec -n application-nonprod <pod> -- \
    bash -c 'cd /shared/dev/shuowei/niletron/Gym && git apply --reverse /tmp/math_last_box_only.patch'

# Then restart the math server (same manual step as above).
```

## Escape hatch: disable at runtime

For A/B testing or emergency rollback without re-deploying:

```bash
# Launch the math server with the flag off (old set-merge behavior):
MATH_KEEP_ONLY_LAST_BOX=0 ng_run "+config_paths=[...]"
```

Default is `"1"` (on). Any value other than `"0"` keeps it on.

## Verify on a running pod

Before and after applying, exercise the exploit path with a crafted
`/verify` POST:

```bash
# Exploit payload: correct answer "42" is in an EARLIER box, last box is wrong.
# Pre-fix behavior (flag=0 or unpatched): sometimes reward=1 (exploit wins).
# Post-fix behavior (flag=1, default): always reward=0 (only last box seen).

PAYLOAD=$(cat <<'JSON'
{
  "task_index": 0,
  "question": "what is 6*7?",
  "expected_answer": "42",
  "response": {
    "id": "test", "created_at": 0, "model": "m", "object": "response",
    "parallel_tool_calls": false, "tool_choice": "none", "tools": [],
    "output": [{
      "type": "message", "role": "assistant", "status": "completed", "id": "m",
      "content": [{"type": "output_text", "text":
        "Working: \\boxed{42}. Actually \\boxed{7}.", "annotations": []}]
    }]
  },
  "responses_create_params": {"input": []}
}
JSON
)

curl -sX POST http://<pod-ip>:<math-port>/verify \
     -H 'content-type: application/json' -d "$PAYLOAD" | jq .reward
```

With the fix active, this MUST return `0.0`.

## Tests

26 tests in `tests/test_last_box_only.py`:

**Unit tests on `_keep_only_last_boxed` (14)**: empty/no-box/single-box
no-ops; two/many-box stripping; nested braces (`\frac{1}{2}`); deeply
nested; escaped braces (`\{`, `\}`); unclosed brace (graceful —
returns input unchanged); multiline; surrounding text preserved.

**Integration via `_verify_in_worker` (8)**: single-box correct/wrong;
multi-box with correct last and wrong earlier (must reward=1);
**multi-box with wrong last and correct earlier (the exploit —
MUST reward=0)**; flood exploit variants; nested box in exploit.

**Feature flag (3)**: `=0` disables preprocessing (all boxes reach
math_verify unmodified); `=1` enables (only last box reaches);
unset → default on.

**Idempotence (2)**: applying the helper twice equals once.

Runs in ~2.7s.

## Why this is opt-out (on by default)

This is a bug fix, not a feature. The set-merge-of-all-boxes
behavior is never what a training run wants: it rewards noise and
creates a direct incentive to game the grader. Flipping the default
on means new server starts are safe without any recipe change.
Existing recipes can opt-out with `MATH_KEEP_ONLY_LAST_BOX=0` if
they need the old behavior for regression comparison.

## Edge case decision: unclosed boxes

If a response contains an unclosed `\boxed{` (no matching `}`), the
helper leaves the string **unchanged** rather than attempting a
dangerous truncation. Rationale: we can't know where the intended
end of the box is, and any guess risks mangling surrounding text.
`math_verify` already handles malformed LaTeX fine (it just fails to
extract), so the model gets reward=0 for malformed output — which is
the right outcome.

If 1 closed box + 1 unclosed fragment, we still return unchanged
(only 1 complete span → nothing to strip). This is intentional.
