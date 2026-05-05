# Slime NaN-reward masking patch

Standalone patch that teaches slime to treat `float('nan')` rewards from a
`custom_rm` (specifically `slime_integration/routing_rm.py`) as a
**sample-level mask**, not as a value-filling hack at the custom_rm layer.

## Why

`routing_rm.py` returns `NaN` when a Gym grader times out or errors. Before
this patch, slime's `_post_process_rewards` would propagate NaN into
`torch.tensor(raw_rewards)`, producing NaN advantages, NaN policy loss, and
a crashed training job. The workaround was to mean-fill NaN at the router
layer — which prevents the crash but biases GRPO/GSPO group statistics when
only some rollouts in a prompt group fail.

With this patch:

1. A NaN reward in `raw_rewards` sets `sample.remove_sample = True`.
   slime's downstream `_convert_samples_to_train_data` already zeros the
   `loss_mask` for such samples, so the policy loss contribution is **zero**.
2. The NaN is replaced with the **group-local mean** of non-NaN peers in
   the same prompt group. A filled value equal to the group mean means
   the advantage (`reward − group_mean`) is exactly **0** for the filled
   sample. Group statistics for valid peers are unaffected.
3. If an entire group is NaN (grader fully down), fill with 0; the
   `remove_sample` flag still prevents gradient. That group produces no
   learning signal for that prompt — the correct behavior.

## Files

| File | Purpose |
|---|---|
| `slime_nan_mask.patch` | Unified diff: adds `slime/utils/nan_masking.py`, edits `slime/ray/rollout.py`, adds `tests/test_nan_reward_masking.py`. |
| `apply_on_pod.sh` | kubectl-based apply script for live P11 pods. |
| `README.md` | This file. |

## Apply locally

```bash
cd /root/slime            # or wherever slime is checked out
git apply --check /path/to/slime_nan_mask.patch   # dry run
git apply        /path/to/slime_nan_mask.patch
python -m pytest tests/test_nan_reward_masking.py -v
```

## Apply on a P11 training pod

```bash
# Auth first
bash ~/verify.sh

# Apply to a single pod
./apply_on_pod.sh shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-0-0

# Or all four nodes
for i in 0 1 2 3; do
    ./apply_on_pod.sh "shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-${i}-0"
done
```

The script:
- Copies the patch to `/tmp/` on each pod.
- Checks for prior application (idempotent — skips if already patched).
- Runs `git apply --check` before applying.
- Runs the new tests in-place to confirm.

## Roll back

```bash
# Local:
cd /root/slime && git apply --reverse /path/to/slime_nan_mask.patch

# On pod:
kubectl exec -n application-nonprod <pod> -- \
    bash -c 'cd /root/slime && git apply --reverse /tmp/slime_nan_mask.patch'
```

## Tests

16 unit tests covering:
- **Mark-remove-sample**: NaN samples get `remove_sample=True`, non-NaN don't.
- **Group-local fill**: fill value = mean of valid peers in same group (not batch mean).
- **No-leak invariant**: after fill, no NaN in raw_rewards or normalized rewards.
- **Group normalization integrity**: valid peers' advantages unaffected by a NaN peer; filled sample's advantage is exactly 0.
- **Variable group size fallback**: non-multiple group sizes fall back to batch mean (still safe via `remove_sample`).
- **No-NaN noop**: no mutations when there are no NaN values (the common fast path).
- **Integer / non-float inputs**: ignored by NaN detection.

Test file: `slime/tests/test_nan_reward_masking.py`.

## Companion patch on the Gym side

`slime_integration/routing_rm.py` (in the `Gym` repo) emits NaN rewards on
any grader failure. With this slime patch in place, `routing_batched_rm`
passes NaN through unchanged; previously it mean-filled at the router layer
and that biased group statistics in the partial-group-failure case. See
the `Stability contract (v3)` docstring in `routing_rm.py` for details.
