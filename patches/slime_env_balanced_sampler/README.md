# Slime env-balanced rollout sampler patch

Adds a `--env-sampling-weights` flag that forces each rollout batch to
contain configured per-env fractions, regardless of how the underlying
prompt dataset is distributed.

## Why

The RLVR1_v2 dataset is heavily skewed toward `instruction_following`
(~33% of prompts). Because GRPO/GSPO advantages are mean-centered
*within* a prompt group, a sample contributes to gradient roughly in
proportion to `p(1-p)` where p is the within-group success rate:

| env | pass rate today | variance `p(1-p)` | gradient value per sample |
|---|---:|---:|---|
| instruction_following | 0.89 | 0.098 | low — already saturated |
| structured_outputs | 0.87 | 0.113 | low |
| calendar | 0.67 | 0.221 | good |
| mcqa | 0.76 | 0.182 | good |
| reasoning_gym | 0.66 | 0.224 | good |
| math | 0.39 | 0.238 | very good — most informative |
| code_gen | 0.23 | 0.177 | good |

With uniform sampling, ~33% of batch compute goes to IF samples whose
gradient contribution is ~4× smaller than math samples. Rebalancing
toward math/code_gen doubles the useful signal per rollout step.

## How it works

Instead of drawing `num_samples` prompts as a contiguous slice from the
shuffled dataset, the patched `RolloutDataSource.get_samples(N)` draws
from per-env buckets independently. Per-env counts are computed by the
largest-remainder method so `len(batch) == N` exactly.

Each env bucket has its own shuffle order and offset pointer. When a
bucket's offset reaches the end, it reshuffles with replacement-cycling
semantics — so a rare env whose target count exceeds its bucket size is
reused across epochs.

See [`slime/utils/env_balanced_sampling.py`](../slime/utils/env_balanced_sampling.py) for the full contract.

## Files changed

| File | Change |
|---|---|
| `slime/utils/env_balanced_sampling.py` | NEW, stdlib-only helper (~190 lines) |
| `slime/rollout/data_source.py` | +~20 lines to wire the sampler into `RolloutDataSource` |
| `slime/utils/arguments.py` | +~30 lines to add `--env-sampling-weights` and `--env-sampling-metadata-key` CLI args |
| `tests/test_env_balanced_sampling.py` | NEW, 18 unit tests |

## Apply locally

```bash
cd /root/slime   # or wherever slime is checked out
git apply --check /path/to/slime_env_balanced_sampler.patch   # dry-run
git apply        /path/to/slime_env_balanced_sampler.patch
python -m pytest tests/test_env_balanced_sampling.py -v
```

## Apply on a P11 training pod

```bash
bash ~/verify.sh    # refresh kubectl auth first
./apply_on_pod.sh shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-0-0
# Or all four nodes:
for i in 0 1 2 3; do
  ./apply_on_pod.sh "shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-${i}-0"
done
```

The apply script is idempotent and patches both `/root/slime` and
`/opt/slime` if present on the pod.

## Roll back

```bash
# Local:
cd /root/slime && git apply --reverse /path/to/slime_env_balanced_sampler.patch

# On pod:
kubectl exec -n application-nonprod <pod> -- \
    bash -c 'cd /root/slime && git apply --reverse /tmp/slime_env_balanced_sampler.patch'
```

## Recipe changes needed to activate

Add to the `ray job submit` args in your training recipe:

```bash
ROLLOUT_ARGS=(
    ...
    --env-sampling-weights '{"math": 0.25, "code_gen": 0.25, "reasoning_gym": 0.15, "mcqa": 0.10, "structured_outputs": 0.10, "instruction_following": 0.10, "calendar": 0.05}'
)
```

The default `--env-sampling-metadata-key=env_type` matches the
routing_rm convention, so no change needed there.

**Important**: the existing `--rollout-shuffle` flag still has effect
on the initial dataset shuffle. Leave it on — each env's bucket is
sampled in its own shuffled order derived from `--rollout-seed`.

## Behavior when weights reference a missing env

Init fails fast with a clear error message. Prevents silently drawing
short batches because a typo in the weights dict mentions an env that
has zero samples.

## Behavior when dataset has an env not in weights

By default those samples are dropped with a WARNING log. Set
`"__other__": <weight>` in the weights dict to include them in the
other-env bucket at that weight:

```bash
--env-sampling-weights '{"math": 0.5, "__other__": 0.5}'
```

## Tests

18 unit tests covering: draw correctness, bucket cycling, error cases,
reproducibility, state_dict round-trip, and an RLVR1_v2-style scenario
that validates the 192-prompt rollout batch matches target fractions.

```bash
python -m pytest tests/test_env_balanced_sampling.py -v
```

Runs in ~0.2s.
