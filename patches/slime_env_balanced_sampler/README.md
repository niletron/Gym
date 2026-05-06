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
| `slime/utils/env_balanced_sampling.py` | NEW, stdlib-only helper (~620 lines; static + adaptive + `get_metrics()`) |
| `slime/rollout/data_source.py` | +~33 lines to wire the sampler (static + adaptive kwargs) into `RolloutDataSource` |
| `slime/utils/arguments.py` | +~95 lines to add CLI args (`--env-sampling-weights`, `--env-sampling-metadata-key`, and six `--env-sampling-*` adaptive knobs) |
| `slime/ray/rollout.py` | +~65 lines: reward-feedback hook inside `_post_process_rewards` + WandB metrics merge inside `_log_rollout_data` (both no-op when no sampler is active) |
| `tests/test_env_balanced_sampling.py` | NEW, 39 unit tests (18 static + 11 adaptive + 10 metrics) |

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

39 unit tests (18 static + 11 adaptive + 10 metrics) covering: draw
correctness, bucket cycling, error cases, reproducibility, state_dict
round-trip, an RLVR1_v2-style 192-prompt scenario, warmup semantics,
EMA smoothing, floor/ceiling clamping (water-filling projection),
refresh interval, thread safety, the module-level active-sampler
registry, and the `get_metrics()` dict shape (all-float values, all
envs present, cumulative counts, initial-weight immutability across
adaptive refreshes, thread-safe concurrent `draw`+`update_rewards`+
`get_metrics`).

```bash
python -m pytest tests/test_env_balanced_sampling.py -v
```

Runs in ~0.3s.

## Adaptive mode (experimental)

Default is **static** — the sampler ships static mode behavior and is a
drop-in replacement for the older patch. Adaptive mode is strictly
opt-in via `--env-sampling-adaptive`.

### What the feedback loop does

1. After each rollout, `_post_process_rewards` buckets the batch's per-
   sample rewards by `sample.metadata["env_type"]` and calls
   `sampler.update_rewards({env: mean_reward_for_env}, rollout_id=...)`.
2. The sampler updates an exponential moving average of each env's
   reward: `ema[env] = alpha * ema[env] + (1-alpha) * reward`.
3. Every `refresh_interval` rollouts past `warmup`, the sampler recomputes
   its weights: `raw_weight[env] = ema[env] * (1 - ema[env])` (binary-
   reward variance proxy). Weights are clamped to `[floor, ceiling]` via
   a water-filling projection (handles multi-env binding correctly) and
   renormalized to sum to 1.

The net effect: saturated envs (pass rate ≈ 1) have variance ≈ 0 and get
clamped down to the floor; envs in the learnable middle (pass rate ≈
0.5) have variance ≈ 0.25 and absorb the extra mass.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--env-sampling-adaptive` | `False` | Turn on adaptive re-weighting. |
| `--env-sampling-ema-alpha` | `0.9` | Smoothing factor (higher = slower reaction). |
| `--env-sampling-refresh-interval` | `10` | Rollouts between re-weight recomputations. |
| `--env-sampling-weight-floor` | `0.02` | Minimum fraction any env can hold (prevents starvation). |
| `--env-sampling-weight-ceiling` | `0.40` | Maximum fraction any env can hold (prevents monopoly). |
| `--env-sampling-warmup` | `5` | Rollouts of static behavior before adaptive kicks in. |

Existing `--env-sampling-weights` are now treated as **initial** weights:
they are the static behavior during warmup and the fallback for envs
with no EMA signal yet. When adaptive is off, they stay static exactly
as before.

### WARNING: fix math grader first

Adaptive mode chases whatever the reward signal says is hard. If a
grader has a false-positive exploit (e.g., `\\boxed` multi-box exploit
in the math grader), adaptive mode will treat the exploited env as
*saturated*, down-sample it, and direct compute toward genuinely-hard
envs — which could be correct OR could be hiding the regression. Run
with `--env-sampling-adaptive=false` until math grading bugs are fixed
and validated. See `scripts/verify_grading_bugs.py` for the current
state.

### Recommended first-try settings

1. **Baseline first.** Run 20 rollouts with `--env-sampling-adaptive=false`
   (static mode) at the weights you shipped last run. Capture per-env
   reward curves in WandB. This is your reference.
2. **Then enable adaptive** with conservative settings:

```bash
ROLLOUT_ARGS=(
    ...
    --env-sampling-weights '{"math": 0.25, "code_gen": 0.25, "reasoning_gym": 0.15, "mcqa": 0.10, "structured_outputs": 0.10, "instruction_following": 0.10, "calendar": 0.05}'
    --env-sampling-adaptive
    --env-sampling-ema-alpha 0.9
    --env-sampling-refresh-interval 10
    --env-sampling-weight-floor 0.03
    --env-sampling-weight-ceiling 0.35
    --env-sampling-warmup 10
)
```

The defaults in this patch are slightly more aggressive
(`floor=0.02, ceiling=0.40, warmup=5`). The recommended first-try
settings above are more conservative — higher warmup, tighter ceiling —
to give EMAs time to stabilize before any env gets down-sampled.

### Thread safety

`update_rewards` may be called concurrently with `draw` (e.g., Ray
actor calling `update_rewards` from one thread while another thread
calls `draw`). The sampler protects its mutable state (EMAs, rollout
counter, live weights dict) with a `threading.Lock`. The per-env shuffle
order is untouched by adaptive mode and remains single-owner.

See `TestAdaptiveMode::test_update_rewards_thread_safe` for the
concurrent-execution smoke test.

## WandB metrics

Every rollout emits sampler telemetry to WandB under the `env_sampler/`
prefix. The hook point is `_log_rollout_data` in `slime/ray/rollout.py` —
the sampler's `get_metrics()` dict is merged into the existing rollout
`log_dict` right before `logging_utils.log(args, log_dict, ...)`. If no
sampler is active (user didn't pass `--env-sampling-weights`) the merge
is a no-op — no keys are emitted and no exception is raised.

### Metric reference

| Key pattern | Meaning |
|---|---|
| `env_sampler/weight/<env>` | Current weight per env. Sums to 1.0. In static mode this is constant; in adaptive mode it drifts as EMAs update. |
| `env_sampler/initial_weight/<env>` | User's original `--env-sampling-weights` input, normalized to sum to 1. **Never changes during the run** — use it as a reference line to see how far adaptive has drifted. |
| `env_sampler/draw_ratio/<env>` | Fraction of the last rollout batch that was this env. Matches `weight/<env>` within largest-remainder rounding. |
| `env_sampler/draw_count/<env>` | Integer count of this env in the last rollout batch. `sum(draw_count/*) == rollout_batch_size`. |
| `env_sampler/cumulative_ratio/<env>` | Fraction of all samples drawn since sampler init. Useful for seeing the lifetime compute-per-env distribution. |
| `env_sampler/ema_reward/<env>` | Pass-rate EMA per env (0.0 in static mode or before any signal). |
| `env_sampler/variance/<env>` | `p * (1 - p)` variance proxy — the raw signal adaptive mode uses to re-weight. |
| `env_sampler/rollout_counter` | Number of `update_rewards()` calls received (= number of rollouts in adaptive mode). |
| `env_sampler/refreshes` | Number of adaptive re-weight events fired so far. Stays at 0 in static mode. |
| `env_sampler/total_draws` | Number of `draw()` calls. |
| `env_sampler/batch_size` | Size of the most recent `draw()` batch. |
| `env_sampler/adaptive` | 1.0 if adaptive mode, 0.0 otherwise (constant). |

All values are `float` — wandb rejects ints/bools/strings for scalar
metrics. Keys are flat strings; the `/` characters inside are wandb's
group separator (so everything nests cleanly in the dashboard sidebar).

### Recommended dashboard panels

For a new run, add these panels to a WandB workspace:

1. **Line chart: `env_sampler/weight/*`** — one line per env. Lets you
   visually watch adaptive re-weighting pull compute toward high-variance
   envs over training. Include `env_sampler/initial_weight/*` as a faded
   reference.
2. **Line chart: `env_sampler/ema_reward/*`** — one line per env. The
   raw signal adaptive is chasing. If you see an env's EMA approach 1.0
   (saturated), expect its weight to shrink toward the floor.
3. **Line chart: `env_sampler/variance/*`** — one line per env. The
   computed `p*(1-p)`. Cross-check against the weight plot.
4. **Stacked area: `env_sampler/draw_ratio/*`** — the actual mix
   going into each rollout. In static mode the stacks are flat bars; in
   adaptive mode they shift. Confirms adaptive is actually reaching the
   sampler (vs. some other bug).
5. **Single-number: `env_sampler/refreshes`** — sanity check that
   adaptive has fired the expected number of times given `refresh_interval`
   and total rollouts.

### Graceful degradation

* No `--env-sampling-weights` → `get_active_sampler()` returns `None` →
  the hook short-circuits. No `env_sampler/*` keys appear on the
  dashboard.
* Static mode → `env_sampler/ema_reward/*` and `env_sampler/variance/*`
  are emitted as `0.0` (not omitted), so if you flip modes mid-run the
  dashboard stays continuous.
* Instrumentation bug → the metrics merge is wrapped in `try/except`
  inside `_log_rollout_data`; a crash there gets `logger.exception`'d
  but the rollout proceeds.
