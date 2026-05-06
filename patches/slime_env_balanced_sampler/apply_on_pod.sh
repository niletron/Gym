#!/bin/bash
# Apply the slime env-balanced sampler patch to P11 training pod(s).
#
# Usage:
#   ./apply_on_pod.sh <pod-name> [pod-name ...]
#   ./apply_on_pod.sh shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-0-0
#
# Patches BOTH /root/slime AND /opt/slime on each pod. Idempotent:
# checks for the presence of the `env_sampling_weights` marker before
# applying, and skips if already in place. Tries strict git apply first,
# falls back to patch --fuzz=3 if the pod's slime has drifted.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pod-name> [pod-name ...]"
    exit 1
fi

NAMESPACE="${NAMESPACE:-application-nonprod}"
PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/slime_env_balanced_sampler.patch"
SLIME_DIRS="${SLIME_DIRS:-/root/slime /opt/slime}"
MARKER="env_sampling_weights"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    exit 1
fi

_apply_one_dir() {
    local pod="$1"
    local dir="$2"

    if ! kubectl exec -n "$NAMESPACE" "$pod" -- test -f "$dir/slime/rollout/data_source.py"; then
        echo "  [$dir] not present on pod — skipping"
        return 0
    fi

    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        grep -q "$MARKER" "$dir/slime/rollout/data_source.py" 2>/dev/null; then
        echo "  [$dir] already patched — skipping"
        return 0
    fi

    echo "  [$dir] strict git apply..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && git apply --check /tmp/slime_env_balanced_sampler.patch 2>/dev/null"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && git apply /tmp/slime_env_balanced_sampler.patch"
        echo "  [$dir] strict git apply OK"
        return 0
    fi

    echo "  [$dir] strict failed (version drift). Trying patch --fuzz=3..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && patch -p1 --dry-run --fuzz=3 < /tmp/slime_env_balanced_sampler.patch >/dev/null 2>&1"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && patch -p1 --fuzz=3 < /tmp/slime_env_balanced_sampler.patch"
        echo "  [$dir] fuzzy patch OK"
        return 0
    fi

    echo "  [$dir] BOTH STRATEGIES FAILED — pod version too far from patch baseline."
    echo "  [$dir] Current _post_process_rewards body on pod (first 30 lines):"
    kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "awk '/def get_samples/,/^    def /' '$dir/slime/rollout/data_source.py'" \
        | head -30 | sed 's/^/    /'
    return 1
}

for POD in "$@"; do
    echo "=== $POD ==="

    echo "[1/3] Copying patch to $POD:/tmp/slime_env_balanced_sampler.patch"
    kubectl cp -n "$NAMESPACE" "$PATCH_FILE" "$POD:/tmp/slime_env_balanced_sampler.patch"

    echo "[2/3] Applying to each slime checkout on the pod"
    failed=0
    for DIR in $SLIME_DIRS; do
        if ! _apply_one_dir "$POD" "$DIR"; then
            failed=1
        fi
    done
    if [[ $failed -eq 1 ]]; then
        echo "  → $POD partially patched. See errors above."
        continue
    fi

    echo "[3/3] Running tests in the first patched checkout"
    for DIR in $SLIME_DIRS; do
        if kubectl exec -n "$NAMESPACE" "$POD" -- test -f "$DIR/tests/test_env_balanced_sampling.py"; then
            kubectl exec -n "$NAMESPACE" "$POD" -- \
                bash -c "cd '$DIR' && python -m pytest tests/test_env_balanced_sampling.py -q 2>&1 | tail -5"
            break
        fi
    done

    echo "  → $POD patched successfully."
    echo
done

echo "Done."
echo
echo "Next: add to your training recipe's ROLLOUT_ARGS:"
echo "  --env-sampling-weights '{\"math\": 0.25, \"code_gen\": 0.25, \"reasoning_gym\": 0.15, \"mcqa\": 0.10, \"structured_outputs\": 0.10, \"instruction_following\": 0.10, \"calendar\": 0.05}'"
echo
echo "Roll back per dir:"
echo "  kubectl exec -n $NAMESPACE <pod> -- bash -c 'cd /root/slime && git apply --reverse /tmp/slime_env_balanced_sampler.patch'"
