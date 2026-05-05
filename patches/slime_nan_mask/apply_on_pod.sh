#!/bin/bash
# Apply the slime NaN-reward masking patch to P11 training pod(s).
#
# Usage:
#   ./apply_on_pod.sh <pod-name> [pod-name ...]
#   ./apply_on_pod.sh shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-0-0
#
# Pods ship TWO slime copies (/root/slime and /opt/slime). The training job's
# ray --working-dir typically points to /root/slime, but PYTHONPATH may
# include /opt/slime as well. This script patches BOTH by default; override
# with SLIME_DIRS="/root/slime" if you only want one.
#
# Slime versions on pods DRIFT from the local checkout — the context lines
# around _post_process_rewards may not match exactly. The script therefore:
#   1. First tries `git apply` (strictest).
#   2. Falls back to `patch -p1 --fuzz=3` (tolerates line-number drift and
#      minor context differences — works as long as the target function
#      still has the recognizable anchor lines).
#   3. If both fail, it prints a diff of what's currently in the pod around
#      the target function so you can regenerate a targeted patch.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pod-name> [pod-name ...]"
    exit 1
fi

NAMESPACE="${NAMESPACE:-application-nonprod}"
PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/slime_nan_mask.patch"
SLIME_DIRS="${SLIME_DIRS:-/root/slime /opt/slime}"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    exit 1
fi

_apply_one_dir() {
    local pod="$1"
    local dir="$2"

    # Probe: does the slime checkout actually exist at this path?
    if ! kubectl exec -n "$NAMESPACE" "$pod" -- test -f "$dir/slime/ray/rollout.py"; then
        echo "  [$dir] not present on pod — skipping"
        return 0
    fi

    # Probe: already patched?
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        grep -q "from slime.utils.nan_masking" "$dir/slime/ray/rollout.py" 2>/dev/null; then
        echo "  [$dir] already patched — skipping"
        return 0
    fi

    echo "  [$dir] strict git apply..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && git apply --check /tmp/slime_nan_mask.patch 2>/dev/null"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && git apply /tmp/slime_nan_mask.patch"
        echo "  [$dir] strict git apply OK"
        return 0
    fi

    echo "  [$dir] strict failed (expected due to version drift). Trying patch --fuzz=3..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && patch -p1 --dry-run --fuzz=3 < /tmp/slime_nan_mask.patch >/dev/null 2>&1"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && patch -p1 --fuzz=3 < /tmp/slime_nan_mask.patch"
        echo "  [$dir] fuzzy patch OK"
        return 0
    fi

    echo "  [$dir] BOTH STRATEGIES FAILED — pod version is too far from the patch baseline."
    echo "  [$dir] Dumping the current _post_process_rewards body for manual inspection:"
    kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "awk '/def _post_process_rewards/,/def _convert_samples_to_train_data/' '$dir/slime/ray/rollout.py'" \
        | head -40 | sed 's/^/    /'
    echo "  [$dir] Regenerate a targeted patch by running:"
    echo "        kubectl cp -n $NAMESPACE '$pod:$dir/slime/ray/rollout.py' /tmp/pod_rollout.py"
    echo "        diff -u /tmp/pod_rollout.py <(your-edited-copy) > targeted.patch"
    return 1
}

for POD in "$@"; do
    echo "=== $POD ==="

    echo "[1/3] Copying patch to $POD:/tmp/slime_nan_mask.patch"
    kubectl cp -n "$NAMESPACE" "$PATCH_FILE" "$POD:/tmp/slime_nan_mask.patch"

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
        if kubectl exec -n "$NAMESPACE" "$POD" -- test -f "$DIR/tests/test_nan_reward_masking.py"; then
            kubectl exec -n "$NAMESPACE" "$POD" -- \
                bash -c "cd '$DIR' && python -m pytest tests/test_nan_reward_masking.py -q 2>&1 | tail -5"
            break
        fi
    done

    echo "  → $POD patched successfully."
    echo
done

echo "Done."
echo "To roll back on a pod (per dir):"
echo "  kubectl exec -n $NAMESPACE <pod> -- \\"
echo "    bash -c 'cd /root/slime && (git apply --reverse /tmp/slime_nan_mask.patch || patch -p1 -R --fuzz=3 < /tmp/slime_nan_mask.patch)'"
