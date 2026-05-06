#!/bin/bash
# Apply the math-grader last-box-only patch to P11 training pod(s).
#
# Usage:
#   ./apply_on_pod.sh <pod-name> [pod-name ...]
#   ./apply_on_pod.sh shuowei-qwen35-slime-rlvr-f55sm0xgt3fp5c-startraycluster-0-3
#
# Patches /shared/dev/shuowei/niletron/Gym on each pod (the active Gym
# checkout driven by the training recipe — NOT /root/slime). Idempotent:
# checks for the `_keep_only_last_boxed` marker before applying, and
# skips if already in place. Tries strict git apply first, falls back to
# patch --fuzz=3 if the pod's Gym has drifted.
#
# IMPORTANT: This script does NOT restart the math server. After
# applying, restart the server_launcher process for math_with_judge
# manually. See README.md for the recipe.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pod-name> [pod-name ...]"
    exit 1
fi

NAMESPACE="${NAMESPACE:-application-nonprod}"
PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/math_last_box_only.patch"
GYM_DIRS="${GYM_DIRS:-/shared/dev/shuowei/niletron/Gym}"
MARKER="_keep_only_last_boxed"
TARGET_REL="resources_servers/math_with_judge/app.py"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    exit 1
fi

_apply_one_dir() {
    local pod="$1"
    local dir="$2"

    if ! kubectl exec -n "$NAMESPACE" "$pod" -- test -f "$dir/$TARGET_REL"; then
        echo "  [$dir] target $TARGET_REL not present — skipping"
        return 0
    fi

    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        grep -q "$MARKER" "$dir/$TARGET_REL" 2>/dev/null; then
        echo "  [$dir] already patched (marker '$MARKER' present) — skipping"
        return 0
    fi

    echo "  [$dir] strict git apply..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && git apply --check /tmp/math_last_box_only.patch 2>/dev/null"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && git apply /tmp/math_last_box_only.patch"
        echo "  [$dir] strict git apply OK"
        return 0
    fi

    echo "  [$dir] strict failed (version drift). Trying patch --fuzz=3..."
    if kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "cd '$dir' && patch -p1 --dry-run --fuzz=3 < /tmp/math_last_box_only.patch >/dev/null 2>&1"; then
        kubectl exec -n "$NAMESPACE" "$pod" -- \
            bash -c "cd '$dir' && patch -p1 --fuzz=3 < /tmp/math_last_box_only.patch"
        echo "  [$dir] fuzzy patch OK"
        return 0
    fi

    echo "  [$dir] BOTH STRATEGIES FAILED — pod version too far from patch baseline."
    echo "  [$dir] Current _verify_in_worker body on pod (first 30 lines):"
    kubectl exec -n "$NAMESPACE" "$pod" -- \
        bash -c "awk '/def _verify_in_worker/,/^def /' '$dir/$TARGET_REL'" \
        | head -30 | sed 's/^/    /'
    return 1
}

for POD in "$@"; do
    echo "=== $POD ==="

    echo "[1/3] Copying patch to $POD:/tmp/math_last_box_only.patch"
    kubectl cp -n "$NAMESPACE" "$PATCH_FILE" "$POD:/tmp/math_last_box_only.patch"

    echo "[2/3] Applying to each Gym checkout on the pod"
    failed=0
    for DIR in $GYM_DIRS; do
        if ! _apply_one_dir "$POD" "$DIR"; then
            failed=1
        fi
    done
    if [[ $failed -eq 1 ]]; then
        echo "  -> $POD partially patched. See errors above."
        continue
    fi

    echo "[3/3] Running tests in the first patched checkout"
    for DIR in $GYM_DIRS; do
        if kubectl exec -n "$NAMESPACE" "$POD" -- \
            test -f "$DIR/resources_servers/math_with_judge/tests/test_last_box_only.py"; then
            kubectl exec -n "$NAMESPACE" "$POD" -- \
                bash -c "cd '$DIR' && python -m pytest resources_servers/math_with_judge/tests/test_last_box_only.py -q 2>&1 | tail -5"
            break
        fi
    done

    echo "  -> $POD patched successfully."
    echo
done

echo "Done."
echo
echo "NEXT (manual step — do NOT automate):"
echo "  Restart the math server_launcher process so the new code loads."
echo "  Find it with:"
echo "    kubectl exec -n $NAMESPACE <pod> -- bash -c \"ps auxf | grep -E 'server_launcher|math_with_judge' | grep -v grep\""
echo "  Then SIGTERM the math server_launcher PID (safe — child of start_ray_cluster)."
echo "  Re-launch with the same env vars the training recipe uses."
echo
echo "Roll back per dir:"
echo "  kubectl exec -n $NAMESPACE <pod> -- bash -c 'cd /shared/dev/shuowei/niletron/Gym && git apply --reverse /tmp/math_last_box_only.patch'"
echo
echo "Disable at runtime without unpatching (env-var escape hatch):"
echo "  Launch the math server with MATH_KEEP_ONLY_LAST_BOX=0"
